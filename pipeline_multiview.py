from argparse import ArgumentParser
import json
import sys
from PIL import Image
import glob
import os
from tqdm import tqdm
import numpy as np
import random
import copy
from PIL import Image
import json
from stable_diffusion_cn import StableDiffusionControlNet
import numpy as np
from PIL import Image
import torch
import os
from time import gmtime, strftime
from omegaconf import OmegaConf
from config.config import Config
from utils import CrossFrameAttnProcessor, tensor_erode, tensor_dilate
from warping import prepare_warp_and_occ, bilinear_sample_function
import torch.nn.functional as F


class SDpipeline():
    def __init__(self, config):
        # init parameters
        self.config = config
        self.expname = config.expname
        self.device = config.model.device
        if not config.model.use_lora:
            lora_weight_list = None
        else:
            lora_weight_list = [[lora_path, weight] for lora_path, weight in zip(config.model.lora_list, config.model.lora_weight)]
        self.init_sd_model(self.device, config.model.cache_dir, 
                            config.model.unet_path, config.model.vae_path,
                            config.model.cn_list, lora_weight_list)
        
        # params
        self.guidance = config.model.guidance
        self.seed = config.systems.seed
        np.random.seed(self.seed)
        self.constant_C = config.model.constant_C
        self.constant_S = config.model.constant_S

        self.dep_to_255_func = lambda x: (x - np.min(x))/(np.max(x) - np.min(x)) * 255
        self.bg_color = config.model.bg_color

        self.num_inference_step = config.model.num_inference_step
        self.proportion = config.model.proportion
        self.controlnet_weight_start = config.model.controlnet_weight_start
        self.controlnet_weight_end = config.model.controlnet_weight_end
        self.batch = config.model.batch
        self.enhancement_scale = config.model.enhancement_scale # scale variance
        self.original_noise_freq = config.model.original_noise_freq
        self.freq_count = 0.  # for auxiliary
        self.use_optional_fixed_views = config.model.use_optional_fixed_views # fix side views
        
        self.the_ref_view = config.model.the_ref_view

        # latent optim
        self.open_optimization = config.model.open_optimization
        self.optim_lr = config.model.optim_lr
        key_optim_steps = config.model.key_optim_steps
        key_optim_steps = [int(stp * self.num_inference_step) for stp in key_optim_steps]
        self.key_optim_steps = key_optim_steps
        print("key_optim_stpes: ", key_optim_steps)
        self.key_epoch_nums = config.model.key_epoch_nums
        self.epoch_nums = config.model.epoch_nums
        self.optim_interval = config.model.optim_interval
        pass

    def init_sd_model(self, device, cache_dir, unet_path, vae_path, cn_list, lora_list=None):
        use_attn_feature_injection = not self.config.model.close_attn
        with torch.no_grad():
            print("initialize sd ..")
            self.sd = StableDiffusionControlNet(device, unet_path=unet_path, vae_path=vae_path, cn_list=cn_list, cache_dir=cache_dir, lora_list=lora_list,
                use_attn_feature_injection=use_attn_feature_injection, attn_processor=CrossFrameAttnProcessor(unet_chunk_size=2, sparse_attn_type='FFC', sparse_attn_switch=True))
            print("scheduler type: ", self.sd.scheduler_type)
        
    def load_dataset(self, data_path, proj_path, data_json, main_prompt):
        frames = data_json['frames']
        extrs = []
        depths = []
        normalized_depths = []
        ops = []
        prompts = []
        masks = []
        nmls = []
        local_nmls = []
        # rgbs = []
        addtional_prompt = self.config.model.addtional_prompt

        for idx, frame in enumerate(frames):
            transform_matrix = frame['transform_matrix'] # this is gl extr
            extr = np.array(transform_matrix)
            extrs.append(extr[None, ...])

            

            depth = np.array(Image.open(os.path.join(data_path, proj_path, frame['depth_file_path']))).astype(np.float32) / 1000. # to meter
            depths.append(depth[None, ...])

            normalized_depth = self.dep_to_255_func(depth).astype(np.uint8)
            normalized_depths.append(Image.fromarray(normalized_depth))

            ops.append(Image.open(os.path.join(data_path, proj_path, frame['op_file_path'])))

            part_prompt = ", fullbody" if idx < len(frames) / 2 else ", upperbody"
            prompts.append(main_prompt + addtional_prompt + part_prompt + frame['view_related_prompt']) #

            mask = Image.open(os.path.join(data_path, proj_path, frame['mask_path']))
            masks.append(np.array(mask)[None, ...])

            if 'global_normal_path' in frame.keys():
                nml = Image.open(os.path.join(data_path, proj_path, frame['global_normal_path']))
                nml = np.array(nml).astype(np.float32) / 127.5 - 1.
                local_nml = np.dot(nml, extr[:3,:3])
                local_nml = Image.fromarray((local_nml*127.5 + 127.5).astype(np.uint8))
                local_nmls.append(local_nml)
                nmls.append(nml[None, ...])

        negative_prompt_list = [
            self.config.model.negative_prompt # 
            ] * len(prompts)
        return extrs, depths, normalized_depths, ops, prompts, negative_prompt_list, masks, nmls, local_nmls
    
    def prepare_warping_dict(self, dataset_num, depths_tensor, masks_tensor, nmls_tensor, intr_matrix_tensor, 
                            extr_tensor_cv):
        eps = 1e-3
        device = self.device
        warp_dict = {}
        # tar
        for idx_i in range(dataset_num):
            depth_tgt = depths_tensor[idx_i:idx_i+1].to(device)
            mask_tgt = masks_tensor[idx_i:idx_i+1].to(device)
            nml_tgt = nmls_tensor[idx_i:idx_i+1].to(device)
            intrinc_tgt = intr_matrix_tensor.unsqueeze(0).to(device)
            extrinc_tgt = extr_tensor_cv[idx_i:idx_i+1].to(device)
            _, _, _, _, xyz_img_global = prepare_warp_and_occ(depth_tgt, depth_tgt, intrinc_tgt, intrinc_tgt,
                extrinc_tgt, extrinc_tgt, occ_thres=0.01, return_xyz_img_global=True)
            xyz_img_global = xyz_img_global.permute(0,2,3,1)
            tgt_xyz_dir = xyz_img_global - extrinc_tgt[:, :3, 3]
            tgt_xyz_dir = torch.nn.functional.normalize(tgt_xyz_dir, dim=3)
            cosine_with_normal = ((-tgt_xyz_dir) * nml_tgt.permute(0,2,3,1)).sum(dim=3, keepdim=True) * self.constant_S + self.constant_C
            cosine_with_normal[~mask_tgt.permute(0,2,3,1)] = 1.
            cosine_with_normal[cosine_with_normal <= 0.] = eps
            warp_dict[f'{idx_i}to{idx_i}_vis_mask'] = cosine_with_normal.cpu()

            # src
            for idx_j in range(dataset_num):
                if idx_j == idx_i: continue
                depth_tgt = depths_tensor[idx_i:idx_i+1].to(device)
                mask_tgt = masks_tensor[idx_i:idx_i+1].to(device)
                color_src = depths_tensor[idx_j:idx_j+1].to(device)
                intrinc_src = intr_matrix_tensor.unsqueeze(0).to(device)
                intrinc_tgt = intrinc_src
                extrinc_tgt = extr_tensor_cv[idx_i:idx_i+1].to(device)
                extrinc_src = extr_tensor_cv[idx_j:idx_j+1].to(device)

                nml_tgt = nmls_tensor[idx_i:idx_i+1].to(device)
                nml_tgt = torch.nn.functional.normalize(nml_tgt, dim=1)

                xyz, occ, _, _, xyz_img_global = prepare_warp_and_occ(depth_tgt, color_src, intrinc_src, intrinc_tgt,
                                                                extrinc_src, extrinc_tgt, occ_thres=0.01, return_xyz_img_global=True)

                # 
                xyz_copy = xyz.reshape(1, 512, 512, 3)
                xy_mask = (xyz_copy[...,0:1]<0) + (xyz_copy[...,0:1]>512) + (xyz_copy[...,1:2]<0) + (xyz_copy[...,1:2]>512)
                xy_mask = ~xy_mask.bool() # 1

                vis_mask = torch.logical_and(~occ, mask_tgt.permute(0,2,3,1)) # 2,4
                vis_mask = torch.logical_and(vis_mask, xy_mask)

                # get src cam to tgt 3d point dir
                xyz_img_global = xyz_img_global.permute(0,2,3,1)
                src2tgt_xyz_dir = xyz_img_global - extrinc_src[:, :3, 3]
                src2tgt_xyz_dir = torch.nn.functional.normalize(src2tgt_xyz_dir, dim=3)

                cosine_with_normal = ((-src2tgt_xyz_dir) * nml_tgt.permute(0,2,3,1)).sum(dim=3, keepdim=True)
                cosine_mask = cosine_with_normal >= np.cos(np.pi * 80 / 180) # ? degree
                # 

                # corrupt
                corrupted_cosine_mask = tensor_dilate(tensor_erode((cosine_mask).float().permute(0,3,1,2))) # B, C, H, W # 3
                corrupted_cosine_mask = torch.logical_and(corrupted_cosine_mask, vis_mask.permute(0,3,1,2))
                
                # update
                final_mask = corrupted_cosine_mask.float() * (cosine_with_normal.permute(0,3,1,2) * self.constant_S + self.constant_C)
                warp_dict[f"{idx_j}to{idx_i}_xyz"] = xyz.cpu()
                warp_dict[f'{idx_j}to{idx_i}_vis_mask'] = final_mask.permute(0,2,3,1).cpu()
                vis_percent = corrupted_cosine_mask.float().sum() / masks_tensor[idx_i:idx_i+1].float().sum()
                warp_dict[f'{idx_j}to{idx_i}_vis_percent'] = vis_percent

        return warp_dict

    def get_two_dict(self, data_type=None):
        assert (self.the_ref_view in [6, 8, 12, 0]), "invalid ref view"
        if self.the_ref_view == 6:
            choosed_src_dict = {
                0: [5,1],
                1: [0,2],
                2: [1,3],
                3: [2,4],
                4: [3,5],
                5: [4,0],
                6: [11,7],
                7: [6,8],
                8: [7,9],
                9: [8,10],
                10: [9,11],
                11: [10,6]
            }
            print("choosed_src_dict")
            print(choosed_src_dict)

            replace_src_dict = \
            {0: 6, 1: 7, 2: 8, 3: 9, 4: 10, 5: 11}
            print("replace_src_dict")
            print(replace_src_dict)
            self.optim_fix_views = [0, 3, 6, 9]
            self.optim_optional_fix_views = []
            return choosed_src_dict, replace_src_dict
        if self.the_ref_view == 8:
            choosed_src_dict = {
            0: [7,1],
            1: [0,2],
            2: [1,3],
            3: [2,4],
            4: [3,5],
            5: [4,6],
            6: [5,7],
            7: [6,0],
            8: [15,9],
            9: [8,10],
            10: [9,11],
            11: [10,12],
            12: [11,13],
            13: [12,14],
            14: [13,15],
            15: [14,8]
            }
            print("choosed_src_dict")
            print(choosed_src_dict)

            replace_src_dict = \
            {0: 8, 1: 9, 2: 10, 3: 11, 4: 12, 5: 13, 6: 14, 7: 15}
            print("replace_src_dict")
            print(replace_src_dict)
            self.optim_fix_views = [0, 4, 8, 12]
            self.optim_optional_fix_views = [2, 6, 10, 14]
            return choosed_src_dict, replace_src_dict
        if self.the_ref_view == 12:
            choosed_src_dict = \
            {0: [10,11,1,2],
             1: [11,0,2,3],
             2: [0,1,3,4],
             3: [1,2,4,5],
             4: [2,3,5,6],
             5: [3,4,6,7],
             6: [4,5,7,8],
             7: [5,6,8,9],
             8: [6,7,9,10],
             9: [7,8,10,11],
             10: [8,9,11,0],
             11: [9,10,0,1],
             12: [22,23,13,14],
             13: [23,12,14,15],
             14: [12,13,15,16],
             15: [13,14,16,17],
             16: [14,15,17,18],
             17: [15,16,18,19],
             18: [16,17,19,20],
             19: [17,18,20,21],
             20: [18,19,21,22],
             21: [19,20,22,23],
             22: [20,21,23,12],
             23: [21,22,12,13],}
            
            print("choosed_src_dict")
            print(choosed_src_dict)

            replace_src_dict = \
            {0: 12, 1: 13, 2: 14, 3: 15, 4: 16, 5: 17, 6: 18, 7: 19, 8: 20, 9: 21, 10: 22, 11: 23}
            print("replace_src_dict")
            print(replace_src_dict)
            self.optim_fix_views = [0, 6, 12, 18]
            self.optim_optional_fix_views = [3, 9, 15, 21]
            return choosed_src_dict, replace_src_dict
        if self.the_ref_view == 0 and data_type == "render_90s":
            choosed_src_dict = {
                0: [3,1],
                1: [0,2],
                2: [1,3],
                3: [2,0],
            }
            print("choosed_src_dict")
            print(choosed_src_dict)

            replace_src_dict = {}
            print("replace_src_dict")
            print(replace_src_dict)
            self.optim_fix_views = [0, 2]
            self.optim_optional_fix_views = []
            return choosed_src_dict, replace_src_dict
        if self.the_ref_view == 0 and data_type == "render_45s":
            choosed_src_dict = {
                0: [7,1],
                1: [0,2],
                2: [1,3],
                3: [2,4],
                4: [3,5],
                5: [4,6],
                6: [5,7],
                7: [6,0],
            }
            print("choosed_src_dict")
            print(choosed_src_dict)

            replace_src_dict = {}
            print("replace_src_dict")
            print(replace_src_dict)
            self.optim_fix_views = [0, 4]
            self.optim_optional_fix_views = [2, 6]
            return choosed_src_dict, replace_src_dict
        raise NotImplementedError

    def get_controlnet_weight(self, decay):
        # decay in 0-1
        if decay is None:
            return self.controlnet_weight_start
        weight_ = [weight_s * decay + weight_e * (1-decay) for weight_s, weight_e in zip(self.controlnet_weight_start, self.controlnet_weight_end)]
        if len(weight_) == 1: weight_ = weight_[0]
        return weight_
    
    def to_img(self, tensor):
        tensor_np = (tensor[:,:3].detach().cpu().permute(0,2,3,1).numpy().clip(0,1) * 255).round().astype('uint8')
        return tensor_np

    def process(self, data_path, proj_name, main_prompt):
        # first get the json
        data_json_path = os.path.join(data_path, proj_name, "transforms.json")
        with open(data_json_path, 'r') as f:
            data_json = json.load(f)
        frames = data_json["frames"]
        
        width, height = data_json['w'], data_json['h']

        intr_matrix = np.identity(3)
        intr_matrix[0,0] = data_json['fl_x']
        intr_matrix[1,1] = data_json['fl_y']
        intr_matrix[0,2] = data_json['cx']
        intr_matrix[1,2] = data_json['cy']

        print("loading dataset ..")
        extrs, depths, normalized_depths, ops, prompts, negative_prompt_list, masks, nmls, local_nmls = self.load_dataset(data_path, proj_name, data_json, main_prompt)

        dataset_num = len(extrs)
        ## validate ##
        assert (dataset_num == 16 and self.the_ref_view == 8) or \
        (dataset_num == 12 and self.the_ref_view == 6) or \
        (dataset_num == 24 and self.the_ref_view == 12) or \
        (dataset_num == 4 and self.the_ref_view == 0) or \
        (dataset_num == 8 and self.the_ref_view == 0 and proj_name == "render_45s"), "dataset or config are not correct"

        # dataset to tensor (cpu)
        do_classifier_free_guidance = self.guidance > 1
        prompt_embeds = self.sd._encode_prompt(prompts, dataset_num, do_classifier_free_guidance, negative_prompt_list, None, None)
        image_tensor_controlnet_op = self.sd.prepare_image(ops, 512, 512, self.sd.controlnet.dtype, do_classifier_free_guidance).cpu()
        image_tensor_controlnet_dep = self.sd.prepare_image(normalized_depths, 512, 512, self.sd.controlnet.dtype, do_classifier_free_guidance).cpu()
        image_tensor_controlnet_local_nml = self.sd.prepare_image(local_nmls, 512, 512, self.sd.controlnet.dtype, do_classifier_free_guidance).cpu()
        generator_list = [torch.Generator(device=self.device).manual_seed(self.seed * i) for i in range(dataset_num)]

        intr_matrix_tensor = torch.from_numpy(intr_matrix).float()  # .to(device)
        extrs_tensor = torch.from_numpy(np.concatenate(extrs)).float()
        extr_tensor_cv = extrs_tensor.clone()
        extr_tensor_cv[:,:,1:3] *= -1
        depths_tensor = torch.from_numpy(np.concatenate(depths)).unsqueeze(1)
        masks_tensor = torch.from_numpy(np.concatenate(masks)).unsqueeze(1).bool()
        nmls_tensor = torch.from_numpy(np.concatenate(nmls)).permute(0,3,1,2)

        # prepare for warping
        print("prepare_warping_dict ..")
        warp_dict = self.prepare_warping_dict(dataset_num, depths_tensor, masks_tensor, nmls_tensor, intr_matrix_tensor, 
                                            extr_tensor_cv)
        
        choosed_src_dict, replace_src_dict = self.get_two_dict(proj_name)

        # create result dir and save config json
        datatime = strftime("%Y-%m-%d_%H%M%S", gmtime())
        result_dir = os.path.join(data_path, "results", self.expname + '_' + datatime)
        os.makedirs(result_dir, exist_ok=True)
        print(f"results writing to {result_dir}")
        # log function
        def log_vis_origins(vis_origin, step):
            if step == self.num_inference_step -1:
                sub_dir = os.path.join(result_dir, "final")
            elif step % (self.num_inference_step // 20) == 0 :
                # return
                sub_dir = os.path.join(result_dir, 'vis', f"{step}")
            else:
                return
            os.makedirs(sub_dir, exist_ok=True)
            for i in range(vis_origin.shape[0]):
                img = vis_origin[i]
                file_name = os.path.split(frames[i]['file_path'])[-1]
                sav_file = os.path.join(sub_dir, file_name)
                Image.fromarray(img).save(sav_file)
        
        frames_copy = []
        for i in range(dataset_num):
            file_name = os.path.split(frames[i]['file_path'])[-1]
            new_frame = {"file_path": os.path.join(".", "final", file_name)}
            new_frame['transform_matrix'] = frames[i]['transform_matrix']
            frames_copy.append(new_frame)
        data_json['frames'] = frames_copy
        print("writing transforms.json ..")
        with open(os.path.join(result_dir, "transforms.json"), 'w') as f:
            json.dump(data_json, f, indent=2)
        print("writing config.json ..")
        self.config.main_prompt = main_prompt
        OmegaConf.save(self.config, os.path.join(result_dir, "config.json"))

        #
        last_latents = None
        optimized_2d = None

        # functions
        def batch_run_sd_step(latents, t, batch, use_space=None, cn_weight_decay=None, step=-1):
            
            # 
            vis_origins=[]
            pred_origin_list = []
            pred_origin_latent_list = []
            pred_noise_unet_list = []
            for i in range(0, dataset_num, batch):
                index_list = list(range(i, min(i+batch, dataset_num)))
                index_list_ex = index_list + [e + dataset_num for e in index_list]

                # 
                extended = False
                batch_size = batch
                if index_list[0] != self.the_ref_view and (not self.config.model.close_attn): #
                    index_list = [self.the_ref_view] + index_list
                    index_list_ex = index_list + [e + dataset_num for e in index_list]
                    extended = True
                    batch_size += 1

                batch_prompts = prompt_embeds[index_list_ex].to(self.device)
                batch_img_op = image_tensor_controlnet_op[index_list_ex].to(self.device)
                batch_img_dep = image_tensor_controlnet_dep[index_list_ex].to(self.device)
                batch_img_local_nml = image_tensor_controlnet_local_nml[index_list_ex].to(self.device)
                batch_generator_list = [generator_list[e] for e in index_list]
                batch_mask = masks_tensor[index_list].to(self.device)
                # rand latents
                batch_latents = latents[index_list]

                control_batch = []
                for cn_type in self.config.model.cn_list:
                    if "openpose" in cn_type: control_batch.append(batch_img_op)
                    if "depth" in cn_type: control_batch.append(batch_img_dep)
                    if "normalbae" in cn_type: control_batch.append(batch_img_local_nml)
                if len(self.config.model.cn_list) == 1: control_batch = control_batch[0]
                controlnet_conditioning_scale = self.get_controlnet_weight(cn_weight_decay)

                pred_noise, pred_origin = self.sd.latent2pred(batch_prompts, batch_latents, control_batch,
                t, batch_size, generator=batch_generator_list, do_classifier_free_guidance=True, \
                    controlnet_conditioning_scale=controlnet_conditioning_scale,)
                if extended and (not self.config.model.close_attn):
                    pred_noise = pred_noise[1:]
                    pred_origin = pred_origin[1:]
                    batch_mask = batch_mask[1:]
                    batch_latents = batch_latents[1:]
                pred_noise_unet_list.append(pred_noise)

                # for visualization
                pred_origin_decode = self.sd.decode_latents(pred_origin).to(self.sd.device)
                vis_origins.append((pred_origin_decode.cpu().permute(0,2,3,1).numpy().clip(0,1) * 255).round().astype('uint8'))

                # mask the pred_origin decode to remove background
                batch_mask_ = batch_mask.repeat(1, pred_origin_decode.shape[1], 1, 1)
                pred_origin_decode[~batch_mask_] = self.bg_color
                pred_origin_list.append(pred_origin_decode)
                pred_origin_latent_list.append(pred_origin)
                continue

            vis_origins_cat = np.concatenate(vis_origins)
            log_vis_origins(vis_origins_cat, infer_step_i)
            pred_noise_unet_list = torch.cat(pred_noise_unet_list)
            pred_origin_list = None if len(pred_origin_list) == 0 else torch.cat(pred_origin_list)
            pred_origin_latent_list = None if len(pred_origin_latent_list) == 0 else torch.cat(pred_origin_latent_list)
            return pred_noise_unet_list, pred_origin_list, pred_origin_latent_list

        def gathering_2D_revise(pred_origin_list, t, max_src_view=None, use_space=None, step=-1):
            # 
            pred_origin_list_new = []
            if self.config.model.paper_mat:
                mat_sav_path = os.path.join(result_dir, 'vis', f"gathering_2D_{step}")
                os.makedirs(mat_sav_path, exist_ok=True)
            # 
            for i in range(len(pred_origin_list)): # as tgt
                pred_origin_viewi = F.interpolate(self.sd.encode_imgs(pred_origin_list[i:i+1]), (512, 512), mode='nearest') # scale it in 512
                #
                pred_origin_viewi_latents_sum = torch.zeros_like(pred_origin_viewi)
                pred_origin_viewi_latents_sum += pred_origin_viewi #

                if self.config.model.paper_mat:
                    pred_origin_img_np = self.to_img(pred_origin_list[i:i+1])
                    Image.fromarray(pred_origin_img_np[0]).save(os.path.join(mat_sav_path, f"target_ori_{i}.png"))
                    pred_origin_encode_np = self.to_img(pred_origin_viewi)
                    Image.fromarray(pred_origin_encode_np[0]).save(os.path.join(mat_sav_path, f"target_ori_encode_{i}.png"))

                vis_sum = torch.zeros_like(pred_origin_viewi)
                vis2_sum = torch.zeros_like(vis_sum)
                vis_tgt = warp_dict[f'{i}to{i}_vis_mask'].permute(0, 3, 1, 2).repeat(1,pred_origin_viewi.shape[1],1,1).float().to(self.device)
                vis_sum += vis_tgt #
                vis2_sum += vis_tgt ** 2 #

                choosed_src_list = choosed_src_dict[i].copy()
                # 
                if max_src_view is not None:
                    np.random.shuffle(choosed_src_list)
                    choosed_src_list = choosed_src_list[:min(len(choosed_src_list), max_src_view)]
                pred_origin_latents_warped = []
                for j in choosed_src_list: #  as src
                    warp_xyz = warp_dict[f'{j}to{i}_xyz'].to(self.device)
                    pred_origin_viewj = pred_origin_list[j:j+1]
                    pred_origin_viewj_warp = F.interpolate(self.sd.encode_imgs(bilinear_sample_function(warp_xyz, pred_origin_viewj, 'border')), (512, 512), mode='nearest') # b,c,h,w
                    vis_ = warp_dict[f'{j}to{i}_vis_mask'].permute(0, 3, 1, 2).to(self.device) # b,c,h,w
                    vis_ = vis_.repeat(1, pred_origin_viewj_warp.shape[1], 1, 1).float()
                    pred_origin_latents_warped.append(pred_origin_viewj_warp)
                    pred_origin_viewi_latents_sum += pred_origin_viewj_warp * vis_ # 
                    vis_sum += vis_ # 
                    vis2_sum += vis_ ** 2 # 

                # mean
                pred_origin_viewi_mean = pred_origin_viewi_latents_sum / vis_sum
                # 

                ############### 
                if self.config.model.enable_enhancement:
                    strength = (vis_sum / torch.sqrt(vis2_sum) * self.enhancement_scale - 1)[:,:1,...] #
                    pred_origin_viewi_n = (pred_origin_viewi - pred_origin_viewi_mean) * (1 + strength) + pred_origin_viewi_mean

                    pred_origin_viewi_latents_sum_n = torch.zeros_like(pred_origin_viewi_n)
                    pred_origin_viewi_latents_sum_n += pred_origin_viewi_n # 

                    for idx, j in enumerate(choosed_src_list): #  as src
                        pred_origin_viewj_warp = pred_origin_latents_warped[idx]
                        pred_origin_viewj_warp_n = (pred_origin_viewj_warp - pred_origin_viewi_mean) * (1 + strength) + pred_origin_viewi_mean
                        vis_ = warp_dict[f'{j}to{i}_vis_mask'].permute(0, 3, 1, 2).to(self.device) # b,c,h,w
                        vis_ = vis_.repeat(1, pred_origin_viewj_warp_n.shape[1], 1, 1).float()
                        pred_origin_viewi_latents_sum_n += pred_origin_viewj_warp_n * vis_ # 
                    pred_origin_viewi_mean_n = pred_origin_viewi_latents_sum_n / vis_sum
                else:
                    pred_origin_viewi_mean_n = pred_origin_viewi_mean
                ############### 

                if i in replace_src_dict.keys():
                    j = replace_src_dict[i] # src for replacement
                    warp_xyz = warp_dict[f'{j}to{i}_xyz'].to(self.device)
                    pred_origin_viewj = pred_origin_list[j:j+1]
                    pred_origin_viewj_warp = F.interpolate(self.sd.encode_imgs(bilinear_sample_function(warp_xyz, pred_origin_viewj, 'border')), (512, 512), mode='nearest') # b,c,h,w
                    vis_ = warp_dict[f'{j}to{i}_vis_mask'].permute(0, 3, 1, 2).to(self.device) # b,c,h,w
                    vis_ = vis_.repeat(1, pred_origin_viewj_warp.shape[1], 1, 1).bool()
                    # replace
                    pred_origin_viewi_mean_n[vis_] = pred_origin_viewj_warp[vis_]

                pred_origin_viewi_mean_64_n = F.interpolate(pred_origin_viewi_mean_n, (64, 64), mode='bilinear') # to 64
                pred_origin_list_new.append(pred_origin_viewi_mean_64_n)
                
            optimized_2d = torch.cat(pred_origin_list_new)
            return optimized_2d


        
            
            return latents_optim.detach().clone()

        def gathering_latents_optimization(latents, epoch_num=1, progress=0, step=-1):
            latents_optim = latents.clone()
            latents_optim.requires_grad = True
            optimizer = torch.optim.Adam([latents_optim], lr=self.optim_lr)
            #
            rand_list = list(range(len(latents)))
            np.random.shuffle(rand_list)
            fixed_views = self.optim_fix_views # 
            optional_fixed_views = [] if progress < self.use_optional_fixed_views else self.optim_optional_fix_views # 
            print(f"starting optimization, fix view is {fixed_views}")
            print(f"optional fix view is {optional_fixed_views}")

            # 
            for epoch in range(epoch_num):
                mean_loss = 0.
                loss_count = 0
                for tgt_id in rand_list:
                    choosed_src_list = choosed_src_dict[tgt_id].copy()
                    np.random.shuffle(choosed_src_list)
                    for src_id in choosed_src_list:
                        loss = 0.
                        latent_tgt = latents_optim[tgt_id:tgt_id+1]
                        if tgt_id in fixed_views: latent_tgt = latent_tgt.detach()
                        if tgt_id in optional_fixed_views and src_id not in fixed_views: latent_tgt = latent_tgt.detach()
                        latent_tgt_decode = self.sd.decode_latents(latent_tgt)
                        
                        latent_src = latents_optim[src_id:src_id+1]
                        if src_id in fixed_views: latent_src = latent_src.detach()
                        if src_id in optional_fixed_views and tgt_id not in fixed_views: latent_src = latent_src.detach()
                        latent_src_decode = self.sd.decode_latents(latent_src)

                        warp_xyz = warp_dict[f"{src_id}to{tgt_id}_xyz"].to(self.device)
                        latent_src_decode_warp = bilinear_sample_function(warp_xyz, latent_src_decode, 'border') # b,c,h,w
                        vis_ = warp_dict[f'{src_id}to{tgt_id}_vis_mask'].permute(0, 3, 1, 2).to(self.device).repeat(1, latent_src_decode_warp.shape[1], 1, 1) # b,c,h,w
                        vis_ = vis_ > 0
                        loss += torch.mean(((latent_tgt_decode - latent_src_decode_warp)**2)[vis_])

                        optimizer.zero_grad()
                        loss.backward()
                        optimizer.step()
                        # print(f"{loss}")

                        mean_loss += loss
                        loss_count += 1

                    if tgt_id in replace_src_dict.keys():
                        loss = 0.
                        latent_tgt = latents_optim[tgt_id:tgt_id+1]
                        latent_tgt_decode = self.sd.decode_latents(latent_tgt)

                        src_id = replace_src_dict[tgt_id]
                        latent_src = latents_optim[src_id:src_id+1]
                        latent_src = latent_src.detach()
                        latent_src_decode = self.sd.decode_latents(latent_src)

                        warp_xyz = warp_dict[f"{src_id}to{tgt_id}_xyz"].to(self.device)
                        latent_src_decode_warp = bilinear_sample_function(warp_xyz, latent_src_decode, 'border') # b,c,h,w
                        vis_ = warp_dict[f'{src_id}to{tgt_id}_vis_mask'].permute(0, 3, 1, 2).to(self.device).repeat(1, latent_src_decode_warp.shape[1], 1, 1) # b,c,h,w
                        vis_ = vis_ > 0
                        loss += torch.mean(((latent_tgt_decode - latent_src_decode_warp)**2)[vis_])

                        optimizer.zero_grad()
                        loss.backward()
                        optimizer.step()
                        # print(f"{loss}")

                        mean_loss += loss
                        loss_count += 1

                mean_loss /= loss_count
                print(f"epoch: {epoch}, loss: {mean_loss}")
            
            return latents_optim.detach().clone()

        batch = self.batch
        self.sd.scheduler.set_timesteps(self.num_inference_step, device=self.sd.device)
        timesteps = self.sd.scheduler.timesteps

        extr_func = lambda x: 1- (1 - x / timesteps[0])**4
        _noise_scale_func = lambda x: min(float(1 - extr_func(x))*5, float(extr_func(x)) / 3)
        # with torch.no_grad():
        pbar = tqdm(range(self.num_inference_step))
        for infer_step_i in pbar:
            t = timesteps[infer_step_i]
            if infer_step_i == 0:
                latents = self.sd.prepare_latents(dataset_num, self.sd.unet.config.in_channels, 512, 512, dtype=prompt_embeds.dtype,
                device=torch.device(self.sd.device), generator=generator_list)
                # 
                if self.open_optimization:
                    latents = gathering_latents_optimization(latents, 20, step=infer_step_i)
                last_latents = latents

                pred_noise_unet_list, pred_origin_list, pred_origin_latent_list = batch_run_sd_step(latents, t, batch, cn_weight_decay=1-float(infer_step_i)/self.num_inference_step, step=infer_step_i)

                _noise_scale = 1

                optimized_2d = pred_origin_latent_list
                continue
            else:
                last_t = timesteps[infer_step_i-1]
                # 
                new_noise = []
                for i in range(0, dataset_num, batch):
                    index_list = list(range(i, min(i+batch, dataset_num)))
                    tmp_new_origins = optimized_2d[index_list]
                    new_noise_ = self.sd.origin2noise(last_latents[index_list], tmp_new_origins, last_t) * _noise_scale
                    new_noise.append(new_noise_)
                new_noise = torch.cat(new_noise)
                # 
                final_noise = new_noise
                # 
                latents = self.sd.get_prev_latents(final_noise, last_t, last_latents, generator_list)
                # optimize latents
                if self.open_optimization and (infer_step_i in self.key_optim_steps or infer_step_i % self.optim_interval == 0): # 
                    epoch_num = self.key_epoch_nums if infer_step_i in self.key_optim_steps else self.epoch_nums #
                    latents = gathering_latents_optimization(latents, epoch_num, float(infer_step_i)/self.num_inference_step, step=infer_step_i)
                
                last_latents = latents

                pred_noise_unet_list, pred_origin_list, pred_origin_latent_list = batch_run_sd_step(latents, t, batch, cn_weight_decay=1-float(infer_step_i)/self.num_inference_step, step=infer_step_i)
                
                if infer_step_i < self.num_inference_step * self.proportion:
                    _noise_scale = 1
                    optimized_2d = pred_origin_latent_list
                    continue
                
                # eval
                if self.config.model.close_original_noise:
                    _noise_scale = 1 #
                    optimized_2d = gathering_2D_revise(pred_origin_list, t, max_src_view=None, step=infer_step_i)
                    continue
                if self.config.model.close_guide_noise:
                    _noise_scale = 1
                    optimized_2d = pred_origin_latent_list
                    continue

                # mix original and guide noise
                self.freq_count += 1.
                if self.freq_count >= self.original_noise_freq: # 
                    _noise_scale = 1
                    optimized_2d = pred_origin_latent_list
                    self.freq_count -= self.original_noise_freq
                    continue
                else: #
                    _noise_scale = 1 #
                    optimized_2d = gathering_2D_revise(pred_origin_list, t, max_src_view=None, step=infer_step_i)


if __name__ == "__main__":
    base_cfg = OmegaConf.structured(Config())
    cli_cfg = OmegaConf.from_cli()
    base_yaml_path = base_cfg.get("config", None)
    yaml_path = cli_cfg.get("config", None)
    if yaml_path is not None:
        yaml_cfg = OmegaConf.load(yaml_path)
    elif base_yaml_path is not None:
        yaml_cfg = OmegaConf.load(base_yaml_path)
    else:
        yaml_cfg = OmegaConf.create()
    args = OmegaConf.merge(base_cfg, yaml_cfg, cli_cfg)  # merge configs

    print(args)

    tar_dir = args.data_dir # 

    # get all dirs
    all_dirs = sorted(glob.glob(os.path.join(tar_dir, "*")))

    # get prompts list
    if args.fix_prompt is not None:
        print(f"fixing prompt: {args.fix_prompt}")
    else:
        assert (args.prompts_list is not None), "lacking prompts list"
        with open(args.prompts_list, 'r') as f:
            print("loading prompts list ..")
            prompts = json.load(f)[args.key]
    
    # index range
    index_range = None
    if args.index_range is not None:
        index_range_str = args.index_range
        index_range_str = index_range_str.split(',')
        index_range = []
        for part in index_range_str:
            if '-' in part:
                start = part.split('-')[0]
                end = part.split('-')[-1]
                index_range += list(range(int(start), int(end) + 1))
            else:
                index_range += [int(part)]
        print("sprcified index range: ", index_range)

    # 
    sdp = SDpipeline(args)

    for adir in all_dirs:
        # print(adir)
        dir_name = os.path.basename(adir)
        print(dir_name)

        input_prompt = None
        if args.fix_prompt:
            input_prompt = args.fix_prompt
        else:
            prompt_idx = int(dir_name.split('_')[0])
            if index_range is not None:
                if prompt_idx not in index_range: continue
            if args.case_name is not None:
                if dir_name not in args.case_name: continue
            input_prompt = prompts[prompt_idx]
        print(f"the prompt is: {input_prompt}")

        data_type = args.data_type  # 
        sdp.process(adir, data_type, input_prompt)

        # break