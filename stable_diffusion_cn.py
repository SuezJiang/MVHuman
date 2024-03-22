import time
import torch.nn.functional as F
import torch.nn as nn
import torch
from transformers import CLIPTextModel, CLIPTokenizer, logging
from diffusers import AutoencoderKL, UNet2DConditionModel, DDPMScheduler, PNDMScheduler, DPMSolverMultistepScheduler, EulerAncestralDiscreteScheduler, LMSDiscreteScheduler
# from icecream import ic
# from models_wota.unet import UNet3DConditionModel
from utils import CrossFrameAttnProcessor
from diffusers import ControlNetModel
from diffusers.models.controlnet import ControlNetOutput
from diffusers.models.modeling_utils import ModelMixin
from diffusers.utils import randn_tensor
from diffusers.utils.import_utils import is_xformers_available
import os
import inspect
import PIL
import numpy as np
from einops import rearrange
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

# suppress partial model loading warning
logging.set_verbosity_error()

def seed_everything(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    #torch.backends.cudnn.deterministic = True
    #torch.backends.cudnn.benchmark = True


class MultiControlNetModel(ModelMixin):
    r"""
    Multiple `ControlNetModel` wrapper class for Multi-ControlNet

    This module is a wrapper for multiple instances of the `ControlNetModel`. The `forward()` API is designed to be
    compatible with `ControlNetModel`.

    Args:
        controlnets (`List[ControlNetModel]`):
            Provides additional conditioning to the unet during the denoising process. You must set multiple
            `ControlNetModel` as a list.
    """

    def __init__(self, controlnets: Union[List[ControlNetModel], Tuple[ControlNetModel]]):
        super().__init__()
        self.nets = nn.ModuleList(controlnets)

    def forward(
        self,
        sample: torch.FloatTensor,
        timestep: Union[torch.Tensor, float, int],
        encoder_hidden_states: torch.Tensor,
        controlnet_cond: List[torch.tensor],
        conditioning_scale: List[float],
        class_labels: Optional[torch.Tensor] = None,
        timestep_cond: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        cross_attention_kwargs: Optional[Dict[str, Any]] = None,
        guess_mode: bool = False,
        return_dict: bool = True,
    ) -> Union[ControlNetOutput, Tuple]:
        for i, (image, scale, controlnet) in enumerate(zip(controlnet_cond, conditioning_scale, self.nets)):
            down_samples, mid_sample = controlnet(
                sample,
                timestep,
                encoder_hidden_states,
                image,
                scale,
                class_labels,
                timestep_cond,
                attention_mask,
                cross_attention_kwargs,
                guess_mode,
                return_dict,
            )

            # merge samples
            if i == 0:
                down_block_res_samples, mid_block_res_sample = down_samples, mid_sample
            else:
                down_block_res_samples = [
                    samples_prev + samples_curr
                    for samples_prev, samples_curr in zip(down_block_res_samples, down_samples)
                ]
                mid_block_res_sample += mid_sample

        return down_block_res_samples, mid_block_res_sample


class StableDiffusionControlNet(nn.Module):
    def __init__(self, device, unet_path=None, vae_path=None, cn_list=None, cache_dir=None, lora_list=None,
                use_attn_feature_injection: bool = False, attn_processor=None):
        super().__init__()

        try:
            with open('./TOKEN', 'r') as f:
                self.token = f.read().replace('\n', '')  # remove the last \n!
                print(f'[INFO] loaded hugging face access token from ./TOKEN!')
        except FileNotFoundError as e:
            self.token = True
            print(f'[INFO] try to load hugging face access token from the default place, make sure you have run `huggingface-cli login`.')

        self.device = device
        self.num_train_timesteps = 1000
        self.min_step = int(self.num_train_timesteps * 0.02)
        self.max_step = int(self.num_train_timesteps * 0.98)

        print(f'[INFO] loading stable diffusion')
        if cache_dir is None:
            hf_cache_home = os.path.expanduser(
                os.getenv("HF_HOME", os.path.join(os.getenv("XDG_CACHE_HOME", "~/.cache"), "huggingface"))
            )
            default_cache_path = os.path.join(hf_cache_home, "diffusers")
            cache_dir = default_cache_path
        print(f'[INFO] loading stable diffusion from cache dir {cache_dir}')

        self.vae = AutoencoderKL.from_pretrained(vae_path, subfolder="vae", use_auth_token=self.token, cache_dir=cache_dir).to(self.device)

        # 2. Load the tokenizer and text encoder to tokenize and encode the text.
        self.tokenizer = CLIPTokenizer.from_pretrained("openai/clip-vit-large-patch14", cache_dir=cache_dir)
        self.text_encoder = CLIPTextModel.from_pretrained("openai/clip-vit-large-patch14", cache_dir=cache_dir).to(self.device)

        # 3. The UNet model for generating the latents.
        self.unet = UNet2DConditionModel.from_pretrained(unet_path, subfolder="unet", use_auth_token=self.token, cache_dir=cache_dir).to(self.device)

        # lora
        if lora_list is not None:
            from utils import load_A1111_lora
            print("load lora")
            # [lora_path, multiplier]
            for lora_pack in lora_list:
                lora_path, multiplier = lora_pack
                print(f"loading {lora_path} with multiplier {multiplier}")
                load_A1111_lora(self.unet, self.text_encoder, lora_path, multiplier=multiplier, device=device, dtype=self.unet.dtype)
        # CF attn
        self.use_attn_feature_injection = use_attn_feature_injection
        if use_attn_feature_injection:
            print("setting cross attn processor")
            assert(attn_processor is not None)
            self.unet.set_attn_processor(processor=attn_processor) # 2 if using guidance

        # Freeze vae and text_encoder
        self.vae.requires_grad_(False)
        self.text_encoder.requires_grad_(False)
        self.unet.requires_grad_(False)
        # if is_xformers_available():
        #     self.unet.enable_xformers_memory_efficient_attention()
        # else:
        #     raise ValueError("xformers is not available. Make sure it is installed correctly")

        # 4. Create a scheduler for inference
        self.scheduler_type = "DPM"
        # self.max_interfere_step = 3
        # self.interfere_scale = 10 # guide noise with interfered origin

        if self.scheduler_type == "PNDM":
            self.scheduler = PNDMScheduler(beta_start=0.00085, beta_end=0.012, beta_schedule="scaled_linear", num_train_timesteps=self.num_train_timesteps)
            self.alphas = self.scheduler.alphas_cumprod.to(self.device)  # for convenience
        
        if self.scheduler_type == "DPM":
            self.scheduler = DPMSolverMultistepScheduler(beta_start=0.00085, beta_end=0.012)
            self.alphas = self.scheduler.alphas_cumprod.to(self.device)  # for convenience
            self.alpha_ts = self.scheduler.alpha_t.to(self.device)
            self.sigma_ts = self.scheduler.sigma_t.to(self.device)
        
        if self.scheduler_type == "DDPM":
            self.scheduler = DDPMScheduler(beta_start=0.00085, beta_end=0.012)
            self.alphas_cumprod = self.scheduler.alphas_cumprod.to(self.device)
        
        self.vae_scale_factor = 2 ** (len(self.vae.config.block_out_channels) - 1)

        # 5. create controlnet unit
        if len(cn_list) == 1:
            controlnet = ControlNetModel.from_pretrained(cn_list[0], cache_dir=cache_dir)
        else:
            controlnet = []
            for cn in cn_list:
                controlnet.append(ControlNetModel.from_pretrained(cn, cache_dir=cache_dir))
        
        if isinstance(controlnet, (list, tuple)):
            controlnet = MultiControlNetModel(controlnet)
        
        self.controlnet = controlnet.to(self.device)
        self.controlnet.requires_grad_(False)
        print(f'[INFO] loaded stable diffusion!')
    
    def set_attn_processor(self, attn_processor):
        if self.use_attn_feature_injection:
            print("setting attn processor ..")
            self.unet.set_attn_processor(processor=attn_processor)

    def prepare_image(
        self,
        image,
        width,
        height,
        dtype,
        do_classifier_free_guidance=False,
    ):
        assert(isinstance(image, list))

        if isinstance(image[0], PIL.Image.Image):
            images = []

            for image_ in image:
                image_ = image_.convert("RGB")
                image_ = image_.resize((width, height), resample=PIL.Image.LANCZOS)
                image_ = np.array(image_)
                image_ = image_[None, :]
                images.append(image_)

            image = images

            image = np.concatenate(image, axis=0)
            image = np.array(image).astype(np.float32) / 255.0
            image = image.transpose(0, 3, 1, 2)
            image = torch.from_numpy(image)
        elif isinstance(image[0], torch.Tensor):
            assert(image[0].shape[2] == 3) or (image[0].shape[3] == 3)
            if image[0].shape[2] == 3:
                image = [image_.unsqueeze(0) for image_ in image]
                image = torch.cat(image, dim=0)
            elif image[0].shape[3] == 3:
                image = torch.cat(image, dim=0)

        image = image.to(device=self.device, dtype=dtype)

        if do_classifier_free_guidance:
            image = torch.cat([image] * 2)

        return image

    # Copied from diffusers.pipelines.stable_diffusion.pipeline_stable_diffusion.StableDiffusionPipeline._encode_prompt
    def _encode_prompt(
        self,
        prompts,
        batch_size,
        do_classifier_free_guidance,
        negative_prompt=None,
        prompt_embeds: Optional[torch.FloatTensor] = None,
        negative_prompt_embeds: Optional[torch.FloatTensor] = None,
        return_separate=False,
    ):
        r"""
        Encodes the prompt into text encoder hidden states.

        Args:
             prompt (`str` or `List[str]`, *optional*):
                prompt to be encoded
            do_classifier_free_guidance (`bool`):
                whether to use classifier free guidance or not
            negative_prompt (`str` or `List[str]`, *optional*):
                The prompt or prompts not to guide the image generation. If not defined, one has to pass
                `negative_prompt_embeds` instead. Ignored when not using guidance (i.e., ignored if `guidance_scale` is
                less than `1`).
            prompt_embeds (`torch.FloatTensor`, *optional*):
                Pre-generated text embeddings. Can be used to easily tweak text inputs, *e.g.* prompt weighting. If not
                provided, text embeddings will be generated from `prompt` input argument.
            negative_prompt_embeds (`torch.FloatTensor`, *optional*):
                Pre-generated negative text embeddings. Can be used to easily tweak text inputs, *e.g.* prompt
                weighting. If not provided, negative_prompt_embeds will be generated from `negative_prompt` input
                argument.
        """

        if prompt_embeds is None:
            # textual inversion: procecss multi-vector tokens if necessary
            # if isinstance(self, TextualInversionLoaderMixin):
            #     prompt = self.maybe_convert_prompt(prompt, self.tokenizer)

            text_inputs = self.tokenizer(
                prompts,
                padding="max_length",
                max_length=self.tokenizer.model_max_length,
                truncation=True,
                return_tensors="pt",
            )
            text_input_ids = text_inputs.input_ids
            untruncated_ids = self.tokenizer(prompts, padding="longest", return_tensors="pt").input_ids

            if untruncated_ids.shape[-1] >= text_input_ids.shape[-1] and not torch.equal(
                text_input_ids, untruncated_ids
            ):
                removed_text = self.tokenizer.batch_decode(
                    untruncated_ids[:, self.tokenizer.model_max_length - 1 : -1]
                )
                print(
                    "[sd_cn]: The following part of your input was truncated because CLIP can only handle sequences up to"
                    f" {self.tokenizer.model_max_length} tokens: {removed_text}"
                )

            if hasattr(self.text_encoder.config, "use_attention_mask") and self.text_encoder.config.use_attention_mask:
                attention_mask = text_inputs.attention_mask.to(self.device)
            else:
                attention_mask = None

            prompt_embeds = self.text_encoder(
                text_input_ids.to(self.device),
                attention_mask=attention_mask,
            )
            prompt_embeds = prompt_embeds[0]

        prompt_embeds = prompt_embeds.to(dtype=self.text_encoder.dtype, device=self.device)

        # bs_embed, seq_len, _ = prompt_embeds.shape
        # # duplicate text embeddings for each generation per prompt, using mps friendly method
        # prompt_embeds = prompt_embeds.repeat(1, num_images_per_prompt, 1)
        # prompt_embeds = prompt_embeds.view(bs_embed * num_images_per_prompt, seq_len, -1)

        # get unconditional embeddings for classifier free guidance
        if do_classifier_free_guidance and negative_prompt_embeds is None:
            uncond_tokens: List[str]
            if negative_prompt is None:
                uncond_tokens = [""] * batch_size
            elif type(prompts) is not type(negative_prompt):
                raise TypeError(
                    f"`negative_prompt` should be the same type to `prompt`, but got {type(negative_prompt)} !="
                    f" {type(prompts)}."
                )
            elif isinstance(negative_prompt, str):
                uncond_tokens = [negative_prompt]
            elif batch_size != len(negative_prompt):
                raise ValueError(
                    f"`negative_prompt`: {negative_prompt} has batch size {len(negative_prompt)}, but `prompt`:"
                    f" {prompts} has batch size {batch_size}. Please make sure that passed `negative_prompt` matches"
                    " the batch size of `prompt`."
                )
            else:
                uncond_tokens = negative_prompt

            # textual inversion: procecss multi-vector tokens if necessary
            # if isinstance(self, TextualInversionLoaderMixin):
            #     uncond_tokens = self.maybe_convert_prompt(uncond_tokens, self.tokenizer)

            max_length = prompt_embeds.shape[1]
            uncond_input = self.tokenizer(
                uncond_tokens,
                padding="max_length",
                max_length=max_length,
                truncation=True,
                return_tensors="pt",
            )

            if hasattr(self.text_encoder.config, "use_attention_mask") and self.text_encoder.config.use_attention_mask:
                attention_mask = uncond_input.attention_mask.to(self.device)
            else:
                attention_mask = None

            negative_prompt_embeds = self.text_encoder(
                uncond_input.input_ids.to(self.device),
                attention_mask=attention_mask,
            )
            negative_prompt_embeds = negative_prompt_embeds[0]

        if return_separate:
            return negative_prompt_embeds, prompt_embeds

        if do_classifier_free_guidance:
            # duplicate unconditional embeddings for each generation per prompt, using mps friendly method
            # seq_len = negative_prompt_embeds.shape[1]

            negative_prompt_embeds = negative_prompt_embeds.to(dtype=self.text_encoder.dtype, device=self.device)

            # negative_prompt_embeds = negative_prompt_embeds.repeat(1, num_images_per_prompt, 1)
            # negative_prompt_embeds = negative_prompt_embeds.view(batch_size * num_images_per_prompt, seq_len, -1)

            # For classifier free guidance, we need to do two forward passes.
            # Here we concatenate the unconditional and text embeddings into a single batch
            # to avoid doing two forward passes
            # print("embeds: ", negative_prompt_embeds.shape, prompt_embeds.shape)
            prompt_embeds = torch.cat([negative_prompt_embeds, prompt_embeds])

        return prompt_embeds
    
    # Copied from diffusers.pipelines.stable_diffusion.pipeline_stable_diffusion.StableDiffusionPipeline.prepare_extra_step_kwargs
    def prepare_extra_step_kwargs(self, generator, eta):
        # prepare extra kwargs for the scheduler step, since not all schedulers have the same signature
        # eta (η) is only used with the DDIMScheduler, it will be ignored for other schedulers.
        # eta corresponds to η in DDIM paper: https://arxiv.org/abs/2010.02502
        # and should be between [0, 1]

        accepts_eta = "eta" in set(inspect.signature(self.scheduler.step).parameters.keys())
        extra_step_kwargs = {}
        if accepts_eta:
            extra_step_kwargs["eta"] = eta

        # check if the scheduler accepts generator
        accepts_generator = "generator" in set(inspect.signature(self.scheduler.step).parameters.keys())
        if accepts_generator:
            extra_step_kwargs["generator"] = generator
        return extra_step_kwargs
    
    # Copied from diffusers.pipelines.stable_diffusion.pipeline_stable_diffusion.StableDiffusionPipeline.prepare_latents
    def prepare_latents(self, batch_size, num_channels_latents, height, width, dtype, device, generator, latents=None):
        shape = (batch_size, num_channels_latents, height // self.vae_scale_factor, width // self.vae_scale_factor)
        if isinstance(generator, list) and len(generator) != batch_size:
            raise ValueError(
                f"You have passed a list of generators of length {len(generator)}, but requested an effective batch"
                f" size of {batch_size}. Make sure the batch size matches the length of the generators."
            )

        if latents is None:
            latents = randn_tensor(shape, generator=generator, device=device, dtype=dtype)
        else:
            latents = latents.to(device)

        # scale the initial noise by the standard deviation required by the scheduler
        latents = latents * self.scheduler.init_noise_sigma
        return latents

    def decode_latents(self, latents, train=False, vae=None):

        latents = 1 / 0.18215 * latents

        if vae is None:
            vae = self.vae

        if train:
            imgs = vae.decode(latents).sample
        else:
            # with torch.no_grad():
            imgs = vae.decode(latents).sample

        # imgs = (imgs / 2 + 0.5).clamp(0, 1)
        imgs = (imgs / 2 + 0.5)

        return imgs

    def encode_imgs(self, imgs):
        # imgs: [B, 3, H, W]

        imgs = 2 * imgs - 1

        posterior = self.vae.encode(imgs).latent_dist
        latents = posterior.sample() * 0.18215

        return latents

    def noise2origin(self, latents, pred_noise, t):
        if self.scheduler_type == "PNDM":
            # alpha_t, sigma_t = self.alpha_ts[t], self.sigma_ts[t]
            # pred_origin = (latents - sigma_t * pred_noise) / alpha_t
            # return pred_origin
            pass
        elif self.scheduler_type == "DPM":
            alpha_t, sigma_t = self.alpha_ts[t], self.sigma_ts[t]
            pred_origin = (latents - sigma_t * pred_noise) / alpha_t
            return pred_origin
        elif self.scheduler_type == "DDPM":
            alpha_prod_t = self.alphas_cumprod[t]
            beta_prod_t = 1 - alpha_prod_t
            pred_origin = (latents - beta_prod_t ** (0.5) * pred_noise) / alpha_prod_t ** (0.5)
            return pred_origin

    def origin2noise(self, latents, pred_origin, t):
        if self.scheduler_type == "DPM":
            alpha_t, sigma_t = self.alpha_ts[t], self.sigma_ts[t]
            pred_noise = -(pred_origin * alpha_t - latents) / sigma_t
            return pred_noise
        elif self.scheduler_type == "DDPM":
            alpha_prod_t = self.alphas_cumprod[t]
            beta_prod_t = 1 - alpha_prod_t
            pred_noise = -(pred_origin * alpha_prod_t ** (0.5) - latents) / beta_prod_t ** (0.5)
            return pred_noise

    def origin2latent(self, pred_origin, pred_noise, t):
        if self.scheduler_type == "DPM":
            alpha_t, sigma_t = self.alpha_ts[t], self.sigma_ts[t]
            latents = pred_noise * sigma_t + pred_origin * alpha_t
            return latents
    
    def latent2pred(
            self,
            text_embeddings,
            latents,
            cn_images,
            t,
            batch_size,
            height=512, width=512,
            guidance_scale=7.5,
            generator=None,
            do_classifier_free_guidance=False,
            controlnet_conditioning_scale=None,
            split_run=False,
        ):
        extra_step_kwargs = self.prepare_extra_step_kwargs(generator, 0.) # eta is 0. here

        with torch.autocast('cuda'):
            # expand the latents if we are doing classifier free guidance
            latent_model_input = torch.cat([latents] * 2) if do_classifier_free_guidance else latents
            latent_model_input = self.scheduler.scale_model_input(latent_model_input, t)

            controlnet_latent_model_input = latent_model_input
            controlnet_prompt_embeds = text_embeddings

            if not split_run:
                down_block_res_samples, mid_block_res_sample = self.controlnet(
                    controlnet_latent_model_input,
                    t,
                    encoder_hidden_states=controlnet_prompt_embeds,
                    controlnet_cond=cn_images,
                    conditioning_scale=controlnet_conditioning_scale,
                    return_dict=False,
                )

                noise_pred = self.unet(
                    latent_model_input,
                    t,
                    encoder_hidden_states=text_embeddings,
                    cross_attention_kwargs=None,
                    down_block_additional_residuals=down_block_res_samples,
                    mid_block_additional_residual=mid_block_res_sample,
                ).sample

                # perform guidance
                if do_classifier_free_guidance:
                    noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
                    noise_pred = noise_pred_uncond + guidance_scale * (noise_pred_text - noise_pred_uncond)
            else:
                raise NotImplementedError
                assert latent_model_input.shape[0] % 2 == 0
                assert do_classifier_free_guidance
                half_ = latent_model_input.shape[0] // 2
                down_block_res_samples, mid_block_res_sample = self.controlnet(
                    controlnet_latent_model_input[:half_],
                    t,
                    encoder_hidden_states=controlnet_prompt_embeds[:half_],
                    controlnet_cond=[cn_img[:half_] for cn_img in cn_images],
                    conditioning_scale=controlnet_conditioning_scale,
                    return_dict=False,
                )
                noise_pred_uncond = self.unet(
                    latent_model_input[:half_],
                    t,
                    encoder_hidden_states=text_embeddings[:half_],
                    cross_attention_kwargs=None,
                    down_block_additional_residuals=down_block_res_samples,
                    mid_block_additional_residual=mid_block_res_sample,
                ).sample

                down_block_res_samples, mid_block_res_sample = self.controlnet(
                    controlnet_latent_model_input[half_:],
                    t,
                    encoder_hidden_states=controlnet_prompt_embeds[half_:],
                    controlnet_cond=[cn_img[half_:] for cn_img in cn_images],
                    conditioning_scale=controlnet_conditioning_scale,
                    return_dict=False,
                )
                noise_pred_text = self.unet(
                    latent_model_input[half_:],
                    t,
                    encoder_hidden_states=text_embeddings[half_:],
                    cross_attention_kwargs=None,
                    down_block_additional_residuals=down_block_res_samples,
                    mid_block_additional_residual=mid_block_res_sample,
                ).sample

                noise_pred = noise_pred_uncond + guidance_scale * (noise_pred_text - noise_pred_uncond)

            # compute the previous noisy sample x_t -> x_t-1
            # latents = self.scheduler.step(noise_pred, t, latents, **extra_step_kwargs).prev_sample
            # compute the predicted x0
            # alpha_t, sigma_t = self.alpha_ts[t], self.sigma_ts[t]
            # pred_origin = (latents - sigma_t * noise_pred) / alpha_t
            pred_origin = self.noise2origin(latents, noise_pred, t)

        return noise_pred, pred_origin

    def get_prev_latents(self, noise_pred, t, latents, generator=None):
        extra_step_kwargs = self.prepare_extra_step_kwargs(generator, 0.) # eta is 0. here
        latents = self.scheduler.step(noise_pred, t, latents, **extra_step_kwargs).prev_sample
        return latents

    def produce_latents(
            self,
            text_embeddings, 
            cn_images,
            batch_size,
            height=512, width=512,
            guidance_scale=7.5, 
            latents=None,
            generator=None,
            timesteps=None,
            do_classifier_free_guidance=False,
            controlnet_conditioning_scale=None,
        ):

        # 6. Prepare latent variables
        num_channels_latents = self.unet.config.in_channels
        latents = self.prepare_latents(
            batch_size,
            num_channels_latents,
            height,
            width,
            text_embeddings.dtype,
            torch.device(self.device),
            generator,
            latents,
        )
        extra_step_kwargs = self.prepare_extra_step_kwargs(generator, 0.) # eta is 0. here

        # scheduler.set_timesteps(num_inference_steps)
        pred_origins = []
        all_latents = []
        with torch.autocast('cuda'):
            for i, t in enumerate(timesteps):
                # # expand the latents if we are doing classifier-free guidance to avoid doing two forward passes.
                # latent_model_input = torch.cat([latents] * 2)

                # # predict the noise residual
                # with torch.no_grad():
                #     noise_pred = unet(latent_model_input, t, encoder_hidden_states=text_embeddings)['sample']

                # # perform guidance
                # noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
                # noise_pred = noise_pred_uncond + guidance_scale * (noise_pred_text - noise_pred_uncond)
                # # compute the previous noisy sample x_t -> x_t-1
                # latents = scheduler.step(noise_pred, t, latents)['prev_sample']
                # inter.append(latents)

                # expand the latents if we are doing classifier free guidance
                latent_model_input = torch.cat([latents] * 2) if do_classifier_free_guidance else latents
                latent_model_input = self.scheduler.scale_model_input(latent_model_input, t)

                controlnet_latent_model_input = latent_model_input
                controlnet_prompt_embeds = text_embeddings

                down_block_res_samples, mid_block_res_sample = self.controlnet(
                    controlnet_latent_model_input,
                    t,
                    encoder_hidden_states=controlnet_prompt_embeds,
                    controlnet_cond=cn_images,
                    conditioning_scale=controlnet_conditioning_scale,
                    return_dict=False,
                )

                # predict the noise residual
                noise_pred = self.unet(
                    latent_model_input,
                    t,
                    encoder_hidden_states=text_embeddings,
                    cross_attention_kwargs=None,
                    down_block_additional_residuals=down_block_res_samples,
                    mid_block_additional_residual=mid_block_res_sample,
                ).sample

                # perform guidance
                if do_classifier_free_guidance:
                    noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
                    noise_pred = noise_pred_uncond + guidance_scale * (noise_pred_text - noise_pred_uncond)

                noise_pred_unet = noise_pred
                pred_origin = self.noise2origin(latents, noise_pred, t)

                

                pred_origins.append(pred_origin)
                all_latents.append(latents)

                pred_noise = self.origin2noise(latents, pred_origin, t)

                # do guidance on unet noise and interfered noise
                pred_noise = noise_pred_unet + self.interfere_scale * (pred_noise - noise_pred_unet)

                # compute the previous noisy sample x_t -> x_t-1
                latents = self.scheduler.step(pred_noise, t, latents, **extra_step_kwargs).prev_sample


        return latents, pred_origins, all_latents

    def prompt_to_img(
            self, 
            prompts, 
            cn_images,
            negative_prompts=None, 
            height=512, width=512, 
            num_inference_steps=20, 
            guidance_scale=7.5, 
            latents=None, 
            generator=None,
            prompt_embeds=None,
            negative_prompt_embeds=None,
            controlnet_conditioning_weights=None,
            early_return=False,
            **kwargs,
        ):
        '''
        cn_images:
         if multi cn:
            (e.g. [[canny_image_1, pose_image_1], [canny_image_2, pose_image_2]])
            (e.g. [[canny_image_1, canny_image_2], [pose_image_1, pose_image_2]])
         else:
            [canny_image_1, canny_image_2]
            tensor or Image, 512, 512, 3
        controlnet_conditioning_weights: [cn_unit1_weight, cn_unit2_weight]
        '''
        batch_size = len(prompts)
        device = self.device

        do_classifier_free_guidance = guidance_scale > 1.0

        # simple check
        assert(isinstance(prompts, list))
        assert(len(controlnet_conditioning_weights) == len(self.controlnet.nets))

        # encode imput prompt
        # [B (2*B), 77, 768]
        prompt_embeds = self._encode_prompt(
            prompts,
            batch_size,
            do_classifier_free_guidance,
            negative_prompts,
            prompt_embeds=prompt_embeds,
            negative_prompt_embeds=negative_prompt_embeds,
        )
        
        # check images
        # image shape, B (2*B), 3, h, w
        if isinstance(self.controlnet, ControlNetModel):
            image_to_controlnet = self.prepare_image(
                image=cn_images,
                width=width,
                height=height,
                dtype=self.controlnet.dtype,
                do_classifier_free_guidance=do_classifier_free_guidance,
            )
        elif isinstance(self.controlnet, MultiControlNetModel):
            images = []

            for image_ in cn_images:
                image_ = self.prepare_image(
                    image=image_,
                    width=width,
                    height=height,
                    dtype=self.controlnet.dtype,
                    do_classifier_free_guidance=do_classifier_free_guidance,
                )

                images.append(image_)

            image_to_controlnet = images
        else:
            assert False
        
        # Prepare timesteps
        self.scheduler.set_timesteps(num_inference_steps, device=device)
        timesteps = self.scheduler.timesteps

        # produce latents
        latents, pred_origins, all_latents = self.produce_latents(
            prompt_embeds, 
            image_to_controlnet,
            batch_size,
            height=512, width=512,
            guidance_scale=7.5, 
            latents=latents,
            generator=generator,
            timesteps=timesteps,
            do_classifier_free_guidance=do_classifier_free_guidance,
            controlnet_conditioning_scale=controlnet_conditioning_weights,
        )

        ret = {}
        ret['latents'] = latents

        # Img latents -> imgs
        imgs = self.decode_latents(latents)  # [B, 3, 512, 512]
        ret['images'] = imgs

        # Img to Numpy
        imgs_np = imgs.detach().cpu().permute(0, 2, 3, 1).numpy()
        imgs_np = (imgs_np.clip(0, 1) * 255).round().astype('uint8')
        ret['images_np'] = imgs_np

        pred_origins = [self.decode_latents(ori) for ori in pred_origins]
        pred_origins = [ori.detach().cpu().permute(0,2,3,1).numpy() for ori in pred_origins]
        pred_origins = [(ori.clip(0, 1) * 255).round().astype('uint8') for ori in pred_origins] # .clip(0, 1)
        ret['pred_origins'] = pred_origins

        all_latents = [self.decode_latents(ori) for ori in all_latents]
        all_latents = [ori.detach().cpu().permute(0,2,3,1).numpy() for ori in all_latents]
        all_latents = [(ori.clip(0, 1) * 255).round().astype('uint8') for ori in all_latents] # .clip(0, 1)
        ret['all_latents'] = all_latents

        return ret