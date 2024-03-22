import os
import numpy as np
import torch
from einops import rearrange
import safetensors
from warping import bilinear_sample_function
import torch.nn.functional as F
from diffusers.utils import _get_model_file, HF_HUB_OFFLINE, DIFFUSERS_CACHE

class CrossFrameAttnProcessor:
    def __init__(self, unet_chunk_size=2, sparse_attn_type='FF', sparse_attn_switch=True):
        self.unet_chunk_size = unet_chunk_size
        self.sparse_attn_type = sparse_attn_type # FF, FFC
        self.sparse_attn_switch = sparse_attn_switch

    def __call__(
            self,
            attn,
            hidden_states,
            encoder_hidden_states=None,
            attention_mask=None):
        batch_size, sequence_length, _ = (
            hidden_states.shape if encoder_hidden_states is None else encoder_hidden_states.shape
        )
        attention_mask = attn.prepare_attention_mask(attention_mask, sequence_length, batch_size)
        query = attn.to_q(hidden_states)

        is_cross_attention = encoder_hidden_states is not None
        if encoder_hidden_states is None:
            encoder_hidden_states = hidden_states
        elif attn.norm_cross is not None:
            encoder_hidden_states = attn.norm_encoder_hidden_states(encoder_hidden_states)

        key = attn.to_k(encoder_hidden_states)
        value = attn.to_v(encoder_hidden_states)

        if self.sparse_attn_switch:
            # Sparse Attention -- first frame
            if not is_cross_attention and self.sparse_attn_type=='FF':
                video_length = key.size()[0] // self.unet_chunk_size
                # former_frame_index = torch.arange(video_length) - 1
                # former_frame_index[0] = 0
                former_frame_index = [0] * video_length
                key = rearrange(key, "(b f) d c -> b f d c", f=video_length)
                key = key[:, former_frame_index]
                key = rearrange(key, "b f d c -> (b f) d c")
                value = rearrange(value, "(b f) d c -> b f d c", f=video_length)
                value = value[:, former_frame_index]
                value = rearrange(value, "b f d c -> (b f) d c")

            # Sparse Attention -- first frame and current
            if not is_cross_attention and self.sparse_attn_type=='FFC':
                video_length = key.size()[0] // self.unet_chunk_size
                former_frame_index = [0] * video_length
                key = rearrange(key, "(b f) d c -> b f d c", f=video_length)
                key = torch.cat([key[:, former_frame_index], key], dim=2)
                key = rearrange(key, "b f d c -> (b f) d c")
                value = rearrange(value, "(b f) d c -> b f d c", f=video_length)
                value = torch.cat([value[:, former_frame_index], value], dim=2)
                value = rearrange(value, "b f d c -> (b f) d c")

        query = attn.head_to_batch_dim(query)
        key = attn.head_to_batch_dim(key)
        value = attn.head_to_batch_dim(value)

        attention_probs = attn.get_attention_scores(query, key, attention_mask)
        hidden_states = torch.bmm(attention_probs, value)
        hidden_states = attn.batch_to_head_dim(hidden_states)

        # linear proj
        hidden_states = attn.to_out[0](hidden_states)
        # dropout
        hidden_states = attn.to_out[1](hidden_states)

        return hidden_states

from safetensors.torch import load_file
from collections import defaultdict
# from diffusers.loaders import LoraLoaderMixin
# import torch 
def load_A1111_lora(unet, text_encoder, checkpoint_path, multiplier, device, dtype):
    
    # load base model
    LORA_PREFIX_UNET = "lora_unet"
    LORA_PREFIX_TEXT_ENCODER = "lora_te"
    # load LoRA weight from .safetensors
    state_dict = load_file(checkpoint_path, device=device)

    updates = defaultdict(dict)
    for key, value in state_dict.items():
        # it is suggested to print out the key, it usually will be something like below
        # "lora_te_text_model_encoder_layers_0_self_attn_k_proj.lora_down.weight"

        layer, elem = key.split('.', 1)
        updates[layer][elem] = value

    index = 0
    # directly update weight in diffusers model
    for layer, elems in updates.items():
        index += 1

        if "text" in layer:
            layer_infos = layer.split(LORA_PREFIX_TEXT_ENCODER + "_")[-1].split("_")
            curr_layer = text_encoder
        else:
            layer_infos = layer.split(LORA_PREFIX_UNET + "_")[-1].split("_")
            curr_layer = unet

        # find the target layer
        temp_name = layer_infos.pop(0)
        while len(layer_infos) > -1:
            try:
                curr_layer = curr_layer.__getattr__(temp_name)
                if len(layer_infos) > 0:
                    temp_name = layer_infos.pop(0)
                elif len(layer_infos) == 0:
                    break
            except Exception:
                if len(temp_name) > 0:
                    temp_name += "_" + layer_infos.pop(0)
                else:
                    temp_name = layer_infos.pop(0)

        # get elements for this layer
        weight_up = elems['lora_up.weight'].to(dtype)
        weight_down = elems['lora_down.weight'].to(dtype)
        alpha = elems['alpha']
        if alpha:
            alpha = alpha.item() / weight_up.shape[1]
        else:
            alpha = 1.0
        
        # if (backup):
        #     original_weights[index] = curr_layer.weight.data.clone().detach()
        # else:
        #     curr_layer.weight.data = original_weights[index].clone().detach()

        # update weight
        if len(weight_up.shape) == 4:
            curr_layer.weight.data += multiplier * alpha * torch.mm(weight_up.squeeze(3).squeeze(2), weight_down.squeeze(3).squeeze(2)).unsqueeze(2).unsqueeze(3)
        else:
            curr_layer.weight.data += multiplier * alpha * torch.mm(weight_up, weight_down)

import torch.nn.functional as F
def tensor_erode(bin_img, ksize=3):
    B, C, H, W = bin_img.shape
    pad = (ksize - 1) // 2
    bin_img = F.pad(bin_img, [pad, pad, pad, pad], mode='constant', value=0)

    patches = bin_img.unfold(dimension=2, size=ksize, step=1)
    patches = patches.unfold(dimension=3, size=ksize, step=1)
    # B x C x H x W x k x k

    eroded, _ = patches.reshape(B, C, H, W, -1).min(dim=-1)
    return eroded
def tensor_dilate(bin_img, ksize=3):
    B, C, H, W = bin_img.shape
    pad = (ksize - 1) // 2
    bin_img = F.pad(bin_img, [pad, pad, pad, pad], mode='constant', value=0)

    patches = bin_img.unfold(dimension=2, size=ksize, step=1)
    patches = patches.unfold(dimension=3, size=ksize, step=1)
    # B x C x H x W x k x k

    eroded, _ = patches.reshape(B, C, H, W, -1).max(dim=-1)
    return eroded
