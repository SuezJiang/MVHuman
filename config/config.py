from dataclasses import dataclass, field
from typing import List, Optional

@dataclass
class System_Config:
    seed: int = 20232023

# 
@dataclass
class Model_Config:
    device: str = 'cuda'
    cache_dir: str = None
    unet_path: str = "stablediffusionapi/chilloutmix"
    vae_path: str = "stablediffusionapi/chilloutmix"
    addtional_prompt: str = ", highly realistic, highres, high quality, detailed"
    negative_prompt: str = "lowres, blur, wrong, low quality, paintings, sketches, bad legs, error legs, bad feet, extremely dark, extremely bright"
    cn_list: List[str] = field(default_factory=lambda: ["lllyasviel/control_v11p_sd15_openpose",
                                                        "lllyasviel/control_v11f1p_sd15_depth",
                                                        "lllyasviel/control_v11p_sd15_normalbae"])
    controlnet_weight_start: List[float] = field(default_factory=lambda: [0.8, 0.8, 0.8])
    controlnet_weight_end: List[float] = field(default_factory=lambda: [0.7, 0.5, 0.5])

    use_lora: bool = False
    lora_list: List[str] = field(default_factory=lambda: [])
    lora_weight: List[float] = field(default_factory=lambda: [])
    guidance: float = 7.5
    bg_color: float = 1. # 0.5

    constant_C: float = 1.
    constant_S: float = 0.2
    deg_thres: float = 80.

    the_ref_view: int = 8
    num_inference_step: int = 150

    batch: int = 1
    proportion: float = 0.05 # 0. - 0.2
    enhancement_scale: float = 1.
    original_noise_freq: float = 2. # 2.5 3 
    use_optional_fixed_views: float = 0.333 # 0.1 - 0.333
    enable_enhancement: bool = True

    # latent optimization
    open_optimization: bool = True
    optim_lr: float = 5e-3
    key_optim_steps: List[float] = field(default_factory=lambda: [0.333, 0.667, 0.9]) # 0.9
    key_epoch_nums: int = 5
    epoch_nums: int = 1
    optim_interval: int = 4
    # use special optimization
    use_special_optimization: bool = False

    # eval options
    close_original_noise: bool = False
    close_guide_noise: bool = False
    close_attn: bool = False

    # paper figure
    paper_mat: bool = False

@dataclass
class Config:
    config: Optional[str] = None
    data_dir: str = "none"
    prompts_list: Optional[str] = None
    key: Optional[str] = "gpt_prompts"
    fix_prompt: Optional[str] = None
    main_prompt: Optional[str] = None  # 
    expname: str = "default"
    index_range: Optional[str] = None
    case_name: Optional[List[str]] = None

    data_type: str = "render_45"

    systems: System_Config = System_Config()
    model: Model_Config = Model_Config()