import tyro
from dataclasses import dataclass
from typing import Tuple, Literal, Dict, Optional


@dataclass
class Options:
    ### model
    # Unet image input size
    input_size: int = 256
    # Unet definition
    down_channels: Tuple[int, ...] = (64, 128, 256, 512, 1024, 1024)
    down_attention: Tuple[bool, ...] = (False, False, False, True, True, True)
    mid_attention: bool = True
    up_channels: Tuple[int, ...] = (1024, 1024, 512, 256)
    up_attention: Tuple[bool, ...] = (True, True, True, False)
    # Unet output size, dependent on the input_size and U-Net structure!
    splat_size: int = 64
    # gaussian render size
    output_size: int = 256

    ### dataset
    # data mode (only support s3 now)
    data_mode: Literal['s3'] = 's3'
    # fovy of the dataset
    fovy: float = 49.1
    # camera near plane
    znear: float = 0.5
    # camera far plane
    zfar: float = 2.5
    # number of all views (input + output)
    num_views: int = 12
    # number of views
    num_input_views: int = 4
    # camera radius
    cam_radius: float = 1.5 # to better use [-1, 1]^3 space
    # num workers
    num_workers: int = 8

    ### training
    # workspace
    workspace: str = './workspace'
    # resume
    resume: Optional[str] = None
    # batch size (per-GPU)
    batch_size: int = 8
    # gradient accumulation
    gradient_accumulation_steps: int = 1
    # training epochs
    num_epochs: int = 30
    # lpips loss weight
    lambda_lpips: float = 1.0
    # gradient clip
    gradient_clip: float = 1.0
    # mixed precision
    mixed_precision: str = 'bf16'
    # learning rate
    lr: float = 4e-4
    # augmentation prob for grid distortion
    prob_grid_distortion: float = 0.5
    # augmentation prob for camera jitter
    prob_cam_jitter: float = 0.5

    ### testing
    # test image path
    test_path: Optional[str] = None

    ### misc
    # nvdiffrast backend setting
    force_cuda_rast: bool = False
    # render fancy video with gaussian scaling effect
    fancy_video: bool = False

    # for stylize setting
    train_cam_num: int = 60 # 80
    edit_cam_num: int = 48 # 60
    # for stylize optimizer setting, something maybe not useful
    edit_train_steps: int = 1500 # set 200 for testing, acutally should be 1500
    # use 500 for testing, default should be 1500
    spatial_lr_scale: float = 3.0
    # gs_lr_scaler: float = 3.0
    # position_lr_init = 0.00016 * gs_lr_scaler
    # gs_lr_end_scaler: float = 2.0
    # position_lr_final = 0.000016 * gs_lr_end_scaler
    # position_lr_delay_mult = 0.01
    # position_lr_max_steps = edit_train_steps
    # color_lr_scaler: float = 3.0
    # opacity_lr_scaler: float = 2.0
    # opacity_lr = 0.05 * opacity_lr_scaler
    # scaling_lr_scaler: float = 2.0
    # scaling_lr = 0.005 * scaling_lr_scaler
    # rotation_lr_scaler: float = 2.0
    # rotation_lr = 0.001 * rotation_lr_scaler
    # color_lr = 0.0125 * color_lr_scaler

    position_lr = 5e-5
    opacity_lr = 5e-5
    scaling_lr = 5e-5 # 5e-4
    rotation_lr = 5e-5 # 5e-4
    # color_lr = 5e-4
    color_lr = 5e-3
    # color_lr = 1e-3

    # for editing process setting
    per_editing_steps: int = 10 # 10 default
    # 目前来看per editing, editing stage没用，因为直接editing...
    edit_begin_step: int = 0
    edit_util_step: int = 1500 # for testing, actually should be 1000
    # text_prompt: str = 'Make it steampunk style' # 'make it to the cartoon style'
    # text_prompt: str = 'Make it terrifying'
    text_prompt: str = "make it a Van Gogh's painting"
    # text_prompt: str = 'Turn him into a clown' # 'make it to the cartoon style'
    # for stylize optimizer
    edit_lambda_l1: float = 100 # 10-》100 效果好一点
    edit_lambda_p: float = 100
    # edit_lambda_l1: float = 10 # 10-》100 效果好一点
    # edit_lambda_p: float = 10

    seed: int = 0

    data_path: str = ''

    style_path: str = "style_image.png"

    lora_path: str = "lora_ckpt/checkpoint-2000"
    controlnet_path: str = "lllyasviel/sd-controlnet-canny"
    sd_path: str = "runwayml/stable-diffusion-v1-5"



# all the default settings
config_defaults: Dict[str, Options] = {}
config_doc: Dict[str, str] = {}

config_doc['lrm'] = 'the default settings for LGM'
config_defaults['lrm'] = Options()

config_doc['small'] = 'small model with lower resolution Gaussians'
config_defaults['small'] = Options(
    input_size=256,
    splat_size=64,
    output_size=256,
    batch_size=8,
    gradient_accumulation_steps=1,
    mixed_precision='bf16',
)

config_doc['big'] = 'big model with higher resolution Gaussians'
config_defaults['big'] = Options(
    input_size=256,
    up_channels=(1024, 1024, 512, 256, 128), # one more decoder
    up_attention=(True, True, True, False, False),
    splat_size=128,
    output_size=512, # render & supervise Gaussians at a higher resolution.
    batch_size=8,
    num_views=8,
    gradient_accumulation_steps=1,
    mixed_precision='bf16',
)

config_doc['tiny'] = 'tiny model for ablation'
config_defaults['tiny'] = Options(
    input_size=256, 
    down_channels=(32, 64, 128, 256, 512),
    down_attention=(False, False, False, False, True),
    up_channels=(512, 256, 128),
    up_attention=(True, False, False, False),
    splat_size=64,
    output_size=256,
    batch_size=16,
    num_views=8,
    gradient_accumulation_steps=1,
    mixed_precision='bf16',
)

AllConfigs = tyro.extras.subcommand_type_from_defaults(config_defaults, config_doc)
