# use prompt to generate a 3DGS, then use this 3DGS the render a colmap dataset for GaussianEditor
import os
import tyro
import glob
import imageio
import numpy as np
import tqdm
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms.functional as TF
from safetensors.torch import load_file
import rembg
import random
import cv2
from PIL import Image

import kiui
from kiui.op import recenter
from kiui.cam import orbit_camera

from core.options import AllConfigs, Options
from core.models import LGM
from mvdream.pipeline_mvdream import MVDreamPipeline
from argparse import ArgumentParser
from stylize_utils import OptimizationParams
from omegaconf import OmegaConf
from threestudio.models.guidance.instructpix2pix_guidance import InstructPix2PixGuidance
from threestudio.models.prompt_processors.stable_diffusion_prompt_processor import StableDiffusionPromptProcessor
from threestudio.utils.perceptual import PerceptualLoss


IMAGENET_DEFAULT_MEAN = (0.485, 0.456, 0.406)
IMAGENET_DEFAULT_STD = (0.229, 0.224, 0.225)

opt = tyro.cli(AllConfigs)

# model
model = LGM(opt)

# resume pretrained checkpoint
if opt.resume is not None:
    if opt.resume.endswith('safetensors'):
        ckpt = load_file(opt.resume, device='cpu')
    else:
        ckpt = torch.load(opt.resume, map_location='cpu')
    model.load_state_dict(ckpt, strict=False)
    print(f'[INFO] Loaded checkpoint from {opt.resume}')
else:
    print(f'[WARN] model randomly initialized, are you sure?')

# device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = model.half().to(device)
model.eval()

rays_embeddings = model.prepare_default_rays(device)

tan_half_fov = np.tan(0.5 * np.deg2rad(opt.fovy))
proj_matrix = torch.zeros(4, 4, dtype=torch.float32, device=device)
proj_matrix[0, 0] = 1 / tan_half_fov
proj_matrix[1, 1] = 1 / tan_half_fov
proj_matrix[2, 2] = (opt.zfar + opt.znear) / (opt.zfar - opt.znear)
proj_matrix[3, 2] = - (opt.zfar * opt.znear) / (opt.zfar - opt.znear)
proj_matrix[2, 3] = 1

# load image dream
pipe_text = MVDreamPipeline.from_pretrained(
    'ashawkey/mvdream-sd2.1-diffusers', # remote weights
    torch_dtype=torch.float16,
    trust_remote_code=True,
    # local_files_only=True,
)
pipe_text = pipe_text.to(device)

def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True

# load rembg
bg_remover = rembg.new_session()
negative_prompt = 'ugly, blurry, pixelated obscure, unnatural colors, poor lighting, dull, unclear, cropped, lowres, low quality, artifacts, duplicate'

def process(opt: Options, prompt, prompt_neg='', input_num_steps=30, guidance_scale=7.5, seed=20240228):
    # save path directly to workspace
    os.makedirs(opt.workspace, exist_ok=True)

    setup_seed(seed)

    mv_image_uint8 = pipe_text(prompt, negative_prompt=prompt_neg, num_inference_steps=input_num_steps, guidance_scale=guidance_scale, elevation=0)
    mv_image_uint8 = (mv_image_uint8 * 255).astype(np.uint8)
    # bg removal
    mv_image = []

    for i in range(4):
        image = rembg.remove(mv_image_uint8[i], session=bg_remover)
        # to white bg
        image = image.astype(np.float32) / 255
        image = recenter(image, image[..., 0] > 0, border_ratio=0.2)
        image = image[..., :3] * image[..., -1:] + (1 - image[..., -1:])
        mv_image.append(image)

    mv_image_grid = np.concatenate([
        np.concatenate([mv_image[1], mv_image[2]], axis=1),
        np.concatenate([mv_image[3], mv_image[0]], axis=1),
    ], axis=0)

    # generate gaussians
    input_image = np.stack([mv_image[1], mv_image[2], mv_image[3], mv_image[0]], axis=0) # [4, 256, 256, 3], float32
    input_image = torch.from_numpy(input_image).permute(0, 3, 1, 2).float().to(device) # [4, 3, 256, 256]
    input_image = F.interpolate(input_image, size=(opt.input_size, opt.input_size), mode='bilinear', align_corners=False)
    input_image = TF.normalize(input_image, IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD)

    input_image = torch.cat([input_image, rays_embeddings], dim=1).unsqueeze(0) # [1, 4, 9, H, W]

    # model.train()

    with torch.autocast(device_type='cuda', dtype=torch.float16):
        gaussians = model.forward_gaussians(input_image) # tensor, no gradient # [1,65536,14]

    gaussians = gaussians.detach().clone().requires_grad_(True) # [:3]: position, fix
    
    model.gs.save_ply(gaussians, os.path.join(opt.workspace, prompt.replace(' ', '_') + '.ply'))
    # ip2p = InstructPix2PixGuidance(OmegaConf.create({"min_step_percent": 0.02, "max_step_percent": 0.98}))

    # prompt_utils = StableDiffusionPromptProcessor({
    #     'pretrained_model_name_or_path': 'runwayml/stable-diffusion-v1-5',
    #     'prompt': "Turn it to a clown",
    #     # 'use_cache': False,
    #     'spawn': False,
    # })()

    # total_views = np.arange(0, 360, 360//opt.train_cam_num, dtype=np.int32)
    # for view in total_views:
    #     cam_poses = torch.from_numpy(orbit_camera(0, view, radius=opt.cam_radius, opengl=True)).unsqueeze(0).to(device)
    #     cam_poses[:, :3, 1:3] *= -1
    #     cam_view = torch.inverse(cam_poses).transpose(1, 2)
    #     cam_view_proj = cam_view @ proj_matrix
    #     cam_pos = - cam_poses[:, :3, 3]
    #     image = model.gs.render(gaussians, cam_view.unsqueeze(0), cam_view_proj.unsqueeze(0), cam_pos.unsqueeze(0), scale_modifier=1)['image']
    #     edited = ip2p(image.squeeze(0).permute(0, 2, 3, 1), image.squeeze(0).permute(0, 2, 3, 1), prompt_utils)['edit_images'].detach().clone()
    #     # edited: 1,H,W,C
    #     print(edited.shape)
    #     image = image.detach().squeeze(0).permute(0, 2, 3, 1).squeeze(0) * 255
    #     # image: 1,H,W,C
    #     image = image.clamp(0, 255).cpu().numpy().astype(np.uint8)
    #     Image.fromarray(image).save(f'{opt.workspace}/frame_{view:03d}.png')
    #     Image.fromarray(edited.clamp(0, 255).cpu().numpy().astype(np.uint8)).save(f'{opt.workspace}/frame_{view:03d}_edited.png')

    images = []
    # generate original video
    with torch.no_grad():
        azimuth = np.arange(0, 360, 2, dtype=np.int32)
        for azi in tqdm.tqdm(azimuth):
                    
            cam_poses = torch.from_numpy(orbit_camera(0, azi, radius=opt.cam_radius, opengl=True)).unsqueeze(0).to(device)

            cam_poses[:, :3, 1:3] *= -1 # invert up & forward direction
                    
            # cameras needed by gaussian rasterizer
            cam_view = torch.inverse(cam_poses).transpose(1, 2) # [V, 4, 4]
            cam_view_proj = cam_view @ proj_matrix # [V, 4, 4]
            cam_pos = - cam_poses[:, :3, 3] # [V, 3]

            image = model.gs.render(gaussians, cam_view.unsqueeze(0), cam_view_proj.unsqueeze(0), cam_pos.unsqueeze(0), scale_modifier=1)['image']
            images.append((image.squeeze(1).permute(0,2,3,1).contiguous().float().cpu().numpy() * 255).astype(np.uint8))

        images = np.concatenate(images, axis=0)
        imageio.mimwrite(os.path.join(opt.workspace, prompt.replace(' ', '_') + '.mp4'), images, fps=30)




process(opt, prompt="a tree", prompt_neg=negative_prompt, seed=202402)

