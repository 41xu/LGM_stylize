# 使用60个视角直接stylize
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
# print("ray_embeddings.shape:", rays_embeddings.shape)

tan_half_fov = np.tan(0.5 * np.deg2rad(opt.fovy))
proj_matrix = torch.zeros(4, 4, dtype=torch.float32, device=device)
proj_matrix[0, 0] = 1 / tan_half_fov
proj_matrix[1, 1] = 1 / tan_half_fov
proj_matrix[2, 2] = (opt.zfar + opt.znear) / (opt.zfar - opt.znear)
proj_matrix[3, 2] = - (opt.zfar * opt.znear) / (opt.zfar - opt.znear)
proj_matrix[2, 3] = 1

# load image dream
pipe = MVDreamPipeline.from_pretrained(
    "ashawkey/imagedream-ipmv-diffusers", # remote weights
    torch_dtype=torch.float16,
    trust_remote_code=True,
    # local_files_only=True,
)
pipe = pipe.to(device)

# load rembg
bg_remover = rembg.new_session()
negative_prompt = 'ugly, blurry, pixelated obscure, unnatural colors, poor lighting, dull, unclear, cropped, lowres, low quality, artifacts, duplicate'


# process function
def process(opt: Options, path):
    name = os.path.splitext(os.path.basename(path))[0]
    print(f'[INFO] Processing {path} --> {name}')
    os.makedirs(opt.workspace, exist_ok=True)

    input_image = kiui.read_image(path, mode='uint8')

    # bg removal
    carved_image = rembg.remove(input_image, session=bg_remover) # [H, W, 4]
    mask = carved_image[..., -1] > 0

    # recenter
    image = recenter(carved_image, mask, border_ratio=0.2)
    
    # generate mv
    image = image.astype(np.float32) / 255.0

    # rgba to rgb white bg
    if image.shape[-1] == 4:
        image = image[..., :3] * image[..., 3:4] + (1 - image[..., 3:4])
    mv_image = pipe('', image, negative_prompt=negative_prompt, guidance_scale=5.0, num_inference_steps=30, elevation=0)
    mv_image = np.stack([mv_image[1], mv_image[2], mv_image[3], mv_image[0]], axis=0) # [4, 256, 256, 3], float32

    # generate gaussians
    input_image = torch.from_numpy(mv_image).permute(0, 3, 1, 2).float().to(device) # [4, 3, 256, 256]
    input_image = F.interpolate(input_image, size=(opt.input_size, opt.input_size), mode='bilinear', align_corners=False)
    input_image = TF.normalize(input_image, IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD)

    input_image = torch.cat([input_image, rays_embeddings], dim=1).unsqueeze(0) # [1, 4, 9, H, W]

    # model.train()

    with torch.autocast(device_type='cuda', dtype=torch.float16):
        gaussians_ = model.forward_gaussians(input_image) # tensor, no gradient
    
    gaussians = gaussians_.detach().clone().requires_grad_(True)

    model.gs.save_ply(gaussians, os.path.join(opt.workspace, name + '.ply'))

    perceptual_loss = PerceptualLoss().eval().to(device)
    optimizer = torch.optim.AdamW([gaussians], lr=opt.lr, weight_decay=0.05, betas=(0.9, 0.95))
    # scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer, max_lr=opt.lr, total_steps=opt.edit_train_steps, pct_start=0.3)

    # stylize
    # in face case, there have 65 views for training, 48 views for stylize
    # in this 360 case, initialize 60 views for training and 48 views for stylize 
    # 上面的实验暂时不太需要，目前直接editing不管重建。。。
    # 所以直接用60个view editing，每次随机选择一个视角编辑
    edit_views = np.arange(0, 360, 360//opt.train_cam_num, dtype=np.int32)
    random.seed(0)
    original_images = []
    for view in edit_views:
        cam_poses = torch.from_numpy(orbit_camera(0, view, radius=opt.cam_radius, opengl=True)).unsqueeze(0).to(device)
        cam_poses[:, :3, 1:3] *= -1
        cam_view = torch.inverse(cam_poses).transpose(1, 2)
        cam_view_proj = cam_view @ proj_matrix  
        cam_pos = - cam_poses[:, :3, 3]
        image = model.gs.render(gaussians, cam_view.unsqueeze(0), cam_view_proj.unsqueeze(0), cam_pos.unsqueeze(0), scale_modifier=1)['image'].squeeze(0).permute(0, 2, 3, 1)
        original_images.append(image)

    prompt_utils = StableDiffusionPromptProcessor({
        'pretrained_model_name_or_path': 'runwayml/stable-diffusion-v1-5',
        'prompt': opt.text_prompt,
        # 'use_cache': False,
        'spawn': False,
    })()

    ip2p = InstructPix2PixGuidance(OmegaConf.create({"min_step_percent": 0.02, "max_step_percent": 0.98}))
    elevation = 0
    random.seed(opt.seed)

    for step in tqdm.tqdm(range(opt.edit_train_steps)):
        optimizer.zero_grad()
        view = random.randint(0, opt.train_cam_num-1)
        cam_poses = torch.from_numpy(orbit_camera(elevation, edit_views[view], radius=opt.cam_radius, opengl=True)).unsqueeze(0).to(device)
        cam_poses[:, :3, 1:3] *= -1
        cam_view = torch.inverse(cam_poses).transpose(1, 2)
        cam_view_proj = cam_view @ proj_matrix
        cam_pos = - cam_poses[:, :3, 3]

        render_image = model.gs.render(gaussians, cam_view.unsqueeze(0), cam_view_proj.unsqueeze(0), cam_pos.unsqueeze(0), scale_modifier=1)['image'].squeeze(0).permute(0, 2, 3, 1)
        train_image = original_images[view]
        gt_image = ip2p(render_image, train_image, prompt_utils)["edit_images"].detach().clone()
        loss = opt.edit_lambda_l1 * torch.nn.functional.l1_loss(render_image, gt_image) + \
                opt.edit_lambda_p * perceptual_loss(render_image.permute(0, 3, 1, 2).contiguous(), gt_image.permute(0, 3, 1, 2).contiguous()) # perceptual input should be 1,C,H,W
        
        loss.backward()
        optimizer.step()
        print(f"[INFO] loss: {loss.detach().item():.6f}")



        if step % 100 == 0: # log
            # fix front view for logging
            vi = 15
            cam_poses_vi = torch.from_numpy(orbit_camera(elevation, edit_views[vi], radius=opt.cam_radius, opengl=True)).unsqueeze(0).to(device)
            cam_poses_vi[:, :3, 1:3] *= -1
            cam_view_vi = torch.inverse(cam_poses_vi).transpose(1, 2)
            cam_view_proj_vi = cam_view_vi @ proj_matrix
            cam_pos_vi = - cam_poses_vi[:, :3, 3]

            render_image_vi = model.gs.render(gaussians, cam_view_vi.unsqueeze(0), cam_view_proj_vi.unsqueeze(0), cam_pos_vi.unsqueeze(0), scale_modifier=1)['image'].squeeze(0).permute(0, 2, 3, 1)
            train_image_vi = original_images[vi]
            gt_image_vi = ip2p(render_image_vi, train_image_vi, prompt_utils)["edit_images"].detach().clone().squeeze(0) * 255
            gt_image_vi = gt_image_vi.clamp(0, 255).cpu().numpy().astype(np.uint8)
            render_image_vi = render_image_vi.squeeze(0) * 255
            render_image_vi = render_image_vi.clamp(0, 255).cpu().detach().numpy().astype(np.uint8)
            Image.fromarray(np.concatenate((gt_image_vi, render_image_vi), axis=1)).save(f'{opt.workspace}/{name}_{step}.png')


    # save gaussians
    model.gs.save_ply(gaussians, os.path.join(opt.workspace, name + '_edit.ply'))
    # render 360 video 
    images = []
    elevation = 0

    with torch.no_grad():
        azimuth = np.arange(0, 360, 2, dtype=np.int32)
        for azi in tqdm.tqdm(azimuth):
                    
            cam_poses = torch.from_numpy(orbit_camera(elevation, azi, radius=opt.cam_radius, opengl=True)).unsqueeze(0).to(device)

            cam_poses[:, :3, 1:3] *= -1 # invert up & forward direction
                    
            # cameras needed by gaussian rasterizer
            cam_view = torch.inverse(cam_poses).transpose(1, 2) # [V, 4, 4]
            cam_view_proj = cam_view @ proj_matrix # [V, 4, 4]
            cam_pos = - cam_poses[:, :3, 3] # [V, 3]

            image = model.gs.render(gaussians, cam_view.unsqueeze(0), cam_view_proj.unsqueeze(0), cam_pos.unsqueeze(0), scale_modifier=1)['image']
            images.append((image.squeeze(1).permute(0,2,3,1).contiguous().float().cpu().numpy() * 255).astype(np.uint8))

        images = np.concatenate(images, axis=0)
        imageio.mimwrite(os.path.join(opt.workspace, name + '_edited.mp4'), images, fps=30)




assert opt.test_path is not None
if os.path.isdir(opt.test_path):
    file_paths = glob.glob(os.path.join(opt.test_path, "*"))
else:
    file_paths = [opt.test_path]
for path in file_paths:
    process(opt, path)

"""
python stylize_multiview.py big --resume pretrained/model_fp16.safetensors --workspace stylize_result --test_path style_test/
"""
