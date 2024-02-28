# load face.ply, then stylize. Compare with GaussianEditor


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

from colmap_loader import *
CUDA_LAUNCH_BLOCKING=1

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


def config_optimizers(opt):
    optimizer_params = OptimizationParams(
        parser = ArgumentParser(description="Stylize script parameters"),
        max_steps = opt.edit_train_steps,
        lr_scaler = opt.gs_lr_scaler,
        lr_final_scaler = opt.gs_lr_end_scaler,
        color_lr_scaler = opt.color_lr_scaler,
        opacity_lr_scaler = opt.opacity_lr_scaler,
        scaling_lr_scaler = opt.scaling_lr_scaler,
        rotation_lr_scaler = opt.rotation_lr_scaler,
    )
    optimizer = OmegaConf.create(vars(optimizer_params))
    return optimizer


def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True

# process function
def process(opt: Options):
    name = "face"
    os.makedirs(opt.workspace, exist_ok=True)

    data_path = opt.data_path

    gaussians = model.gs.load_ply(os.path.join(opt.workspace, 'face.ply')).to(device)
    gaussians.unsqueeze_(0).requires_grad_(True)

    optimizer = torch.optim.AdamW([gaussians], lr=opt.lr, weight_decay=0.05, betas=(0.9, 0.95))

    # stylize
    # in face case, there have 65 views for training, 48 views for stylize
    # in this 360 case, initialize 60 views for training and 48 views for stylize
    perceptual_loss = PerceptualLoss().eval().to(device)

    camlist = []

    try:
        cameras_extrinsic_file = os.path.join(data_path, 'sparse/0', 'images.bin')
        cameras_intrinsic_file = os.path.join(data_path, 'sparse/0', 'cameras.bin')
        cam_extrinsics = read_extrinsics_binary(cameras_extrinsic_file)
        cam_intrinsics = read_intrinsics_binary(cameras_intrinsic_file)
    except:
        cameras_extrinsic_file = os.path.join(data_path, 'sparse/0', 'images.txt')
        cameras_intrinsic_file = os.path.join(data_path, 'sparse/0', 'cameras.txt')
        cam_extrinsics = read_extrinsics_binary(cameras_extrinsic_file)
        cam_intrinsics = read_intrinsics_binary(cameras_intrinsic_file)

    cam_infos_unsorted = readColmapCameras(cam_extrinsics=cam_extrinsics, cam_intrinsics=cam_intrinsics, images_folder=os.path.join(data_path, "images"))
    cam_infos = sorted(cam_infos_unsorted, key=lambda x: x.image_name)

    nerf_normalization = getNerfppNorm(cam_infos)

    cameras_extent = nerf_normalization['radius']

    # cameraList_from_camInfos
    for id, c in enumerate(cam_infos):
        camlist.append(loadCam(device, id, c))

    # total: 65views, random chose 48views for editing
    # TODO: 这里我暂时没写完
    # original 3DGS rendering function
    image_height, image_width = camlist[0].image_height, camlist[0].image_width

    view_index = random.sample(range(0, len(camlist)), min(len(camlist), opt.edit_cam_num))
    edit_cameras = [camlist[i] for i in view_index]
    # train_images = edit_cameras.original_image # image clamped into 0,1
    edit_images = {}

    prompt_utils = StableDiffusionPromptProcessor({
        'pretrained_model_name_or_path': 'runwayml/stable-diffusion-v1-5',
        'prompt': opt.text_prompt,
        'spawn': False,
    })()
            
    ip2p = InstructPix2PixGuidance(OmegaConf.create({"min_step_percent": 0.02, "max_step_percent": 0.98}))
    view_index_stack = list(range(len(edit_cameras)))
    elevation = 0
    for step in tqdm.tqdm(range(opt.edit_train_steps)):
        optimizer.zero_grad()

        if not view_index_stack:
            view_index_stack = list(range(len(edit_cameras)))

        view_index = random.choice(view_index_stack)
        view_index_stack.remove(view_index)

        cam = edit_cameras[view_index]

        render_image = model.gs.render_face(gaussians, cam)['image'] # CHW
        render_image = render_image.unsqueeze(0).permute(0, 2, 3, 1) # 1,H,W,C
        original_image = cam.original_image.unsqueeze(0).permute(0, 2, 3, 1)

        if view_index not in edit_images or (opt.per_editing_steps > 0 and opt.edit_begin_step < step < opt.edit_util_step and step % opt.per_editing_steps == 0):
            result = ip2p(render_image, original_image, prompt_utils)
            edit_images[view_index] = result["edit_images"].detach().clone() # 1,H,W,C
        
        gt_image = edit_images[view_index]

        loss = opt.edit_lambda_l1 * torch.nn.functional.l1_loss(render_image, gt_image) + \
                opt.edit_lambda_p * perceptual_loss(render_image.permute(0, 3, 1, 2).contiguous(), gt_image.permute(0, 3, 1, 2).contiguous()) # perceptual input should be 1,C,H,W
        loss += F.mse_loss(render_image, gt_image)

        loss.backward()
        optimizer.step()

        print(f"[INFO] loss: {loss.detach().item():.6f}")

        # save rendered image
        tmp = gt_image.squeeze(0) * 255 
        tmp = tmp.clamp(0, 255).cpu().numpy().astype(np.uint8)
        Image.fromarray(tmp).save(f'{opt.workspace}/{name}_edited_{step}.png')


    model.gs.save_ply(gaussians, os.path.join(opt.workspace, name + '_edit.ply'))
    images = []
    elevation = 0

    with torch.no_grad():
        for cam in camlist:
            image = model.gs.render_face(gaussians, cam)['image']
            tmp = image.permute(1, 2, 0) * 255
            tmp.clamp(0, 255).cpu().numpy().astype(np.uint8)
            Image.fromarray(tmp).save(f'{opt.workspace}/{name}_{cam.image_name}.png')

    # with torch.no_grad():
    #     azimuth = np.arange(0, 360, 2, dtype=np.int32)
    #     for azi in tqdm.tqdm(azimuth):
                    
    #         cam_poses = torch.from_numpy(orbit_camera(elevation, azi, radius=opt.cam_radius, opengl=True)).unsqueeze(0).to(device)

    #         cam_poses[:, :3, 1:3] *= -1 # invert up & forward direction
                    
    #         # cameras needed by gaussian rasterizer
    #         cam_view = torch.inverse(cam_poses).transpose(1, 2) # [V, 4, 4]
    #         cam_view_proj = cam_view @ proj_matrix # [V, 4, 4]
    #         cam_pos = - cam_poses[:, :3, 3] # [V, 3]

    #         image = model.gs.render(gaussians, cam_view.unsqueeze(0), cam_view_proj.unsqueeze(0), cam_pos.unsqueeze(0), scale_modifier=1)['image']
    #         images.append((image.squeeze(1).permute(0,2,3,1).contiguous().float().cpu().numpy() * 255).astype(np.uint8))

    #     images = np.concatenate(images, axis=0)
    #     imageio.mimwrite(os.path.join(opt.workspace, name + '_edited.mp4'), images, fps=30)

 



    # images = []
    # for cam in camlist:
    #     # model.gs.render(render_size=(camlist[0].image_height, camlist[0].image_width))
    #     image = model.gs.render_face(gaussians, cam)['image']
    #     images.append(image)

    # total_views = np.arange(0, 360, 360//opt.train_cam_num, dtype=np.int32)
    # view_index = random.sample(range(0, opt.train_cam_num), min(opt.train_cam_num, opt.edit_cam_num))
    # edit_views = [total_views[i] for i in view_index]
    # images = []
    # # render some images for train and editing
    # for azi in tqdm.tqdm(total_views):
    #     cam_poses = torch.from_numpy(orbit_camera(0, azi, radius=opt.cam_radius, opengl=True)).unsqueeze(0).to(device)
    #     cam_poses[:, :3, 1:3] *= -1
    #     cam_view = torch.inverse(cam_poses).transpose(1, 2)
    #     cam_view_proj = cam_view @ proj_matrix
    #     cam_pos = - cam_poses[:, :3, 3]
    #     image = model.gs.render(gaussians, cam_view.unsqueeze(0), cam_view_proj.unsqueeze(0), cam_pos.unsqueeze(0), scale_modifier=1)['image']
    #     images.append(image)
    # train_images = [images[i] for i in view_index] # use for editing
    # edit_images = {}

    # prompt_utils = StableDiffusionPromptProcessor({
    #     'pretrained_model_name_or_path': 'runwayml/stable-diffusion-v1-5',
    #     'prompt': opt.text_prompt,
    #     # 'use_cache': False,
    #     'spawn': False,
    # })()

    # # edit
    # ip2p = InstructPix2PixGuidance(OmegaConf.create({"min_step_percent": 0.02, "max_step_percent": 0.98}))
    # view_index_stack = list(range(len(edit_views)))
    # elevation = 0

    # for step in tqdm.tqdm(range(opt.edit_train_steps)):
    #     # model.train()
    #     optimizer.zero_grad()

    #     if not view_index_stack:
    #         view_index_stack = list(range(len(edit_views)))
        
    #     view_index = random.choice(view_index_stack)
    #     view_index_stack.remove(view_index)
        
    #     cam_pos = torch.from_numpy(orbit_camera(elevation, edit_views[view_index], radius=opt.cam_radius, opengl=True)).unsqueeze(0).to(device)
    #     cam_pos[:, :3, 1:3] *= -1
    #     cam_view = torch.inverse(cam_pos).transpose(1, 2)
    #     cam_view_proj = cam_view @ proj_matrix
    #     cam_pos = - cam_pos[:, :3, 3]

    #     render_image = model.gs.render(gaussians, cam_view.unsqueeze(0), cam_view_proj.unsqueeze(0), cam_pos.unsqueeze(0), scale_modifier=1)['image']
    #     # render_image shape and train_image shape: [1,1,3,512,512]
    #     render_image_reshape = render_image.squeeze(0).permute(0, 2, 3, 1)

    #     # loss calculate
    #     if view_index not in edit_images or (opt.per_editing_steps > 0 and opt.edit_begin_step < step < opt.edit_util_step and step % opt.per_editing_steps == 0):
    #         train_img = train_images[view_index].squeeze(0).permute(0, 2, 3, 1) # 1,H,W,C
    #         result = ip2p(render_image_reshape, train_img, prompt_utils)
    #         # render_image shape should be 1,512,512,3
    #         edit_images[view_index] = result["edit_images"].detach().clone() # 1,H,W,C
    #         tmp = edit_images[view_index].squeeze(0) * 255
    #         tmp = tmp.clamp(0, 255).cpu().numpy().astype(np.uint8)
    #         # tmp2 = render_image_reshape.squeeze(0) * 255
    #         # tmp2 = tmp2.clamp(0, 255).cpu().detach().numpy().astype(np.uint8)
    #         image = Image.fromarray(tmp).save(f'{opt.workspace}/{name}_{step}.png')

    #     # print(edit_images.keys(), len(edit_images.keys()))
    #     gt_image = edit_images[view_index] 
        


    #     loss = opt.edit_lambda_l1 * torch.nn.functional.l1_loss(render_image_reshape, gt_image) + \
    #             opt.edit_lambda_p * perceptual_loss(render_image_reshape.permute(0, 3, 1, 2).contiguous(), gt_image.permute(0, 3, 1, 2).contiguous()) # perceptual input should be 1,C,H,W
    #      # TODO: loss add model loss
        
    #     # mse loss for rendering
    #     loss += F.mse_loss(render_image.squeeze(0).permute(0, 2, 3, 1), gt_image)
        
    #     loss.backward()
    #     # l[-1]['params'].require_grad = False
    #     with torch.no_grad():
    #         gaussians.grad[..., :11] = 0
    #         # grad clipping      
    #     optimizer.step()
    #     # scheduler.step()

    #     print(f"[INFO] loss: {loss.detach().item():.6f}")

    # # save gaussians
    # model.gs.save_ply(gaussians, os.path.join(opt.workspace, name + '_edit.ply'))
    # # render 360 video 
    # images = []
    # elevation = 0

    # with torch.no_grad():
    #     azimuth = np.arange(0, 360, 2, dtype=np.int32)
    #     for azi in tqdm.tqdm(azimuth):
                    
    #         cam_poses = torch.from_numpy(orbit_camera(elevation, azi, radius=opt.cam_radius, opengl=True)).unsqueeze(0).to(device)

    #         cam_poses[:, :3, 1:3] *= -1 # invert up & forward direction
                    
    #         # cameras needed by gaussian rasterizer
    #         cam_view = torch.inverse(cam_poses).transpose(1, 2) # [V, 4, 4]
    #         cam_view_proj = cam_view @ proj_matrix # [V, 4, 4]
    #         cam_pos = - cam_poses[:, :3, 3] # [V, 3]

    #         image = model.gs.render(gaussians, cam_view.unsqueeze(0), cam_view_proj.unsqueeze(0), cam_pos.unsqueeze(0), scale_modifier=1)['image']
    #         images.append((image.squeeze(1).permute(0,2,3,1).contiguous().float().cpu().numpy() * 255).astype(np.uint8))

    #     images = np.concatenate(images, axis=0)
    #     imageio.mimwrite(os.path.join(opt.workspace, name + '_edited.mp4'), images, fps=30)



setup_seed(20240226)
process(opt)

"""
python face_stylize.py big --resume pretrained/model_fp16.safetensors --workspace face --data_path /home/shiyaoxu/datasets/face
"""
