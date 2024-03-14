# add different lr
# main function
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
from core.models import LGM, GaussianModel
from mvdream.pipeline_mvdream import MVDreamPipeline
from argparse import ArgumentParser
from stylize_utils import OptimizationParams
from omegaconf import OmegaConf
from threestudio.models.guidance.instructpix2pix_guidance import InstructPix2PixGuidance
from threestudio.models.prompt_processors.stable_diffusion_prompt_processor import StableDiffusionPromptProcessor
from threestudio.utils.perceptual import PerceptualLoss

from nn_loss import match_colors_for_image_set, NNLoss


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


def get_expon_lr_func(
    lr_init, lr_final, lr_delay_steps=0, lr_delay_mult=1.0, max_steps=1000000
):
    """
    Copied from Plenoxels

    Continuous learning rate decay function. Adapted from JaxNeRF
    The returned rate is lr_init when step=0 and lr_final when step=max_steps, and
    is log-linearly interpolated elsewhere (equivalent to exponential decay).
    If lr_delay_steps>0 then the learning rate will be scaled by some smooth
    function of lr_delay_mult, such that the initial learning rate is
    lr_init*lr_delay_mult at the beginning of optimization but will be eased back
    to the normal learning rate when steps>lr_delay_steps.
    :param conf: config subtree 'lr' or similar
    :param max_steps: int, the number of steps during optimization.
    :return HoF which takes step as input
    """

    def helper(step):
        if step < 0 or (lr_init == 0.0 and lr_final == 0.0):
            # Disable this parameter
            return 0.0
        if lr_delay_steps > 0:
            # A kind of reverse cosine decay.
            delay_rate = lr_delay_mult + (1 - lr_delay_mult) * np.sin(
                0.5 * np.pi * np.clip(step / lr_delay_steps, 0, 1)
            )
        else:
            delay_rate = 1.0
        t = np.clip(step / max_steps, 0, 1)
        log_lerp = np.exp(np.log(lr_init) * (1 - t) + np.log(lr_final) * t)
        return delay_rate * log_lerp

    return helper

def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True


class Gaussians:
    def __init__(self, gaussians, device):
        self.xyz = gaussians[..., :3].detach().clone().to(device).requires_grad_(True)
        self.opacity = gaussians[..., 3:4].detach().clone().to(device).requires_grad_(True)
        self.scale = gaussians[..., 4:7].detach().clone().to(device).requires_grad_(True)
        self.rotation = gaussians[..., 7:11].detach().clone().to(device).requires_grad_(True)
        self.rgb = gaussians[..., 11:].detach().clone().to(device).requires_grad_(True)
    
    def training_setup(self, opt):
        l = [
                {
                    "params": [self.xyz],
                    "lr": opt.position_lr_init * opt.spatial_lr_scale,
                    "name": "xyz",
                },
                {
                    "params": [self.opacity],
                    "lr": opt.opacity_lr,
                    "name": "opacity",
                },
                {
                    "params": [self.scale],
                    "lr": opt.scaling_lr,
                    "name": "scaling",
                },
                {
                    "params": [self.rotation],
                    "lr": opt.rotation_lr,
                    "name": "rotation",
                },
                {
                    "params": [self.rgb],
                    "lr": opt.color_lr,
                    "name": "feature",
                }
            ]

        self.optimizer = torch.optim.Adam(l, lr=0.0, eps=1e-15)

        self.xyz_scheduler_args = get_expon_lr_func(lr_init=opt.position_lr_init*opt.spatial_lr_scale,
                                                            lr_final=opt.position_lr_final*opt.spatial_lr_scale,
                                                            lr_delay_mult=opt.position_lr_delay_mult,
                                                            max_steps=opt.position_lr_max_steps)

    
    def update_lr(self, step):
        for param_group in self.optimizer.param_groups:
            if param_group['name'] == 'xyz':
                lr = self.xyz_scheduler_args(step)
                param_group["lr"] = lr
    
    def capture(self):
        return torch.cat([self.xyz, self.opacity, self.scale, self.rotation, self.rgb], dim=-1) # [B, N, 14]


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
    # TODO: here use more mv, num_frames=20 maybe
    mv_image = pipe('', image, negative_prompt=negative_prompt, guidance_scale=5.0, num_inference_steps=30, elevation=0)
    mv_image = np.stack([mv_image[1], mv_image[2], mv_image[3], mv_image[0]], axis=0) # [4, 256, 256, 3], float32

    # generate gaussians
    input_image = torch.from_numpy(mv_image).permute(0, 3, 1, 2).float().to(device) # [4, 3, 256, 256]
    input_image = F.interpolate(input_image, size=(opt.input_size, opt.input_size), mode='bilinear', align_corners=False)
    input_image = TF.normalize(input_image, IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD)

    input_image = torch.cat([input_image, rays_embeddings], dim=1).unsqueeze(0) # [1, 4, 9, H, W]

    # model.train()

    Gaussian = GaussianModel()

    with torch.autocast(device_type='cuda', dtype=torch.float16):
        gaussians = model.forward_gaussians(input_image) # tensor, no gradient

    Gaussian.load(gaussians)
    Gaussian.traininig_setup(opt)

    print(f"gaussians.shape: {gaussians.shape}, Gaussian.save_gaussians.shape: {Gaussian.save_gaussians().shape}")
    
    model.gs.save_ply(gaussians, os.path.join(opt.workspace, name + '.ply'))

    perceptual_loss = PerceptualLoss().eval().to(device)


    # stylize
    # in face case, there have 65 views for training, 48 views for stylize
    # in this 360 case, initialize 60 views for training and 48 views for stylize

    total_views = np.arange(0, 360, 360//opt.train_cam_num, dtype=np.int32)
    images = []
    # render some images for train and editing
    for azi in tqdm.tqdm(total_views):
        cam_poses = torch.from_numpy(orbit_camera(0, azi, radius=opt.cam_radius, opengl=True)).unsqueeze(0).to(device)
        cam_poses[:, :3, 1:3] *= -1
        cam_view = torch.inverse(cam_poses).transpose(1, 2)
        cam_view_proj = cam_view @ proj_matrix
        cam_pos = - cam_poses[:, :3, 3]
        image = model.gs.render(Gaussian.save_gaussians(), cam_view.unsqueeze(0), cam_view_proj.unsqueeze(0), cam_pos.unsqueeze(0), scale_modifier=1)['image'].squeeze(0)
        images.append(image) # image.shape: 1,3,512,512 [content_image]
    images = torch.cat(images, dim=0) # N_img,3,512,512
    N_img, C, H, W = images.shape
    # resize style image such that its long side matches the long side of content image
    style_img = imageio.imread(opt.style_path).astype(np.float32) / 255.0
    style_h, style_w = style_img.shape[:2]
    content_long_side = max(images[0].shape[1], images[0].shape[2])
    if style_h > style_w:
        style_img = cv2.resize(
            style_img,
            (int(content_long_side / style_h * style_w), content_long_side),
            interpolation=cv2.INTER_AREA,
        )
    else:
        style_img = cv2.resize(
            style_img,
            (content_long_side, int(content_long_side / style_w * style_h)),
            interpolation=cv2.INTER_AREA,
        )
    style_img = cv2.resize(
        style_img,
        (style_img.shape[1] // 2, style_img.shape[0] //2),
        interpolation=cv2.INTER_AREA,
    ) # this is to replace the downsampling in optimization loop

    imageio.imwrite(
        os.path.join(opt.workspace, "style_image.png"),
        np.clip(style_img * 255.0, 0.0, 255.0).astype(np.uint8),
    )
    style_img = torch.from_numpy(style_img).to("cuda") # 160,256,3

    # color transfer
    content_images, color_tf = match_colors_for_image_set(images, style_img) # color_tf: [4,4]
    dtype = images.dtype
    for i in range(len(images)):
        tmp = torch.cat((images[i].reshape(-1, 3), torch.ones((H*W, 1), dtype=dtype, device=device)), dim=-1)
        tmp = tmp @ color_tf[:3,:4].T
        images[i] = tmp.reshape(3, H, W)
    
    nn_loss_fn = NNLoss(device="cuda")
    for step in tqdm.tqdm(range(opt.edit_train_steps)):
        view_stack = list(range(len(total_views)))
        view_index = random.choice(view_stack)
        gt_image = images[view_index].unsqueeze(0)

        cam_pos = torch.from_numpy(orbit_camera(0, total_views[view_index], radius=opt.cam_radius, opengl=True)).unsqueeze(0).to(device)
        cam_pos[:, :3, 1:3] *= -1
        cam_view = torch.inverse(cam_pos).transpose(1, 2)
        cam_view_proj = cam_view @ proj_matrix
        cam_pos = - cam_pos[:, :3, 3]

        render_image = model.gs.render(Gaussian.save_gaussians(), cam_view.unsqueeze(0), cam_view_proj.unsqueeze(0), cam_pos.unsqueeze(0), scale_modifier=1)['image'].squeeze(0)
        # 1,3,512,512

        nn_loss, _, content_loss = nn_loss_fn(
            F.interpolate(render_image, size=None, scale_factor=0.5, mode="bilinear"),
            style_img.permute(2, 0, 1).unsqueeze(0),
            loss_names=["nn_loss", "content_loss"],
            contents=F.interpolate(gt_image, size=None, scale_factor=0.5, mode="bilinear"),
        )
        loss = nn_loss*10 + content_loss * 0.05
        print(nn_loss, content_loss)
        # reconstruct loss:
        # reconstruct = F.mse_loss(render_image, gt_image)
        # loss += reconstruct
        # print(nn_loss, content_loss, reconstruct)
        loss.backward()
        Gaussian.optimizer.step()
        Gaussian.optimizer.zero_grad()

        

        print(f"[INFO] loss: {loss.detach().item():.6f}")

        with torch.no_grad():
            if step % 10 == 0: # log
                # fix front view for logging
                vi = 0
                cam_poses_vi = torch.from_numpy(orbit_camera(0, total_views[vi], radius=opt.cam_radius, opengl=True)).unsqueeze(0).to(device)
                cam_poses_vi[:, :3, 1:3] *= -1
                cam_view_vi = torch.inverse(cam_poses_vi).transpose(1, 2)
                cam_view_proj_vi = cam_view_vi @ proj_matrix
                cam_pos_vi = - cam_poses_vi[:, :3, 3]
                render_image_vi = model.gs.render(Gaussian.save_gaussians(), cam_view_vi.unsqueeze(0), cam_view_proj_vi.unsqueeze(0), cam_pos_vi.unsqueeze(0), scale_modifier=1)['image'].squeeze(0).permute(0, 2, 3, 1)
                render_image_vi = render_image_vi.squeeze(0) * 255
                render_image_vi = render_image_vi.clamp(0, 255).cpu().detach().numpy().astype(np.uint8)
                Image.fromarray(render_image_vi).save(f'{opt.workspace}/{name}_{step}.png')

            
    gaussians = Gaussian.save_gaussians()
    # save gaussians
    model.gs.save_ply(gaussians, os.path.join(opt.workspace, name + '_edit.ply'))
    # render 360 video 
    images = []
    elevation = 0

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
        imageio.mimwrite(os.path.join(opt.workspace, name + '_edited.mp4'), images, fps=30)




assert opt.test_path is not None
if os.path.isdir(opt.test_path):
    file_paths = glob.glob(os.path.join(opt.test_path, "*"))
else:
    file_paths = [opt.test_path]
for path in file_paths:
    setup_seed(20240226)
    process(opt, path)

"""
python stylize.py big --resume pretrained/model_fp16.safetensors --workspace stylize_result --test_path style_test/
"""
