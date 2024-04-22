
import cv2
import numpy as np
import torch
import torch.nn.functional as F
from diffusers import DDIMScheduler, StableDiffusionControlNetImg2ImgPipeline, ControlNetModel
from diffusers.utils.import_utils import is_xformers_available
from diffusers.image_processor import PipelineImageInput
from tqdm import tqdm

import threestudio
from threestudio.models.prompt_processors.base import PromptProcessorOutput
from threestudio.utils.base import BaseObject
from threestudio.utils.misc import C, parse_version
from threestudio.utils.typing import *


@threestudio.register("stable-diffusion-controlnet-guidance")
class StableDiffusionControlNetGuidance(BaseObject):
    @dataclass
    class Config(BaseObject.Config):
        cache_dir: Optional[str] = None
        ddim_scheduler_name_or_path: str = "CompVis/stable-diffusion-v1-4"
        sd_path: str = "runwayml/stable-diffusion-v1-5"
        controlnet_path: str = "lllyasviel/sd-controlnet-canny"
        lora_path: str = "lora_ckpt/checkpoint-2000"

        enable_memory_efficient_attention: bool = False
        enable_sequential_cpu_offload: bool = False
        enable_attention_slicing: bool = False
        enable_channels_last_format: bool = False
        guidance_scale: float = 7.5
        condition_scale: float = 1.5
        grad_clip: Optional[
            Any
        ] = None  # field(default_factory=lambda: [0, 2.0, 8.0, 1000])
        half_precision_weights: bool = True

        fixed_size: int = -1

        min_step_percent: float = 0.02
        max_step_percent: float = 0.98

        diffusion_steps: int = 20

        use_sds: bool = False # if not use_sds: use mse


    cfg: Config

    def configure(self) -> None:
        threestudio.info(f"Loading StableDiffusionControlNetPix2Pix ...")
        self.weights_dtype = (torch.float16 if self.cfg.half_precision_weights else torch.float32)

        self.controlnet = ControlNetModel.from_pretrained(self.cfg.controlnet_path)
        self.pipe = StableDiffusionControlNetImg2ImgPipeline.from_pretrained(self.cfg.sd_path, controlnet=self.controlnet).to(self.device)
        self.pipe.load_lora_weights(self.cfg.lora_path)
        self.scheduler = DDIMScheduler.from_pretrained(
            self.cfg.ddim_scheduler_name_or_path,
            subfolder="scheduler",
            torch_dtype=self.weights_dtype,
            cache_dir=self.cfg.cache_dir,)
        self.scheduler.set_timesteps(self.cfg.diffusion_steps)
        
        # Create model
        self.vae = self.pipe.vae.eval()
        self.unet = self.pipe.unet.eval()

        for p in self.vae.parameters():
            p.requires_grad_(False)
        for p in self.unet.parameters():
            p.requires_grad_(False)
        
        self.num_train_timesteps = self.scheduler.config.num_train_timesteps
        self.set_min_max_steps() # default value

        self.alphas: Float[Tensor, "..."] = self.scheduler.alphas_cumprod.to(self.device)
        self.grad_clip_val: Optional[float] = None

        threestudio.info(f"Loaded StableDiffusionControlNetPix2Pix!")
    
    @torch.cuda.amp.autocast(enabled=False)
    def set_min_max_steps(self, min_step_percent=0.02, max_step_percent=0.98):
        self.min_step = int(self.num_train_timesteps * min_step_percent)
        self.max_step = int(self.num_train_timesteps * max_step_percent)

    @torch.cuda.amp.autocast(enabled=False)
    def forward_unet(
        self,
        latents: Float[Tensor, "..."],
        t: Float[Tensor, "..."],
        encoder_hidden_states: Float[Tensor, "..."],
    ) -> Float[Tensor, "..."]:
        input_dtype = latents.dtype
        return self.unet(
            latents.to(self.weights_dtype),
            t.to(self.weights_dtype),
            encoder_hidden_states=encoder_hidden_states.to(self.weights_dtype),
        ).sample.to(input_dtype)

    @torch.cuda.amp.autocast(enabled=False)
    def encode_images(
        self, imgs: Float[Tensor, "B 3 H W"]
    ) -> Float[Tensor, "B 4 DH DW"]:
        input_dtype = imgs.dtype
        imgs = imgs * 2.0 - 1.0
        posterior = self.vae.encode(imgs.to(self.weights_dtype)).latent_dist
        latents = posterior.sample() * self.vae.config.scaling_factor
        return latents.to(input_dtype)

    @torch.cuda.amp.autocast(enabled=False)
    def encode_cond_images(
        self, imgs: Float[Tensor, "B 3 H W"]
    ) -> Float[Tensor, "B 4 DH DW"]:
        input_dtype = imgs.dtype
        imgs = imgs * 2.0 - 1.0
        posterior = self.vae.encode(imgs.to(self.weights_dtype)).latent_dist
        latents = posterior.mode()
        uncond_image_latents = torch.zeros_like(latents)
        latents = torch.cat([latents, latents, uncond_image_latents], dim=0)
        return latents.to(input_dtype)

    @torch.cuda.amp.autocast(enabled=False)
    def decode_latents(
        self, latents: Float[Tensor, "B 4 DH DW"]
    ) -> Float[Tensor, "B 3 H W"]:
        input_dtype = latents.dtype
        latents = 1 / self.vae.config.scaling_factor * latents
        image = self.vae.decode(latents.to(self.weights_dtype)).sample
        image = (image * 0.5 + 0.5).clamp(0, 1)
        return image.to(input_dtype)
    
    def edit_latents(
        self,
        text_embeddings: Float[Tensor, "BB 77 768"],
        latents: Float[Tensor, "B 4 DH DW"],
        image_cond_latents: Float[Tensor, "B 4 DH DW"],
        t: Int[Tensor, "B"],
    ) -> Float[Tensor, "B 4 DH DW"]:
        self.scheduler.config.num_train_timesteps = t.item()
        self.scheduler.set_timesteps(self.cfg.diffusion_steps)
        with torch.no_grad():
            # add noise
            noise = torch.randn_like(latents)
            latents = self.scheduler.add_noise(latents, noise, t)  # type: ignore
            threestudio.debug("Start editing...")
            # sections of code used from https://github.com/huggingface/diffusers/blob/main/src/diffusers/pipelines/stable_diffusion/pipeline_stable_diffusion_instruct_pix2pix.py
            for i, t in enumerate(self.scheduler.timesteps):
                # predict the noise residual with unet, NO grad!
                with torch.no_grad():
                    # pred noise
                    latent_model_input = torch.cat([latents] * 3)
                    latent_model_input = torch.cat(
                        [latent_model_input, image_cond_latents], dim=1
                    )

                    noise_pred = self.forward_unet(
                        latent_model_input, t, encoder_hidden_states=text_embeddings
                    )

                # perform classifier-free guidance
                noise_pred_text, noise_pred_image, noise_pred_uncond = noise_pred.chunk(
                    3
                )
                noise_pred = (
                    noise_pred_uncond
                    + self.cfg.guidance_scale * (noise_pred_text - noise_pred_image)
                    + self.cfg.condition_scale * (noise_pred_image - noise_pred_uncond)
                )

                # get previous sample, continue loop
                latents = self.scheduler.step(noise_pred, t, latents).prev_sample
            threestudio.debug("Editing finished.")
        return latents
    @torch.no_grad()
    def __call__(
            self,
            prompt: Union[str, List[str]] = None,
            negative_prompt: Optional[Union[str, List[str]]] = None,
            strength: float = 0.8,
            guidance_scale: float = 7.5,
            controlnet_conditioning_scale: Union[float, List[float]] = 0.5,
            # control_guidance_start: Union[float, List[float]] = 0.0,
            # control_guidance_end: Union[float, List[float]] = 1.0,
            num_inference_steps: int = 50,
            cross_attention_kwargs: Optional[Dict[str, Any]] = None,
            image: PipelineImageInput = None,
            control_image: PipelineImageInput = None,
            **kwagrs):
        control_guidance_start, control_guidance_end = (1 * [0.0], 1 * [1.0]) # hash code

        # 1. check inputs. 省略
                