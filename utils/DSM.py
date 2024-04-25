"""
Diffusion Sanitization Model.

From: https://github.com/ethz-spylab/diffusion_denoised_smoothing
"""

import torch
import torch.nn as nn 
from utils.improved_diffusion.script_util import (
    model_and_diffusion_defaults,
    create_model_and_diffusion,
    args_to_dict,
)
from transformers import AutoModelForImageClassification

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

class Args:
    image_size=32
    num_channels=128
    num_res_blocks=3
    num_heads=4
    num_heads_upsample=-1
    attention_resolutions="16,8"
    dropout=0.3
    learn_sigma=True
    sigma_small=False
    class_cond=False
    diffusion_steps=4000
    noise_schedule="cosine"
    timestep_respacing=""
    use_kl=False
    predict_xstart=False
    rescale_timesteps=True
    rescale_learned_sigmas=True
    use_checkpoint=False
    use_scale_shift_norm=True


class DiffusionSanitizationModel(nn.Module):
    def __init__(self, path):
        super().__init__()
        model, diffusion = create_model_and_diffusion(
            **args_to_dict(Args(), model_and_diffusion_defaults().keys())
        )
            
        if DEVICE == "cpu":
            model.load_state_dict(
                torch.load(path ,map_location=torch.device('cpu'))
            )
        else:
            model.load_state_dict(
                torch.load(path)
            )
        model.eval().to(DEVICE)

        self.model = model 
        self.diffusion = diffusion

    
    def sanitize(self, x_start, t=400, add_noise=True):
        """
        Sanitize an image that potentially contains steganographic material.

        Parameters
        ---------
        x_start : tensor
            A batch of tensor images (batch_size, channels, H, W)
        t : int
            Number of timesteps to add/remove noise
        add_noise : bool
            If true, add noise t steps, then denoise t steps. If false, denoise t steps.

        Returns
        -------
        out : tensor
            A batch of sanitized tensor images (batch_size, channels, H, W)
        """
        t_batch = torch.tensor([t] * len(x_start)).to(DEVICE)

        if add_noise:
            noise = torch.randn_like(x_start)
            x_start = self.diffusion.q_sample(x_start=x_start, t=t_batch, noise=noise)

        with torch.no_grad():
            out = self.diffusion.p_sample(
                    self.model,
                    x_start,
                    t_batch,
                    clip_denoised=True
                )['pred_xstart']

        return torch.clip(out, 0, 1)

    def sanitize_ddim(self, x_start, t=400, add_noise=True):
        """
        Sanitize an image that potentially contains steganographic material using DDIM instead of DDPM.

        Parameters
        ---------
        x_start : tensor
            A batch of tensor images (batch_size, channels, H, W)
        t : int
            Number of timesteps to add/remove noise
        add_noise : bool
            If true, add noise t steps, then denoise t steps. If false, denoise t steps.

        Returns
        -------
        out : tensor
            A batch of sanitized tensor images (batch_size, channels, H, W)
        """
        t_batch = torch.tensor([t] * len(x_start)).to(DEVICE)

        if add_noise:
            noise = torch.randn_like(x_start)
            x_start = self.diffusion.q_sample(x_start=x_start, t=t_batch, noise=noise)

        # ddim sample instead of p sample.
        with torch.no_grad():
            out = self.diffusion.ddim_sample(
                    self.model,
                    x_start,
                    t_batch,
                    clip_denoised=True
                )['pred_xstart']

        return torch.clip(out, 0, 1)