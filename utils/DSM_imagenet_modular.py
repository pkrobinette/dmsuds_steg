"""
Diffusion Sanitization Model.

From: https://github.com/ethz-spylab/diffusion_denoised_smoothing
"""

import torch
import torch.nn as nn 
from utils.guided_diffusion.script_util import (
    model_and_diffusion_defaults,
    create_model_and_diffusion,
    args_to_dict,
)
from transformers import AutoModelForImageClassification

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

class Args:
    image_size=256
    num_channels=256
    num_res_blocks=2
    num_heads=4
    num_heads_upsample=-1
    num_head_channels=64
    attention_resolutions="32,16,8"
    channel_mult=""
    dropout=0.0
    class_cond=False
    use_checkpoint=False
    use_scale_shift_norm=True
    resblock_updown=True
    use_fp16=False
    use_new_attention_order=False
    clip_denoised=True
    num_samples=10000
    batch_size=16
    use_ddim=False
    model_path=""
    classifier_path=""
    classifier_scale=1.0
    learn_sigma=True
    diffusion_steps=1000
    noise_schedule="linear"
    timestep_respacing=None
    use_kl=False
    predict_xstart=False
    rescale_timesteps=False
    rescale_learned_sigmas=False


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