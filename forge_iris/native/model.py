# Iris 3B support for Forge Neo
# huggingface_guess registration, following the MiniMax H3 extension's model.py (forge_h3/native/model.py).

import os

import torch
from huggingface_guess import latent
from huggingface_guess.model_list import BASE, ModelType

from . import detect as detection

# absolute path: the loader joins it with its own huggingface folder, and os.path.join keeps an absolute second part
CONFIG_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "huggingface", "iris-3b")


class Iris(BASE):
    huggingface_repo = CONFIG_DIR

    unet_config = {"image_model": "iris"}

    # rectified flow: the DiT embeds timestep = sigma * 1000; the "Iris" scheduler supplies the sigma grid
    sampling_settings = {"shift": 4.0, "multiplier": 1000}

    # ComfyUI-Iris runs the same weights with memory_usage_factor 0.3
    memory_usage_factor = 0.3
    supported_inference_dtypes = [torch.bfloat16, torch.float16, torch.float32]

    unet_extra_config = {}
    # pixel-space: the "latent" is the RGB image itself, so the Forge RGB preview is exact
    latent_format = latent.RGB

    # the checkpoint carries no VAE; the loader's vae bucket stays empty and the extension's loader hook
    # answers it with an empty stand-in module
    vae_key_prefix = ["vae."]
    text_encoder_key_prefix = ["text_encoders."]

    unet_target = "transformer"

    def clip_target(self, state_dict={}):
        # the same Qwen3-VL 4B the Forge Neo loads natively for Krea 2
        return {"qwen3vl_4b.transformer": "text_encoder"}

    def model_type(self, state_dict):
        return ModelType.FLOW


def detect(state_dict: dict, key_prefix: str) -> dict | None:
    return detection.detect(state_dict, key_prefix)
