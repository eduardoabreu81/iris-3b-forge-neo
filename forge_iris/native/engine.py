# Iris 3B support for Forge Neo
# The Iris diffusion engine: text-to-image and image-to-image in pixel space, no VAE.
#
# The Forge pipeline always decodes and encodes through decode_first_stage / encode_first_stage, so both are the
# identity (the DiT denoises RGB in [-1, 1] directly, as the PiD engine does). The sampler's "latent" is the image.

import torch
from backend import memory_management
from backend.diffusion_engine.base import ForgeDiffusionEngine, ForgeObjects
from backend.patcher.clip import CLIP
from backend.patcher.unet import UnetPatcher
from backend.patcher.vae import VAE

from .model import Iris
from .text_engine import TEXT_LENGTH, IrisTextEngine


class IrisEngine(ForgeDiffusionEngine):
    matched_guesses = [Iris]

    def __init__(self, estimated_config, huggingface_components):
        super().__init__(estimated_config, huggingface_components)

        clip = CLIP(model_dict={"qwen3vl_4b": huggingface_components["text_encoder"]},
                    tokenizer_dict={"qwen3vl_4b": huggingface_components["tokenizer"]})

        # the empty stand-in the loader hook answered the pipeline's VAE slot with
        vae = VAE(model=huggingface_components["vae"])
        vae.upscale_ratio = 1
        vae.downscale_ratio = 1
        vae.latent_channels = 3

        unet = UnetPatcher.from_model(model=huggingface_components["transformer"], diffusers_scheduler=None,
                                      k_predictor=self._get_predictor(), config=estimated_config)

        self.text_engine_iris = IrisTextEngine(text_encoder=clip.cond_stage_model.qwen3vl_4b,
                                               tokenizer=clip.tokenizer.qwen3vl_4b,
                                               dit=unet.model)

        self.forge_objects = ForgeObjects(unet=unet, clip=clip, vae=vae, clipvision=None)
        self.forge_objects_original = self.forge_objects.shallow_copy()
        self.forge_objects_after_applying_lora = self.forge_objects.shallow_copy()

        self.is_iris = True

    @torch.inference_mode()
    def get_learned_conditioning(self, prompt: list[str]):
        memory_management.load_models_gpu([self.forge_objects.clip.patcher])
        return self.text_engine_iris(prompt)

    @torch.inference_mode()
    def get_prompt_lengths_on_ui(self, prompt: str) -> tuple[int, int]:
        token_count = self.text_engine_iris.tokenize(prompt)
        return token_count, max(TEXT_LENGTH, token_count)

    @torch.inference_mode()
    def encode_first_stage(self, x: torch.Tensor):
        # pixels are their own latent
        return x

    @torch.inference_mode()
    def decode_first_stage(self, x: torch.Tensor):
        return x
