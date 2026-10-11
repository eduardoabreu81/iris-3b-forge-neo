# Iris 3B support for Forge Neo
# Runtime hooks into Forge Neo's loader. Every hook hands anything that is not Iris to the original function
# untouched; Forge Neo itself stays unchanged. Follows the MiniMax H3 extension's patches.py.

import logging
import os
import types

import torch
from backend import loader
from backend.nn import krea
from huggingface_guess import detection, model_list

from . import model
from .engine import IrisEngine
from .transformer import IrisDiT

logger = logging.getLogger("forge_iris")

# a Qwen3-VL 4B language tower: 36 layers, hidden 2560
TE_KEYS = ("model.layers.0.self_attn.q_norm.weight", "model.layers.35.self_attn.q_proj.weight")
MISSING_TE = ("Iris needs its Qwen3-VL 4B text encoder: select qwen3vl_4b_bf16.safetensors under "
              "VAE / Text Encoder")

_applied = False


class _PixelPassthrough(torch.nn.Module):
    """The Iris has no VAE: the DiT denoises RGB directly, so the pipeline's VAE slot holds this empty stand-in.
    Forge's base engine writes latent_format onto it; the engine's decode/encode are the identity and never use it."""

    def __init__(self):
        super().__init__()
        self.config = types.SimpleNamespace(latent_channels=3)


def _hook_detection() -> None:
    original = detection.detect_unet_config

    def detect_unet_config(state_dict, key_prefix, *args, **kwargs):
        dit_config = model.detect(state_dict, key_prefix)
        if dit_config is not None:
            return dit_config
        return original(state_dict, key_prefix, *args, **kwargs)

    detection.detect_unet_config = detect_unet_config


def _hook_components() -> None:
    original = loader.load_huggingface_component

    def load_huggingface_component(guess, component_name, lib_name, cls_name, repo_path, state_dict):
        if not isinstance(guess, model.Iris):
            return original(guess, component_name, lib_name, cls_name, repo_path, state_dict)

        if cls_name == "AutoencoderKLIris":
            return _PixelPassthrough()
        if cls_name == "IrisTransformer2DModel":
            # the Krea 2 branch is Forge's generic DiT load: dtype, quantization (INT8 ConvRot) and device handling;
            # it imports backend.nn.krea.SingleStreamDiT inside the function, so the swap routes the build to us
            original_krea = krea.SingleStreamDiT
            krea.SingleStreamDiT = IrisDiT
            try:
                return original(guess, component_name, lib_name, "Krea2Transformer2DModel", repo_path, state_dict)
            finally:
                krea.SingleStreamDiT = original_krea
        if cls_name == "Qwen3VLModel":
            # the Forge loads the 4B natively for Krea 2; the hook only explains a missing text encoder
            if not isinstance(state_dict, dict) or len(state_dict) <= 16 or not all(k in state_dict for k in TE_KEYS):
                raise ValueError(MISSING_TE)

        return original(guess, component_name, lib_name, cls_name, repo_path, state_dict)

    loader.load_huggingface_component = load_huggingface_component


def _register_scheduler() -> None:
    from modules import sd_schedulers, shared

    from . import scheduler as iris_scheduler

    if iris_scheduler.LABEL in sd_schedulers.schedulers_map:
        return
    entry = sd_schedulers.Scheduler(iris_scheduler.NAME, iris_scheduler.LABEL, iris_scheduler.sigmas,
                                    aliases=list(iris_scheduler.ALIASES))
    sd_schedulers.all_schedulers.append(entry)
    if entry.label not in shared.opts.hide_schedulers:
        sd_schedulers.schedulers.append(entry)
    sd_schedulers.schedulers_map.update({entry.name: entry, entry.label: entry,
                                         **{alias: entry for alias in (entry.aliases or [])}})


def apply() -> None:
    global _applied
    if _applied:
        return

    _hook_detection()
    if model.Iris not in model_list.models:
        model_list.models.append(model.Iris)
    if IrisEngine not in loader.possible_models:
        loader.possible_models = (*loader.possible_models, IrisEngine)
    _hook_components()
    try:
        _register_scheduler()
    except Exception as e:
        # the model still works: pick Simple, or paste the schedules from the wiki
        logger.warning(f"[Iris 3B] could not add the Iris scheduler: {e}")
    try:
        from . import presets

        presets.register()
    except Exception as e:
        # the model still works: dpmpp_2m, Simple, 50 steps, CFG 3
        logger.warning(f"[Iris 3B] could not add the iris UI preset: {e}")

    _applied = True
    print(f"[Iris 3B] native backend enabled (torch {torch.__version__}, {os.path.basename(model.CONFIG_DIR)})")
