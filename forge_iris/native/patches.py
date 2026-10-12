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

from .. import checkpoints
from . import model
from .engine import IrisEngine
from .transformer import IrisDiT

logger = logging.getLogger("forge_iris")

# a Qwen3-VL 4B language tower: 36 layers, hidden 2560. Forge renames the language keys inside its own branch, so
# the hook sees them as the file stores them: model.language_model.* (Comfy-Org's qwen3vl_4b_bf16) or model.*
TE_PREFIXES = ("model.language_model.", "model.")
TE_KEYS = ("layers.0.self_attn.q_norm.weight", "layers.35.self_attn.q_proj.weight")
TE_NORM = "layers.0.input_layernorm.weight"
TE_HIDDEN = 2560
MISSING_TE = ("Iris needs its Qwen3-VL 4B text encoder: select qwen3vl_4b_bf16.safetensors under "
              "VAE / Text Encoder")
WRONG_TE = ("Iris needs the Qwen3-VL 4B text encoder, and the one selected under VAE / Text Encoder is another "
            "size: select qwen3vl_4b_bf16.safetensors")


def check_text_encoder(state_dict) -> None:
    """Raise the user-facing message when the selected text encoder cannot be the Qwen3-VL 4B language tower."""
    if not isinstance(state_dict, dict) or len(state_dict) <= 16:
        raise ValueError(MISSING_TE)
    prefix = next((p for p in TE_PREFIXES if all(f"{p}{k}" in state_dict for k in TE_KEYS)), None)
    if prefix is None:
        raise ValueError(MISSING_TE)
    norm = state_dict.get(f"{prefix}{TE_NORM}")
    if norm is not None and hasattr(norm, "shape") and norm.shape[0] != TE_HIDDEN:
        raise ValueError(WRONG_TE)

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
        if checkpoints.kind_of(state_dict, key_prefix=key_prefix) == checkpoints.FOREIGN:
            # Forge would only say "Failed to recognize model"
            raise ValueError(checkpoints.FOREIGN_MESSAGE)
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
            check_text_encoder(state_dict)

        return original(guess, component_name, lib_name, cls_name, repo_path, state_dict)

    loader.load_huggingface_component = load_huggingface_component


def _register_scheduler() -> None:
    from modules import sd_schedulers, shared

    from . import scheduler as iris_scheduler

    for name, label, function, aliases in (
            (iris_scheduler.NAME, iris_scheduler.LABEL, iris_scheduler.sigmas, iris_scheduler.ALIASES),
            (iris_scheduler.TURBO_NAME, iris_scheduler.TURBO_LABEL, iris_scheduler.turbo_sigmas,
             iris_scheduler.TURBO_ALIASES)):
        if label in sd_schedulers.schedulers_map:
            continue
        entry = sd_schedulers.Scheduler(name, label, function, aliases=list(aliases))
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
        from . import sampler

        sampler.register()
    except Exception as e:
        # the model still works: Euler is exact first order on the flow, or DPM++ 2M SDE with eta 0
        logger.warning(f"[Iris 3B] could not add the Iris DPM++ 2M sampler: {e}")
    # Forge's built-in LoRA extension loads before this one; when it does not, the script retries per generation
    from . import lora

    lora.install()
    try:
        from . import presets

        presets.register()
    except Exception as e:
        # the model still works: Iris DPM++ 2M, Iris, 50 steps, CFG 3 by hand
        logger.warning(f"[Iris 3B] could not add the iris UI preset: {e}")

    _applied = True
    print(f"[Iris 3B] native backend enabled (torch {torch.__version__}, {os.path.basename(model.CONFIG_DIR)})")
