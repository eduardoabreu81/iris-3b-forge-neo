# Iris 3B support for Forge Neo
# Checks that the installed Forge Neo has every piece the Iris backend builds on, before anything is patched.
# Same mechanism as the MiniMax H3 extension's compat.py (forge_h3/native/compat.py).

import importlib
import inspect

# what the backend imports or patches, by module; an older Forge Neo misses some of these
REQUIRED = {
    "backend.attention": ("attention_function",),
    "backend.memory_management": ("load_models_gpu",),
    "backend.operations": ("using_forge_operations",),
    "backend.state_dict": ("load_state_dict",),
    "backend.loader": ("load_huggingface_component", "possible_models"),
    "backend.nn.krea": ("SingleStreamDiT",),
    "backend.diffusion_engine.base": ("ForgeDiffusionEngine", "ForgeObjects"),
    "backend.patcher.clip": ("CLIP",),
    "backend.patcher.unet": ("UnetPatcher",),
    "backend.patcher.vae": ("VAE",),
    "backend.text_processing._comfy": ("SDClipModel", "SDTokenizer", "INF"),
    "backend.text_processing.emphasis": ("EmphasisNone", "uses_emphasis"),
    "modules.sd_schedulers": ("Scheduler", "all_schedulers", "schedulers", "schedulers_map"),
    "modules_forge.presets": ("PresetArch", "SAMPLERS", "SCHEDULERS", "STEPS", "CFG", "register"),
    "huggingface_guess.detection": ("detect_unet_config",),
    "huggingface_guess.model_list": ("models", "BASE", "ModelType"),
    "huggingface_guess.latent": ("RGB", "LatentFormat"),
}

# the transformer goes through this branch of the loader, the text encoder through that one
LOADER_BRANCHES = ("Krea2Transformer2DModel", "Qwen3VLModel")


def check() -> list[str]:
    problems: list[str] = []

    for module_name, names in REQUIRED.items():
        try:
            module = importlib.import_module(module_name)
        except Exception as e:
            problems.append(f"{module_name} could not be imported ({type(e).__name__}: {e})")
            continue
        missing = [n for n in names if not hasattr(module, n)]
        if missing:
            problems.append(f"{module_name} is missing {', '.join(missing)}")

    if not problems:
        from backend import loader

        source = inspect.getsource(loader.load_huggingface_component)
        missing = [b for b in LOADER_BRANCHES if f'"{b}"' not in source]
        if missing:
            problems.append(f"the model loader has no {', '.join(missing)} branch")

    return problems


def report(problems: list[str]) -> None:
    print("[Iris 3B] this Forge Neo is too old or too new for the extension, so it stays disabled "
          "(Forge Neo itself keeps working as usual):")
    for problem in problems:
        print(f"          {problem}")
