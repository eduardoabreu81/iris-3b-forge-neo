"""Minimal stand-ins for the Forge Neo modules the extension imports, for CPU tests without Forge.

Only the names the extension touches exist; anything else raises AttributeError, so a test notices when the
extension starts depending on more of Forge. Same shape as the MiniMax H3 extension's forge_stubs.
"""

import types


def module(name, **attributes):
    result = types.ModuleType(name)
    for key, value in attributes.items():
        setattr(result, key, value)
    return result


class FakeHFTokenizer:
    """One token per character, ids above the special range; PAD 151643 never appears in text."""

    def encode(self, text, add_special_tokens=True):
        return [1000 + ord(c) for c in text]


class FakeSDTokenizer:
    def __init__(self, tokenizer_object=None, **kwargs):
        self.tokenizer = tokenizer_object


class FakeSDClipModel:
    """encode_token_weights as Forge's does with a layer list: states (B, L, T, D) marked with the token count,
    mask (B, T) 1 until the first PAD."""

    def __init__(self, text_encoder=None, **kwargs):
        self.text_encoder = text_encoder

    def encode_token_weights(self, token_weight_pairs):
        import torch

        tokens = [pair[0] for pair in token_weight_pairs[0]]
        length, dim, layers = len(tokens), 24, 12
        states = torch.full((1, layers, length, dim), float(length))
        pad = 151643
        mask = torch.tensor([[0.0 if token == pad else 1.0 for token in tokens]])
        return states, None, {"attention_mask": mask}


class FakeForgeDiffusionEngine:
    def __init__(self, estimated_config=None, huggingface_components=None):
        self.model_config = estimated_config

    def _get_predictor(self):
        return types.SimpleNamespace()


def text_processing():
    """backend.args and backend.text_processing, enough to import forge_iris.native.text_engine."""
    comfy = module("backend.text_processing._comfy", EMBEDDINGS=list, INF=float("inf"),
                   SDTokenizer=FakeSDTokenizer, SDClipModel=FakeSDClipModel)
    emphasis = module("backend.text_processing.emphasis", uses_emphasis=lambda text: False,
                      EmphasisNone=type("EmphasisNone", (), {}))
    package = module("backend.text_processing", emphasis=emphasis, _comfy=comfy)
    args = module("backend.args", dynamic_args=types.SimpleNamespace(last_extra_generation_params={}))
    return {"backend.args": args, "backend.text_processing": package,
            "backend.text_processing.emphasis": emphasis, "backend.text_processing._comfy": comfy}


def guess_modules():
    latent = module("huggingface_guess.latent",
                    RGB=type("RGB", (), {"latent_channels": 3}), LatentFormat=object)
    model_list = module("huggingface_guess.model_list", BASE=object, models=[],
                        ModelType=types.SimpleNamespace(FLOW="flow"))
    detection = module("huggingface_guess.detection", detect_unet_config=lambda *a, **k: None)
    return {"huggingface_guess": module("huggingface_guess", latent=latent, model_list=model_list,
                                        detection=detection),
            "huggingface_guess.latent": latent, "huggingface_guess.model_list": model_list,
            "huggingface_guess.detection": detection}


def engine_modules():
    """Everything forge_iris.native.engine imports from Forge Neo and huggingface_guess."""
    stubs = text_processing()
    stubs.update(guess_modules())
    memory = module("backend.memory_management", load_models_gpu=lambda patchers, memory_required=0: None,
                    load_model_gpu=lambda patcher: None)
    base = module("backend.diffusion_engine.base", ForgeDiffusionEngine=FakeForgeDiffusionEngine,
                  ForgeObjects=types.SimpleNamespace)
    stubs.update({
        "backend": module("backend", memory_management=memory, args=stubs["backend.args"],
                          text_processing=stubs["backend.text_processing"]),
        "backend.memory_management": memory,
        "backend.diffusion_engine": module("backend.diffusion_engine", base=base),
        "backend.diffusion_engine.base": base,
        "backend.patcher": module("backend.patcher"),
        "backend.patcher.clip": module("backend.patcher.clip", CLIP=object),
        "backend.patcher.unet": module("backend.patcher.unet", UnetPatcher=object),
        "backend.patcher.vae": module("backend.patcher.vae", VAE=object),
    })
    return stubs


def patch_modules():
    """Everything forge_iris.native.patches imports: engine_modules plus the loader hooks' targets."""
    stubs = engine_modules()
    stubs.update({
        "backend.loader": module("backend.loader", load_huggingface_component=None, possible_models=()),
        "backend.nn": module("backend.nn"),
        "backend.nn.krea": module("backend.nn.krea", SingleStreamDiT=type("SingleStreamDiT", (), {})),
    })
    stubs["backend"].loader = stubs["backend.loader"]
    stubs["backend"].nn = stubs["backend.nn"]
    return stubs
