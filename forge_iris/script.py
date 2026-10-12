# Iris 3B support for Forge Neo
# The script's generation callbacks: the pixel-space guards Forge has no hook for, and the noise in full resolution.
#
# The Iris is the first text-to-image pixel-space model on Forge Neo (PiD is img2img-only), so the sampler's
# noise has to be replaced: Forge builds it as an image latent at 1/8 resolution, for txt2img and img2img alike.
# before_sampling swaps the RNG and the noise for (batch, 3, height, width), as the MiniMax H3 extension does for
# its packed latent.
#
# before_process runs before Forge loads the checkpoint, so p.sd_model there is still the previous model: the guards
# read the selected checkpoint's header instead (checkpoints.file_kind, as the MiniMax H3 extension's select_h3).
#
# img2img also runs the fine-tunes' single-pass tasks (tasks.py) through Forge's own pipeline, so model loading,
# LoRAs typed in the prompt, saving and infotext work as usual: the input is prepared to the task's size, the prompt
# keeps only its extra-network tags (the fine-tunes are conditioned on the empty prompt), and the sampling is one
# Euler step from the task's sigma to 0. Forge always mixes noise into an img2img input (sigma * noise +
# (1 - sigma) * image); with the image itself as the noise that mix is the image, as the tasks need.

import re

from . import checkpoints, tasks
from .contracts import IrisError, set_pending_error, set_task

MODE_IMAGE = "Image"
MODE_RESTORE = "Restore x4"
MODE_DEPTH = "Depth"
MODES = (MODE_IMAGE, MODE_RESTORE, MODE_DEPTH)
_TASKS = {MODE_RESTORE: tasks.RESTORE, MODE_DEPTH: tasks.DEPTH}
_SIGMAS = {tasks.RESTORE: tasks.RESTORE_SIGMA, tasks.DEPTH: tasks.DEPTH_SIGMA}
_EXTRA_NETWORK = re.compile(r"<[^<>]+>")

DEPTH_CHECKPOINT = ("[Iris 3B] Depth needs the Iris depth model as the checkpoint (iris-3b-depth_bf16 or "
                    "iris-3b-depth_int8_convrot from nynxz/iris-3b).")


def selected_kind(p) -> str | None:
    """What this generation runs (checkpoints.IRIS, DEPTH, FOREIGN or None): the checkpoint Forge is about to load,
    override included."""
    from modules import sd_models, shared

    name = getattr(p, "override_settings", {}).get("sd_model_checkpoint", shared.opts.sd_model_checkpoint)
    info = sd_models.get_closet_checkpoint_match(name) if name else None
    if info is None:
        # nothing to read: the model in memory is the one Forge will keep
        return checkpoints.IRIS if getattr(p.sd_model, "is_iris", False) else None
    return checkpoints.file_kind(info.filename)


def is_img2img(p) -> bool:
    # Forge Neo's processing objects have no is_img2img (A1111's); only img2img carries init_images
    return getattr(p, "init_images", None) is not None


def _refuse(message: str) -> None:
    error = IrisError(message)
    set_pending_error(error)
    print(error)


def script_before_process(p, mode: str = MODE_IMAGE) -> None:
    # a refusal, or a task, from an earlier generation must not reach this one
    set_pending_error(None)
    set_task(None)
    p.iris_task = None
    _reset_memory_factor(p.sd_model)
    kind = selected_kind(p)
    task = _TASKS.get(mode) if is_img2img(p) else None
    if kind == checkpoints.DEPTH:
        if task != tasks.DEPTH:
            return _refuse(checkpoints.DEPTH_MESSAGE)
    # a FOREIGN build never loads: the loader's detection hook explains it
    elif kind != checkpoints.IRIS:
        return
    elif task == tasks.DEPTH:
        return _refuse(DEPTH_CHECKPOINT)
    from modules import shared

    from .native import lora

    # Iris LoRA formats (the Turbo LoRA, other FFN widths); in place once Forge's LoRA extension is imported
    lora.install()

    if getattr(p, "enable_hr", False):
        return _refuse("[Iris 3B] Hires. fix is not supported yet: its latent upscale assumes a VAE, and the Iris "
                       "latent is the image itself. Generate at the target size instead.")
    if shared.opts.sd_vae_decode_method != "Full":
        return _refuse("[Iris 3B] needs Settings > VAE > VAE for Decoding = Full: the approximate decoders assume a "
                       "VAE (and keep only the first image of a batch), and the Iris output is the image itself.")
    if not is_img2img(p):
        return
    if shared.opts.sd_vae_encode_method != "Full":
        return _refuse("[Iris 3B] img2img needs Settings > VAE > VAE for Encoding = Full: the approximate encoders "
                       "assume a VAE, and the Iris input image is its own latent.")
    if task is not None:
        return _prepare_task(p, task, mode)
    if getattr(p, "resize_mode", 0) == 3:
        # "Just resize (latent)" interpolates the latent to 1/8 resolution; in pixel space the plain image
        # resize (mode 0) is already the right thing
        p.resize_mode = 0
        print("[Iris 3B] resize mode 'Just resize (latent)' is not applicable in pixel space; using 'Just resize'")


def _prepare_task(p, task: str, mode: str) -> None:
    import torch
    from PIL import Image

    if getattr(p, "image_mask", None) is not None:
        return _refuse(f"[Iris 3B] {mode} works on the whole image: clear the inpaint mask.")
    if not getattr(p, "init_images", None):
        return _refuse(f"[Iris 3B] {mode} needs an input image.")
    image = p.init_images[0].convert("RGB")
    if task == tasks.RESTORE:
        reference, work = tasks.restore_inputs(image)
        p.init_images = [tasks.to_image(work)]
        p.iris_task = {"task": task, "reference": reference}
    else:
        size = tasks.depth_size(*image.size)
        p.init_images = [image.resize(size, Image.Resampling.LANCZOS)]
        p.iris_task = {"task": task, "size": (image.height, image.width)}
    p.width, p.height = p.init_images[0].size
    p.resize_mode = 0
    # the fine-tunes are conditioned on the empty prompt; LoRA tags (an upscaler LoRA) stay for Forge to apply.
    # Without tags the prompt is a space, not "": the text engine encodes both as the empty prompt, and
    # sd-dynamic-prompts raises StopIteration on an empty template
    p.prompt = " ".join(_EXTRA_NETWORK.findall(p.prompt or "")) or " "
    p.negative_prompt = ""
    p.cfg_scale = 1.0
    p.batch_size = 1
    p.steps = 1
    p.denoising_strength = 1.0
    p.initial_noise_multiplier = 1.0
    p.sampler_name, p.scheduler = "Euler", "Automatic"
    sigma = _SIGMAS[task]
    p.sampler_noise_scheduler_override = lambda steps: torch.tensor([sigma, 0.0])
    # Forge's extra img2img noise would add to the image-as-noise; Settings overrides apply after before_process
    p.override_settings = {**(getattr(p, "override_settings", None) or {}), "img2img_extra_noise": 0}
    p.extra_generation_params["Iris task"] = mode
    set_task(task)


def _reset_memory_factor(model) -> None:
    config = getattr(model, "model_config", None)
    if getattr(model, "is_iris", False) and config is not None:
        config.__dict__.pop("memory_usage_factor", None)


def postprocess_image(p, pp) -> None:
    """The task's output from the pass's: the restored 4x image with its colour fix, or the depth map."""
    task = getattr(p, "iris_task", None)
    if not task:
        return
    y = tasks.to_tensor(pp.image)
    if task["task"] == tasks.RESTORE:
        out = tasks.restore_output(y, task["reference"])
    else:
        out = tasks.depth_map(y, task["size"])
    pp.image = tasks.to_image(out)


def postprocess(p) -> None:
    set_task(None)
    _reset_memory_factor(p.sd_model)


def before_sampling(p, noise, x=None) -> None:
    # the hires pass builds its noise from the upscaled samples, already full resolution (p.height/width are still
    # the first pass's there)
    if not getattr(p.sd_model, "is_iris", False) or getattr(p, "is_hr_pass", False):
        return
    task = getattr(p, "iris_task", None)
    if task and x is not None:
        # the image as its own noise: Forge's sigma * noise + (1 - sigma) * image is then the image
        p.modified_noise = x.clone().to(device=noise.device, dtype=noise.dtype)
        if task["task"] == tasks.RESTORE:
            # Forge reserves activation memory for the whole image; the pass only ever holds one tile
            area = x.shape[-1] * x.shape[-2]
            config = p.sd_model.model_config
            config.memory_usage_factor = type(config).memory_usage_factor * min(1.0, tasks.TILE ** 2 / area)
        return
    # ImageRNG takes the shape of one sample and stacks one noise per seed
    shape = (3, p.height, p.width)
    if tuple(noise.shape[1:]) == shape:
        return
    from modules import rng

    # "Resize seed from" divides by 8 for the image latent; in pixel space the source noise is full resolution
    resize_h, resize_w = p.seed_resize_from_h, p.seed_resize_from_w
    if resize_h > 0 and resize_w > 0:
        resize_h, resize_w = resize_h * 8, resize_w * 8
    # the samplers that add noise on the way (ancestral, SDE, res_multistep) draw from p.rng too
    p.rng = rng.ImageRNG(shape, p.seeds, subseeds=p.subseeds, subseed_strength=p.subseed_strength,
                         seed_resize_from_h=resize_h, seed_resize_from_w=resize_w)
    p.modified_noise = p.rng.next().to(device=noise.device, dtype=noise.dtype)
