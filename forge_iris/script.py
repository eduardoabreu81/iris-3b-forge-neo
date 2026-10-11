# Iris 3B support for Forge Neo
# The script's generation callbacks: the pixel-space guards Forge has no hook for, and the txt2img noise in
# full resolution.
#
# The Iris is the first text-to-image pixel-space model on Forge Neo (PiD is img2img-only), so the sampler's
# noise has to be replaced: Forge builds it as an image latent at 1/8 resolution and interpolates init latents
# accordingly. before_sampling swaps the RNG and the noise for (batch, 3, height, width), as the MiniMax H3
# extension does for its packed latent.

from .contracts import IrisError, set_pending_error


def script_before_process(p) -> None:
    if not getattr(p.sd_model, "is_iris", False):
        return
    if getattr(p, "enable_hr", False):
        error = IrisError("[Iris 3B] Hires. fix is not supported yet: its latent upscale assumes a VAE, and the "
                          "Iris latent is the image itself. Generate at the target size instead.")
        set_pending_error(error)
        print(error)
        return
    if not getattr(p, "is_img2img", False):
        return
    from modules import shared

    if shared.opts.sd_vae_encode_method != "Full":
        error = IrisError("[Iris 3B] img2img needs Settings > VAE > VAE for Encoding = Full: the approximate "
                          "encoders assume a VAE, and the Iris input image is its own latent.")
        set_pending_error(error)
        print(error)
        return
    if getattr(p, "resize_mode", 0) == 3:
        # "Just resize (latent)" interpolates the latent to 1/8 resolution; in pixel space the plain image
        # resize (mode 0) is already the right thing
        p.resize_mode = 0
        print("[Iris 3B] resize mode 'Just resize (latent)' is not applicable in pixel space; using 'Just resize'")


def before_sampling(p, noise) -> None:
    if not getattr(p.sd_model, "is_iris", False):
        return
    shape = (noise.shape[0], 3, p.height, p.width)
    if tuple(noise.shape) == shape:
        return
    from modules import rng

    # the samplers that add noise on the way (ancestral, SDE, res_multistep) draw from p.rng too
    p.rng = rng.ImageRNG(shape, p.seeds, subseeds=p.subseeds, subseed_strength=p.subseed_strength,
                         seed_resize_from_h=p.seed_resize_from_h, seed_resize_from_w=p.seed_resize_from_w)
    p.modified_noise = p.rng.next().to(device=noise.device, dtype=noise.dtype)
