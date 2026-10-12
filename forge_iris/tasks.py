# Iris 3B support for Forge Neo
# The single-pass tasks of Iris-3B's fine-tunes, as speridlabs/iris-3b runs them (src/iris3b/downstream/, commit
# a8d15239): image restoration and 4x upscaling, and monocular depth.
#
# Both are one forward pass of the whole DiT with the empty prompt, no sampling loop:
# - restore: the low-quality image upsampled bicubically 4x, in [-1, 1], at model time 500 (sigma 0.5), output
#   x - 0.5 * v. The fine-tune only runs on 1024x1024 tiles: a short side below 1024 is enlarged for the pass and
#   resized back, larger outputs are covered by tiles at stride 512 fused with a Gaussian window, and a wavelet colour
#   fix keeps the restored high frequencies on the bicubic upsample's low frequencies.
# - depth: RGB in [-1, 1] at model time 1000, the output is relative log depth (-1 near, +1 far). The long side is
#   capped at 1024 and both sides rounded to the 16-pixel patch.
#
# The restore weights are whatever the loaded checkpoint carries: the official upscaler, or the base model with an
# upscaler LoRA in the prompt. This module holds the pure image math; script.py fits it into Forge's img2img.

import torch
import torch.nn.functional as F

RESTORE = "restore"
DEPTH = "depth"

RESTORE_SCALE = 4
RESTORE_SIGMA = 0.5
DEPTH_SIGMA = 1.0
TILE = 1024
PATCH = 16
DEPTH_MAX_SIDE = 1024

# colour fix: a-trous binomial pyramid, effective radius 16 px
_DILATIONS = (1, 2, 4, 8, 16)
_KERNEL = torch.tensor([[1.0, 2.0, 1.0], [2.0, 4.0, 2.0], [1.0, 2.0, 1.0]]) / 16.0


def to_tensor(image) -> torch.Tensor:
    """PIL RGB -> [1, 3, H, W] float32 in [0, 1]."""
    import numpy as np

    return torch.from_numpy(np.array(image.convert("RGB"))).permute(2, 0, 1)[None].float() / 255


def to_image(x: torch.Tensor):
    """[1, 3, H, W] in [0, 1] -> PIL RGB."""
    from PIL import Image

    return Image.fromarray((x.clamp(0, 1)[0].permute(1, 2, 0) * 255).round().byte().cpu().numpy())


# region restore


def restore_sizes(width: int, height: int) -> tuple[tuple[int, int], tuple[int, int]]:
    """((out_h, out_w), (work_h, work_w)): the 4x output, and the size the pass runs at (short side >= one tile)."""
    out = (height * RESTORE_SCALE, width * RESTORE_SCALE)
    if min(out) > TILE:
        return out, out
    ratio = TILE / min(out)
    return out, tuple(max(TILE, round(side * ratio)) for side in out)


def restore_inputs(image) -> tuple[torch.Tensor, torch.Tensor]:
    """(reference, work) for a PIL image: the bicubic 4x upsample in [0, 1], and the pass input at its work size."""
    lq = to_tensor(image)
    reference = F.interpolate(lq, scale_factor=RESTORE_SCALE, mode="bicubic", align_corners=False)
    out, work = restore_sizes(lq.shape[-1], lq.shape[-2])
    x = reference
    if work != out:
        x = F.interpolate(x, size=work, mode="bicubic", align_corners=False, antialias=True)
    return reference, x


def restore_output(y: torch.Tensor, reference: torch.Tensor, color_fix: bool = True) -> torch.Tensor:
    """The pass's [1, 3, h, w] output in [0, 1] -> the final 4x image: back to the output size, then the colour fix."""
    if y.shape[-2:] != reference.shape[-2:]:
        y = F.interpolate(y, size=reference.shape[-2:], mode="bicubic", align_corners=False, antialias=True)
    if color_fix:
        y = wavelet_color_fix(y, reference)
    return y.clamp(0, 1)


def _blur(image: torch.Tensor, dilation: int) -> torch.Tensor:
    kernel = _KERNEL.to(image)[None, None].repeat(image.shape[1], 1, 1, 1)
    padded = F.pad(image, (dilation,) * 4, mode="replicate")
    return F.conv2d(padded, kernel, groups=image.shape[1], dilation=dilation)


def _split(image: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """(high, low) frequency parts; they sum back to the input."""
    high, low = torch.zeros_like(image), image
    for dilation in _DILATIONS:
        blurred = _blur(low, dilation)
        high, low = high + (low - blurred), blurred
    return high, low


def wavelet_color_fix(content: torch.Tensor, reference: torch.Tensor) -> torch.Tensor:
    """``content``'s detail on ``reference``'s colour; both the same size."""
    return _split(content)[0] + _split(reference)[1]


def tile_positions(size: int, tile: int, stride: int) -> list[int]:
    """Start offsets along one axis; the last tile is flush with the far edge."""
    if size <= tile:
        return [0]
    return [*range(0, size - tile, stride), size - tile]


def gaussian_window(tile: int, device) -> torch.Tensor:
    """[1, 1, tile, tile] fusion weights, upstream's exactly (its centres are tile / 2 vertically and
    (tile - 1) / 2 horizontally)."""
    index = torch.arange(tile, device=device, dtype=torch.float32)
    cols = torch.exp(-((index - (tile - 1) / 2) ** 2) / (tile * tile) / 0.02)
    rows = torch.exp(-((index - tile / 2) ** 2) / (tile * tile) / 0.02)
    return (rows[:, None] * cols[None, :])[None, None]


def tiled(fn, x: torch.Tensor, tile: int | None = None, stride: int | None = None) -> torch.Tensor:
    """Apply ``fn`` to overlapping ``tile`` crops of [B, C, H, W] (default TILE at stride TILE / 2) and fuse them
    with the Gaussian window."""
    tile = tile or TILE
    stride = stride or tile // 2
    height, width = x.shape[-2:]
    if height <= tile and width <= tile:
        return fn(x)
    # fused in float32, as upstream: the window falls to ~1e-11 in a tile's corners, which float16 flushes to zero
    # (and an image corner covered by one tile only then divides 0 by 0)
    window = gaussian_window(tile, x.device)
    out = torch.zeros(x.shape, device=x.device, dtype=torch.float32)
    weight = torch.zeros((x.shape[0], 1, *x.shape[2:]), device=x.device, dtype=torch.float32)
    for top in tile_positions(height, tile, stride):
        for left in tile_positions(width, tile, stride):
            crop = (..., slice(top, top + tile), slice(left, left + tile))
            out[crop] += fn(x[crop].contiguous()).float() * window
            weight[crop] += window
    return (out / weight).to(x.dtype)


def restore_forward(fn, x: torch.Tensor) -> torch.Tensor:
    """The DiT pass over the work image: zero-padded to the patch (as upstream), tiled, cropped back."""
    height, width = x.shape[-2:]
    x = F.pad(x, (0, -width % PATCH, 0, -height % PATCH))
    return tiled(fn, x)[..., :height, :width]


# endregion

# region depth


def depth_size(width: int, height: int, max_side: int = DEPTH_MAX_SIDE) -> tuple[int, int]:
    """(width, height) the depth pass runs at: long side capped, both rounded to the patch."""
    scale = min(1.0, max_side / max(width, height)) if max_side else 1.0
    return tuple(max(PATCH, round(side * scale / PATCH) * PATCH) for side in (width, height))


def depth_map(output: torch.Tensor, size: tuple[int, int]) -> torch.Tensor:
    """The pass's [1, 3, h, w] output in [0, 1] (relative log depth mapped from [-1, 1]; the folded model writes the
    same value to all three channels) -> a [1, 3, H, W] greyscale map at size (H, W), near bright, scaled to its own
    2nd-98th percentiles as upstream's colorize does."""
    depth = output.mean(dim=1, keepdim=True) * 2 - 1
    if depth.shape[-2:] != tuple(size):
        depth = F.interpolate(depth, size=size, mode="bilinear", align_corners=False)
    low, high = torch.quantile(depth.flatten().float(), torch.tensor([0.02, 0.98], dtype=torch.float32))
    near = ((high - depth) / max(float(high - low), 1e-6)).clamp(0, 1)
    return near.repeat(1, 3, 1, 1)


# endregion
