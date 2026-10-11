# Iris 3B support for Forge Neo
# The "Iris" sigma schedule, as ComfyUI-Iris's iris_sigmas: the upstream rectified-flow grid with shift 4.
#
# sigma = 1 - linspace(1, 0.001, n + 1) runs 0 .. 0.999 unshifted (0.001 is the model's training floor); the
# shift maps each through 4s / (1 + 3s) and the grid flips, so the first sigma is ~0.999 and the last is 0.

import torch

NAME = "iris"
LABEL = "Iris"
ALIASES = ["iris_schedule"]
SHIFT = 4.0


def sigmas(n, sigma_min=None, sigma_max=None, device=None):
    sigma = 1.0 - torch.linspace(1.0, 0.001, n + 1, dtype=torch.float64)
    return (SHIFT * sigma / (1.0 + (SHIFT - 1.0) * sigma)).flip(0).to(torch.float32).to(device or "cpu")
