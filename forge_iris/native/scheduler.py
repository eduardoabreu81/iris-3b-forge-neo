# Iris 3B support for Forge Neo
# The "Iris" sigma schedule, as ComfyUI-Iris's iris_sigmas: the upstream rectified-flow grid with shift 4.
#
# sigma = 1 - linspace(1, 0.001, n + 1) runs 0 .. 0.999 unshifted (0.001 is the model's training floor); the
# shift maps each through 4s / (1 + 3s) and the grid flips, so the first sigma is ~0.999 and the last is 0.
#
# "Iris Turbo" is the few-step student's grid (pentaCoxian/iris-3b-turbo, distill.turbo_times): shift(1 - i / n)
# for i < n, from pure noise (sigma 1) down, then 0. With the LCM sampler (predict x0, re-noise at the next sigma)
# it is the Turbo LoRA's sampling exactly: at 4 steps sigma = 1, 0.923, 0.8, 0.571.

import torch

NAME = "iris"
LABEL = "Iris"
ALIASES = ["iris_schedule"]
SHIFT = 4.0


TURBO_NAME = "iris_turbo"
TURBO_LABEL = "Iris Turbo"
TURBO_ALIASES = ["iris_turbo_schedule"]


def _shift(sigma):
    return SHIFT * sigma / (1.0 + (SHIFT - 1.0) * sigma)


def sigmas(n, sigma_min=None, sigma_max=None, device=None):
    sigma = 1.0 - torch.linspace(1.0, 0.001, n + 1, dtype=torch.float64)
    return _shift(sigma).flip(0).to(torch.float32).to(device or "cpu")


def turbo_sigmas(n, sigma_min=None, sigma_max=None, device=None):
    grid = _shift(1.0 - torch.arange(n, dtype=torch.float64) / n)
    return torch.cat((grid, grid.new_zeros(1))).to(torch.float32).to(device or "cpu")
