# Iris 3B support for Forge Neo
# The "Iris DPM++ 2M" sampler: upstream's FlowDPM-Solver++(2M) (speridlabs/iris-3b, src/iris3b/flow/solver.py).
#
# Forge's "DPM++ 2M" is the k-diffusion formula for variance-exploding models (sigma = e^-t, no alpha), which is a
# different integrator on a rectified flow. This one is the data-prediction multistep solver on the flow schedule
# alpha = 1 - sigma, lambda = log((1 - sigma) / sigma): first order on the first step, second order after, and
# first order again on the last step, which lands on sigma 0 as the exact projection x <- x0.

import math

import torch

NAME = "Iris DPM++ 2M"
ALIASES = ["iris_dpmpp_2m"]
# lambda is infinite at sigma 1; a schedule that starts there is nudged inside, as Forge does for its SNR samplers
SIGMA_MAX = 1.0 - 1e-4


def _lambda(sigma: float) -> float:
    return math.log((1.0 - sigma) / sigma)


@torch.no_grad()
def sample_iris_dpmpp_2m(model, x, sigmas, extra_args=None, callback=None, disable=None):
    from tqdm.auto import trange

    extra_args = {} if extra_args is None else extra_args
    s_in = x.new_ones([x.shape[0]])
    previous = None
    for i in trange(len(sigmas) - 1, disable=disable):
        s, t = min(float(sigmas[i]), SIGMA_MAX), float(sigmas[i + 1])
        denoised = model(x, sigmas[i] * s_in, **extra_args)
        if callback is not None:
            callback({"x": x, "i": i, "sigma": sigmas[i], "sigma_hat": sigmas[i], "denoised": denoised})
        if t <= 0.0:
            x = denoised
        else:
            h = _lambda(t) - _lambda(s)
            phi = math.expm1(-h)
            x = (t / s) * x - (1.0 - t) * phi * denoised
            if previous is not None:
                s_last, denoised_last = previous
                r = (_lambda(s) - _lambda(s_last)) / h
                x = x - 0.5 * (1.0 - t) * phi * (denoised - denoised_last) / r
        previous = (s, denoised)
    return x


def register() -> None:
    from modules import sd_samplers, sd_samplers_common, sd_samplers_kdiffusion

    from . import scheduler

    data = sd_samplers_common.SamplerData(
        NAME, lambda model: sd_samplers_kdiffusion.KDiffusionSampler(sample_iris_dpmpp_2m, model),
        ALIASES, {"scheduler": scheduler.NAME})
    sd_samplers.add_sampler(data)
