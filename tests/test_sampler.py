# Iris 3B support for Forge Neo
# The Iris DPM++ 2M sampler against upstream's FlowDPMSolver (speridlabs/iris-3b a8d15239,
# src/iris3b/flow/solver.py), transcribed here: same grid, same velocity model, same result.

import math
import unittest

import torch

from forge_iris.native.sampler import sample_iris_dpmpp_2m
from forge_iris.native.scheduler import sigmas as iris_sigmas


def upstream_sample(velocity, z, steps, shift=4.0, order=2):
    """FlowDPMSolver.sample with cfg_scale 1, prediction "v"."""
    def lam(t):
        return math.inf if t <= 0.0 else math.log((1.0 - t) / t)

    sigma = 1.0 - torch.linspace(1.0, 0.001, steps + 1, dtype=torch.float64)
    grid = (shift * sigma / (1 + (shift - 1) * sigma)).flip(0).tolist()
    x, history = z, []
    for i in range(1, steps + 1):
        s, t = grid[i - 1], grid[i]
        x0 = x - s * velocity(x, s)
        history.append((s, x0))
        if min(i, order, steps + 1 - i) == 1:
            h = lam(t) - lam(s)
            x = (t / s) * x - (1.0 - t) * math.expm1(-h) * x0
        else:
            (s1, x0_1), (s0, x0_0) = history[-2], history[-1]
            h, h0 = lam(t) - lam(s0), lam(s0) - lam(s1)
            phi = math.expm1(-h)
            x = (t / s0) * x - (1.0 - t) * phi * x0_0 - 0.5 * (1.0 - t) * phi * (x0_0 - x0_1) / (h0 / h)
        if len(history) > 2:
            history.pop(0)
    return x


def velocity(x, sigma):
    # a smooth, sigma-dependent stand-in for the DiT's velocity
    return torch.tanh(x) * (1.0 + sigma) - 0.3 * x


class TestIrisSampler(unittest.TestCase):
    def test_matches_upstream_solver(self):
        torch.manual_seed(0)
        z = torch.randn(2, 3, 8, 8, dtype=torch.float64)
        for steps in (1, 2, 5, 50):
            with self.subTest(steps=steps):
                # Forge's denoiser contract: model(x, sigma) -> x0 (flow: x - sigma * v)
                def model(x, sigma):
                    return x - sigma[:, None, None, None] * velocity(x, sigma[0].item())

                sigmas = iris_sigmas(steps).to(torch.float64)
                ours = sample_iris_dpmpp_2m(model, z.clone(), sigmas, disable=True)
                torch.testing.assert_close(ours, upstream_sample(velocity, z.clone(), steps))

    def test_schedule_starting_at_one(self):
        # another scheduler may start at sigma 1, where lambda is infinite: the step stays finite
        sigmas = torch.tensor([1.0, 0.5, 0.0], dtype=torch.float64)
        out = sample_iris_dpmpp_2m(lambda x, s: x * 0.5, torch.ones(1, 3, 4, 4, dtype=torch.float64), sigmas,
                                   disable=True)
        self.assertTrue(bool(torch.isfinite(out).all()))


if __name__ == "__main__":
    unittest.main()
