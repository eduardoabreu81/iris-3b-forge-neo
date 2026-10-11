# Iris 3B support for Forge Neo
# The "Iris" sigma grid: shifted rectified flow, first ~0.999, last exactly 0, n + 1 points.

import unittest

try:
    import torch
except ImportError:  # the grid math itself needs tensors, like Forge's own schedulers
    torch = None


@unittest.skipIf(torch is None, "torch not installed")
class TestScheduler(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        from forge_iris.native import scheduler

        cls.scheduler = scheduler

    def test_grid_shape(self):
        sigmas = self.scheduler.sigmas(50)
        self.assertEqual(sigmas.shape[0], 51)

    def test_reference_values(self):
        # the first sigma is the shifted 0.999 training ceiling, the last is exactly zero
        sigmas = self.scheduler.sigmas(50)
        self.assertAlmostEqual(sigmas[0].item(), 4.0 * 0.999 / (1.0 + 3.0 * 0.999), places=6)
        self.assertEqual(sigmas[-1].item(), 0.0)

    def test_descending(self):
        sigmas = self.scheduler.sigmas(20)
        self.assertTrue(bool((sigmas[1:] < sigmas[:-1]).all()))

    def test_device(self):
        sigmas = self.scheduler.sigmas(5, device="cpu")
        self.assertEqual(str(sigmas.device), "cpu")

    def test_matches_comfyui_iris(self):
        # the same math as ComfyUI-Iris's iris_sigmas at denoise 1
        import torch

        steps = 30
        sigma = 1.0 - torch.linspace(1.0, 0.001, steps + 1, dtype=torch.float64)
        expected = (4.0 * sigma / (1.0 + 3.0 * sigma)).flip(0).to(torch.float32)
        actual = self.scheduler.sigmas(steps)
        self.assertTrue(torch.allclose(actual, expected))


if __name__ == "__main__":
    unittest.main()
