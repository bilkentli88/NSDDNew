from __future__ import annotations

import importlib.util
import sys
import unittest
from pathlib import Path

import numpy as np
try:
    import torch
except ModuleNotFoundError:
    raise unittest.SkipTest("PyTorch is required for the legacy neural-model tests")

ROOT = Path(__file__).resolve().parents[1]
MODULE_PATH = ROOT / "experiments" / "run_principal_scalar.py"
spec = importlib.util.spec_from_file_location("scalar_experiments_test", MODULE_PATH)
assert spec is not None and spec.loader is not None
scalar = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = scalar
spec.loader.exec_module(scalar)


class CoreInvariantTests(unittest.TestCase):
    def test_true_delay_is_inside_admissible_interval(self) -> None:
        grid = np.linspace(-5.0, 5.0, 501)
        tau = scalar.tau_true_np(grid)
        self.assertTrue(np.all(tau >= scalar.TAU_MIN))
        self.assertTrue(np.all(tau <= scalar.TAU_MAX))

    def test_learned_delay_parameterization_is_bounded(self) -> None:
        model = scalar.RestrictedLearnedModel()
        x = torch.linspace(-5.0, 5.0, 101).reshape(-1, 1)
        tau = model.tau(x).detach().numpy()
        self.assertTrue(np.all(tau >= scalar.TAU_MIN))
        self.assertTrue(np.all(tau <= scalar.TAU_MAX))

    def test_linear_interpolation_is_exact_for_linear_signal(self) -> None:
        time = np.linspace(-scalar.TAU_MAX, 0.0, scalar.HIST_STEPS + 1)
        values = 1.7 + 2.3 * time
        query = -0.337
        expected = 1.7 + 2.3 * query
        actual = scalar.interpolate_numpy(values, query)
        self.assertAlmostEqual(actual, expected, places=12)

    def test_ramp_matched_history_is_centered_degenerate(self) -> None:
        states = np.array([-0.7, -0.1, 0.4, 1.0])
        r = 0.31
        # For x_j(t)=c_j+t and x_j(t_j)=s, q_{s,j}(r)=s-r for every j.
        q = np.full_like(states, fill_value=0.2 - r, dtype=float)
        centered = q - q.mean()
        self.assertTrue(np.allclose(centered, 0.0))


if __name__ == "__main__":
    unittest.main()
