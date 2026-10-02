from __future__ import annotations
import sys
import unittest
from pathlib import Path
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from experiments.strengthening import certificate as cert
from experiments.strengthening import numpy_data as data


class ProfileAndDataTests(unittest.TestCase):
    def test_squared_profile_agrees_with_affine_least_squares(self):
        slopes = np.array([1.2, -.1, .7, 1.9, -.8])
        delays = np.array([.15, .267, .5])
        profile = cert.profile_residual(slopes, .3, delays)
        for index, delay in enumerate(delays):
            history = .3-data.C5*delay+data.RHO_STRONG*data.D5*delay**2
            design = np.column_stack([np.ones(5), history])
            fit = design @ np.linalg.lstsq(design, slopes, rcond=None)[0]
            self.assertAlmostEqual(profile[index], np.mean((slopes-fit)**2), places=12)

    def test_noise_preserves_history_and_initial_value(self):
        clean, _ = data.make_strong_dataset([0.])
        noisy = data.add_observation_noise(clean, .01, 123)
        np.testing.assert_array_equal(noisy[:, :data.HIST_STEPS+1], clean[:, :data.HIST_STEPS+1])
        self.assertFalse(np.array_equal(noisy[:, data.HIST_STEPS+1:], clean[:, data.HIST_STEPS+1:]))

    def test_forced_ramps_satisfy_generating_equation(self):
        trajectories, inputs, _ = data.make_forced_ramp_dataset([-.4, .2])
        for row, forcing in zip(trajectories, inputs):
            x = row[data.HIST_STEPS:-1]
            np.testing.assert_allclose(data.a_true_np(x)+data.b_true_np(x)*(x-data.tau_true_np(x))+forcing, 1., atol=1e-12)

    def test_interval_membership_and_model_scope_are_separate(self):
        geometry = pd.DataFrame([dict(state=0., true_delay=.3, estimated_delay=.3,
                interval_lower=.25, interval_upper=.35, interval_width=.1,
                profile_minimum=0., eta_95=1., geometry_accepted=True,
                verdict="identifiable", verdict_reason="resolved connected profile")])
        profiles = pd.DataFrame(dict(state=[0., 0.], candidate_delay=[.15, .5], profile_value=[0., 0.]))
        candidates = pd.DataFrame([dict(state=0., learned_delay=.3, model_class="restricted_learned"),
                                   dict(state=0., learned_delay=.4, model_class="restricted_learned"),
                                   dict(state=0., learned_delay=.3, model_class="unrestricted_learned")])
        result = cert.evaluate_candidates(geometry, profiles, candidates, .035)
        self.assertEqual(result.candidate_accepted.tolist(), [True, False, False])
        self.assertEqual(result.candidate_in_empirical_set.tolist(), [True, True, True])


if __name__ == "__main__":
    unittest.main()
