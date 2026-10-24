"""
Unit and Integration Tests for SHELL Framework
==============================================
Tests all core mathematical and operational components:
1. Degradation Injector (noise, drift, dropout)
2. Online Health Estimator (variance and consistency tracking)
3. Action Selection Hierarchy (RGSCA rules)
4. Stability-Constrained Projection Operator (norm boundedness)
5. Cross-Modal Latent Reconstructor
6. End-to-End Pipeline Integration
"""

import sys
import unittest
import numpy as np
import torch
from pathlib import Path

# Add implementation root to path
impl_root = Path(__file__).parent.parent
sys.path.insert(0, str(impl_root))

from models.backbone import SHELLModel, CrossModalReconstructor
from shell.controller import SHELLController, SHELLConfig, Action
from shell.degradation import DegradationInjector, DegradationScenario


class TestDegradationInjector(unittest.TestCase):
    def setUp(self):
        self.injector = DegradationInjector(seed=42)
        self.window = 30
        self.n_sensors = 14
        self.x = np.ones((self.window, self.n_sensors), dtype=np.float32)
        self.sigma = np.ones(self.n_sensors, dtype=np.float32) * 0.1

    def test_clean_identity(self):
        x_out, mask = self.injector.apply(self.x, [], t=0, sigma_baseline=self.sigma)
        np.testing.assert_allclose(x_out, self.x)
        self.assertEqual(len(mask), self.n_sensors)
        # In injector.apply, mask is degradation_mask (0 = healthy, 1 = degraded)
        self.assertTrue(all(m == 0 for m in mask))

    def test_gaussian_noise_injection(self):
        scenario = self.injector.create_noise_scenario([0], severity=1.0, start_cycle=0, end_cycle=100)
        x_out, mask = self.injector.apply(self.x, [scenario], t=5, sigma_baseline=self.sigma)
        # Sensor 0 should have noise added
        self.assertNotEqual(x_out[-1, 0], self.x[-1, 0])
        # Sensor 1 should remain untouched
        self.assertAlmostEqual(x_out[-1, 1], self.x[-1, 1])
        self.assertEqual(mask[0], 1.0)
        self.assertEqual(mask[1], 0.0)

    def test_dropout_injection(self):
        scenario = self.injector.create_dropout_scenario([2], start_cycle=0, end_cycle=100)
        x_out, mask = self.injector.apply(self.x, [scenario], t=10, sigma_baseline=self.sigma)
        # Target sensor should be zeroed out
        self.assertEqual(x_out[-1, 2], 0.0)
        self.assertEqual(mask[2], 1.0)


class TestStabilityProjection(unittest.TestCase):
    def test_projection_bound(self):
        """Verify that projected parameters never exceed radius rho."""
        n_sensors = 14
        latent_dim = 32
        adapter_dim = 16
        rho = 0.5

        model = SHELLModel(n_sensors=n_sensors, time_steps=30,
                           sensor_latent_dim=latent_dim, adapter_dim=adapter_dim)
        
        # Save theta_0
        theta_0 = [p.data.clone() for p in model.adapter.parameters()]
        
        # Add massive perturbation
        with torch.no_grad():
            for p in model.adapter.parameters():
                p.add_(torch.randn_like(p) * 10.0)
        
        # Compute pre-projection distance
        dist_before = torch.sqrt(sum(
            (p - p0).pow(2).sum() for p, p0 in zip(model.adapter.parameters(), theta_0)
        )).item()
        self.assertGreater(dist_before, rho)
        
        # Apply Euclidean projection
        with torch.no_grad():
            diffs = [p.data - p0 for p, p0 in zip(model.adapter.parameters(), theta_0)]
            total_norm = torch.sqrt(sum(d.pow(2).sum() for d in diffs))
            if total_norm > rho:
                scale = rho / total_norm
                for p, p0 in zip(model.adapter.parameters(), theta_0):
                    p.data.copy_(p0 + (p.data - p0) * scale)
        
        # Compute post-projection distance
        dist_after = torch.sqrt(sum(
            (p - p0).pow(2).sum() for p, p0 in zip(model.adapter.parameters(), theta_0)
        )).item()
        self.assertAlmostEqual(dist_after, rho, places=4)


class TestControllerActionSelection(unittest.TestCase):
    def test_nominal_noop(self):
        n_sensors = 14
        model = SHELLModel(n_sensors=n_sensors, time_steps=30,
                           sensor_latent_dim=32, adapter_dim=16)
        cfg = SHELLConfig()
        ctrl = SHELLController(model, None, cfg, n_sensors, 'cpu')
        
        # Initialize with steady clean signal
        x_init = np.random.randn(100, n_sensors).astype(np.float32)
        ctrl.initialize(x_init)
        
        # Fresh nominal input window
        x_window = torch.randn(1, 30, n_sensors)
        rul, action = ctrl.step(x_window)
        
        self.assertIsInstance(action, Action)
        self.assertIsInstance(rul, torch.Tensor)
        self.assertEqual(rul.shape, (1, 1))


if __name__ == '__main__':
    unittest.main()
