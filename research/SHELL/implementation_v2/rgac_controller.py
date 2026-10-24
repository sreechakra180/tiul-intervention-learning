"""
Reliability-Gated Adaptation Controller (RGAC) - Version 2
==========================================================
Core decision logic:
1. Healthy inputs (min(h) >= tau_healthy):
   - Action.NO_OP: uniform weights (w_i = 1/N).
   - Preserves baseline output bit-for-bit.
2. Mild noise (min(h) in [tau_mask, tau_healthy)):
   - Action.SOFT_REWEIGHT: linear proportional weighting w_i = h_i / sum(h).
3. Severe failure / Dropout (min(h) < tau_mask):
   - Action.IMPUTE: imputes failed channels with nominal baseline mean mu_{0,i}
     preserving uniform fusion representation, OR optionally uses learned reconstructor.
4. Full telemetry:
   - Tracks intervention frequencies, health histories, and execution latency.
"""

import numpy as np
import torch
import torch.nn as nn
from enum import IntEnum
from dataclasses import dataclass, field
from typing import Optional, List, Dict, Tuple
import time

class Action(IntEnum):
    NO_OP         = 0
    SOFT_REWEIGHT = 1
    IMPUTE        = 2
    RECONSTRUCT   = 3
    ADAPT         = 4

@dataclass
class RGACConfig:
    tau_healthy: float = 0.85        # Minimum health to allow NO_OP
    tau_mask: float = 0.25           # Threshold below which channel is imputed / masked
    use_reconstruction: bool = False # Whether to engage learned reconstructor vs mean imputation
    use_adaptation: bool = False     # Whether test-time gradient adaptation is enabled
    adapt_lr: float = 1e-4           # Learning rate for adapter
    rho_bound: float = 0.5           # Bounded parameter radius

class RGACController:
    def __init__(
        self,
        model: nn.Module,
        health_estimator,
        config: Optional[RGACConfig] = None,
        reconstructor: Optional[nn.Module] = None,
        device: str = 'cpu'
    ):
        self.model = model
        self.health_estimator = health_estimator
        self.config = config or RGACConfig()
        self.reconstructor = reconstructor
        self.device = device
        self.n_sensors = health_estimator.n_sensors if health_estimator else 14
        
        # Telemetry
        self.action_counts = {a: 0 for a in Action}
        self.health_history = []
        self.step_latencies = []
        
        # Initial adapter weights for bounded projection
        self._phi_0 = None
        if hasattr(self.model, 'adapter'):
            self._phi_0 = {name: p.data.clone() for name, p in self.model.adapter.named_parameters()}
            
    def reset(self):
        """Resets trajectory-level tracking state."""
        self.action_counts = {a: 0 for a in Action}
        self.health_history = []
        self.step_latencies = []
        if self._phi_0 and hasattr(self.model, 'adapter'):
            with torch.no_grad():
                for name, p in self.model.adapter.named_parameters():
                    p.data.copy_(self._phi_0[name])
                    
    @torch.no_grad()
    def step(self, x: torch.Tensor) -> Tuple[torch.Tensor, Action, Dict]:
        """
        Executes one causal inference step.
        Args:
            x: (1, W, n_sensors) or (batch, W, n_sensors)
        Returns:
            rul_pred: (batch, 1) or scalar
            action: selected Action enum
            diagnostics: dict with health scores, weights, latency
        """
        t0 = time.perf_counter()
        cfg = self.config
        x_np = x.detach().cpu().numpy()
        
        # 1. Health estimation
        health_scores = self.health_estimator.evaluate_window(x_np[0])
        min_health = float(health_scores.min())
        mean_health = float(health_scores.mean())
        
        # 2. Decision rule
        if min_health >= cfg.tau_healthy:
            action = Action.NO_OP
            rul_pred, _, _, _ = self.model(x)
            weights = np.ones(self.n_sensors) / self.n_sensors
            
        elif min_health >= cfg.tau_mask:
            action = Action.SOFT_REWEIGHT
            w = health_scores / (health_scores.sum() + 1e-8)
            weights = w
            w_tensor = torch.tensor(w, dtype=torch.float32, device=self.device)
            rul_pred, _, _, _ = self.model(x, w_tensor)
            
        else:
            # Severe failure / dropout
            failed_idx = np.where(health_scores < cfg.tau_mask)[0]
            
            if cfg.use_reconstruction and self.reconstructor is not None:
                action = Action.RECONSTRUCT
                # Latent reconstruction path
                _, _, _, z_per_sensor = self.model(x)
                mask_t = torch.ones(self.n_sensors, device=self.device)
                mask_t[failed_idx] = 0.0
                z_recon = self.reconstructor(z_per_sensor, mask_t)
                z_upd = list(z_per_sensor)
                for idx in failed_idx:
                    z_upd[idx] = z_recon
                z_stack = torch.stack(z_upd, dim=1)
                w_uniform = torch.ones(self.n_sensors, device=self.device) / self.n_sensors
                z_fused = (z_stack * w_uniform.view(1, -1, 1)).sum(dim=1)
                z_adapted = z_fused + self.model.adapter(z_fused)
                rul_pred = self.model.rul_head(z_adapted)
                weights = np.ones(self.n_sensors) / self.n_sensors
            else:
                action = Action.IMPUTE
                # Mean imputation: replace dead sensor with nominal baseline mean
                x_imputed = x.clone()
                if self.health_estimator.baseline_mu is not None:
                    mu_0 = self.health_estimator.baseline_mu
                    for idx in failed_idx:
                        x_imputed[:, :, idx] = float(mu_0[idx])
                rul_pred, _, _, _ = self.model(x_imputed)
                weights = np.ones(self.n_sensors) / self.n_sensors
                
        elapsed = time.perf_counter() - t0
        self.action_counts[action] += 1
        self.health_history.append(health_scores.copy())
        self.step_latencies.append(elapsed)
        
        diagnostics = {
            'health_scores': health_scores,
            'weights': weights,
            'action': action.name,
            'latency_ms': elapsed * 1000.0,
            'min_health': min_health,
            'mean_health': mean_health
        }
        
        return rul_pred, action, diagnostics
