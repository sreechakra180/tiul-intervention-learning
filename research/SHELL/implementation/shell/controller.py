"""
SHELL Controller: Reliability-Gated Stability-Constrained Adaptation (RGSCA)
=============================================================================

This module implements the core SHELL decision controller that:
1. Estimates per-sensor health scores
2. Selects the appropriate corrective action from a prioritized hierarchy
3. Executes the selected action (reweight, mask, reconstruct, adapt, fallback)
4. Enforces the bounded-update stability constraint on adapter updates

The action hierarchy (cheapest-first):
    a_0: No-op      — continue normal inference
    a_1: Reweight   — update fusion weights from health scores
    a_2: Mask       — zero-weight severely failed sensors
    a_3: Reconstruct— reconstruct failed sensor latents from healthy ones
    a_4: Adapt      — update adapter parameters (stability-constrained)
    a_5: Fallback   — reset adapter to initialization

See research/SHELL/02_problem_formulation.md for full mathematical description.
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from dataclasses import dataclass, field
from typing import Optional, List, Dict, Tuple
from enum import IntEnum
import time


class Action(IntEnum):
    """Enumeration of SHELL corrective actions in order of increasing cost."""
    NO_OP       = 0
    REWEIGHT    = 1
    MASK        = 2
    RECONSTRUCT = 3
    ADAPT       = 4
    FALLBACK    = 5


@dataclass
class SHELLConfig:
    """
    Configuration for the SHELL controller.
    
    All threshold parameters are tuned on validation data (NOT test data).
    Default values are set based on preliminary experimentation on FD001.
    """
    # Health score estimation
    ema_alpha: float = 0.1          # EWA coefficient for variance estimation
    health_kappa: float = 3.0       # Sharpness parameter for health score sigmoid
    
    # Action selection thresholds
    tau_healthy: float = 0.85       # Health mean above which no action needed
    tau_reweight: float = 0.60      # Minimum mean health for reweighting
    tau_mask: float = 0.25          # Individual health below which sensor is masked
    tau_reconstruct: float = 0.40   # Health mean above which reconstruction is attempted
    tau_consistency: float = 0.50   # Cross-modal consistency below which recon is triggered
    tau_uncertainty: float = 1.5    # Prediction entropy above which adaptation is triggered
    tau_drift: float = 0.10         # Distribution drift above which adaptation is triggered
    tau_knowledge: float = 2.0      # Max allowed adapter drift before fallback
    
    # Stability-constrained update
    delta_max_base: float = 0.5     # Base maximum adapter Frobenius norm
    gradient_clip: float = 1.0      # Gradient clipping for adapter updates
    
    # Adaptation hyperparameters
    adapt_lr: float = 1e-3          # Learning rate for adapter updates
    adapt_steps: int = 1            # Number of gradient steps per adaptation event
    lambda_kr: float = 0.1          # Knowledge retention regularization weight
    lambda_consistency: float = 0.1 # Consistency regularization weight
    
    # Resource budget
    max_adapt_per_window: int = 5   # Maximum adapter updates per evaluation window
    window_size: int = 50           # Window size for budget tracking (cycles)
    
    # Sensor restoration
    restoration_warmup: int = 10    # Cycles after restoration before trusting sensor fully
    
    # Thresholds for fusion temperature
    fusion_temperature: float = 0.5  # Temperature for softmax over health scores


@dataclass
class ControllerState:
    """Mutable state maintained by the SHELL controller across timesteps."""
    # Health score tracking (exponential moving average of sensor variance)
    sigma_hat: np.ndarray = field(default_factory=lambda: np.ones(14))   # (n_sensors,)
    sigma_baseline: np.ndarray = field(default_factory=lambda: np.ones(14))
    mu_hat: np.ndarray = field(default_factory=lambda: np.zeros(14))
    health_scores: np.ndarray = field(default_factory=lambda: np.ones(14))
    
    # Reliability weights
    reliability_weights: np.ndarray = field(default_factory=lambda: np.ones(14) / 14)
    
    # Action tracking
    action_history: List[Action] = field(default_factory=list)
    adapt_count_in_window: int = 0
    window_step: int = 0
    
    # Sensor restoration tracking
    # When sensor i was last masked: mask_start[i] = cycle
    mask_start: Dict[int, int] = field(default_factory=dict)
    restoration_countdown: Dict[int, int] = field(default_factory=dict)
    
    # Step counter
    step: int = 0


class HealthEstimator:
    """
    Estimates per-sensor health scores using online statistics.
    
    Health score h_i(t) ∈ [0, 1] is computed from:
    - Relative increase in signal variance vs. baseline
    - Cross-sensor consistency (how well sensor agrees with others in latent space)
    
    This is a heuristic estimator. It requires calibration on clean data.
    """
    
    def __init__(self, n_sensors: int, config: SHELLConfig):
        self.n_sensors = n_sensors
        self.config = config
        self.alpha = config.ema_alpha
        self.kappa = config.health_kappa
    
    def initialize(self, state: ControllerState, x_baseline: np.ndarray):
        """
        Initialize baseline statistics from clean sensor data.
        
        Args:
            state: ControllerState to update
            x_baseline: (T_baseline, n_sensors) clean sensor readings
        """
        state.sigma_baseline = np.std(x_baseline, axis=0) + 1e-6
        state.mu_hat = np.mean(x_baseline, axis=0)
        state.sigma_hat = state.sigma_baseline.copy()
        state.health_scores = np.ones(self.n_sensors)
    
    def update(
        self,
        state: ControllerState,
        x_current: np.ndarray,
        z_per_sensor: Optional[List[torch.Tensor]] = None,
    ) -> np.ndarray:
        """
        Update health score estimates from current observation.
        
        Args:
            state: ControllerState to update in place
            x_current: (n_sensors,) current sensor readings
            z_per_sensor: optional list of latent representations for cross-modal check
        
        Returns:
            health_scores: (n_sensors,) updated health scores
        """
        alpha = self.alpha
        
        # Update running variance estimate (EMA)
        state.mu_hat = (1 - alpha) * state.mu_hat + alpha * x_current
        squared_dev = (x_current - state.mu_hat) ** 2
        state.sigma_hat = (1 - alpha) * state.sigma_hat + alpha * squared_dev
        
        # Compute health from relative variance increase
        # Higher ratio of current variance to baseline → lower health
        # Threshold at 1.8 so nominal variance (ratio ~1.0) yields health ~0.92
        variance_ratio = np.sqrt(state.sigma_hat) / (state.sigma_baseline + 1e-6)
        raw_health = 1.0 / (1.0 + np.exp(self.kappa * (variance_ratio - 1.8)))
        
        # Handle complete dropout (sensor reads exactly 0 for non-zero sensors)
        dropout_mask = (np.abs(x_current) < 1e-6) & (state.sigma_baseline > 0.01)
        raw_health[dropout_mask] = 0.0
        
        state.health_scores = raw_health
        return state.health_scores
    
    def compute_cross_consistency(
        self,
        z_per_sensor: List[torch.Tensor],
        health_scores: np.ndarray,
    ) -> float:
        """
        Compute cross-modal consistency among healthy sensors' latents.
        
        C(t) ∈ [0, 1]; 1 = perfectly consistent latent representations
        """
        if not z_per_sensor or len(z_per_sensor) < 2:
            return 1.0
        
        healthy_idx = np.where(health_scores > 0.3)[0]
        if len(healthy_idx) < 2:
            return 0.0
        
        # Compute pairwise cosine similarity between healthy sensor latents
        similarities = []
        with torch.no_grad():
            for i in healthy_idx:
                for j in healthy_idx:
                    if i >= j:
                        continue
                    z_i = z_per_sensor[i]
                    z_j = z_per_sensor[j]
                    cos_sim = F.cosine_similarity(z_i, z_j, dim=-1).mean().item()
                    similarities.append(cos_sim)
        
        return float(np.mean(similarities)) if similarities else 1.0
    
    def compute_reliability_weights(
        self,
        health_scores: np.ndarray,
        temperature: float = 0.5,
    ) -> np.ndarray:
        """
        Convert health scores to fusion reliability weights via softmax.
        
        w_i = softmax(h_i / temperature)
        
        Lower temperature → more aggressive reweighting
        """
        h = health_scores / (temperature + 1e-8)
        w = np.exp(h - h.max())  # numerically stable softmax
        w = w / (w.sum() + 1e-8)
        return w


class SHELLController:
    """
    Main SHELL decision controller implementing RGSCA.
    
    The controller maintains running state across timesteps and selects
    the appropriate action based on current sensor health, uncertainty,
    distribution drift, and resource availability.
    
    Usage:
        controller = SHELLController(model, reconstructor, config, n_sensors)
        controller.initialize(x_baseline, dataloader)
        
        for x_batch, y_true in test_stream:
            rul_pred, action = controller.step(x_batch)
            record_action(action, rul_pred, y_true)
    """
    
    def __init__(
        self,
        model,                      # SHELLModel instance
        reconstructor,              # CrossModalReconstructor instance
        config: SHELLConfig,
        n_sensors: int = 14,
        device: str = 'cpu',
    ):
        self.model = model
        self.reconstructor = reconstructor
        self.config = config
        self.n_sensors = n_sensors
        self.device = device
        
        self.health_estimator = HealthEstimator(n_sensors, config)
        self.state = ControllerState(
            sigma_hat=np.ones(n_sensors),
            sigma_baseline=np.ones(n_sensors),
            mu_hat=np.zeros(n_sensors),
            health_scores=np.ones(n_sensors),
            reliability_weights=np.ones(n_sensors) / n_sensors,
        )
        
        # Optimizer for adapter (only active when action a_4 is selected)
        self.optimizer = torch.optim.Adam(
            self.model.adapter.parameters(),
            lr=config.adapt_lr
        )
        
        # Store initial adapter weights for knowledge retention loss
        self._phi_0 = {
            name: param.data.clone()
            for name, param in self.model.adapter.named_parameters()
        }
        
        # Metrics tracking
        self.action_counts = {a: 0 for a in Action}
        self.adaptation_times = []
        self.rul_predictions = []
        self.health_history = []
    
    def initialize(self, x_baseline: np.ndarray):
        """
        Initialize health estimator with clean sensor statistics.
        
        Args:
            x_baseline: (T_baseline, n_sensors) clean sensor readings
                        These are the last N cycles from the training set
                        for the held-out evaluation engines.
        """
        self.health_estimator.initialize(self.state, x_baseline)
        print(f"[SHELL] Initialized with baseline stats from {len(x_baseline)} samples")
        print(f"[SHELL] Baseline sigma: {self.state.sigma_baseline.round(4)}")
    
    def _compute_delta_max(self, h_mean: float) -> float:
        """
        Compute maximum allowed adapter update norm based on sensor health.
        
        δ_max = δ_base · h_mean
        
        Lower health → tighter constraint on parameter updates.
        This is the key coupling between sensor health and model stability.
        """
        return self.config.delta_max_base * h_mean
    
    def _compute_distribution_drift(
        self,
        z_adapted: torch.Tensor,
        reference_mean: Optional[torch.Tensor] = None,
    ) -> float:
        """
        Estimate distribution drift as deviation of current batch
        latent distribution from source distribution.
        
        Returns a scalar drift estimate in [0, ∞).
        """
        if reference_mean is None:
            return 0.0
        
        current_mean = z_adapted.detach().mean(dim=0)
        drift = (current_mean - reference_mean).norm(p=2).item()
        return drift
    
    def _compute_prediction_entropy(self, rul_pred: torch.Tensor) -> float:
        """
        Approximate prediction uncertainty for regression via prediction variance.
        
        For RUL regression, we use a simple proxy: deviation from the
        expected range [0, 125].
        
        In Phase 1, this is a simple range check; Phase 2 would use an ensemble.
        """
        rul_val = rul_pred.detach().squeeze().mean().item()
        # Normalized uncertainty: how far from the center of the range
        uncertainty = abs(rul_val - 62.5) / 62.5
        return uncertainty
    
    def _select_action(
        self,
        health_scores: np.ndarray,
        cross_consistency: float,
        uncertainty: float,
        drift: float,
    ) -> Action:
        """
        Select the cheapest sufficient corrective action.
        
        This implements the decision rule from §4 of the problem formulation.
        """
        h_mean = health_scores.mean()
        h_min = health_scores.min()
        cfg = self.config
        
        # Check resource budget
        budget_available = (
            self.state.adapt_count_in_window < cfg.max_adapt_per_window
        )
        
        # Decision hierarchy (cheapest first)
        if h_mean >= cfg.tau_healthy:
            return Action.NO_OP
        
        elif h_mean >= cfg.tau_reweight:
            return Action.REWEIGHT
        
        elif h_min < cfg.tau_mask:
            # At least one sensor has failed severely
            if h_mean >= cfg.tau_reconstruct and cross_consistency >= cfg.tau_consistency:
                if budget_available:
                    return Action.RECONSTRUCT
                else:
                    return Action.MASK
            else:
                return Action.MASK
        
        elif (uncertainty > cfg.tau_uncertainty or drift > cfg.tau_drift):
            if budget_available:
                adapter_drift = self.model.adapter_drift()
                if adapter_drift > cfg.tau_knowledge:
                    return Action.FALLBACK
                else:
                    return Action.ADAPT
            else:
                return Action.REWEIGHT
        
        else:
            return Action.REWEIGHT
    
    def _execute_reweight(self, health_scores: np.ndarray) -> torch.Tensor:
        """Update reliability weights from health scores."""
        weights = self.health_estimator.compute_reliability_weights(
            health_scores, self.config.fusion_temperature
        )
        self.state.reliability_weights = weights
        return torch.tensor(weights, dtype=torch.float32, device=self.device)
    
    def _execute_mask(self, health_scores: np.ndarray) -> torch.Tensor:
        """Zero out severely failed sensors in reliability weights."""
        weights = health_scores.copy()
        weights[weights < self.config.tau_mask] = 0.0
        weight_sum = weights.sum()
        if weight_sum < 1e-8:
            # All sensors failed: fallback to uniform weights
            weights = np.ones(self.n_sensors) / self.n_sensors
        else:
            weights = weights / weight_sum
        self.state.reliability_weights = weights
        return torch.tensor(weights, dtype=torch.float32, device=self.device)
    
    def _execute_reconstruct(
        self,
        x: torch.Tensor,
        health_scores: np.ndarray,
        z_per_sensor: List[torch.Tensor],
    ) -> tuple:
        """
        Reconstruct failed sensor latents and update weights.
        
        Returns updated (reliability_weights, z_per_sensor_updated)
        """
        mask = (health_scores >= self.config.tau_mask).astype(np.float32)
        mask_tensor = torch.tensor(mask, device=self.device)
        
        failed_indices = np.where(mask == 0)[0]
        
        if len(failed_indices) > 0 and self.reconstructor is not None:
            with torch.no_grad():
                z_recon = self.reconstructor(z_per_sensor, mask_tensor)
                # Replace failed sensor latents with reconstructed ones
                z_per_sensor_updated = list(z_per_sensor)
                for idx in failed_indices:
                    z_per_sensor_updated[idx] = z_recon
        else:
            z_per_sensor_updated = z_per_sensor
        
        # After reconstruction, restore healthy weighting to reconstructed sensors
        effective_health = health_scores.copy()
        effective_health[failed_indices] = 0.85
        weights = self._execute_reweight(effective_health)
        return weights, z_per_sensor_updated
    
    def _execute_adapt(
        self,
        z_adapted: torch.Tensor,
        h_mean: float,
    ) -> None:
        """
        Update adapter with stability-constrained gradient step.
        
        Stability mechanism:
        1. Compute adapter update gradient
        2. Apply gradient clipping
        3. Apply optimizer step
        4. Project adapter back to bounded region (δ_max based on h_mean)
        """
        cfg = self.config
        
        # Entropy minimization loss (for unlabeled test-time adaptation)
        # In RUL regression, we use teacher consistency loss instead
        # Teacher: frozen backbone + frozen head applied to clean-equivalent output
        rul_pred, _, z_adapted, _ = self.model(
            self._current_x,
            torch.tensor(self.state.reliability_weights, dtype=torch.float32, device=self.device)
        )
        
        # Pseudo-label consistency: prediction should be similar to frozen teacher
        with torch.no_grad():
            rul_teacher, _, _, _ = self.model(self._current_x)
        
        # Consistency loss (adapter should not drift far from teacher prediction)
        task_loss = F.mse_loss(rul_pred, rul_teacher.detach())
        
        # Knowledge retention loss (prevent adapter from drifting from initialization)
        kr_loss = 0.0
        for name, param in self.model.adapter.named_parameters():
            init_p = self._phi_0[name].to(param.device)
            kr_loss = kr_loss + (param - init_p).pow(2).sum()
        
        total_loss = task_loss + cfg.lambda_kr * kr_loss
        
        self.optimizer.zero_grad()
        total_loss.backward()
        
        # Gradient clipping
        nn.utils.clip_grad_norm_(
            self.model.adapter.parameters(),
            max_norm=cfg.gradient_clip
        )
        
        self.optimizer.step()
        
        # Bounded-update projection (Claim 1 from problem formulation)
        delta_max = self._compute_delta_max(h_mean)
        self.model.project_adapter(delta_max)
        
        self.state.adapt_count_in_window += 1
    
    def _execute_fallback(self):
        """Reset adapter to initialization state."""
        if self.model._adapter_init_params is None:
            return
        for name, param in self.model.adapter.named_parameters():
            init_p = self.model._adapter_init_params[name]
            param.data.copy_(init_p)
        print(f"[SHELL] Fallback: adapter reset to initialization (step {self.state.step})")
    
    @torch.no_grad()
    def step(
        self,
        x: torch.Tensor,
        return_details: bool = False,
    ) -> Tuple[torch.Tensor, Action]:
        """
        Execute one SHELL timestep.
        
        Args:
            x: (batch, time_steps, n_sensors) — current observation batch
            return_details: if True, return dict with full diagnostics
        
        Returns:
            rul_pred: (batch, 1) — RUL prediction
            action: Action — action taken this step
        """
        t0 = time.perf_counter()
        
        # Store for possible use in _execute_adapt
        self._current_x = x
        
        # Step 1: Estimate sensor health from current batch
        x_np = x.detach().cpu().numpy()
        x_last = x_np[:, -1, :]  # use last timestep in window for health estimation
        x_mean_last = x_last.mean(axis=0)  # batch average
        
        health_scores = self.health_estimator.update(self.state, x_mean_last)
        
        # Reliability weights tensor
        w_tensor = torch.tensor(
            self.state.reliability_weights,
            dtype=torch.float32, device=self.device
        )
        
        # Step 2: Forward pass with current weights (no grad for inference)
        rul_pred, z_fused, z_adapted, z_per_sensor = self.model(x, w_tensor)
        
        # Step 3: Compute system diagnostics
        cross_consistency = self.health_estimator.compute_cross_consistency(
            z_per_sensor, health_scores)
        uncertainty = self._compute_prediction_entropy(rul_pred)
        drift = 0.0  # Phase 1: simplified drift estimate
        
        # Step 4: Select action
        action = self._select_action(health_scores, cross_consistency, uncertainty, drift)
        
        # Step 5: Execute action
        if action == Action.NO_OP:
            pass  # Use current rul_pred as-is
        
        elif action == Action.REWEIGHT:
            w_tensor = self._execute_reweight(health_scores)
            with torch.no_grad():
                rul_pred, z_fused, z_adapted, z_per_sensor = self.model(x, w_tensor)
        
        elif action == Action.MASK:
            w_tensor = self._execute_mask(health_scores)
            with torch.no_grad():
                rul_pred, z_fused, z_adapted, z_per_sensor = self.model(x, w_tensor)
        
        elif action == Action.RECONSTRUCT:
            w_tensor, z_per_sensor_upd = self._execute_reconstruct(
                x, health_scores, z_per_sensor)
            # Re-fuse with reconstructed latents
            z_upd_stack = torch.stack(z_per_sensor_upd, dim=1)
            w = w_tensor.to(self.device)
            w = w / (w.sum() + 1e-8)
            z_fused_upd = (z_upd_stack * w.view(1, -1, 1)).sum(dim=1)
            z_adapted_upd = z_fused_upd + self.model.adapter(z_fused_upd)
            rul_pred = self.model.rul_head(z_adapted_upd)
        
        elif action == Action.ADAPT:
            # Execute adaptation (needs gradient)
            with torch.enable_grad():
                self._execute_adapt(z_adapted, health_scores.mean())
            # Re-run forward pass after adaptation
            with torch.no_grad():
                w_tensor = self._execute_reweight(health_scores)
                rul_pred, z_fused, z_adapted, z_per_sensor = self.model(x, w_tensor)
        
        elif action == Action.FALLBACK:
            self._execute_fallback()
            with torch.no_grad():
                rul_pred, z_fused, z_adapted, z_per_sensor = self.model(x)
        
        # Step 6: Update tracking state
        self.action_counts[action] += 1
        self.state.action_history.append(action)
        self.state.step += 1
        self.state.window_step += 1
        
        # Reset window budget counter
        if self.state.window_step >= self.config.window_size:
            self.state.window_step = 0
            self.state.adapt_count_in_window = 0
        
        # Track metrics
        self.health_history.append(health_scores.copy())
        elapsed = time.perf_counter() - t0
        self.adaptation_times.append(elapsed)
        
        if return_details:
            return rul_pred, action, {
                'health_scores': health_scores,
                'reliability_weights': self.state.reliability_weights.copy(),
                'cross_consistency': cross_consistency,
                'uncertainty': uncertainty,
                'drift': drift,
                'adapter_drift': self.model.adapter_drift(),
                'elapsed_ms': elapsed * 1000,
            }
        
        return rul_pred, action
    
    def get_action_summary(self) -> Dict[str, int]:
        """Return action frequency counts."""
        return {a.name: self.action_counts[a] for a in Action}
    
    def get_mean_health(self) -> float:
        """Return mean health score across all timesteps."""
        if not self.health_history:
            return 1.0
        return float(np.mean([h.mean() for h in self.health_history]))
