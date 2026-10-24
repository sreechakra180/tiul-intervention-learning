"""
Candidate Action Generator for Counterfactual Intervention Governor (CIG)
==========================================================================
Generates candidate corrections and their shadow predictions on the frozen model:
- a_0: NO_OP (default frozen inference)
- a_1: SOFT_REWEIGHT (linear health-proportional fusion weights)
- a_2: MEAN_IMPUTE (imputes suspicious channels with nominal baseline mean)
- a_3: RECONSTRUCT (reconstructs suspicious latents via CrossModalReconstructor)
- a_4: HARD_MASK (zero-weights suspicious channels)
"""

import numpy as np
import torch
import torch.nn as nn
from enum import IntEnum
from typing import Dict, Tuple, List, Optional

class CandidateAction(IntEnum):
    NO_OP         = 0
    SOFT_REWEIGHT = 1
    MEAN_IMPUTE   = 2
    RECONSTRUCT   = 3
    HARD_MASK     = 4

class CandidateActionGenerator:
    def __init__(
        self,
        model: nn.Module,
        baseline_mu: np.ndarray,
        reconstructor: Optional[nn.Module] = None,
        tau_suspicious: float = 0.70,
        device: str = 'cpu'
    ):
        self.model = model
        self.baseline_mu = baseline_mu
        self.reconstructor = reconstructor
        self.tau_suspicious = tau_suspicious
        self.device = device
        self.n_sensors = len(baseline_mu)
        
    @torch.no_grad()
    def generate_candidate_predictions(
        self,
        x_tensor: torch.Tensor,
        health_scores: np.ndarray
    ) -> Dict[CandidateAction, float]:
        """
        Executes all 5 candidate actions in shadow mode on the frozen model.
        Args:
            x_tensor: (1, W, n_sensors)
            health_scores: (n_sensors,) in [0, 1]
        Returns:
            preds: Dict mapping CandidateAction -> scalar RUL prediction
        """
        preds = {}
        suspicious_idx = np.where(health_scores < self.tau_suspicious)[0]
        
        # a_0: NO_OP (Uniform weights, unmodified input)
        rul_0, _, _, z_per_sensor = self.model(x_tensor)
        preds[CandidateAction.NO_OP] = float(rul_0.squeeze().item())
        
        # a_1: SOFT_REWEIGHT
        w_soft = health_scores / (health_scores.sum() + 1e-8)
        w_soft_t = torch.tensor(w_soft, dtype=torch.float32, device=self.device)
        rul_1, _, _, _ = self.model(x_tensor, w_soft_t)
        preds[CandidateAction.SOFT_REWEIGHT] = float(rul_1.squeeze().item())
        
        # a_2: MEAN_IMPUTE
        if len(suspicious_idx) > 0:
            x_imp = x_tensor.clone()
            for idx in suspicious_idx:
                x_imp[:, :, idx] = float(self.baseline_mu[idx])
            rul_2, _, _, _ = self.model(x_imp)
            preds[CandidateAction.MEAN_IMPUTE] = float(rul_2.squeeze().item())
        else:
            preds[CandidateAction.MEAN_IMPUTE] = preds[CandidateAction.NO_OP]
            
        # a_3: RECONSTRUCT
        if len(suspicious_idx) > 0 and self.reconstructor is not None:
            mask_t = torch.ones(self.n_sensors, device=self.device)
            mask_t[suspicious_idx] = 0.0
            z_recon = self.reconstructor(z_per_sensor, mask_t)
            z_upd = list(z_per_sensor)
            for idx in suspicious_idx:
                z_upd[idx] = z_recon
            z_stack = torch.stack(z_upd, dim=1)
            w_uni = torch.ones(self.n_sensors, device=self.device) / self.n_sensors
            z_fused = (z_stack * w_uni.view(1, -1, 1)).sum(dim=1)
            z_adapted = z_fused + self.model.adapter(z_fused)
            rul_3 = self.model.rul_head(z_adapted)
            preds[CandidateAction.RECONSTRUCT] = float(rul_3.squeeze().item())
        else:
            preds[CandidateAction.RECONSTRUCT] = preds[CandidateAction.NO_OP]
            
        # a_4: HARD_MASK
        if len(suspicious_idx) > 0 and len(suspicious_idx) < self.n_sensors:
            w_mask = np.ones(self.n_sensors, dtype=np.float32)
            w_mask[suspicious_idx] = 0.0
            w_mask = w_mask / (w_mask.sum() + 1e-8)
            w_mask_t = torch.tensor(w_mask, dtype=torch.float32, device=self.device)
            rul_4, _, _, _ = self.model(x_tensor, w_mask_t)
            preds[CandidateAction.HARD_MASK] = float(rul_4.squeeze().item())
        else:
            preds[CandidateAction.HARD_MASK] = preds[CandidateAction.NO_OP]
            
        return preds
        
    def extract_action_feature_vector(
        self,
        x_window: np.ndarray,
        health_scores: np.ndarray,
        baseline_sigma: np.ndarray,
        action: CandidateAction,
        pred_a: float,
        pred_0: float
    ) -> np.ndarray:
        """
        Extracts feature vector for Action Risk Estimator Q(x, a).
        Features:
        1. min(h), mean(h), std(h), count(h < 0.70)
        2. Sorted lowest 2 health scores
        3. Max variance ratio relative to baseline
        4. Max absolute deviation from nominal mean (normalized)
        5. Dropout flag (is any channel exactly 0)
        6. pred_0 (baseline predicted RUL)
        7. pred_a (candidate predicted RUL)
        8. delta_pred = pred_a - pred_0
        9. abs(delta_pred)
        10. Action one-hot encoding (5 dims)
        Total dimension: 4 + 2 + 1 + 1 + 1 + 1 + 1 + 1 + 1 + 5 = 18 features
        """
        h_sorted = np.sort(health_scores)
        curr_std = np.std(x_window, axis=0)
        curr_last = x_window[-1, :]
        
        var_ratios = curr_std / (baseline_sigma + 1e-6)
        mean_devs = np.abs(curr_last - self.baseline_mu) / (baseline_sigma + 1e-6)
        
        has_dropout = 1.0 if np.any(np.abs(x_window) < 1e-4) else 0.0
        n_suspicious = float(np.sum(health_scores < self.tau_suspicious))
        
        delta = pred_a - pred_0
        
        # Action one-hot
        one_hot = np.zeros(len(CandidateAction), dtype=np.float32)
        one_hot[int(action)] = 1.0
        
        feats = [
            float(np.min(health_scores)),
            float(np.mean(health_scores)),
            float(np.std(health_scores)),
            n_suspicious,
            float(h_sorted[0]),
            float(h_sorted[1] if len(h_sorted) > 1 else h_sorted[0]),
            float(np.max(var_ratios)),
            float(np.max(mean_devs)),
            has_dropout,
            pred_0 / 125.0,        # Normalized to [0, 1]
            pred_a / 125.0,
            delta / 25.0,          # Scaled shift
            abs(delta) / 25.0,
        ]
        
        feats.extend(one_hot.tolist())
        return np.array(feats, dtype=np.float32)
