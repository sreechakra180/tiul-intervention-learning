"""
Counterfactual Intervention Governor (CIG) - Implementation
============================================================
Working Title: When Should a Frozen Prognostic Model Be Repaired?
               Risk-Gated Sensor Intervention Under Degradation

Key Mechanism:
1. Evaluates candidate sensor corrections in shadow mode on the frozen model.
2. An action-conditioned risk estimator predicts expected error R_hat(a) for each candidate.
3. Compares estimated error reduction Delta_R(a) = R_hat(a_0) - R_hat(a) against safety margin gamma and cost.
4. COMMITS the correction only when credible evidence demonstrates benefit;
   otherwise REJECTS and falls back safely to default frozen prediction.
"""

import time
import numpy as np
import torch
import torch.nn as nn
from dataclasses import dataclass, field
from typing import Dict, Tuple, List, Optional
from action_generator import CandidateAction, CandidateActionGenerator
from risk_estimator import CalibratedActionRiskEstimator

@dataclass
class CIGConfig:
    # Safety margin: minimum estimated error reduction (in cycles) required to intervene
    safety_margin_gamma: float = 0.50
    
    # Action cost penalties (favor simpler interventions over aggressive ones)
    action_costs: Dict[CandidateAction, float] = field(default_factory=lambda: {
        CandidateAction.NO_OP: 0.0,
        CandidateAction.SOFT_REWEIGHT: 0.10,
        CandidateAction.MEAN_IMPUTE: 0.20,
        CandidateAction.HARD_MASK: 0.30,
        CandidateAction.RECONSTRUCT: 0.40,
    })
    
    # Nominal health threshold below which shadow evaluation triggers
    tau_nominal: float = 0.85
    
    # Suspicious threshold passed to action generator
    tau_suspicious: float = 0.70

class CounterfactualInterventionGovernor:
    def __init__(
        self,
        model: nn.Module,
        health_estimator,
        risk_estimator: CalibratedActionRiskEstimator,
        action_generator: CandidateActionGenerator,
        config: Optional[CIGConfig] = None,
        device: str = 'cpu'
    ):
        self.model = model
        self.health_estimator = health_estimator
        self.risk_estimator = risk_estimator
        self.action_generator = action_generator
        self.config = config or CIGConfig()
        self.device = device
        
        # Telemetry
        self.total_steps = 0
        self.committed_actions = {a: 0 for a in CandidateAction}
        self.rejected_count = 0
        self.latencies = []
        
    def reset(self):
        """Resets engine trajectory state."""
        self.total_steps = 0
        self.committed_actions = {a: 0 for a in CandidateAction}
        self.rejected_count = 0
        self.latencies = []
        
    @torch.no_grad()
    def step(self, x: torch.Tensor) -> Tuple[float, CandidateAction, Dict]:
        """
        Executes one causal inference step.
        Args:
            x: (1, W, n_sensors)
        Returns:
            final_pred: float RUL prediction
            selected_action: CandidateAction committed (or NO_OP if rejected)
            diagnostics: telemetry dict
        """
        t0 = time.perf_counter()
        cfg = self.config
        x_np = x.detach().cpu().numpy()[0]
        
        # 1. Causal health estimation
        health_scores = self.health_estimator.evaluate_window(x_np)
        min_health = float(np.min(health_scores))
        mean_health = float(np.mean(health_scores))
        
        # Fast path: If all sensors appear completely healthy, NO_OP immediately
        if min_health >= cfg.tau_nominal:
            rul_0, _, _, _ = self.model(x)
            pred_0 = float(rul_0.squeeze().item())
            elapsed = time.perf_counter() - t0
            self.total_steps += 1
            self.committed_actions[CandidateAction.NO_OP] += 1
            self.latencies.append(elapsed)
            
            return pred_0, CandidateAction.NO_OP, {
                'action_committed': CandidateAction.NO_OP.name,
                'rejected': False,
                'reason': 'nominal_fast_path',
                'min_health': min_health,
                'mean_health': mean_health,
                'latency_ms': elapsed * 1000.0,
                'risk_estimates': {CandidateAction.NO_OP.name: 0.0},
                'estimated_benefit': 0.0
            }
            
        # 2. Suspicious sensor detected: Shadow evaluation of candidate actions
        cand_preds = self.action_generator.generate_candidate_predictions(x, health_scores)
        pred_0 = cand_preds[CandidateAction.NO_OP]
        
        # 3. Estimate risk for each candidate action
        risk_estimates = {}
        for act in CandidateAction:
            pred_a = cand_preds[act]
            feat = self.action_generator.extract_action_feature_vector(
                x_window=x_np,
                health_scores=health_scores,
                baseline_sigma=self.health_estimator.baseline_sigma,
                action=act,
                pred_a=pred_a,
                pred_0=pred_0
            )
            risk_estimates[act] = self.risk_estimator.predict_risk(feat)
            
        # 4. Counterfactual Decision Rule
        risk_0 = risk_estimates[CandidateAction.NO_OP]
        
        # Calculate net benefit for each alternative action
        benefits = {}
        for act in CandidateAction:
            if act == CandidateAction.NO_OP:
                continue
            cost = cfg.action_costs.get(act, 0.0)
            net_benefit = (risk_0 - risk_estimates[act]) - cost
            benefits[act] = net_benefit
            
        best_act = max(benefits.keys(), key=lambda a: benefits[a])
        best_net_benefit = benefits[best_act]
        
        # 5. Commit or Reject
        if best_net_benefit > cfg.safety_margin_gamma:
            # Commit intervention
            committed_act = best_act
            final_pred = cand_preds[committed_act]
            rejected = False
            reason = f'committed_{committed_act.name}_net_benefit_{best_net_benefit:.2f}'
            self.committed_actions[committed_act] += 1
        else:
            # Reject intervention and safely retain default frozen prediction
            committed_act = CandidateAction.NO_OP
            final_pred = pred_0
            rejected = True
            reason = f'rejected_best_benefit_{best_net_benefit:.2f}_below_gamma_{cfg.safety_margin_gamma:.2f}'
            self.rejected_count += 1
            self.committed_actions[CandidateAction.NO_OP] += 1
            
        elapsed = time.perf_counter() - t0
        self.total_steps += 1
        self.latencies.append(elapsed)
        
        diagnostics = {
            'action_committed': committed_act.name,
            'rejected': rejected,
            'reason': reason,
            'min_health': min_health,
            'mean_health': mean_health,
            'latency_ms': elapsed * 1000.0,
            'risk_estimates': {a.name: round(r, 2) for a, r in risk_estimates.items()},
            'candidate_predictions': {a.name: round(p, 2) for a, p in cand_preds.items()},
            'estimated_benefit': round(best_net_benefit, 3)
        }
        
        return final_pred, committed_act, diagnostics
