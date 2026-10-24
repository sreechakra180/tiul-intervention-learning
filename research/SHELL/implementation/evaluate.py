"""
SHELL Evaluation Script
========================
Evaluates SHELL and all baselines under all degradation scenarios.

IMPORTANT: All thresholds and hyperparameters for SHELL were tuned 
on the VALIDATION SET, not the test set. Test labels are only 
revealed at this final evaluation stage.

Baselines evaluated:
    B0: Static (no adaptation, no degradation)
    B1: Static + Degradation (no adaptation)
    B2: Tent (entropy minimization TTA)
    B3: CoTTA-style (weight restoration + augmentation averaging)
    B4: Reliability-reweight only (no model adaptation)
    B5: EWC adapter update (no health gating)
    B6: SHELL-noHealth (random gating instead of health score)
    B7: SHELL-noStability (no bounded-update projection)
    B8: SHELL-noRecon (no reconstruction option)
    B9: SHELL (full proposed method)

Usage:
    python evaluate.py --checkpoint ./checkpoints/shell_backbone_FD001_seed42.pt \
                       --data_dir ./data/raw --subset FD001 --seed 42
"""

import os
import sys
import argparse
import json
import time
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from copy import deepcopy

sys.path.insert(0, str(Path(__file__).parent))

from data.download_cmapss import load_and_preprocess, create_sequences
from models.backbone import SHELLModel, CrossModalReconstructor
from shell.controller import SHELLController, SHELLConfig, Action
from shell.degradation import DegradationInjector, DegradationScenario


def set_seed(seed: int):
    import random
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def nasa_score(pred: np.ndarray, true: np.ndarray) -> float:
    """Asymmetric NASA scoring function for RUL estimation."""
    d = pred - true
    score = np.where(d < 0, np.exp(-d / 13) - 1, np.exp(d / 10) - 1)
    return float(score.sum())


def compute_metrics(pred: np.ndarray, true: np.ndarray) -> dict:
    """Compute all evaluation metrics."""
    rmse = float(np.sqrt(np.mean((pred - true) ** 2)))
    mae = float(np.mean(np.abs(pred - true)))
    ns = nasa_score(pred, true)
    return {'rmse': rmse, 'mae': mae, 'nasa_score': ns}


# ============================================================
# BASELINE IMPLEMENTATIONS
# ============================================================

class StaticBaseline:
    """B0/B1: No adaptation. Runs frozen pretrained model."""
    
    def __init__(self, model: SHELLModel, device: str):
        self.model = model
        self.device = device
        model.eval()
        for p in model.parameters():
            p.requires_grad_(False)
    
    @torch.no_grad()
    def predict(self, x: torch.Tensor) -> np.ndarray:
        rul, _, _, _ = self.model(x)
        return rul.squeeze().cpu().numpy()


class TentBaseline:
    """
    B2: Tent (Entropy Minimization TTA).
    Adapted for regression via prediction variance / pseudo-label loss.
    """
    def __init__(self, model: SHELLModel, lr: float = 1e-4, device: str = 'cpu'):
        self.model = deepcopy(model)
        self.device = device
        self.model.freeze_backbone_and_head()
        self.optimizer = torch.optim.Adam(self.model.adapter.parameters(), lr=lr)
        self.ref_model = deepcopy(model)
        self.ref_model.eval()
        for p in self.ref_model.parameters():
            p.requires_grad_(False)
    
    def predict(self, x: torch.Tensor) -> np.ndarray:
        self.model.train()
        rul_pred, _, _, _ = self.model(x)
        with torch.no_grad():
            ref_pred, _, _, _ = self.ref_model(x)
        loss = F.mse_loss(rul_pred, ref_pred)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        self.model.eval()
        with torch.no_grad():
            rul_out, _, _, _ = self.model(x)
        return rul_out.squeeze().detach().cpu().numpy()


class CoTTABaseline:
    """
    B3: CoTTA (Continual Test-Time Adaptation).
    Uses stochastic weight restoration.
    """
    def __init__(self, model: SHELLModel, lr: float = 1e-4, p_restore: float = 0.05, device: str = 'cpu'):
        self.model = deepcopy(model)
        self.device = device
        self.p_restore = p_restore
        self.model.freeze_backbone_and_head()
        self.optimizer = torch.optim.Adam(self.model.adapter.parameters(), lr=lr)
        self._phi_0 = {
            name: param.data.clone()
            for name, param in self.model.adapter.named_parameters()
        }
        self.ref_model = deepcopy(model)
        self.ref_model.eval()
        for p in self.ref_model.parameters():
            p.requires_grad_(False)

    def predict(self, x: torch.Tensor) -> np.ndarray:
        self.model.train()
        rul_pred, _, _, _ = self.model(x)
        with torch.no_grad():
            ref_pred, _, _, _ = self.ref_model(x)
        loss = F.mse_loss(rul_pred, ref_pred)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        with torch.no_grad():
            for name, param in self.model.adapter.named_parameters():
                mask = torch.rand_like(param) < self.p_restore
                param.data[mask] = self._phi_0[name].to(param.device)[mask]
        self.model.eval()
        with torch.no_grad():
            rul_out, _, _, _ = self.model(x)
        return rul_out.squeeze().detach().cpu().numpy()


class ReliabilityReweightBaseline:
    """
    B4: Reliability-weighted fusion only, no model adaptation.
    Fusion weights from sensor variance statistics.
    """
    def __init__(self, model: SHELLModel, device: str):
        self.model = deepcopy(model)
        self.device = device
        self.model.eval()
        for p in self.model.parameters():
            p.requires_grad_(False)
        self.n_sensors = model.n_sensors
        self.sigma_hat = np.ones(self.n_sensors)
        self.sigma_baseline = np.ones(self.n_sensors) * 0.01
        self.mu_hat = np.zeros(self.n_sensors)
        self.alpha = 0.1
    
    @torch.no_grad()
    def predict(self, x: torch.Tensor) -> np.ndarray:
        x_np = x.cpu().numpy()
        x_last = x_np[:, -1, :].mean(axis=0)
        self.mu_hat = (1 - self.alpha) * self.mu_hat + self.alpha * x_last
        sq_dev = (x_last - self.mu_hat) ** 2
        self.sigma_hat = (1 - self.alpha) * self.sigma_hat + self.alpha * sq_dev
        ratio = np.sqrt(self.sigma_hat) / (self.sigma_baseline + 1e-6)
        health = 1.0 / (1.0 + np.exp(3.0 * (ratio - 1.0)))
        w = np.exp(health / 0.5)
        w = w / (w.sum() + 1e-8)
        w_tensor = torch.tensor(w, dtype=torch.float32, device=self.device)
        rul, _, _, _ = self.model(x, w_tensor)
        return rul.squeeze().cpu().numpy()
    
    def initialize(self, x_baseline: np.ndarray):
        self.sigma_baseline = np.std(x_baseline, axis=0) + 1e-6


class EWCBaseline:
    """
    B5: EWC regularized adapter update.
    """
    def __init__(self, model: SHELLModel, lr: float = 1e-4, ewc_lambda: float = 0.5, device: str = 'cpu'):
        self.model = deepcopy(model)
        self.device = device
        self.ewc_lambda = ewc_lambda
        self.model.freeze_backbone_and_head()
        self.optimizer = torch.optim.Adam(self.model.adapter.parameters(), lr=lr)
        self._phi_0 = {
            name: param.data.clone()
            for name, param in self.model.adapter.named_parameters()
        }
        self.ref_model = deepcopy(model)
        self.ref_model.eval()
        for p in self.ref_model.parameters():
            p.requires_grad_(False)

    def predict(self, x: torch.Tensor) -> np.ndarray:
        self.model.train()
        rul_pred, _, _, _ = self.model(x)
        with torch.no_grad():
            ref_pred, _, _, _ = self.ref_model(x)
        task_loss = F.mse_loss(rul_pred, ref_pred)
        ewc_loss = 0.0
        for name, param in self.model.adapter.named_parameters():
            init_p = self._phi_0[name].to(param.device)
            ewc_loss = ewc_loss + (param - init_p).pow(2).sum()
        total_loss = task_loss + self.ewc_lambda * ewc_loss
        self.optimizer.zero_grad()
        total_loss.backward()
        nn.utils.clip_grad_norm_(self.model.adapter.parameters(), 1.0)
        self.optimizer.step()
        self.model.eval()
        with torch.no_grad():
            rul_out, _, _, _ = self.model(x)
        return rul_out.squeeze().detach().cpu().numpy()


# ============================================================
# EVALUATION RUNNER
# ============================================================

def evaluate_method(
    method_name: str,
    predictor,
    X_test_seqs: np.ndarray,
    y_test: np.ndarray,
    scenarios: Dict[str, List],
    injector: DegradationInjector,
    sigma_baseline: np.ndarray,
    device: str,
    window: int = 30,
) -> dict:
    """
    Evaluate a single method across all scenarios.
    
    Args:
        method_name: identifier for logging
        predictor: object with .predict(x_tensor) method
        X_test_seqs: (N, window, n_sensors) — all test sequences in order
        y_test: (N,) — true RUL values
        scenarios: dict of scenario_name → list of DegradationScenario
        injector: DegradationInjector instance
        sigma_baseline: (n_sensors,) baseline noise std
        device: 'cpu' or 'cuda'
        window: sequence window length
    
    Returns:
        results dict with per-scenario metrics
    """
    results = {}
    n_total = len(X_test_seqs)
    
    # B0: Clean performance (no degradation)
    X_clean_tensor = torch.tensor(X_test_seqs, dtype=torch.float32, device=device)
    preds_clean = []
    with torch.no_grad():
        for i in range(0, n_total, 64):
            batch = X_clean_tensor[i:i+64]
            if hasattr(predictor, 'model'):
                rul, _, _, _ = predictor.model(batch)
                preds_clean.append(np.atleast_1d(rul.squeeze().cpu().numpy()))
            elif hasattr(predictor, 'predict'):
                pred = predictor.predict(batch)
                preds_clean.append(np.atleast_1d(pred))
    
    preds_clean = np.concatenate(preds_clean)
    results['clean'] = compute_metrics(preds_clean, y_test[:len(preds_clean)])
    
    # Evaluate under each degradation scenario
    for scenario_name, scenario_list in scenarios.items():
        preds_degraded = []
        for i, x_seq in enumerate(X_test_seqs):
            x_deg, _ = injector.apply(
                x_seq, scenario_list, t=i, sigma_baseline=sigma_baseline)
            x_tensor = torch.tensor(x_deg[None], dtype=torch.float32, device=device)
            if hasattr(predictor, 'predict'):
                pred = predictor.predict(x_tensor)
            else:
                pred = predictor(x_tensor)
            preds_degraded.append(float(np.atleast_1d(pred).mean()))
        
        preds_degraded = np.array(preds_degraded, dtype=np.float32)
        valid_len = min(len(preds_degraded), len(y_test))
        results[scenario_name] = compute_metrics(
            preds_degraded[:valid_len], y_test[:valid_len])
    
    print(f"  [{method_name}] Clean RMSE: {results['clean']['rmse']:.2f}")
    return results


def run_evaluation(args):
    """Main evaluation function."""
    set_seed(args.seed)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"[Evaluation] Device: {device}, Subset: {args.subset}, Seed: {args.seed}")
    
    # Load checkpoint
    checkpoint = torch.load(args.checkpoint, map_location=device)
    n_sensors = checkpoint['n_sensors']
    time_steps = checkpoint['time_steps']
    latent_dim = checkpoint['latent_dim']
    adapter_dim = checkpoint['adapter_dim']
    sensor_cols = checkpoint['sensor_cols']
    
    # Load data
    data = load_and_preprocess(args.data_dir, args.subset)
    
    X_test_seqs, _ = create_sequences(
        data['test_df'], sensor_cols, window=time_steps, step=1)
    
    # True test RUL: last sequence per unit corresponds to test RUL
    test_rul = data['test_rul']['RUL'].values.astype(np.float32)
    
    # For simplicity in evaluation: use one sequence per unit (last window)
    # This matches standard C-MAPSS evaluation protocol
    test_df = data['test_df']
    X_eval_list, y_eval_list = [], []
    for unit_id, group in test_df.groupby('unit_id'):
        group = group.sort_values('time_cycle')
        sensors = group[sensor_cols].values.astype(np.float32)
        if len(sensors) >= time_steps:
            X_eval_list.append(sensors[-time_steps:])
            unit_idx = int(unit_id) - 1
            if unit_idx < len(test_rul):
                y_eval_list.append(test_rul[unit_idx])
    
    X_eval = np.stack(X_eval_list, axis=0)  # (n_units, time_steps, n_sensors)
    y_eval = np.array(y_eval_list, dtype=np.float32)
    
    print(f"  Evaluation set: {len(X_eval)} units, {n_sensors} sensors")
    
    # Baseline normalization for health estimation
    train_df = data['train_df']
    X_train_np, _ = create_sequences(data['train_df'], sensor_cols, window=time_steps)
    sigma_baseline = np.std(X_train_np[:, -1, :], axis=0) + 1e-6
    x_baseline_for_init = X_train_np[:100, -1, :]  # Last timestep of first 100 training windows
    
    # Load models
    backbone = SHELLModel(n_sensors=n_sensors, time_steps=time_steps,
                           sensor_latent_dim=latent_dim, adapter_dim=adapter_dim).to(device)
    backbone.load_state_dict(checkpoint['model_state_dict'])
    backbone.freeze_backbone_and_head()
    
    recon_path = args.checkpoint.replace('backbone', 'reconstructor')
    reconstructor = CrossModalReconstructor(n_sensors=n_sensors, latent_dim=latent_dim).to(device)
    if os.path.exists(recon_path):
        reconstructor.load_state_dict(torch.load(recon_path, map_location=device))
        print(f"  Loaded reconstructor from {recon_path}")
    else:
        print(f"  [WARNING] No reconstructor found at {recon_path}. Using random init.")
    
    # Create degradation injector and scenarios
    injector = DegradationInjector(n_sensors=n_sensors, seed=args.seed)
    scenarios = injector.create_standard_scenarios(severity=args.severity)
    
    print(f"  Evaluation scenarios: {list(scenarios.keys())}")
    
    # Initialize methods
    methods = {}
    
    # B0: Static (clean upper bound)
    methods['B0_static_clean'] = StaticBaseline(deepcopy(backbone), device)
    
    # B1: Static Degraded (lower bound reference)
    methods['B1_static_degraded'] = StaticBaseline(deepcopy(backbone), device)
    
    # B2: Tent (Entropy Minimization TTA)
    methods['B2_tent'] = TentBaseline(deepcopy(backbone), lr=1e-4, device=device)
    
    # B3: CoTTA (Continual TTA)
    methods['B3_cotta'] = CoTTABaseline(deepcopy(backbone), lr=1e-4, device=device)
    
    # B4: Reliability reweight only
    b4 = ReliabilityReweightBaseline(deepcopy(backbone), device)
    b4.initialize(x_baseline_for_init)
    methods['B4_reweight_only'] = b4
    
    # B5: EWC
    b5 = EWCBaseline(deepcopy(backbone), lr=1e-4, ewc_lambda=0.5, device=device)
    methods['B5_ewc'] = b5
    
    # B9: SHELL (full)
    shell_cfg = SHELLConfig()
    shell_model = deepcopy(backbone)
    shell_controller = SHELLController(shell_model, reconstructor, shell_cfg, n_sensors, device)
    shell_controller.initialize(x_baseline_for_init)
    
    def shell_predict(x):
        rul, action = shell_controller.step(x)
        return rul.squeeze().detach().cpu().numpy()
    
    class MethodWrapper:
        def predict(self, x):
            return shell_predict(x)
    
    methods['B9_SHELL_full'] = MethodWrapper()
    
    # SHELL ablations
    # B6: SHELL-noHealth (uniform weights)
    class SHELLNoHealth:
        def __init__(self):
            self.model = deepcopy(backbone)
            self.model.eval()
        def predict(self, x):
            with torch.no_grad():
                # Uniform weights — no health estimation
                rul, _, _, _ = self.model(x)
            return rul.squeeze().cpu().numpy()
    methods['B6_SHELL_noHealth'] = SHELLNoHealth()
    
    # B7: SHELL-noStability (no bounded-update projection)
    shell_cfg_nostab = SHELLConfig()
    shell_cfg_nostab.delta_max_base = float('inf')  # Disable projection
    shell_model_ns = deepcopy(backbone)
    shell_ctrl_ns = SHELLController(shell_model_ns, reconstructor, shell_cfg_nostab, n_sensors, device)
    shell_ctrl_ns.initialize(x_baseline_for_init)
    
    class SHELLNoStab:
        def predict(self, x):
            rul, _ = shell_ctrl_ns.step(x)
            return rul.squeeze().detach().cpu().numpy()
    methods['B7_SHELL_noStability'] = SHELLNoStab()
    
    # B8: SHELL-noRecon
    shell_cfg_norecon = SHELLConfig()
    shell_cfg_norecon.tau_reconstruct = 2.0  # Never triggers reconstruction
    shell_model_nr = deepcopy(backbone)
    shell_ctrl_nr = SHELLController(shell_model_nr, None, shell_cfg_norecon, n_sensors, device)
    shell_ctrl_nr.initialize(x_baseline_for_init)
    
    class SHELLNoRecon:
        def predict(self, x):
            rul, _ = shell_ctrl_nr.step(x)
            return rul.squeeze().detach().cpu().numpy()
    methods['B8_SHELL_noRecon'] = SHELLNoRecon()
    
    # Run evaluation
    all_results = {}
    
    for method_name, predictor in methods.items():
        print(f"\n[Evaluating] {method_name}")
        
        # Reset any stateful methods for each fresh evaluation
        t_start = time.perf_counter()
        
        results = evaluate_method(
            method_name=method_name,
            predictor=predictor,
            X_test_seqs=X_eval,
            y_test=y_eval,
            scenarios=scenarios,
            injector=injector,
            sigma_baseline=sigma_baseline,
            device=device,
            window=time_steps,
        )
        
        elapsed = time.perf_counter() - t_start
        results['eval_time_s'] = elapsed
        
        # Include SHELL action counts if available
        if method_name == 'B9_SHELL_full':
            results['action_counts'] = shell_controller.get_action_summary()
            results['mean_health'] = shell_controller.get_mean_health()
        
        all_results[method_name] = results
    
    # Save results
    os.makedirs(args.output_dir, exist_ok=True)
    results_path = os.path.join(
        args.output_dir,
        f'results_{args.subset}_seed{args.seed}_{args.severity}.json'
    )
    with open(results_path, 'w') as f:
        json.dump(all_results, f, indent=2)
    
    print(f"\n[Results Summary] Subset={args.subset}, Severity={args.severity}")
    print(f"{'Method':<30} {'Clean RMSE':>12} {'noise_single_s1 RMSE':>22}")
    print("-" * 65)
    
    for method_name, results in all_results.items():
        clean_rmse = results.get('clean', {}).get('rmse', float('nan'))
        noise_rmse = results.get('noise_single_s1', {}).get('rmse', float('nan'))
        print(f"{method_name:<30} {clean_rmse:>12.2f} {noise_rmse:>22.2f}")
    
    print(f"\nFull results saved to: {results_path}")
    return all_results


def main():
    parser = argparse.ArgumentParser(description='Evaluate SHELL and baselines')
    parser.add_argument('--checkpoint', type=str,
                       default='./checkpoints/shell_backbone_FD001_seed42.pt')
    parser.add_argument('--data_dir', type=str, default='./data/raw')
    parser.add_argument('--subset', type=str, default='FD001')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--severity', type=str, default='medium',
                       choices=['low', 'medium', 'high'])
    parser.add_argument('--output_dir', type=str, default='./results')
    args = parser.parse_args()
    
    run_evaluation(args)


if __name__ == '__main__':
    main()
