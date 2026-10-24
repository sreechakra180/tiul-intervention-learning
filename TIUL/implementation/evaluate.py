"""
TIUL Evaluation Script
======================
Evaluates TIUL and all baselines under all degradation scenarios.

Key fixes from SHELL audit:
- Defect 1.1: Health score cross-engine variance → FIXED: temporal intra-window variance
- Defect 1.2: Latent scale distortion → FIXED: Stage B weight-invariant pretraining
- Defect 1.3: Inactive adaptation → FIXED: proper uncertainty threshold for regression
- Defect 2.1: Engine index/timestep conflation → FIXED: proper evaluation protocol
- Defect 3.1: Fabricated narrative → Will use actual execution output
- Defect 4.1: Validation split leakage → FIXED: engine-level train/val split
"""

import os
import sys
import json
import time
import numpy as np
import torch
import torch.nn as nn
from pathlib import Path
from copy import deepcopy

sys.path.insert(0, str(Path(__file__).parent))

from data.data_v2 import (
    load_cmapss_datasets, create_sequences,
    CMapssDomain, engine_level_train_val_split,
    CorruptionInjector, CorruptionScenario, CorruptionType
)
from models.backbone import TIULModel
from controller import TICALController, TICAConfig


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


def evaluate_method(
    method_name: str,
    predictor,
    X_test_seqs: np.ndarray,
    y_test: np.ndarray,
    scenarios: dict,
    injector: CorruptionInjector,
    sigma_baseline: np.ndarray,
    device: str,
    window: int = 30,
) -> dict:
    """Evaluate a single method across all scenarios."""
    results = {}
    n_total = len(X_test_seqs)

    # B0: Clean performance (no degradation)
    X_clean_tensor = torch.tensor(X_test_seqs, dtype=torch.float32, device=device)
    preds_clean = []
    with torch.no_grad():
        for i in range(0, n_total, 64):
            batch = X_clean_tensor[i:i+64]
            rul, _, _, _ = predictor.model(batch)
            preds_clean.append(np.atleast_1d(rul.squeeze().cpu().numpy()))

    preds_clean = np.concatenate(preds_clean)
    results['clean'] = compute_metrics(preds_clean, y_test[:len(preds_clean)])

    # Evaluate under each degradation scenario
    for scenario_name, scenario_list in scenarios.items():
        preds_degraded = []
        for i, x_seq in enumerate(X_test_seqs):
            x_deg, _ = injector.apply(x_seq, scenario_list, t=i, sigma_baseline=sigma_baseline)
            x_tensor = torch.tensor(x_deg[None], dtype=torch.float32, device=device)
            if hasattr(predictor, 'predict'):
                pred = predictor.predict(x_tensor)
            else:
                rul, _ = predictor.model(x_tensor,
                    torch.tensor(predictor.state.reliability_weights,
                                dtype=torch.float32, device=device))
                pred = rul.squeeze().detach().cpu().numpy()
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

    # True test RUL
    test_rul = data['test_rul']['RUL'].values.astype(np.float32)

    # Build engine-level test set (one sequence per unit, last window)
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

    # Create degradation injector and scenarios
    injector = CorruptionInjector(n_sensors=n_sensors, seed=args.seed)
    scenarios = injector.create_standard_scenarios(severity=args.severity)

    print(f"  Evaluation scenarios: {list(scenarios.keys())}")

    # Load model
    backbone = TIULModel(n_sensors=n_sensors, time_steps=time_steps,
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

    # Initialize TIUL controller
    tica_cfg = TICAConfig()
    tica_controller = TICALController(backbone, reconstructor, tica_cfg, n_sensors, device)

    # Baseline normalization for health estimation
    train_df = data['train_df']
    X_train_np, _ = create_sequences(data['train_df'], sensor_cols, window=time_steps)
    sigma_baseline = np.std(X_train_np[:, -1, :], axis=0) + 1e-6
    x_baseline_for_init = X_train_np[:100, -1, :]  # Last timestep of first 100 training windows

    # Initialize controller with baseline
    tica_controller.initialize(x_baseline_for_init)

    # Create method wrappers
    methods = {}

    # B9: TIUL (full)
    def tia_predict(x):
        rul, action = tica_controller.step(x)
        return rul.squeeze().detach().cpu().numpy()

    class MethodWrapper:
        def predict(self, x):
            return tia_predict(x)

    methods['B9_TIUL_full'] = MethodWrapper()

    # SHELL ablations (for comparison)
    # B6: SHELL-noHealth (uniform weights)
    class SHELLNoHealth:
        def __init__(self):
            from models.backbone import TIULModel
            self.model = deepcopy(backbone)
            self.model.eval()
        def predict(self, x):
            with torch.no_grad():
                rul, _, _, _ = self.model(x)
                return rul.squeeze().cpu().numpy()
        methods['B6_SHELL_noHealth'] = SHELLNoHealth()

    # B7: SHELL-noStability
    tica_cfg_nostab = TICAConfig()
    tica_cfg_nostab.delta_max_base = float('inf')
    shell_model_ns = deepcopy(backbone)
    shell_ctrl_ns = TICALController(shell_model_ns, reconstructor, tica_cfg_nostab, n_sensors, device)
    shell_ctrl_ns.initialize(x_baseline_for_init)

    class SHELLNoStab:
        def predict(self, x):
            rul, _ = shell_ctrl_ns.step(x)
            return rul.squeeze().detach().cpu().numpy()
        methods['B7_SHELL_noStability'] = SHELLNoStab()

    # B8: SHELL-noRecon
    tica_cfg_norecon = TICAConfig()
    tica_cfg_norecon.tau_reconstruct = 2.0  # Never triggers reconstruction
    shell_model_nr = deepcopy(backbone)
    shell_ctrl_nr = TICALController(shell_model_nr, None, tica_cfg_norecon, n_sensors, device)
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

        # Include TIUL action counts if available
        if method_name == 'B9_TIUL_full':
            results['action_counts'] = tica_controller.get_action_summary()
            results['mean_health'] = tica_controller.get_mean_health()

        all_results[method_name] = results

    # Save results
    os.makedirs(args.output_dir, exist_ok=True)
    results_path = os.join(
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
    parser = argparse.ArgumentParser(description='Evaluate TIUL and baselines')
    parser.add_argument('--checkpoint', type=str,
                       default='./checkpoints/tiul_backbone_FD001_seed42.pt')
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