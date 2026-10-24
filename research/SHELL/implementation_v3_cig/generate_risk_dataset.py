"""
Offline Action-Training Dataset Generator for Action Risk Estimator
===================================================================
Uses training engines only (Units 1-80):
- Injects synthetic corruptions (clean, noise, dropout, drift)
- Computes causal health scores
- Evaluates candidate actions in shadow mode on the frozen model
- Calculates actual RUL error R*(a) = |pred(a) - y_true| for each candidate
- Saves feature-target dataset to implementation_v3_cig/data/action_risk_dataset.npz
"""

import os
import sys
import numpy as np
import torch
from pathlib import Path

base_dir = Path(__file__).parent.parent
sys.path.insert(0, str(base_dir / 'implementation'))
sys.path.insert(0, str(base_dir / 'implementation_v2'))
sys.path.insert(0, str(base_dir / 'implementation_v3_cig'))

from data_v2 import load_data_partitioned, create_sequences_by_engine, ACTIVE_SENSORS
from models.backbone import SHELLModel, CrossModalReconstructor
from health_estimator_v2 import TemporalHealthEstimator
from action_generator import CandidateActionGenerator, CandidateAction
from shell.degradation import DegradationInjector

def generate_action_risk_dataset(subsample_step: int = 4, seed: int = 42):
    np.random.seed(seed)
    torch.manual_seed(seed)
    device = 'cpu'
    
    # Load disjoint train data (Units 1-80)
    data = load_data_partitioned(str(base_dir / 'implementation' / 'data' / 'raw'), train_units_max=80)
    
    # 1. Fit reference health estimator on training sequences
    X_train_raw, y_train_raw, _ = create_sequences_by_engine(data['train_df'], ACTIVE_SENSORS, window=30, step=subsample_step)
    
    health_estimator = TemporalHealthEstimator()
    health_estimator.fit_reference(X_train_raw)
    
    # 2. Load frozen model
    ckpt_path = base_dir / 'implementation' / 'checkpoints' / 'shell_backbone_FD001_seed42.pt'
    ckpt = torch.load(ckpt_path, map_location=device)
    model = SHELLModel(n_sensors=14, time_steps=30, sensor_latent_dim=32, adapter_dim=16).to(device)
    model.load_state_dict(ckpt['model_state_dict'])
    model.eval()
    
    # 3. Load reconstructor
    recon_path = base_dir / 'implementation' / 'checkpoints' / 'shell_reconstructor_FD001_seed42.pt'
    reconstructor = CrossModalReconstructor(n_sensors=14, latent_dim=32).to(device)
    if os.path.exists(recon_path):
        reconstructor.load_state_dict(torch.load(recon_path, map_location=device))
        reconstructor.eval()
    else:
        reconstructor = None
        
    action_gen = CandidateActionGenerator(
        model=model,
        baseline_mu=health_estimator.baseline_mu,
        reconstructor=reconstructor,
        device=device
    )
    
    injector = DegradationInjector(n_sensors=14, seed=seed)
    n_samples = len(X_train_raw)
    n_sensors = 14
    
    print(f"[CIG Dataset Generation] Processing {n_samples} training windows from Units 1-80...")
    
    X_features = []
    y_risks = []
    action_indices = []
    meta_info = []
    
    for i in range(n_samples):
        x_clean = X_train_raw[i].copy()
        y_true = float(y_train_raw[i])
        
        # Stochastic corruption regime
        r = np.random.rand()
        if r < 0.25:
            # Clean
            x_curr = x_clean
            corr_type = 'clean'
        elif r < 0.50:
            # Gaussian noise on 1 or 2 channels
            n_noisy = np.random.randint(1, 3)
            noisy_idx = np.random.choice(n_sensors, n_noisy, replace=False).tolist()
            sc = injector.create_noise_scenario(noisy_idx, severity=float(np.random.uniform(1.0, 2.5)), seed=i)
            x_curr, _ = injector.apply(x_clean, [sc], t=i, sigma_baseline=health_estimator.baseline_sigma)
            corr_type = 'noise'
        elif r < 0.75:
            # Dropout on 1 or 2 channels
            n_drop = np.random.randint(1, 3)
            drop_idx = np.random.choice(n_sensors, n_drop, replace=False).tolist()
            sc = injector.create_dropout_scenario(drop_idx)
            sc.deg_type = sc.deg_type.__class__['MODALITY_DROPOUT']
            x_curr, _ = injector.apply(x_clean, [sc], t=i, sigma_baseline=health_estimator.baseline_sigma)
            corr_type = 'dropout'
        else:
            # Bias drift on 1 channel
            drift_idx = [np.random.choice(n_sensors, 1)[0]]
            sc = injector.create_drift_scenario(drift_idx, drift_rate=float(np.random.uniform(0.02, 0.06)))
            x_curr, _ = injector.apply(x_clean, [sc], t=i, sigma_baseline=health_estimator.baseline_sigma)
            corr_type = 'drift'
            
        # Compute health scores
        health = health_estimator.evaluate_window(x_curr)
        x_tensor = torch.tensor(x_curr[None], dtype=torch.float32, device=device)
        
        # Shadow candidate evaluation
        cand_preds = action_gen.generate_candidate_predictions(x_tensor, health)
        pred_0 = cand_preds[CandidateAction.NO_OP]
        
        # Generate feature and risk target for each of the 5 actions
        for act in CandidateAction:
            pred_a = cand_preds[act]
            true_risk = abs(pred_a - y_true) # Actual absolute error in cycles
            
            feat_vec = action_gen.extract_action_feature_vector(
                x_window=x_curr,
                health_scores=health,
                baseline_sigma=health_estimator.baseline_sigma,
                action=act,
                pred_a=pred_a,
                pred_0=pred_0
            )
            
            X_features.append(feat_vec)
            y_risks.append(true_risk)
            action_indices.append(int(act))
            
        if (i + 1) % 1000 == 0:
            print(f"  Processed {i+1}/{n_samples} samples...")
            
    X_features = np.stack(X_features, axis=0) # (N * 5, d)
    y_risks = np.array(y_risks, dtype=np.float32)  # (N * 5,)
    action_indices = np.array(action_indices, dtype=np.int32)
    
    out_dir = base_dir / 'implementation_v3_cig' / 'data'
    os.makedirs(out_dir, exist_ok=True)
    out_path = out_dir / 'action_risk_dataset.npz'
    
    np.savez_compressed(
        out_path,
        X=X_features,
        y=y_risks,
        actions=action_indices,
        baseline_sigma=health_estimator.baseline_sigma,
        baseline_mu=health_estimator.baseline_mu
    )
    
    print(f"\n[CIG Dataset Saved] {X_features.shape[0]} total action-evaluation samples.")
    print(f"Feature dimension: {X_features.shape[1]}")
    print(f"Mean risk across all actions: {np.mean(y_risks):.2f} cycles")
    print(f"Dataset saved to: {out_path}")

if __name__ == '__main__':
    generate_action_risk_dataset(subsample_step=4)
