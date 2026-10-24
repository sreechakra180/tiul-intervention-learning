"""
Train and Calibrate the Action Risk Estimator for CIG
=====================================================
- Trains on Units 1-80 action risk dataset
- Validates on held-out validation engines (Units 81-100)
- Computes calibration curves and Pearson correlation
- Saves checkpoint to implementation_v3_cig/checkpoints/action_risk_estimator.pt
- Saves metrics to audit/17_risk_estimator_calibration.json
"""

import os
import sys
import json
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
from risk_estimator import CalibratedActionRiskEstimator
from shell.degradation import DegradationInjector

def train_and_calibrate():
    data_path = base_dir / 'implementation_v3_cig' / 'data' / 'action_risk_dataset.npz'
    npz = np.load(data_path)
    X_train = npz['X']
    y_train = npz['y']
    
    # Generate validation dataset on Units 81-100
    data = load_data_partitioned(str(base_dir / 'implementation' / 'data' / 'raw'), train_units_max=80)
    X_val_seqs, y_val_seqs, _ = create_sequences_by_engine(data['val_df'], ACTIVE_SENSORS, window=30, step=8)
    
    health_estimator = TemporalHealthEstimator()
    health_estimator.fit_reference(X_val_seqs)
    
    ckpt_path = base_dir / 'implementation' / 'checkpoints' / 'shell_backbone_FD001_seed42.pt'
    ckpt = torch.load(ckpt_path, map_location='cpu')
    model = SHELLModel(n_sensors=14, time_steps=30, sensor_latent_dim=32, adapter_dim=16)
    model.load_state_dict(ckpt['model_state_dict'])
    model.eval()
    
    recon_path = base_dir / 'implementation' / 'checkpoints' / 'shell_reconstructor_FD001_seed42.pt'
    reconstructor = CrossModalReconstructor(n_sensors=14, latent_dim=32)
    if os.path.exists(recon_path):
        reconstructor.load_state_dict(torch.load(recon_path, map_location='cpu'))
        reconstructor.eval()
    else:
        reconstructor = None
        
    action_gen = CandidateActionGenerator(
        model=model,
        baseline_mu=health_estimator.baseline_mu,
        reconstructor=reconstructor
    )
    
    injector = DegradationInjector(n_sensors=14, seed=999)
    X_val, y_val = [], []
    
    for i in range(len(X_val_seqs)):
        x_clean = X_val_seqs[i].copy()
        y_true = float(y_val_seqs[i])
        
        # Inject corruptions on validation
        r = i % 4
        if r == 0:
            x_curr = x_clean
        elif r == 1:
            sc = injector.create_noise_scenario([0], severity=1.5, seed=i)
            x_curr, _ = injector.apply(x_clean, [sc], t=i, sigma_baseline=health_estimator.baseline_sigma)
        elif r == 2:
            sc = injector.create_dropout_scenario([5])
            sc.deg_type = sc.deg_type.__class__['MODALITY_DROPOUT']
            x_curr, _ = injector.apply(x_clean, [sc], t=i, sigma_baseline=health_estimator.baseline_sigma)
        else:
            sc = injector.create_drift_scenario([2], drift_rate=0.03)
            x_curr, _ = injector.apply(x_clean, [sc], t=i, sigma_baseline=health_estimator.baseline_sigma)
            
        health = health_estimator.evaluate_window(x_curr)
        x_t = torch.tensor(x_curr[None], dtype=torch.float32)
        cand_preds = action_gen.generate_candidate_predictions(x_t, health)
        pred_0 = cand_preds[CandidateAction.NO_OP]
        
        for act in CandidateAction:
            pred_a = cand_preds[act]
            true_risk = abs(pred_a - y_true)
            feat = action_gen.extract_action_feature_vector(
                x_window=x_curr,
                health_scores=health,
                baseline_sigma=health_estimator.baseline_sigma,
                action=act,
                pred_a=pred_a,
                pred_0=pred_0
            )
            X_val.append(feat)
            y_val.append(true_risk)
            
    X_val = np.stack(X_val, axis=0)
    y_val = np.array(y_val, dtype=np.float32)
    
    print(f"[Training Risk Estimator] Train samples: {X_train.shape[0]}, Val samples: {X_val.shape[0]}")
    
    estimator = CalibratedActionRiskEstimator(in_features=X_train.shape[1], hidden_dim=64)
    metrics = estimator.fit(X_train, y_train, X_val, y_val, epochs=40, lr=1e-3)
    
    print(f"  Training Loss: {metrics['train_loss']:.4f}")
    print(f"  Validation MAE: {metrics['val_mae']:.2f} cycles")
    print(f"  Validation RMSE: {metrics['val_rmse']:.2f} cycles")
    print(f"  Pearson Correlation: {metrics['pearson_corr']:.4f}")
    
    # Save checkpoint
    ckpt_dir = base_dir / 'implementation_v3_cig' / 'checkpoints'
    os.makedirs(ckpt_dir, exist_ok=True)
    model_save_path = ckpt_dir / 'action_risk_estimator.pt'
    estimator.save(str(model_save_path))
    
    # Save calibration report
    calib_report = {
        'train_samples': int(X_train.shape[0]),
        'val_samples': int(X_val.shape[0]),
        'features_dim': int(X_train.shape[1]),
        'final_train_loss': float(metrics['train_loss']),
        'val_mae': float(metrics['val_mae']),
        'val_rmse': float(metrics['val_rmse']),
        'pearson_correlation': float(metrics['pearson_corr']),
        'model_path': str(model_save_path),
        'mean_true_val_risk': float(np.mean(y_val)),
        'std_true_val_risk': float(np.std(y_val))
    }
    
    report_path = base_dir / 'audit' / '17_risk_estimator_calibration.json'
    with open(report_path, 'w') as f:
        json.dump(calib_report, f, indent=2)
        
    print(f"[Complete] Model saved to {model_save_path}")
    print(f"[Complete] Calibration metrics saved to {report_path}")

if __name__ == '__main__':
    train_and_calibrate()
