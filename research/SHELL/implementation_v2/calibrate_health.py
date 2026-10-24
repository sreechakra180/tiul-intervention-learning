"""
Calibration and Validation of Causal Temporal Health Estimator (V2.2)
====================================================================
Evaluates health estimation on held-out validation engines (Units 81-100):
- Clean sequences: false positive rate (FPR) and mean health.
- Injected Gaussian noise: precision, recall, F1.
- Injected sensor dropout: precision, recall, F1.
- Injected sensor drift: precision, recall, F1.
Saves results to audit/10_health_calibration_results.csv
"""

import sys
import csv
import numpy as np
from pathlib import Path

base_dir = Path(__file__).parent.parent
sys.path.insert(0, str(base_dir / 'implementation_v2'))
sys.path.insert(0, str(base_dir / 'implementation'))

from data_v2 import load_data_partitioned, create_sequences_by_engine, ACTIVE_SENSORS
from health_estimator_v2 import TemporalHealthEstimator, HealthConfig
from shell.degradation import DegradationInjector

def run_calibration():
    data = load_data_partitioned(str(base_dir / 'implementation' / 'data' / 'raw'), train_units_max=80)
    
    # Fit reference on train sequences
    X_train_seqs, _, _ = create_sequences_by_engine(data['train_df'], ACTIVE_SENSORS, window=30, step=5)
    estimator = TemporalHealthEstimator()
    estimator.fit_reference(X_train_seqs)
    
    # Extract validation sequences (Units 81-100)
    X_val_seqs, _, _ = create_sequences_by_engine(data['val_df'], ACTIVE_SENSORS, window=30, step=5)
    n_val = len(X_val_seqs)
    n_sensors = 14
    
    injector = DegradationInjector(n_sensors=14, seed=42)
    
    scenarios_to_test = [
        ('clean', 'none', -1),
        ('noise_s0_sev1.5', 'gaussian_noise', 0),
        ('noise_s3_sev1.5', 'gaussian_noise', 3),
        ('dropout_s5', 'modality_dropout', 5),
        ('dropout_s8', 'modality_dropout', 8),
        ('drift_s2', 'bias_drift', 2),
    ]
    
    csv_rows = []
    print(f"[Health Calibration] Testing on {n_val} validation sequences across {len(scenarios_to_test)} conditions...")
    
    for sc_name, sc_type, target_sensor in scenarios_to_test:
        tp, fp, tn, fn = 0, 0, 0, 0
        health_means = []
        target_healths = []
        non_target_healths = []
        
        for i in range(n_val):
            x_seq = X_val_seqs[i].copy()
            
            # Apply degradation
            if sc_type == 'gaussian_noise':
                sc = injector.create_noise_scenario([target_sensor], severity=1.5, seed=i)
                x_deg, _ = injector.apply(x_seq, [sc], t=i, sigma_baseline=estimator.baseline_sigma)
            elif sc_type == 'modality_dropout':
                sc = injector.create_dropout_scenario([target_sensor])
                sc.deg_type = injector.create_noise_scenario([0]).deg_type.__class__['MODALITY_DROPOUT']
                x_deg, _ = injector.apply(x_seq, [sc], t=i, sigma_baseline=estimator.baseline_sigma)
            elif sc_type == 'bias_drift':
                sc = injector.create_drift_scenario([target_sensor], drift_rate=0.05)
                x_deg, _ = injector.apply(x_seq, [sc], t=i, sigma_baseline=estimator.baseline_sigma)
            else:
                x_deg = x_seq
                
            health = estimator.evaluate_window(x_deg)
            health_means.append(float(health.mean()))
            
            # Classification threshold: sensor is "flagged" if health < 0.70
            flagged = (health < 0.70)
            
            for s in range(n_sensors):
                is_actually_degraded = (s == target_sensor)
                if flagged[s]:
                    if is_actually_degraded:
                        tp += 1
                    else:
                        fp += 1
                else:
                    if is_actually_degraded:
                        fn += 1
                    else:
                        tn += 1
                        
            if target_sensor >= 0:
                target_healths.append(float(health[target_sensor]))
                other_idx = [s for s in range(n_sensors) if s != target_sensor]
                non_target_healths.extend(health[other_idx].tolist())
            else:
                non_target_healths.extend(health.tolist())
                
        precision = tp / (tp + fp) if (tp + fp) > 0 else (1.0 if target_sensor == -1 else 0.0)
        recall = tp / (tp + fn) if (tp + fn) > 0 else (1.0 if target_sensor == -1 else 0.0)
        fpr = fp / (fp + tn) if (fp + tn) > 0 else 0.0
        f1 = (2 * precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0
        
        target_h_mean = float(np.mean(target_healths)) if target_healths else 1.0
        non_target_h_mean = float(np.mean(non_target_healths))
        
        print(f"  {sc_name:<16} | FPR: {fpr*100:5.2f}% | Precision: {precision:5.2f} | Recall: {recall:5.2f} | F1: {f1:5.2f} | Target Health: {target_h_mean:.3f}")
        
        csv_rows.append({
            'scenario': sc_name,
            'degradation_type': sc_type,
            'target_sensor': target_sensor if target_sensor >= 0 else 'None',
            'n_sequences': n_val,
            'false_positive_rate': f"{fpr:.4f}",
            'precision': f"{precision:.4f}",
            'recall': f"{recall:.4f}",
            'f1_score': f"{f1:.4f}",
            'mean_overall_health': f"{np.mean(health_means):.4f}",
            'target_sensor_mean_health': f"{target_h_mean:.4f}",
            'healthy_sensors_mean_health': f"{non_target_h_mean:.4f}"
        })
        
    out_csv = base_dir / 'audit' / '10_health_calibration_results.csv'
    fieldnames = [
        'scenario', 'degradation_type', 'target_sensor', 'n_sequences',
        'false_positive_rate', 'precision', 'recall', 'f1_score',
        'mean_overall_health', 'target_sensor_mean_health', 'healthy_sensors_mean_health'
    ]
    with open(out_csv, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(csv_rows)
        
    print(f"\n[Calibration Complete] Results written to {out_csv}")

if __name__ == '__main__':
    run_calibration()
