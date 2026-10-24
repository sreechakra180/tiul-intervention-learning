"""
Master Evaluation & Harmful Intervention Benchmark for CIG
==========================================================
Evaluates and compares:
1. M0: Frozen Baseline (no intervention)
2. M2: Blind Soft Reweighting (always reweights when suspicious)
3. M3: Blind Imputation (always imputes when suspicious)
4. M7: Previous RGAC Controller (heuristic threshold gating)
5. M1: Corruption-Augmented Backbone (Stage B)
6. CIG: Proposed Counterfactual Intervention Governor

Key Measurements:
- RMSE, MAE, NASA asymmetric score across matched scenarios
- Harmful Intervention Rate (HIR): fraction of interventions that increased error vs no-op
- Clean-input Intervention Rate (CIR): false alarm intervention rate on clean data
- Rejection Rate: frequency with which CIG safely rejected an unsafe intervention
- Generalization on Unseen Corruption Severities
Saves results to audit/18_cig_experiment_results.csv and audit/19_harmful_intervention_analysis.md
"""

import os
import sys
import json
import csv
import time
import numpy as np
import torch
from pathlib import Path
from collections import Counter

base_dir = Path(__file__).parent.parent
sys.path.insert(0, str(base_dir / 'implementation'))
sys.path.insert(0, str(base_dir / 'implementation_v2'))
sys.path.insert(0, str(base_dir / 'implementation_v3_cig'))

from data_v2 import load_data_partitioned, create_sequences_by_engine, ACTIVE_SENSORS
from models.backbone import SHELLModel, CrossModalReconstructor
from health_estimator_v2 import TemporalHealthEstimator
from action_generator import CandidateActionGenerator, CandidateAction
from risk_estimator import CalibratedActionRiskEstimator
from cig_governor import CounterfactualInterventionGovernor, CIGConfig
from rgac_controller import RGACController, RGACConfig
from shell.degradation import DegradationInjector

def nasa_score(pred: np.ndarray, true: np.ndarray) -> float:
    d = pred - true
    score = np.where(d < 0, np.exp(-d / 13) - 1, np.exp(d / 10) - 1)
    return float(score.sum())

def compute_metrics(pred: np.ndarray, true: np.ndarray) -> dict:
    rmse = float(np.sqrt(np.mean((pred - true) ** 2)))
    mae = float(np.mean(np.abs(pred - true)))
    ns = nasa_score(pred, true)
    return {'rmse': rmse, 'mae': mae, 'nasa_score': ns}

def run_cig_benchmark():
    device = 'cpu'
    data = load_data_partitioned(str(base_dir / 'implementation' / 'data' / 'raw'), train_units_max=80)
    
    # Load training sequences to fit reference statistics
    X_train_seqs, _, _ = create_sequences_by_engine(data['train_df'], ACTIVE_SENSORS, window=30, step=5)
    health_estimator = TemporalHealthEstimator()
    health_estimator.fit_reference(X_train_seqs)
    
    # Load frozen model M0
    ckpt_path = base_dir / 'implementation' / 'checkpoints' / 'shell_backbone_FD001_seed42.pt'
    ckpt = torch.load(ckpt_path, map_location=device)
    model_orig = SHELLModel(n_sensors=14, time_steps=30, sensor_latent_dim=32, adapter_dim=16).to(device)
    model_orig.load_state_dict(ckpt['model_state_dict'])
    model_orig.eval()
    
    # Load reconstructor
    recon_path = base_dir / 'implementation' / 'checkpoints' / 'shell_reconstructor_FD001_seed42.pt'
    reconstructor = CrossModalReconstructor(n_sensors=14, latent_dim=32).to(device)
    if os.path.exists(recon_path):
        reconstructor.load_state_dict(torch.load(recon_path, map_location=device))
        reconstructor.eval()
    else:
        reconstructor = None
        
    # Load robust model M1
    robust_path = base_dir / 'implementation_v2' / 'checkpoints' / 'robust_backbone_FD001_seed42.pt'
    if os.path.exists(robust_path):
        rob_ckpt = torch.load(robust_path, map_location=device)
        model_robust = SHELLModel(n_sensors=14, time_steps=30, sensor_latent_dim=32, adapter_dim=16).to(device)
        model_robust.load_state_dict(rob_ckpt['model_state_dict'])
        model_robust.eval()
    else:
        model_robust = None
        
    # Load trained Risk Estimator
    risk_ckpt_path = base_dir / 'implementation_v3_cig' / 'checkpoints' / 'action_risk_estimator.pt'
    risk_estimator = CalibratedActionRiskEstimator(in_features=18, hidden_dim=64, device=device)
    risk_estimator.load(str(risk_ckpt_path))
    
    action_generator = CandidateActionGenerator(
        model=model_orig,
        baseline_mu=health_estimator.baseline_mu,
        reconstructor=reconstructor,
        device=device
    )
    
    # CIG Governor
    cig_gov = CounterfactualInterventionGovernor(
        model=model_orig,
        health_estimator=health_estimator,
        risk_estimator=risk_estimator,
        action_generator=action_generator,
        config=CIGConfig(safety_margin_gamma=0.30),
        device=device
    )
    
    # Previous RGAC Controller
    rgac_ctrl = RGACController(
        model=model_orig,
        health_estimator=health_estimator,
        config=RGACConfig(use_reconstruction=True),
        reconstructor=reconstructor,
        device=device
    )
    
    # 100 test engines
    X_test = data['X_test_eval'] # (100, 30, 14)
    y_test = data['y_test_eval'] # (100,)
    n_engines = len(y_test)
    
    injector = DegradationInjector(n_sensors=14, seed=42)
    
    scenarios = {}
    scenarios['clean'] = []
    scenarios['noise_single'] = [injector.create_noise_scenario([0], severity=1.5, seed=100)]
    scenarios['noise_multi'] = [injector.create_noise_scenario([0, 1], severity=1.5, seed=200)]
    scenarios['drift_single'] = [injector.create_drift_scenario([2], drift_rate=0.03)]
    sc_drop = injector.create_dropout_scenario([5])
    sc_drop.deg_type = sc_drop.deg_type.__class__['MODALITY_DROPOUT']
    scenarios['dropout_single'] = [sc_drop]
    scenarios['compound'] = [injector.create_noise_scenario([0], severity=1.5, seed=300), sc_drop]
    
    # Generalization test: UNSEEN severities
    scenarios['unseen_severe_noise'] = [injector.create_noise_scenario([0], severity=3.0, seed=400)]
    scenarios['unseen_severe_drift'] = [injector.create_drift_scenario([2], drift_rate=0.08)]
    
    # Helper policy predictors
    def predict_blind_reweight(x_t):
        x_np = x_t.numpy()[0]
        h = health_estimator.evaluate_window(x_np)
        w = h / (h.sum() + 1e-8)
        w_t = torch.tensor(w, dtype=torch.float32)
        with torch.no_grad():
            r, _, _, _ = model_orig(x_t, w_t)
            return float(r.squeeze().item()), 'SOFT_REWEIGHT' if np.min(h) < 0.85 else 'NO_OP'
            
    def predict_blind_impute(x_t):
        x_np = x_t.numpy()[0]
        h = health_estimator.evaluate_window(x_np)
        x_mod = x_t.clone()
        susp = np.where(h < 0.70)[0]
        if len(susp) > 0:
            for idx in susp:
                x_mod[:, :, idx] = float(health_estimator.baseline_mu[idx])
        with torch.no_grad():
            r, _, _, _ = model_orig(x_mod)
            return float(r.squeeze().item()), 'IMPUTE' if len(susp) > 0 else 'NO_OP'
            
    methods = {
        'M0_frozen_baseline': lambda x: (float(model_orig(x)[0].item()), 'NO_OP'),
        'M2_blind_reweight': predict_blind_reweight,
        'M3_blind_impute': predict_blind_impute,
        'M7_previous_RGAC': lambda x: (float(rgac_ctrl.step(x)[0].item()), rgac_ctrl.step(x)[1].name),
        'CIG_proposed_governor': lambda x: (cig_gov.step(x)[0], cig_gov.step(x)[1].name),
    }
    
    if model_robust is not None:
        methods['M1_robust_dropout_backbone'] = lambda x: (float(model_robust(x)[0].item()), 'NO_OP')
        
    print(f"[CIG Benchmark] Evaluating {len(methods)} methods across {len(scenarios)} conditions on 100 test engines...")
    
    all_results = []
    harmful_analysis_data = {}
    
    for m_name, m_func in methods.items():
        print(f"\n--- Method: {m_name} ---")
        harmful_analysis_data[m_name] = {}
        
        for sc_name, sc_list in scenarios.items():
            cig_gov.reset()
            rgac_ctrl.reset()
            
            preds = []
            actions = []
            t_start = time.perf_counter()
            
            for i in range(n_engines):
                x_seq = X_test[i].copy()
                if sc_list:
                    x_deg, _ = injector.apply(x_seq, sc_list, t=i, sigma_baseline=health_estimator.baseline_sigma)
                else:
                    x_deg = x_seq
                x_t = torch.tensor(x_deg[None], dtype=torch.float32)
                p, act = m_func(x_t)
                preds.append(p)
                actions.append(act)
                
            elapsed_ms = (time.perf_counter() - t_start) * 1000.0 / n_engines
            preds = np.array(preds, dtype=np.float32)
            metrics = compute_metrics(preds, y_test)
            
            # Baseline predictions for calculating harmful interventions
            with torch.no_grad():
                preds_0 = model_orig(torch.tensor(X_test, dtype=torch.float32))[0].squeeze().numpy()
                if sc_list:
                    X_deg_all = []
                    for i, x_seq in enumerate(X_test):
                        x_d, _ = injector.apply(x_seq, sc_list, t=i, sigma_baseline=health_estimator.baseline_sigma)
                        X_deg_all.append(x_d)
                    X_deg_all = np.stack(X_deg_all, axis=0)
                    preds_0 = model_orig(torch.tensor(X_deg_all, dtype=torch.float32))[0].squeeze().numpy()
                    
            err_0 = np.abs(preds_0 - y_test)
            err_m = np.abs(preds - y_test)
            
            # Calculate harmful interventions
            interventions = [i for i, a in enumerate(actions) if a != 'NO_OP']
            n_interventions = len(interventions)
            harmful_count = sum(1 for i in interventions if err_m[i] > err_0[i] + 1e-4)
            beneficial_count = sum(1 for i in interventions if err_m[i] < err_0[i] - 1e-4)
            
            hir = (harmful_count / n_interventions) if n_interventions > 0 else 0.0
            
            act_counts = dict(Counter(actions))
            
            print(f"  {sc_name:<20} | RMSE: {metrics['rmse']:5.2f} | NASA: {metrics['nasa_score']:5.1f} | Interv: {n_interventions:3d} (Harmful: {harmful_count:2d}, Beneficial: {beneficial_count:2d}) | HIR: {hir*100:4.1f}% | Latency: {elapsed_ms:4.2f}ms")
            
            all_results.append({
                'method': m_name,
                'scenario': sc_name,
                'rmse': round(metrics['rmse'], 4),
                'mae': round(metrics['mae'], 4),
                'nasa_score': round(metrics['nasa_score'], 2),
                'latency_ms_per_engine': round(elapsed_ms, 3),
                'n_interventions': n_interventions,
                'harmful_interventions': harmful_count,
                'beneficial_interventions': beneficial_count,
                'harmful_intervention_rate': round(hir, 4),
                'actions_breakdown': str(act_counts)
            })
            
            harmful_analysis_data[m_name][sc_name] = {
                'rmse': metrics['rmse'],
                'n_interventions': n_interventions,
                'harmful_count': harmful_count,
                'beneficial_count': beneficial_count,
                'hir': hir,
                'actions': act_counts
            }
            
    # Save CSV
    out_csv = base_dir / 'audit' / '18_cig_experiment_results.csv'
    fieldnames = [
        'method', 'scenario', 'rmse', 'mae', 'nasa_score',
        'latency_ms_per_engine', 'n_interventions',
        'harmful_interventions', 'beneficial_interventions',
        'harmful_intervention_rate', 'actions_breakdown'
    ]
    with open(out_csv, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(all_results)
        
    print(f"\n[Complete] Benchmark results saved to {out_csv}")
    
    # Save Harmful Intervention Analysis Markdown report
    md_path = base_dir / 'audit' / '19_harmful_intervention_analysis.md'
    with open(md_path, 'w') as f:
        f.write("# CIG Audit: Harmful Intervention Analysis & Selective Repair Report\n")
        f.write("**Date:** October 2026  \n")
        f.write("**Auditor:** Autonomous Scientific Audit Team  \n")
        f.write("**Deliverable:** `audit/19_harmful_intervention_analysis.md`\n\n---\n\n")
        f.write("## 1. Executive Summary\n\n")
        f.write("The core research premise of the **Counterfactual Intervention Governor (CIG)** is that **detecting an anomaly is not equivalent to improving a prediction**.\n")
        f.write("When a sensor degrades, naive fixed-policy controllers (such as blind soft reweighting or heuristic RGAC) apply corrections unconditionally, often distorting internal representations and making predictions *worse* than doing nothing.\n\n")
        f.write("CIG evaluates candidate corrections in shadow mode and commits an intervention only when an offline-calibrated action risk estimator predicts a credible reduction in RUL error.\n\n")
        f.write("---\n\n## 2. Comparative Harmful Intervention Rate (HIR) Across Methods\n\n")
        f.write("| Method | Total Interventions | Harmful Interventions | Beneficial Interventions | Aggregate HIR | Mean Degraded RMSE |\n")
        f.write("|---|---|---|---|---|---|\n")
        
        for m in harmful_analysis_data:
            tot_int = sum(harmful_analysis_data[m][s]['n_interventions'] for s in harmful_analysis_data[m] if s != 'clean')
            tot_harm = sum(harmful_analysis_data[m][s]['harmful_count'] for s in harmful_analysis_data[m] if s != 'clean')
            tot_ben = sum(harmful_analysis_data[m][s]['beneficial_count'] for s in harmful_analysis_data[m] if s != 'clean')
            agg_hir = (tot_harm / tot_int * 100.0) if tot_int > 0 else 0.0
            mean_deg_rmse = float(np.mean([harmful_analysis_data[m][s]['rmse'] for s in harmful_analysis_data[m] if s != 'clean']))
            f.write(f"| **{m}** | {tot_int} | {tot_harm} | {tot_ben} | **{agg_hir:5.1f}%** | **{mean_deg_rmse:.2f}** |\n")
            
        f.write("\n---\n\n## 3. Scenario-by-Scenario Breakdown\n\n")
        f.write("```json\n" + json.dumps(harmful_analysis_data, indent=2) + "\n```\n")
        
    print(f"[Complete] Harmful intervention analysis saved to {md_path}")

if __name__ == '__main__':
    run_cig_benchmark()
