# TIUL: Transferable Intervention Utility Learning & Robust Sensor Adaptation

This repository contains the official code implementations for **TIUL (Transferable Intervention Utility Learning)** and **SHELL (Sensor Health-gated Evolutionary Lifelong Learning)** architectures. 

The codebase provides end-to-end pipelines for robust Remaining Useful Life (RUL) estimation, sensor degradation simulation, test-time adaptation (TTA), counterfactual intervention governance, and cross-domain transfer learning on multi-sensor turbofan benchmarks (NASA C-MAPSS FD001 and FD002).

---

## Codebase Architecture

```
tiul-intervention-learning/
├── TIUL/
│   └── implementation/
│       ├── models/
│       │   └── backbone.py               # TIUL backbone with Stage B pretraining & weight invariance
│       ├── data/
│       │   ├── download_cmapss.py        # Automated C-MAPSS dataset downloader & preprocessor
│       │   ├── data_v2.py                # Sequence generator, unit-disjoint splits, corruption injector
│       │   └── __init__.py
│       ├── controller.py                 # TICA decision controller (temporal variance health & bounded updates)
│       ├── data_v2.py                    # Root data interface for TIUL
│       ├── train_backbone.py             # Stage B corruption-augmented training script
│       ├── evaluate.py                   # Benchmark suite (Static, Tent, CoTTA, TIUL)
│       └── run_cross_domain_experiments.py # Cross-domain transfer (FD001 -> FD002)
│
├── research/
│   └── SHELL/
│       ├── implementation/               # SHELL Baseline implementation
│       │   ├── models/backbone.py        # Baseline multi-head sensor encoder & reconstructor
│       │   ├── shell/controller.py       # Reliability-gated adaptation controller
│       │   ├── shell/degradation.py      # Sensor fault injection routines
│       │   ├── data/download_cmapss.py   # Dataset downloader
│       │   ├── data/synthetic_cmapss.py  # Synthetic telemetry generator for unit tests
│       │   ├── train_backbone.py         # Backbone training
│       │   ├── evaluate.py               # 10-baseline evaluation suite
│       │   └── tests/test_shell.py       # Unit & integration test suite
│       │
│       ├── implementation_v2/            # SHELL v2 (Robust Backbone & Calibrated Health)
│       │   ├── calibrate_health.py       # Health metric calibration
│       │   ├── health_estimator_v2.py    # Temporal intra-window variance estimator
│       │   ├── rgac_controller.py        # Reliability-Gated Adaptation Controller v2
│       │   ├── train_robust_backbone.py  # Robust pretraining
│       │   ├── evaluate_all.py           # Evaluation across degradation regimes
│       │   └── data_v2.py                # Dataset utilities
│       │
│       └── implementation_v3_cig/        # SHELL v3 (Counterfactual Intervention Governor - CIG)
│           ├── cig_governor.py           # Selective intervention governor
│           ├── action_generator.py       # Candidate adaptation action generator
│           ├── risk_estimator.py         # Intervention risk prediction model
│           ├── train_risk_estimator.py   # Risk model training
│           ├── generate_risk_dataset.py  # Counterfactual risk dataset generator
│           ├── evaluate_cig.py           # CIG evaluation
│           └── run_decision_quality_benchmark.py # Decision regret & regret-vs-oracle benchmarks
│
├── requirements.txt                      # Python dependencies
└── .gitignore                            # Excludes checkpoints, raw datasets, and research docs
```

---

## Key Algorithmic Components

### 1. TIUL (Transferable Intervention Utility Learning)
- **Stage B Corruption-Augmented Pretraining:** Samples Dirichlet sensor weights and random sensor dropouts during training so the latent fusion head is mathematically invariant to sensor weight shifts.
- **TICA Controller:** Leverages intra-window temporal variance rather than cross-engine statistics to evaluate sensor health in $[0, 1]$ reliably.
- **Bounded Adaptation:** Restricts parameter shift via Frobenius-norm projections scaled by aggregate sensor health, preventing catastrophic test-time divergence.
- **Strict Unit Partitioning:** Split by engine Unit ID (e.g. engines 1-80 train, 81-100 val) to ensure zero sequence leakage across temporal windows.

### 2. CIG Governor (Counterfactual Intervention Governor)
- **Selective Intervention:** Default to no-op when signals are nominal, intervening only when estimated risk of inaction exceeds the risk of intervention.
- **Regret Truncation:** Evaluated against counterfactual oracle choices to eliminate unnecessary adaptations on healthy sensors.

---

## Installation & Setup

1. **Clone the repository:**
   ```bash
   git clone https://github.com/sreechakra180/tiul-intervention-learning.git
   cd tiul-intervention-learning
   ```

2. **Create a virtual environment & install requirements:**
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: .\venv\Scripts\activate
   pip install -r requirements.txt
   ```

---

## Running TIUL

### 1. Download & Prepare NASA C-MAPSS Dataset
```bash
python TIUL/implementation/data/download_cmapss.py --data_dir TIUL/implementation/data/raw
```

### 2. Train TIUL Backbone (Stage B Pretraining)
```bash
python TIUL/implementation/train_backbone.py \
    --data_dir TIUL/implementation/data/raw \
    --subset FD001 \
    --epochs 100 \
    --batch_size 64 \
    --save_dir TIUL/checkpoints
```

### 3. Evaluate TIUL Against Baselines
```bash
python TIUL/implementation/evaluate.py \
    --checkpoint TIUL/checkpoints/tiul_backbone_FD001.pt \
    --data_dir TIUL/implementation/data/raw \
    --subset FD001
```

### 4. Cross-Domain Transfer (FD001 → FD002)
```bash
python TIUL/implementation/run_cross_domain_experiments.py \
    --source_domain FD001 \
    --target_domain FD002 \
    --epochs 80 \
    --corrupt_prob 0.35
```

---

## Running SHELL & CIG Governor

### Baseline SHELL
```bash
python research/SHELL/implementation/evaluate.py \
    --checkpoint research/SHELL/implementation/checkpoints/shell_backbone_FD001_seed42.pt \
    --data_dir research/SHELL/implementation/data/raw \
    --subset FD001 \
    --seed 42
```

### CIG Decision Quality Benchmark
```bash
python research/SHELL/implementation_v3_cig/run_decision_quality_benchmark.py
```

---

## License & Attribution

This codebase is provided for research and development into transferable intervention policies and robust test-time adaptation for multi-sensor systems.
