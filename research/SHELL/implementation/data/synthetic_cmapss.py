"""
Synthetic C-MAPSS-like Data Generator (Fallback)
=================================================
If the NASA C-MAPSS dataset cannot be downloaded, this script generates
synthetic multivariate sensor time series that mimic the statistical 
properties of C-MAPSS FD001 for initial prototype testing.

This data is SYNTHETIC and does NOT replace the real C-MAPSS dataset.
Experiments reported in the paper MUST use the real dataset.

Synthetic data is used ONLY for:
- Unit testing of the SHELL controller and degradation modules
- Smoke testing of the training and evaluation pipelines
- Rapid iteration before real data is available

NOTE: Any results obtained on synthetic data must be clearly labeled
as "synthetic" and cannot substitute for real-data experiments.
"""

import numpy as np
import os
import pandas as pd
from pathlib import Path
from typing import Optional


def generate_engine_trajectory(
    n_cycles: int,
    n_sensors: int = 14,
    noise_level: float = 0.02,
    degradation_start: Optional[int] = None,
    degradation_rate: float = 0.005,
    seed: int = 42,
) -> np.ndarray:
    """
    Generate a single synthetic engine sensor trajectory.
    
    Simple model:
    - Healthy phase: sensor readings near baseline with small noise
    - Degradation phase: sensors drift monotonically (some up, some down)
    
    Returns:
        (n_cycles, n_sensors) array of normalized sensor readings ∈ [0, 1]
    """
    rng = np.random.RandomState(seed)
    
    if degradation_start is None:
        degradation_start = n_cycles // 3
    
    # Baseline sensor values (between 0.3 and 0.7 for diversity)
    baselines = rng.uniform(0.3, 0.7, size=n_sensors)
    
    # Degradation direction (some sensors go up, some go down under degradation)
    deg_direction = rng.choice([-1, 1], size=n_sensors) * rng.uniform(0.3, 1.0, size=n_sensors)
    
    data = np.zeros((n_cycles, n_sensors), dtype=np.float32)
    
    for t in range(n_cycles):
        noise = rng.normal(0, noise_level, size=n_sensors)
        
        if t < degradation_start:
            # Healthy phase
            data[t] = baselines + noise
        else:
            # Degradation phase
            deg_progress = (t - degradation_start) * degradation_rate
            data[t] = baselines + deg_direction * deg_progress + noise
    
    # Clip to [0, 1] range (normalized)
    data = np.clip(data, 0.0, 1.0)
    return data


def generate_synthetic_cmapss(
    n_train_units: int = 100,
    n_test_units: int = 100,
    min_cycles: int = 100,
    max_cycles: int = 400,
    n_sensors: int = 14,
    max_rul: int = 125,
    noise_level: float = 0.02,
    seed: int = 42,
    output_dir: str = './data/raw',
) -> dict:
    """
    Generate synthetic C-MAPSS-like dataset and save in the same format.
    
    Returns:
        dict with same structure as load_and_preprocess()
    """
    rng = np.random.RandomState(seed)
    os.makedirs(output_dir, exist_ok=True)
    
    sensor_cols = [f'sensor_{i+1}' for i in range(n_sensors)]
    
    print(f"[Synthetic C-MAPSS] Generating {n_train_units} train + {n_test_units} test units")
    
    # Generate training data (run-to-failure)
    train_records = []
    for unit_id in range(1, n_train_units + 1):
        n_cycles = rng.randint(min_cycles, max_cycles)
        deg_rate = rng.uniform(0.002, 0.01)
        traj = generate_engine_trajectory(
            n_cycles, n_sensors, noise_level, 
            degradation_rate=deg_rate, seed=seed + unit_id
        )
        
        for t in range(n_cycles):
            row = [unit_id, t + 1, 0.0, 0.0, 0.0] + list(traj[t])
            train_records.append(row)
    
    # Column names
    cols = ['unit_id', 'time_cycle', 'op_setting_1', 'op_setting_2', 'op_setting_3']
    cols += sensor_cols
    
    train_df = pd.DataFrame(train_records, columns=cols)
    
    # Generate test data (stopped before failure)
    test_records = []
    test_rul = []
    for unit_id in range(1, n_test_units + 1):
        total_cycles = rng.randint(min_cycles, max_cycles)
        deg_rate = rng.uniform(0.002, 0.01)
        traj = generate_engine_trajectory(
            total_cycles, n_sensors, noise_level,
            degradation_rate=deg_rate, seed=seed + 1000 + unit_id
        )
        
        # Stop at a random point before failure
        stop_at = rng.randint(total_cycles // 2, total_cycles - 10)
        true_rul = min(total_cycles - stop_at, max_rul)
        test_rul.append(true_rul)
        
        for t in range(stop_at):
            row = [unit_id, t + 1, 0.0, 0.0, 0.0] + list(traj[t])
            test_records.append(row)
    
    test_df = pd.DataFrame(test_records, columns=cols)
    
    # Compute RUL for training
    def compute_rul(df, max_rul=125):
        df = df.copy()
        max_cycles = df.groupby('unit_id')['time_cycle'].max().reset_index()
        max_cycles.columns = ['unit_id', 'max_cycle']
        df = df.merge(max_cycles, on='unit_id')
        df['RUL'] = (df['max_cycle'] - df['time_cycle']).clip(upper=max_rul)
        df.drop(columns=['max_cycle'], inplace=True)
        return df
    
    train_df = compute_rul(train_df, max_rul)
    
    # MinMax normalize using training stats
    norm_stats = {}
    for col in sensor_cols:
        col_min = train_df[col].min()
        col_max = train_df[col].max()
        col_range = col_max - col_min + 1e-8
        train_df[col] = (train_df[col] - col_min) / col_range
        test_df[col] = (test_df[col] - col_min) / col_range
        norm_stats[col] = {'min': col_min, 'max': col_max}
    
    # Save to files in same format as real C-MAPSS
    train_path = os.path.join(output_dir, 'train_FD001_synthetic.txt')
    test_path = os.path.join(output_dir, 'test_FD001_synthetic.txt')
    rul_path = os.path.join(output_dir, 'RUL_FD001_synthetic.txt')
    
    train_df.drop(columns=['RUL']).to_csv(train_path, sep=' ', header=False, index=False)
    test_df.to_csv(test_path, sep=' ', header=False, index=False)
    
    with open(rul_path, 'w') as f:
        for rul in test_rul:
            f.write(f'{rul}\n')
    
    print(f"  Train: {len(train_df['unit_id'].unique())} units, {len(train_df)} records")
    print(f"  Test:  {len(test_df['unit_id'].unique())} units, {len(test_df)} records")
    print(f"  Saved to: {output_dir}")
    
    # Create test_rul DataFrame matching load_and_preprocess format
    rul_df = pd.DataFrame({'RUL': test_rul, 'unit_id': range(1, n_test_units + 1)})
    test_last = test_df.groupby('unit_id').last().reset_index()
    test_last = test_last.merge(rul_df, on='unit_id')
    
    return {
        'train_df': train_df,
        'test_df': test_df,
        'test_rul': test_last,
        'sensor_cols': sensor_cols,
        'norm_stats': norm_stats,
        'subset': 'FD001_synthetic',
        'is_synthetic': True,
    }


if __name__ == '__main__':
    data = generate_synthetic_cmapss(
        n_train_units=100, n_test_units=100, seed=42,
        output_dir='./data/raw'
    )
    print("Synthetic data generation complete.")
    print(f"Train shape: {data['train_df'].shape}")
    print(f"Test shape: {data['test_df'].shape}")
