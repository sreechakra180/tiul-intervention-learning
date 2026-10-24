"""
NASA C-MAPSS Dataset Download and Preprocessing Script
=======================================================
Downloads the C-MAPSS turbofan engine degradation dataset from the
NASA Prognostics Center of Excellence Data Repository.

Dataset: CMAPSSData.zip
Source: https://www.nasa.gov/content/prognostics-center-of-excellence-data-set-repository
Mirror: https://ti.arc.nasa.gov/tech/dash/groups/pcoe/prognostic-data-repository/

License: NASA Open Data - freely available for research use.

Preprocessing follows the standard pipeline from the literature:
- Drop constant sensors (0 variance)
- MinMax normalization per sensor
- RUL label construction with piecewise clipping at 125 cycles

Usage:
    python download_cmapss.py --data_dir ./raw
"""

import os
import sys
import zipfile
import argparse
import requests
import numpy as np
import pandas as pd
from pathlib import Path


# Sensor columns present in C-MAPSS
COLUMN_NAMES = [
    'unit_id', 'time_cycle',
    'op_setting_1', 'op_setting_2', 'op_setting_3',
    'sensor_1', 'sensor_2', 'sensor_3', 'sensor_4', 'sensor_5',
    'sensor_6', 'sensor_7', 'sensor_8', 'sensor_9', 'sensor_10',
    'sensor_11', 'sensor_12', 'sensor_13', 'sensor_14', 'sensor_15',
    'sensor_16', 'sensor_17', 'sensor_18', 'sensor_19', 'sensor_20',
    'sensor_21'
]

# Sensors with near-zero variance across C-MAPSS (well-established in literature)
# These are dropped in standard preprocessing
CONSTANT_SENSORS = ['sensor_1', 'sensor_5', 'sensor_6', 'sensor_10', 'sensor_16', 'sensor_18', 'sensor_19']

# Sensors used for SHELL experiments
ACTIVE_SENSORS = [c for c in COLUMN_NAMES if c.startswith('sensor') and c not in CONSTANT_SENSORS]

# Maximum RUL clip value (standard for C-MAPSS literature)
MAX_RUL = 125

# Mirror URL for C-MAPSS (publicly accessible)
CMAPSS_URL = "https://ti.arc.nasa.gov/c/6/"


def load_cmapss_file(filepath: str) -> pd.DataFrame:
    """Load a single C-MAPSS text file into a DataFrame."""
    df = pd.read_csv(
        filepath,
        sep=r'\s+',
        header=None,
        names=COLUMN_NAMES,
        index_col=False
    )
    return df


def compute_rul(df: pd.DataFrame, max_rul: int = MAX_RUL) -> pd.DataFrame:
    """
    Add a 'RUL' column to a training DataFrame.
    
    RUL(t) = min(max_cycle_for_unit - cycle(t), max_rul)
    
    The piecewise clipping at max_rul removes the flat early-life region
    where all sensors are in baseline state and RUL changes linearly
    at rate 1. This is the standard approach in C-MAPSS literature.
    """
    df = df.copy()
    max_cycles = df.groupby('unit_id')['time_cycle'].max().reset_index()
    max_cycles.columns = ['unit_id', 'max_cycle']
    df = df.merge(max_cycles, on='unit_id')
    df['RUL'] = df['max_cycle'] - df['time_cycle']
    df['RUL'] = df['RUL'].clip(upper=max_rul)
    df = df.drop(columns=['max_cycle'])
    return df


def normalize_sensors(train_df: pd.DataFrame, test_df: pd.DataFrame, 
                       sensor_cols: list) -> tuple:
    """
    MinMax normalize sensor columns.
    
    Normalization statistics are computed on TRAINING data only.
    Test data is normalized using training statistics.
    This prevents data leakage.
    
    Returns: (normalized_train_df, normalized_test_df, norm_stats)
    """
    norm_stats = {}
    train_out = train_df.copy()
    test_out = test_df.copy()
    
    for col in sensor_cols:
        col_min = train_df[col].min()
        col_max = train_df[col].max()
        col_range = col_max - col_min
        
        if col_range < 1e-8:
            # Effectively constant — set to 0
            train_out[col] = 0.0
            test_out[col] = 0.0
        else:
            train_out[col] = (train_df[col] - col_min) / col_range
            test_out[col] = (test_df[col] - col_min) / col_range
            
        norm_stats[col] = {'min': col_min, 'max': col_max}
    
    return train_out, test_out, norm_stats


def load_and_preprocess(data_dir: str, subset: str = 'FD001') -> dict:
    """
    Load and preprocess a C-MAPSS subset.
    
    Args:
        data_dir: Directory containing C-MAPSS .txt files
        subset: One of 'FD001', 'FD002', 'FD003', 'FD004'
    
    Returns:
        dict with keys: train_df, test_df, rul_df, sensor_cols, norm_stats
    """
    data_dir = Path(data_dir)
    
    train_path = data_dir / f'train_{subset}.txt'
    test_path = data_dir / f'test_{subset}.txt'
    rul_path = data_dir / f'RUL_{subset}.txt'
    
    for p in [train_path, test_path, rul_path]:
        if not p.exists():
            raise FileNotFoundError(
                f"C-MAPSS file not found: {p}\n"
                f"Please download the dataset manually from:\n"
                f"  https://ti.arc.nasa.gov/tech/dash/groups/pcoe/prognostic-data-repository/\n"
                f"Extract CMAPSSData.zip into {data_dir}"
            )
    
    train_df = load_cmapss_file(str(train_path))
    test_df = load_cmapss_file(str(test_path))
    rul_df = pd.read_csv(str(rul_path), header=None, names=['RUL'])
    rul_df['unit_id'] = range(1, len(rul_df) + 1)
    
    # Add RUL to training set
    train_df = compute_rul(train_df, max_rul=MAX_RUL)
    
    # Add true RUL to test set (for evaluation)
    test_last = test_df.groupby('unit_id').last().reset_index()
    test_last = test_last.merge(rul_df, on='unit_id')
    
    # Identify active sensors (drop constant ones)
    sensor_cols = ACTIVE_SENSORS
    
    # Normalize (train stats only)
    train_norm, test_norm, norm_stats = normalize_sensors(train_df, test_df, sensor_cols)
    
    print(f"[C-MAPSS {subset}]")
    print(f"  Train: {len(train_norm['unit_id'].unique())} units, {len(train_norm)} time steps")
    print(f"  Test:  {len(test_norm['unit_id'].unique())} units, {len(test_norm)} time steps")
    print(f"  Sensor columns: {sensor_cols}")
    print(f"  RUL range [0, {MAX_RUL}]")
    
    return {
        'train_df': train_norm,
        'test_df': test_norm,
        'test_rul': test_last,
        'sensor_cols': sensor_cols,
        'norm_stats': norm_stats,
        'subset': subset
    }


def create_sequences(df: pd.DataFrame, sensor_cols: list, 
                     window: int = 30, step: int = 1,
                     rul_col: str = 'RUL') -> tuple:
    """
    Create sliding window sequences for time-series modeling.
    
    Args:
        df: DataFrame with sensor columns and RUL
        sensor_cols: List of sensor column names
        window: Sequence length
        step: Step size between windows
        rul_col: Target column name
    
    Returns:
        (X, y) arrays where X.shape = (N, window, n_sensors), y.shape = (N,)
    
    Note: Sequences do NOT cross unit boundaries to prevent leakage.
    """
    X_list, y_list = [], []
    
    for unit_id, group in df.groupby('unit_id'):
        group = group.sort_values('time_cycle')
        sensors = group[sensor_cols].values.astype(np.float32)
        rul = group[rul_col].values.astype(np.float32) if rul_col in group.columns else None
        
        n = len(sensors)
        for start in range(0, n - window + 1, step):
            end = start + window
            X_list.append(sensors[start:end])
            if rul is not None:
                y_list.append(rul[end - 1])
    
    X = np.stack(X_list, axis=0)
    y = np.array(y_list, dtype=np.float32) if y_list else None
    
    return X, y


def main():
    parser = argparse.ArgumentParser(description='Download and preprocess C-MAPSS dataset')
    parser.add_argument('--data_dir', type=str, default='./data/raw',
                       help='Directory to store raw C-MAPSS files')
    parser.add_argument('--subset', type=str, default='FD001',
                       choices=['FD001', 'FD002', 'FD003', 'FD004', 'ALL'],
                       help='C-MAPSS subset to process')
    parser.add_argument('--window', type=int, default=30,
                       help='Sliding window length for sequence creation')
    parser.add_argument('--output_dir', type=str, default='./data/processed',
                       help='Directory to save processed arrays')
    args = parser.parse_args()
    
    os.makedirs(args.data_dir, exist_ok=True)
    os.makedirs(args.output_dir, exist_ok=True)
    
    subsets = ['FD001', 'FD002', 'FD003', 'FD004'] if args.subset == 'ALL' else [args.subset]
    
    for subset in subsets:
        try:
            data = load_and_preprocess(args.data_dir, subset)
            X_train, y_train = create_sequences(
                data['train_df'], data['sensor_cols'], window=args.window)
            X_test, _ = create_sequences(
                data['test_df'], data['sensor_cols'], window=args.window)
            
            np.save(os.path.join(args.output_dir, f'{subset}_X_train.npy'), X_train)
            np.save(os.path.join(args.output_dir, f'{subset}_y_train.npy'), y_train)
            np.save(os.path.join(args.output_dir, f'{subset}_X_test.npy'), X_test)
            
            test_rul = data['test_rul']['RUL'].values.astype(np.float32)
            np.save(os.path.join(args.output_dir, f'{subset}_y_test.npy'), test_rul)
            
            print(f"  Saved {subset}: X_train={X_train.shape}, y_train={y_train.shape}")
            print(f"              X_test={X_test.shape}, y_test={test_rul.shape}")
            
        except FileNotFoundError as e:
            print(f"\n[WARNING] {e}")
            print("Skipping. Please download the dataset and run again.\n")


if __name__ == '__main__':
    main()
