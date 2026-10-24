"""
Rigorous Engine-Disjoint Data Pipeline for C-MAPSS FD001 (Version 2)
=====================================================================
- Disjoint partitioning by Unit ID:
    * Training: Units 1 to 80 (80 engines)
    * Validation: Units 81 to 100 (20 engines)
    * Testing: Official test_FD001.txt (100 engines)
- Normalization statistics (min/max) fitted strictly on Training engines (1-80).
- Zero sliding window leakage between splits.
- Piecewise linear RUL with standard clipping (RUL_max = 125).
"""

import os
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, Tuple, List

COLUMN_NAMES = [
    'unit_id', 'time_cycle',
    'op_setting_1', 'op_setting_2', 'op_setting_3',
    'sensor_1', 'sensor_2', 'sensor_3', 'sensor_4', 'sensor_5',
    'sensor_6', 'sensor_7', 'sensor_8', 'sensor_9', 'sensor_10',
    'sensor_11', 'sensor_12', 'sensor_13', 'sensor_14', 'sensor_15',
    'sensor_16', 'sensor_17', 'sensor_18', 'sensor_19', 'sensor_20',
    'sensor_21'
]

CONSTANT_SENSORS = [
    'sensor_1', 'sensor_5', 'sensor_6', 'sensor_10',
    'sensor_16', 'sensor_18', 'sensor_19'
]

ACTIVE_SENSORS = [
    c for c in COLUMN_NAMES if c.startswith('sensor') and c not in CONSTANT_SENSORS
]

MAX_RUL = 125

def load_cmapss_file(filepath: str) -> pd.DataFrame:
    df = pd.read_csv(filepath, sep=r'\s+', header=None, names=COLUMN_NAMES)
    return df

def compute_piecewise_rul(df: pd.DataFrame, max_rul: int = MAX_RUL) -> pd.DataFrame:
    df = df.copy()
    max_cycles = df.groupby('unit_id')['time_cycle'].max().reset_index()
    max_cycles.columns = ['unit_id', 'max_cycle']
    df = df.merge(max_cycles, on='unit_id')
    df['RUL'] = df['max_cycle'] - df['time_cycle']
    df['RUL'] = df['RUL'].clip(upper=max_rul)
    df = df.drop(columns=['max_cycle'])
    return df

def fit_normalizer(train_df: pd.DataFrame, sensor_cols: List[str]) -> Dict[str, Dict[str, float]]:
    norm_stats = {}
    for col in sensor_cols:
        c_min = float(train_df[col].min())
        c_max = float(train_df[col].max())
        norm_stats[col] = {'min': c_min, 'max': c_max, 'range': max(c_max - c_min, 1e-8)}
    return norm_stats

def apply_normalizer(df: pd.DataFrame, sensor_cols: List[str], norm_stats: Dict[str, Dict[str, float]]) -> pd.DataFrame:
    df_out = df.copy()
    for col in sensor_cols:
        c_min = norm_stats[col]['min']
        c_range = norm_stats[col]['range']
        df_out[col] = (df[col] - c_min) / c_range
    return df_out

def create_sequences_by_engine(
    df: pd.DataFrame,
    sensor_cols: List[str],
    window: int = 30,
    step: int = 1,
    rul_col: str = 'RUL'
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Creates sequences strictly bounded within each unit.
    Returns:
        X: (N, window, n_sensors)
        y: (N,) target RUL at sequence end
        unit_ids: (N,) unit ID for each sequence
    """
    X_list, y_list, unit_list = [], [], []
    for unit_id, group in df.groupby('unit_id'):
        group = group.sort_values('time_cycle')
        sensors = group[sensor_cols].values.astype(np.float32)
        ruls = group[rul_col].values.astype(np.float32) if rul_col in group.columns else None
        
        n = len(sensors)
        for start in range(0, n - window + 1, step):
            end = start + window
            X_list.append(sensors[start:end])
            unit_list.append(unit_id)
            if ruls is not None:
                y_list.append(ruls[end - 1])
                
    X = np.stack(X_list, axis=0)
    y = np.array(y_list, dtype=np.float32) if y_list else None
    unit_ids = np.array(unit_list, dtype=np.int32)
    return X, y, unit_ids

def load_data_partitioned(data_dir: str, train_units_max: int = 80) -> dict:
    """
    Loads FD001 and partitions into:
    - Train (units 1 to train_units_max)
    - Val (units train_units_max+1 to 100)
    - Test (100 official test units)
    """
    data_dir = Path(data_dir)
    raw_train = load_cmapss_file(str(data_dir / 'train_FD001.txt'))
    raw_test = load_cmapss_file(str(data_dir / 'test_FD001.txt'))
    raw_rul = pd.read_csv(str(data_dir / 'RUL_FD001.txt'), header=None, names=['RUL'])
    raw_rul['unit_id'] = range(1, len(raw_rul) + 1)
    
    # Add piecewise RUL to train
    train_with_rul = compute_piecewise_rul(raw_train)
    
    # Split train and val by unit_id
    train_split_df = train_with_rul[train_with_rul['unit_id'] <= train_units_max].copy()
    val_split_df = train_with_rul[train_with_rul['unit_id'] > train_units_max].copy()
    
    # Fit normalizer strictly on train_split_df
    norm_stats = fit_normalizer(train_split_df, ACTIVE_SENSORS)
    
    # Apply normalizer
    train_norm = apply_normalizer(train_split_df, ACTIVE_SENSORS, norm_stats)
    val_norm = apply_normalizer(val_split_df, ACTIVE_SENSORS, norm_stats)
    test_norm = apply_normalizer(raw_test, ACTIVE_SENSORS, norm_stats)
    
    # Extract endpoint test evaluation set (final window per test engine)
    X_test_eval_list, y_test_eval_list = [], []
    test_rul_vals = raw_rul['RUL'].values.astype(np.float32)
    
    for unit_id, group in test_norm.groupby('unit_id'):
        group = group.sort_values('time_cycle')
        sensors = group[ACTIVE_SENSORS].values.astype(np.float32)
        if len(sensors) >= 30:
            X_test_eval_list.append(sensors[-30:])
            unit_idx = int(unit_id) - 1
            if unit_idx < len(test_rul_vals):
                y_test_eval_list.append(test_rul_vals[unit_idx])
                
    X_test_eval = np.stack(X_test_eval_list, axis=0) # (100, 30, 14)
    y_test_eval = np.array(y_test_eval_list, dtype=np.float32) # (100,)
    
    # Validation endpoint evaluation set (final window of each validation engine)
    X_val_eval_list, y_val_eval_list = [], []
    for unit_id, group in val_norm.groupby('unit_id'):
        group = group.sort_values('time_cycle')
        sensors = group[ACTIVE_SENSORS].values.astype(np.float32)
        ruls = group['RUL'].values.astype(np.float32)
        if len(sensors) >= 30:
            X_val_eval_list.append(sensors[-30:])
            y_val_eval_list.append(ruls[-1])
            
    X_val_eval = np.stack(X_val_eval_list, axis=0) # (20, 30, 14)
    y_val_eval = np.array(y_val_eval_list, dtype=np.float32) # (20,)
    
    return {
        'train_df': train_norm,
        'val_df': val_norm,
        'test_df': test_norm,
        'raw_rul': raw_rul,
        'sensor_cols': ACTIVE_SENSORS,
        'norm_stats': norm_stats,
        'X_val_eval': X_val_eval,
        'y_val_eval': y_val_eval,
        'X_test_eval': X_test_eval,
        'y_test_eval': y_test_eval,
    }
