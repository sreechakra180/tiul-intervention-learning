"""
TIUL Data Module: NASA C-MAPSS Cross-Domain Loading

Loads NASA C-MAPSS FD001 and FD002 datasets with cross-domain train/val/test splits.
Supports corruption injection, normalization isolation, and domain-shifted evaluation.

Dataset Origins:
- NASA C-MAPSS (Commercial Modular Aero-Propulsion System Simulation)
- PCoE at NASA Ames
- FD001: 100 training engines, 100 test engines, 1 operating condition (sea level)
- FD002: 260 training engines, 259 test engines, 6 operating conditions

All degradation scenarios are applied ONLY during evaluation.
Training uses clean data. This prevents test-time information leakage.
"""

import numpy as np
import torch
from pathlib import Path
import json
from typing import Dict, List, Optional, Tuple
from copy import deepcopy
from dataclasses import dataclass, field
from enum import Enum


# ============================================================
# C-MAPSS Data Loading & Preprocessing
# ============================================================

# Active sensor channels from C-MAPSS (14 of 21, after filtering 7 zero-variance)
# Sensors 2,3,4 = temperatures; 7,8,11,12,13,14 = pressures/flows/fan/speed
C_MAPSS_ACTIVE_SENSORS = [1, 2, 3, 4, 6, 7, 8, 9, 11, 12, 13, 14, 17, 20, 21]
# 0-indexed mapping for the 14 active sensors used throughout TIUL
ACTIVE_SENSOR_MAP = {s: i for i, s in enumerate(C_MAPSS_ACTIVE_SENSORS)}


def parse_cmapss_file(filepath: str, limit_rows: int = None) -> dict:
    """Parse a NASA C-MAPSS text file.

    Format: Each row has 24 columns (3 op settings + 21 sensor readings + RUL at end)
    But standard format has: op_setting_1, op_setting_2, op_setting_3,
    sensor_1 through sensor_21, RUL
    """
    data = []
    with open(filepath, 'r') as f:
        for i, line in enumerate(f):
            if limit_rows is not None and i >= limit_rows:
                break
            parts = line.strip().split()
            if len(parts) < 24:
                continue
            try:
                row = {
                    'unit_number': int(parts[0]),
                    'time_cycle': int(parts[1]),
                    'op_setting_1': float(parts[2]),
                    'op_setting_2': float(parts[3]),
                    'op_setting_3': float(parts[4]),
                }
                # Sensors 1-21
                for j in range(21):
                    row[f'sensor_{j+1}'] = float(parts[5 + j])
                # RUL at end
                row['RUL'] = float(parts[26])
                data.append(row)
            except (ValueError, IndexError):
                continue
    return data


def load_cmapss_datasets(data_dir: str) -> dict:
    """Load all C-MAPSS FD001 and FD002 dataset files."""

    result = {
        'train_FD001': parse_cmapss_file(f"{data_dir}/train_FD001.txt"),
        'test_FD001': parse_cmapss_file(f"{data_dir}/test_FD001.txt"),
        'RUL_FD001': parse_cmapss_file(f"{data_dir}/RUL_FD001.txt"),
        'train_FD002': parse_cmapss_file(f"{data_dir}/train_FD002.txt"),
        'test_FD002': parse_cmapss_file(f"{data_dir}/test_FD002.txt"),
    }

    # Verify files loaded
    for key, val in result.items():
        if not val:
            raise ValueError(f"Failed to load {key} from {data_dir}")

    return result

# ============================================================
# Normalization & Sequence Creation
# ============================================================

def compute_minmax(train_data: list, sensor_cols: list) -> dict:
    """Compute MinMax normalization parameters from training data.

    Fitted STRICTLY on training data. Test data uses these parameters.
    """
    mins = {}
    maxs = {}
    for col in sensor_cols:
        values = [row[col] for row in train_data if col in row]
        if values:
            mins[col] = min(values)
            maxs[col] = max(values)
        else:
            mins[col] = 0.0
            maxs[col] = 1.0
    return {'mins': mins, 'maxs': maxs}


def minmax_normalize(data: list, norm_params: dict, sensor_cols: list) -> list:
    """Apply MinMax normalization using provided parameters."""
    result = deepcopy(data)
    for row in result:
        for col in sensor_cols:
            if col in row and norm_params['maxs'][col] > norm_params['mins'][col]:
                row[col] = (row[col] - norm_params['mins'][col]) / (
                    norm_params['maxs'][col] - norm_params['mins'][col]
                )
            else:
                row[col] = 0.0
    return result


def create_sequences(data: list, sensor_cols: list, window: int = 30, step: int = 1) -> tuple:
    """Create sliding window sequences from C-MAPSS data.

    Returns:
        X: (n_sequences, window, n_sensors) float32 tensor
        y: (n_sequences,) float32 tensor (RUL at end of window)
        seq_units: list of unit IDs for each sequence
    """
    sequences = []
    targets = []
    seq_units = []

    # Group by unit_id
    from collections import defaultdict
    unit_groups = defaultdict(list)
    for row in data:
        unit_groups[row['unit_number']].append(row)

    for unit_id, group in unit_groups.items():
        # Sort by time cycle
        group = sorted(group, key=lambda r: r['time_cycle'])

        # Get sensor matrix
        n_sensors = len(sensor_cols)
        sensor_matrix = np.zeros((len(group), n_sensors), dtype=np.float32)

        for i, row in enumerate(group):
            for j, col in enumerate(sensor_cols):
                if col in row:
                    sensor_matrix[i, j] = row[col]

        # Create sliding windows
        for start in range(0, len(group) - window + 1, step):
            end = start + window
            if end > len(group):
                break

            X_win = sensor_matrix[start:end]  # (window, n_sensors)
            y_val = group[end - 1]['RUL']  # RUL at end of window

            sequences.append(X_win)
            targets.append(y_val)
            seq_units.append(unit_id)

    if not sequences:
        raise ValueError("No sequences created — data may be too short for window size")

    X = np.array(sequences, dtype=np.float32)  # (n_seq, window, n_sensors)
    y = np.array(targets, dtype=np.float32)     # (n_seq,)

    return X, y, seq_units


def clip_rul(y: np.ndarray, rul_max: float = 125.0) -> np.ndarray:
    """Clip RUL at maximum value (standard C-MAPSS protocol)."""
    y_clipped = np.minimum(y, rul_max)
    return y_clipped


# ============================================================
# Domain Definition & Splitting
# ============================================================

class CMapssDomain:
    """Represents a C-MAPSS domain (FD001 or FD002) with operating condition info."""

    def __init__(self, name: str, n_train: int, n_test: int, n_operating_conditions: int = 1,
                 sensor_cols: list = None):
        self.name = name
        self.n_train = n_train
        self.n_test = n_test
        self.n_operating_conditions = n_operating_conditions
        self.sensor_cols = sensor_cols or ACTIVE_SENSOR_MAP.keys()

    def __repr__(self):
        return f"CMapssDomain({self.name}, train={self.n_train}, test={self.n_test}, " \
               f"ops={self.n_operating_conditions})"


def define_domains(raw_data: dict) -> tuple:
    """Define C-MAPSS domains from raw loaded data."""

    # FD001: 100 train, 100 test, 1 operating condition
    fd001_train = len(raw_data['train_FD001'])
    fd001_test = len(raw_data['test_FD001'])
    domain_S = CMapssDomain('FD001', fd001_train, fd001_test, n_operating_conditions=1)

    # FD002: 260 train, 259 test, 6 operating conditions
    fd002_train = len(raw_data['train_FD002'])
    fd002_test = len(raw_data['test_FD002'])
    domain_T = CMapssDomain('FD002', fd002_train, fd002_test, n_operating_conditions=6)

    return domain_S, domain_T


def engine_level_train_val_split(domain: CMapssDomain, raw_data: dict,
                                 train_engines: int = None, val_engines: int = None) -> dict:
    """Split data by engine Unit ID — zero sequence overlap between splits.

    This is the critical fix from the SHELL audit (Defect 4.1): validation split must
    be by engine Unit ID, not by random window split.
    """
    from collections import defaultdict

    if train_engines is None:
        # Default: use first N engines for training, rest for validation
        # FD001: engines 1-80 train, 81-100 validation
        # FD002: engines 1-200 train, 201-260 validation (rough 80/20)
        if domain.name == 'FD001':
            train_engines = 80
            val_engines = 20
        elif domain.name == 'FD002':
            train_engines = 200
            val_engines = 60
        else:
            train_engines = 80
            val_engines = 20

    # Group data by unit number
    unit_groups = defaultdict(list)
    for row in raw_data['train_FD001'] if domain.name == 'FD001' else raw_data['train_FD002']:
        unit_groups[row['unit_number']].append(row)

    # Select training engines (lowest numbered)
    train_unit_ids = sorted(unit_groups.keys())[:train_engines]
    val_unit_ids = sorted(unit_groups.keys())[train_engines:train_engines+val_engines]

    # Partition data
    train_data = []
    val_data = []
    test_data = raw_data[f'test_{domain.name}']

    for row in test_data:
        test_data_row = {k: v for k, v in row.items()}
        test_data_row['source'] = 'test'
        test_data.append(test_data_row)

    for uid in train_unit_ids:
        for row in unit_groups[uid]:
            row_copy = {k: v for k, v in row.items()}
            row_copy['source'] = 'train'
            train_data.append(row_copy)

    for uid in val_unit_ids:
        for row in unit_groups[uid]:
            row_copy = {k: v for k, v in row.items()}
            row_copy['source'] = 'val'
            val_data.append(row_copy)

    return {
        'train': train_data,
        'validation': val_data,
        'test': test_data,
        'train_unit_ids': train_unit_ids,
        'val_unit_ids': val_unit_ids,
        'test_unit_ids': [row['unit_number'] for row in test_data],
    }# ============================================================
# Corruption Injection (Evaluation-Only)
# ============================================================

class CorruptionType(Enum):
    NONE = 'none'
    GAUSSIAN_NOISE = 'gaussian_noise'
    BIAS_DRIFT = 'bias_drift'
    TEMPORAL_DROPOUT = 'temporal_dropout'
    MODALITY_DROPOUT = 'modality_dropout'
    DISTRIBUTION_SHIFT = 'distribution_shift'
    RESTORATION = 'restoration'


@dataclass
class CorruptionScenario:
    """Describes a single corruption scenario for evaluation."""
    deg_type: CorruptionType = CorruptionType.NONE
    sensor_indices: List[int] = field(default_factory=list)
    severity: float = 0.0
    start_cycle: int = 0
    end_cycle: int = -1  # -1 means "forever"
    restoration_at: Optional[int] = None
    drift_rate: float = 0.0
    dropout_period: int = 10
    affine_alpha: float = 1.0
    affine_beta: float = 0.0
    seed: int = 42

    def is_active(self, t: int) -> bool:
        """Returns True if corruption is active at cycle t."""
        if t < self.start_cycle:
            return False
        if self.end_cycle != -1 and t >= self.end_cycle:
            return False
        if self.restoration_at is not None and t >= self.restoration_at:
            return False
        return True

    def is_restored(self, t: int) -> bool:
        """Returns True if corruption has been restored at cycle t."""
        if self.restoration_at is None:
            return False
        return t >= self.restoration_at


class CorruptionInjector:
    """Applies reproducible corruption scenarios to multivariate sensor data.

    Designed for C-MAPSS evaluation: applies corruption to test sequences
    while keeping training data clean. All corruptions are injected exclusively
    at evaluation time.
    """

    def __init__(self, n_sensors: int = 14, seed: int = 42):
        self.n_sensors = n_sensors
        self.seed = seed
        self.rng = np.random.RandomState(seed)

        # Track drift state per scenario (for gradual drift)
        self._drift_state = {}

    def create_noise_scenario(
        self,
        sensor_indices: List[int],
        severity: float = 0.5,
        start_cycle: int = 0,
        end_cycle: int = -1,
        restoration_at: Optional[int] = None,
        seed: int = 42,
    ) -> CorruptionScenario:
        """Create a Gaussian noise amplification scenario.

        Severity = multiplicative factor on baseline noise std.
        severity=0.1 → 10% increase; severity=1.0 → 100% increase (doubled σ)
        """
        return CorruptionScenario(
            deg_type=CorruptionType.GAUSSIAN_NOISE,
            sensor_indices=sensor_indices,
            severity=severity,
            start_cycle=start_cycle,
            end_cycle=end_cycle,
            restoration_at=restoration_at,
            seed=seed,
        )

    def create_drift_scenario(
        self,
        sensor_indices: List[int],
        drift_rate: float = 0.01,
        start_cycle: int = 0,
        end_cycle: int = -1,
        restoration_at: Optional[int] = None,
    ) -> CorruptionScenario:
        """Create a gradual bias drift scenario.

        Drift increases linearly: b(t) = drift_rate * (t - start_cycle)
        """
        return CorruptionScenario(
            deg_type=CorruptionType.BIAS_DRIFT,
            sensor_indices=sensor_indices,
            drift_rate=drift_rate,
            start_cycle=start_cycle,
            end_cycle=end_cycle,
            restoration_at=restoration_at,
        )

    def create_dropout_scenario(
        self,
        sensor_indices: List[int],
        dropout_period: int = 50,
        start_cycle: int = 0,
        end_cycle: int = -1,
        restoration_at: Optional[int] = None,
    ) -> CorruptionScenario:
        """Create a complete sensor dropout scenario.

        Sensor reads exactly 0 for dropout_period cycles starting at start_cycle.
        """
        return CorruptionScenario(
            deg_type=CorruptionType.MODALITY_DROPOUT,
            sensor_indices=sensor_indices,
            dropout_period=dropout_period,
            start_cycle=start_cycle,
            end_cycle=start_cycle + dropout_period if end_cycle == -1 else end_cycle,
            restoration_at=restoration_at,
        )

    def create_distribution_shift_scenario(
        self,
        sensor_indices: List[int],
        alpha: float = 0.5,
        beta: float = 0.1,
        start_cycle: int = 0,
        end_cycle: int = -1,
        restoration_at: Optional[int] = None,
    ) -> CorruptionScenario:
        """Create an affine distribution shift scenario.

        x' = alpha * x + beta
        alpha < 1 → amplitude compression; alpha > 1 → amplification
        beta > 0 → positive offset; beta < 0 → negative offset
        """
        return CorruptionScenario(
            deg_type=CorruptionType.DISTRIBUTION_SHIFT,
            sensor_indices=sensor_indices,
            affine_alpha=alpha,
            affine_beta=beta,
            start_cycle=start_cycle,
            end_cycle=end_cycle,
            restoration_at=restoration_at,
        )

    def create_restoration_scenario(
        self,
        sensor_indices: List[int],
        start_cycle: int = 0,
        restoration_at: int = 60,
    ) -> CorruptionScenario:
        """Create a sensor restoration scenario.

        Sensor returns to clean operation at restoration_at cycle.
        """
        return CorruptionScenario(
            deg_type=CorruptionType.RESTORATION,
            sensor_indices=sensor_indices,
            start_cycle=start_cycle,
            restoration_at=restoration_at,
        )

    def create_standard_scenarios(
        self,
        severity: str = 'medium',
    ) -> dict:
        """Create a standard set of evaluation scenarios for TIUL.

        These are the scenarios evaluated in the paper's experiments.
        Seeds and parameters are fixed for reproducibility.

        Args:
            severity: 'low', 'medium', or 'high'

        Returns:
            dict mapping scenario_name → list of CorruptionScenario
        """
        # Severity parameters
        params = {
            'low':    {'noise': 0.5,  'drift': 0.005, 'dropout': 20},
            'medium': {'noise': 1.0,  'drift': 0.02,  'dropout': 50},
            'high':   {'noise': 2.0,  'drift': 0.05,  'dropout': 100},
        }[severity]

        # Sensor groups (by physical measurement type in C-MAPPS, 0-indexed)
        temp_sensors = [0, 1]           # Temperatures (sensor_2, sensor_3)
        pressure_sensors = [2, 3, 4]    # Pressures (sensor_4, sensor_7, sensor_8)
        speed_sensors = [5, 6]          # Speeds (sensor_9, sensor_11)
        flow_sensors = [7, 8, 9]         # Flows (sensor_12, sensor_13, sensor_14)

        scenarios = {}

        # Scenario 1: Single sensor Gaussian noise
        scenarios['noise_single_s1'] = [
            self.create_noise_scenario([0], params['noise'], start_cycle=0, seed=100)
        ]

        # Scenario 2: Multi-sensor noise (temperature group)
        scenarios['noise_temp_group'] = [
            self.create_noise_scenario(temp_sensors, params['noise'], start_cycle=0, seed=200)
        ]

        # Scenario 3: Gradual drift on one sensor
        scenarios['drift_single'] = [
            self.create_drift_scenario([2], params['drift'], start_cycle=0)
        ]

        # Scenario 4: Multi-sensor drift (pressure group)
        scenarios['drift_pressure'] = [
            self.create_drift_scenario(pressure_sensors, params['drift'], start_cycle=0)
        ]

        # Scenario 5: Temporal dropout (single sensor)
        scenarios['dropout_single'] = [
            self.create_dropout_scenario([0], params['dropout'], start_cycle=0)
        ]

        # Scenario 6: Multi-sensor dropout
        scenarios['dropout_multimodal'] = [
            self.create_dropout_scenario(temp_sensors, params['dropout'], start_cycle=0)
        ]

        # Scenario 7: Noise + restoration event
        scenarios['noise_then_restore'] = [
            self.create_noise_scenario(
                [0], params['noise'],
                start_cycle=0,
                restoration_at=60,
                seed=300
            )
        ]

        # Scenario 8: Compound degradation (noise + drift simultaneously)
        scenarios['compound_noise_drift'] = [
            self.create_noise_scenario([0], params['noise'], start_cycle=0, seed=400),
            self.create_drift_scenario([1], params['drift'], start_cycle=0),
        ]

        # Scenario 9: Distribution shift
        scenarios['distribution_shift'] = [
            self.create_distribution_shift_scenario(
                speed_sensors, alpha=0.7, beta=0.05, start_cycle=0)
        ]

        # Scenario 10: Severe multi-modal dropout with restoration
        scenarios['severe_multimodal_restore'] = [
            self.create_dropout_scenario(
                temp_sensors + pressure_sensors[:1],
                params['dropout'],
                start_cycle=0,
                restoration_at=60
            )
        ]

        return scenarios

    def apply(
        self,
        x: np.ndarray,
        scenarios: List[CorruptionScenario],
        t: int,
        sigma_baseline: Optional[np.ndarray] = None,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Apply all active scenarios to a single observation.

        Args:
            x: (time_steps, n_sensors) — clean sensor sequence
            scenarios: list of CorruptionScenario objects
            t: current cycle index
            sigma_baseline: (n_sensors,) baseline noise std for noise injection

        Returns:
            x_degraded: (time_steps, n_sensors) — degraded sensor sequence
            degradation_mask: (n_sensors,) binary array, 1 = degraded
        """
        x_out = x.copy().astype(np.float64)
        degradation_mask = np.zeros(self.n_sensors, dtype=np.float32)

        if sigma_baseline is None:
            sigma_baseline = np.ones(self.n_sensors) * 0.01

        for scenario in scenarios:
            if not scenario.is_active(t):
                continue

            for idx in scenario.sensor_indices:
                if idx >= self.n_sensors:
                    continue

                degradation_mask[idx] = 1.0

                if scenario.deg_type == CorruptionType.GAUSSIAN_NOISE:
                    noise_std = sigma_baseline[idx] * scenario.severity
                    rng = np.random.RandomState(scenario.seed + t + idx)
                    noise = rng.normal(0, noise_std, size=x_out.shape[0])
                    x_out[:, idx] += noise

                elif scenario.deg_type == CorruptionType.BIAS_DRIFT:
                    drift = scenario.drift_rate * (t - scenario.start_cycle)
                    x_out[:, idx] += drift

                elif scenario.deg_type == CorruptionType.MODALITY_DROPOUT:
                    x_out[:, idx] = 0.0

                elif scenario.deg_type == CorruptionType.DISTRIBUTION_SHIFT:
                    x_out[:, idx] = (
                        scenario.affine_alpha * x_out[:, idx] + scenario.affine_beta
                    )

                elif scenario.deg_type == CorruptionType.RESTORATION:
                    # Restoration: sensor returns to clean operation
                    # TIUL's restoration recovery protocol handles this transition
                    pass  # No corruption applied; signal is clean

        return x_out.astype(np.float32), degradation_mask

    def inject_into_sequences(
        self,
        X_seqs: np.ndarray,
        scenarios: dict,
        y_true: np.ndarray,
        sigma_baseline: np.ndarray,
        window: int = 30,
    ) -> dict:
        """Inject corruptions into all test sequences across all scenarios.

        Args:
            X_seqs: (n_units, window, n_sensors) — all test sequences
            scenarios: dict of scenario_name → list of CorruptionScenario
            y_true: (n_units,) — true RUL values
            sigma_baseline: (n_sensors,) baseline noise std
            window: sequence window length

        Returns:
            results dict with per-scenario metrics
        """
        results = {}
        n_total = len(X_seqs)

        # Evaluate under each degradation scenario
        for scenario_name, scenario_list in scenarios.items():
            preds_degraded = []
            for i, x_seq in enumerate(X_seqs):
                x_deg, _ = self.apply(x_seq, scenario_list, t=i, sigma_baseline=sigma_baseline)
                x_tensor = torch.tensor(x_deg[None], dtype=torch.float32)
                # Forward pass would happen in evaluation script
                # For now, record that this scenario was applied
                preds_degraded.append(float(np.mean(x_deg[:, 0])))  # placeholder

            # Compute metrics vs true RUL
            valid_len = min(len(preds_degraded), len(y_true))
            # metrics = compute_metrics(
            #     np.array(preds_degraded[:valid_len], dtype=np.float32),
            #     y_true[:valid_len]
            # )

            results[scenario_name] = {
                'n_evaluated': valid_len,
                'scenarios_applied': len(scenario_list),
            }

        return results


# ============================================================
# Utility Functions
# ============================================================

def get_sensor_names() -> List[str]:
    """Return the 14 active sensor names in TIUL convention."""
    return [f's{i+1}' for i in range(14)]


def get_cmapss_sensor_indices() -> List[int]:
    """Return the C-MAPSS sensor indices (1-based)."""
    return C_MAPSS_ACTIVE_SENSORS


# Example usage
if __name__ == '__main__':
    # Quick test
    print("TIUL Data Module loaded")
    print(f"Active sensors: {len(ACTIVE_SENSOR_MAP)}")
    print(f"Sensor map: {ACTIVE_SENSOR_MAP}")

    # Create injector and scenarios
    inj = CorruptionInjector(n_sensors=14, seed=42)
    scenarios = inj.create_standard_scenarios(severity='medium')
    print(f"Created {len(scenarios)} standard scenarios")
    print(f"Scenario keys: {list(scenarios.keys())[:5]}...")