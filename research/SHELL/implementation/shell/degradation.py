"""
SHELL Degradation Injection Module
===================================
Implements controllable, reproducible sensor degradation scenarios
for experimental evaluation.

All degradation scenarios are applied ONLY during evaluation.
Training uses clean data. This prevents test-time information leakage.

Degradation types:
    1. Gaussian noise amplification (increased σ)
    2. Additive bias / gradual drift
    3. Temporal dropout (windows of zero readings)
    4. Complete sensor failure (modality dropout)
    5. Distribution shift (affine transform)
    6. Recovery / restoration event

Usage:
    injector = DegradationInjector(n_sensors=14, seed=42)
    scenario = injector.create_scenario('noise', sensor_idx=3, severity=0.5)
    x_degraded, deg_mask = injector.apply(x_clean, scenario, t)
"""

import numpy as np
from dataclasses import dataclass, field
from typing import Optional, List, Tuple
from enum import Enum


class DegradationType(Enum):
    NONE            = 'none'
    GAUSSIAN_NOISE  = 'gaussian_noise'
    BIAS_DRIFT      = 'bias_drift'
    TEMPORAL_DROPOUT= 'temporal_dropout'
    MODALITY_DROPOUT= 'modality_dropout'
    DISTRIBUTION_SHIFT = 'distribution_shift'
    RESTORATION     = 'restoration'


@dataclass
class DegradationScenario:
    """
    Describes a single degradation scenario applied to specific sensors.
    
    Fields:
        deg_type:       Type of degradation
        sensor_indices: List of sensor indices affected
        severity:       Severity level (interpretation depends on deg_type)
        start_cycle:    Cycle at which degradation begins (inclusive)
        end_cycle:      Cycle at which degradation ends (-1 = never)
        restoration_at: If not None, cycle at which sensor recovers
        drift_rate:     Rate of drift increase per cycle (for BIAS_DRIFT)
        dropout_period: Length of dropout windows (for TEMPORAL_DROPOUT)
        affine_alpha:   Scaling factor (for DISTRIBUTION_SHIFT)
        affine_beta:    Offset (for DISTRIBUTION_SHIFT)
        seed:           Random seed for reproducible noise
    """
    deg_type: DegradationType = DegradationType.NONE
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
        """Returns True if degradation is active at cycle t."""
        if t < self.start_cycle:
            return False
        if self.end_cycle != -1 and t >= self.end_cycle:
            return False
        if self.restoration_at is not None and t >= self.restoration_at:
            return False
        return True
    
    def is_restored(self, t: int) -> bool:
        """Returns True if sensor has been restored at cycle t."""
        if self.restoration_at is None:
            return False
        return t >= self.restoration_at


class DegradationInjector:
    """
    Applies reproducible degradation scenarios to multivariate sensor data.
    
    Designed for C-MAPSS evaluation: applies degradation to test sequences
    while keeping training data clean.
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
    ) -> DegradationScenario:
        """
        Create a Gaussian noise amplification scenario.
        
        Severity = multiplicative factor on baseline noise std.
        severity=0.1 → 10% increase; severity=1.0 → 100% increase (doubled σ)
        """
        return DegradationScenario(
            deg_type=DegradationType.GAUSSIAN_NOISE,
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
    ) -> DegradationScenario:
        """
        Create a gradual bias drift scenario.
        
        Drift increases linearly: b(t) = drift_rate * (t - start_cycle)
        """
        return DegradationScenario(
            deg_type=DegradationType.BIAS_DRIFT,
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
    ) -> DegradationScenario:
        """
        Create a complete sensor dropout scenario.
        
        Sensor reads exactly 0 for dropout_period cycles starting at start_cycle.
        """
        return DegradationScenario(
            deg_type=DegradationType.MODALITY_DROPOUT,
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
    ) -> DegradationScenario:
        """
        Create an affine distribution shift scenario.
        
        x' = alpha * x + beta
        alpha < 1 → amplitude compression; alpha > 1 → amplification
        beta > 0 → positive offset; beta < 0 → negative offset
        """
        return DegradationScenario(
            deg_type=DegradationType.DISTRIBUTION_SHIFT,
            sensor_indices=sensor_indices,
            affine_alpha=alpha,
            affine_beta=beta,
            start_cycle=start_cycle,
            end_cycle=end_cycle,
            restoration_at=restoration_at,
        )
    
    def apply(
        self,
        x: np.ndarray,
        scenarios: List[DegradationScenario],
        t: int,
        sigma_baseline: Optional[np.ndarray] = None,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Apply all active scenarios to a single observation.
        
        Args:
            x: (time_steps, n_sensors) — clean sensor sequence
            scenarios: list of DegradationScenario objects
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
                
                if scenario.deg_type == DegradationType.GAUSSIAN_NOISE:
                    noise_std = sigma_baseline[idx] * scenario.severity
                    rng = np.random.RandomState(scenario.seed + t + idx)
                    noise = rng.normal(0, noise_std, size=x_out.shape[0])
                    x_out[:, idx] += noise
                
                elif scenario.deg_type == DegradationType.BIAS_DRIFT:
                    drift = scenario.drift_rate * (t - scenario.start_cycle)
                    x_out[:, idx] += drift
                
                elif scenario.deg_type == DegradationType.MODALITY_DROPOUT:
                    x_out[:, idx] = 0.0
                
                elif scenario.deg_type == DegradationType.DISTRIBUTION_SHIFT:
                    x_out[:, idx] = (
                        scenario.affine_alpha * x_out[:, idx] + scenario.affine_beta
                    )
                
                elif scenario.deg_type == DegradationType.RESTORATION:
                    # Restoration: sensor returns to clean operation
                    # SHELL's restoration recovery protocol handles this transition
                    pass  # No degradation applied; signal is clean
        
        return x_out.astype(np.float32), degradation_mask
    
    def create_standard_scenarios(
        self,
        severity: str = 'medium',
    ) -> dict:
        """
        Create a standard set of evaluation scenarios for SHELL.
        
        These are the scenarios evaluated in the paper's experiments.
        Seeds and parameters are fixed for reproducibility.
        
        Args:
            severity: 'low', 'medium', or 'high'
        
        Returns:
            dict mapping scenario_name → list of DegradationScenario
        """
        # Severity parameters
        params = {
            'low':    {'noise': 0.5,  'drift': 0.005, 'dropout': 20},
            'medium': {'noise': 1.0,  'drift': 0.02,  'dropout': 50},
            'high':   {'noise': 2.0,  'drift': 0.05,  'dropout': 100},
        }[severity]
        
        # Sensor groups (by physical measurement type in C-MAPSS)
        temp_sensors = [0, 1]          # Temperatures (sensor_2, sensor_3)
        pressure_sensors = [2, 3, 4]   # Pressures (sensor_4, sensor_7, sensor_8)
        speed_sensors = [5, 6]         # Speeds (sensor_9, sensor_11)
        flow_sensors = [7, 8, 9]       # Flows (sensor_12, sensor_13, sensor_14)
        
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
