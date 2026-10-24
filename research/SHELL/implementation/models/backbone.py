"""
SHELL Backbone Model
====================
Temporal encoder for multivariate sensor time series (C-MAPSS).
Produces per-sensor latent representations and a fused prediction.

Architecture:
- Per-sensor 1D CNN encoder (sensor-specific feature extraction)
- Cross-sensor attention (learns inter-sensor relationships)
- RUL prediction head

Sensors are treated as separate modalities to enable per-modality
health estimation, masking, and reconstruction in SHELL.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, List


class SensorEncoder(nn.Module):
    """
    Lightweight 1D CNN encoder for a single sensor channel.
    
    Input:  (batch, time_steps, 1) — single sensor time series
    Output: (batch, latent_dim)   — sensor latent representation
    """
    
    def __init__(self, time_steps: int = 30, latent_dim: int = 32):
        super().__init__()
        self.time_steps = time_steps
        self.latent_dim = latent_dim
        
        self.conv_layers = nn.Sequential(
            nn.Conv1d(1, 16, kernel_size=5, padding=2),
            nn.GELU(),
            nn.Conv1d(16, 32, kernel_size=3, padding=1),
            nn.GELU(),
            nn.AdaptiveAvgPool1d(8),  # → (batch, 32, 8)
        )
        
        self.projection = nn.Sequential(
            nn.Linear(32 * 8, latent_dim),
            nn.LayerNorm(latent_dim),
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (batch, time_steps, 1) or (batch, time_steps) single sensor
        Returns:
            z: (batch, latent_dim)
        """
        if x.dim() == 2:
            x = x.unsqueeze(-1)
        # (batch, time_steps, 1) → (batch, 1, time_steps) for Conv1d
        x = x.permute(0, 2, 1)
        h = self.conv_layers(x)        # (batch, 32, 8)
        h = h.view(h.size(0), -1)     # (batch, 256)
        z = self.projection(h)         # (batch, latent_dim)
        return z


class MultiSensorBackbone(nn.Module):
    """
    Multimodal backbone that encodes each sensor independently,
    then fuses via weighted attention.
    
    This design supports per-sensor masking and latent reconstruction
    as required by SHELL.
    
    Input:  (batch, time_steps, n_sensors)
    Output: (batch, latent_dim), List[(batch, latent_dim)] per-sensor
    """
    
    def __init__(
        self,
        n_sensors: int = 14,
        time_steps: int = 30,
        sensor_latent_dim: int = 32,
        fusion_dim: int = 64,
    ):
        super().__init__()
        self.n_sensors = n_sensors
        self.time_steps = time_steps
        self.sensor_latent_dim = sensor_latent_dim
        self.fusion_dim = fusion_dim
        
        # Per-sensor encoders (independent, not shared)
        self.sensor_encoders = nn.ModuleList([
            SensorEncoder(time_steps, sensor_latent_dim)
            for _ in range(n_sensors)
        ])
        
        # Fusion MLP
        self.fusion_head = nn.Sequential(
            nn.Linear(sensor_latent_dim, fusion_dim),
            nn.GELU(),
        )
        
    def encode_sensors(
        self,
        x: torch.Tensor,
        reliability_weights: Optional[torch.Tensor] = None,
    ) -> tuple:
        """
        Encode each sensor independently.
        
        Args:
            x: (batch, time_steps, n_sensors)
            reliability_weights: (n_sensors,) weights for fusion [default: uniform]
        
        Returns:
            z_fused: (batch, sensor_latent_dim) weighted average of sensor latents
            z_per_sensor: list of (batch, sensor_latent_dim) per sensor
        """
        batch_size = x.size(0)
        z_list = []
        
        for i, encoder in enumerate(self.sensor_encoders):
            z_i = encoder(x[:, :, i])  # (batch, sensor_latent_dim)
            z_list.append(z_i)
        
        # Stack: (batch, n_sensors, sensor_latent_dim)
        z_stack = torch.stack(z_list, dim=1)
        
        # Apply reliability weights
        if reliability_weights is None:
            reliability_weights = torch.ones(
                self.n_sensors, device=x.device) / self.n_sensors
        
        # Normalize weights
        w = reliability_weights.to(x.device)
        w = w / (w.sum() + 1e-8)
        
        # Weighted sum: (batch, sensor_latent_dim)
        z_fused = (z_stack * w.view(1, -1, 1)).sum(dim=1)
        
        return z_fused, z_list
    
    def forward(
        self,
        x: torch.Tensor,
        reliability_weights: Optional[torch.Tensor] = None,
    ) -> tuple:
        """
        Args:
            x: (batch, time_steps, n_sensors)
            reliability_weights: (n_sensors,) optional
        Returns:
            z_fused: (batch, sensor_latent_dim)
            z_per_sensor: list of per-sensor latents
        """
        return self.encode_sensors(x, reliability_weights)


class RULHead(nn.Module):
    """
    Regression head for Remaining Useful Life (RUL) prediction.
    
    Input:  (batch, fusion_dim)
    Output: (batch, 1) — predicted RUL value
    """
    
    def __init__(self, input_dim: int = 32, hidden_dim: int = 64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim, 32),
            nn.GELU(),
            nn.Linear(32, 1),
        )
    
    def forward(self, z: torch.Tensor) -> torch.Tensor:
        """
        Args:
            z: (batch, input_dim)
        Returns:
            rul: (batch, 1)
        """
        return self.net(z)


class SHELLModel(nn.Module):
    """
    Complete SHELL model: backbone + adapter + RUL head.
    
    Design principles:
    - Backbone (frozen after pretraining): MultiSensorBackbone
    - Adapter (updated by SHELL): lightweight bottleneck on fused representation
    - RUL Head (frozen after pretraining): RULHead
    
    The adapter is the only component updated during test-time adaptation.
    This follows the PEFT paradigm established in the hardware-aware CL literature.
    """
    
    def __init__(
        self,
        n_sensors: int = 14,
        time_steps: int = 30,
        sensor_latent_dim: int = 32,
        adapter_dim: int = 16,
    ):
        super().__init__()
        self.n_sensors = n_sensors
        self.time_steps = time_steps
        self.sensor_latent_dim = sensor_latent_dim
        self.adapter_dim = adapter_dim
        
        # Backbone (frozen during test-time adaptation)
        self.backbone = MultiSensorBackbone(
            n_sensors=n_sensors,
            time_steps=time_steps,
            sensor_latent_dim=sensor_latent_dim,
        )
        
        # Lightweight adapter (updated by SHELL)
        # Bottleneck: sensor_latent_dim → adapter_dim → sensor_latent_dim
        self.adapter = nn.Sequential(
            nn.Linear(sensor_latent_dim, adapter_dim),
            nn.GELU(),
            nn.Linear(adapter_dim, sensor_latent_dim),
        )
        # Initialize adapter as identity (additive residual)
        nn.init.zeros_(self.adapter[0].weight)
        nn.init.zeros_(self.adapter[0].bias)
        nn.init.zeros_(self.adapter[2].weight)
        nn.init.zeros_(self.adapter[2].bias)
        
        # RUL prediction head (frozen during test-time adaptation)
        self.rul_head = RULHead(sensor_latent_dim, hidden_dim=64)
        
        # Track adapter initial state for bounded-update projection
        self._adapter_init_params = None
    
    def freeze_backbone_and_head(self):
        """Freeze all parameters except the adapter."""
        for param in self.backbone.parameters():
            param.requires_grad = False
        for param in self.rul_head.parameters():
            param.requires_grad = False
        for param in self.adapter.parameters():
            param.requires_grad = True
        
        # Store initial adapter params for bounded-update projection
        self._adapter_init_params = {
            name: param.data.clone()
            for name, param in self.adapter.named_parameters()
        }
    
    def adapter_drift(self) -> float:
        """
        Compute Frobenius distance of adapter from its initialization.
        
        Used for the bounded-update projection and forgetting monitoring.
        Returns float value.
        """
        if self._adapter_init_params is None:
            return 0.0
        
        total_drift = 0.0
        for name, param in self.adapter.named_parameters():
            init_p = self._adapter_init_params[name].to(param.device)
            total_drift += (param.data - init_p).norm(p='fro').item() ** 2
        return total_drift ** 0.5
    
    def project_adapter(self, delta_max: float):
        """
        Project adapter parameters to satisfy the bounded-update constraint.
        
        If ||phi - phi_0||_F > delta_max, scale the deviation back.
        
        This implements Claim 1 from the problem formulation:
        ||Delta_phi||_F <= delta_max
        
        Note: This is applied to the cumulative drift, not per-step.
        A tighter per-step version would clip gradient norms instead.
        """
        if self._adapter_init_params is None:
            return
        
        current_drift = self.adapter_drift()
        if current_drift > delta_max and current_drift > 1e-8:
            scale = delta_max / current_drift
            for name, param in self.adapter.named_parameters():
                init_p = self._adapter_init_params[name].to(param.device)
                deviation = param.data - init_p
                param.data = init_p + scale * deviation
    
    def forward(
        self,
        x: torch.Tensor,
        reliability_weights: Optional[torch.Tensor] = None,
    ) -> tuple:
        """
        Forward pass.
        
        Args:
            x: (batch, time_steps, n_sensors)
            reliability_weights: (n_sensors,) optional
        
        Returns:
            rul_pred: (batch, 1)
            z_fused: (batch, sensor_latent_dim) — pre-adapter
            z_adapted: (batch, sensor_latent_dim) — post-adapter
            z_per_sensor: list of per-sensor latents
        """
        z_fused, z_per_sensor = self.backbone(x, reliability_weights)
        
        # Adapter (residual connection)
        z_adapted = z_fused + self.adapter(z_fused)
        
        # RUL prediction
        rul_pred = self.rul_head(z_adapted)
        
        return rul_pred, z_fused, z_adapted, z_per_sensor


class CrossModalReconstructor(nn.Module):
    """
    Lightweight cross-modal reconstructor for missing sensor latents.
    
    Trained offline using masked autoencoder objective:
    given a subset of sensor latents (the "healthy" ones),
    reconstruct the masked sensor's latent representation.
    
    Architecture: simple MLP (no generative model to keep cost low)
    
    Input:  (batch, n_active_sensors * latent_dim) — concatenated healthy latents
    Output: (batch, latent_dim)                    — reconstructed latent
    """
    
    def __init__(
        self,
        n_sensors: int = 14,
        latent_dim: int = 32,
        hidden_dim: int = 128,
    ):
        super().__init__()
        self.n_sensors = n_sensors
        self.latent_dim = latent_dim
        
        # We train one reconstructor per possible target sensor
        # For Phase 1, we train a single shared reconstructor
        # conditioned on a mask indicating which sensors are active
        self.encoder_in = nn.Sequential(
            nn.Linear(n_sensors * latent_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.GELU(),
        )
        
        self.decoder_out = nn.Linear(hidden_dim // 2, latent_dim)
    
    def forward(
        self,
        z_per_sensor: List[torch.Tensor],
        mask: torch.Tensor,
    ) -> torch.Tensor:
        """
        Reconstruct the representation for masked sensors.
        
        Args:
            z_per_sensor: list of (batch, latent_dim) — all sensor latents
            mask: (n_sensors,) binary tensor, 1 = available, 0 = to reconstruct
        
        Returns:
            z_reconstructed: (batch, latent_dim) — reconstructed aggregate
        """
        # Zero out masked sensors for reconstruction input
        masked_z = []
        for i, z_i in enumerate(z_per_sensor):
            if mask[i] > 0:
                masked_z.append(z_i)
            else:
                masked_z.append(torch.zeros_like(z_i))
        
        # Concatenate all sensor latents (masked ones are zeros)
        z_cat = torch.cat(masked_z, dim=-1)  # (batch, n_sensors * latent_dim)
        h = self.encoder_in(z_cat)
        z_recon = self.decoder_out(h)
        return z_recon
