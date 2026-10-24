
"""
TIUL Backbone Model
===================
Temporal encoder for multivariate sensor time series (C-MAPSS).
Produces per-sensor latent representations and a fused prediction.

Key design principles (lessons from SHELL audit):
1. Weight-invariant RUL head: pretrained with Dirichlet weight sampling + sensor dropout (Stage B)
2. Lightweight adapter (updated by TIUL)
3. Frozen backbone and RUL head during test-time adaptation
4. Cross-modal reconstructor for missing sensor latents
5. Proper uncertainty estimation for regression (via ensemble of adapter perturbations)

Architecture:
- Per-sensor 1D CNN encoder (sensor-specific feature extraction)
- Cross-sensor attention (learns inter-sensor relationships)
- Weight-invariant RUL prediction head
- Adapter bottleneck (updated during TTA)
- Cross-modal reconstructor

Sensors are treated as separate modalities to enable per-modality
health estimation, masking, and reconstruction in TIUL.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, List
import numpy as np


# ============================================================
# SensorEncoder: Lightweight 1D CNN per sensor channel
# ============================================================

class SensorEncoder(nn.Module):
    """Lightweight 1D CNN encoder for a single sensor channel.

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


# ============================================================
# CrossSensorAttention: Learns inter-sensor relationships
# ============================================================

class CrossSensorAttention(nn.Module):
    """Attention mechanism that learns weighted relationships between sensor latents.

    Allows the model to learn which sensors are most relevant for the current
    operating condition and degradation state. This is used for fusion and
    is modulation-invariant.
    """

    def __init__(self, latent_dim: int = 32, n_sensors: int = 14):
        super().__init__()
        self.latent_dim = latent_dim
        self.n_sensors = n_sensors

        # Q, K, V projections
        self.q_proj = nn.Linear(latent_dim, latent_dim)
        self.k_proj = nn.Linear(latent_dim, latent_dim)
        self.v_proj = nn.Linear(latent_dim, latent_dim)

        # Output projection
        self.out_proj = nn.Linear(latent_dim, latent_dim)

        # Scale factor
        self.scale = latent_dim ** -0.5

        # Sensor bias (learned per-sensor importance)
        self.sensor_bias = nn.Parameter(torch.zeros(n_sensors))

    def forward(self, z_per_sensor: List[torch.Tensor]) -> torch.Tensor:
        """
        Args:
            z_per_sensor: list of (batch, latent_dim) per sensor latents

        Returns:
            z_fused: (batch, latent_dim) — attention-weighted combination
        """
        # Stack latents: (batch, n_sensors, latent_dim)
        z_stack = torch.stack(z_per_sensor, dim=1)

        # Project Q, K, V
        Q = self.q_proj(z_stack)
        K = self.k_proj(z_stack)
        V = self.v_proj(z_stack)

        # Attention scores: (batch, n_sensors, n_sensors)
        attn_scores = torch.bmm(Q, K.transpose(-2, -1)) * self.scale

        # Add sensor bias
        attn_scores = attn_scores + self.sensor_bias.unsqueeze(0).unsqueeze(0)

        # Softmax
        attn_weights = F.softmax(attn_scores, dim=-1)

        # Apply to values
        attended = torch.bmm(attn_weights, V)  # (batch, n_sensors, latent_dim)
        attended = self.out_proj(attended)

        # Output: mean across sensors (attention provides inter-sensor context)
        z_fused = attended.mean(dim=1)  # (batch, latent_dim)

        return z_fused


# ============================================================
# Adapter: Lightweight bottleneck updated during TTA
# ============================================================

class Adapter(nn.Module):
    """Lightweight adapter bottleneck updated by TIUL during test-time adaptation.

    Design:
    - Bottleneck: sensor_latent_dim → adapter_dim → sensor_latent_dim
    - Initialized as identity (additive residual with zero perturbation)
    - Updated via projected gradient descent with bounded update constraint

    The adapter is the only component updated during test-time adaptation.
    This follows the PEFT paradigm.
    """

    def __init__(self, sensor_latent_dim: int = 32, adapter_dim: int = 8):
        super().__init__()
        self.sensor_latent_dim = sensor_latent_dim
        self.adapter_dim = adapter_dim

        self.adapter = nn.Sequential(
            nn.Linear(sensor_latent_dim, adapter_dim),
            nn.GELU(),
            nn.Linear(adapter_dim, sensor_latent_dim),
        )

        # Initialize as identity (zero perturbation)
        nn.init.zeros_(self.adapter[0].weight)
        nn.init.zeros_(self.adapter[0].bias)
        nn.init.zeros_(self.adapter[2].weight)
        nn.init.zeros_(self.adapter[2].bias)

        self._adapter_init_params = None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.adapter(x)

    def initialize_params(self):
        """Store initial adapter params for bounded-update projection."""
        self._adapter_init_params = {
            name: param.data.clone()
            for name, param in self.named_parameters()
        }

    def adapter_drift(self) -> float:
        """Compute Frobenius distance of adapter from initialization."""
        if self._adapter_init_params is None:
            return 0.0
        total_drift = 0.0
        for name, param in self.named_parameters():
            init_p = self._adapter_init_params[name].to(param.device)
            total_drift += (param.data - init_p).norm(p='fro').item() ** 2
        return total_drift ** 0.5

    def project_adapter(self, delta_max: float):
        """Project adapter to satisfy ||phi - phi_0||_F <= delta_max."""
        if self._adapter_init_params is None:
            return
        current_drift = self.adapter_drift()
        if current_drift > delta_max and current_drift > 1e-8:
            scale = delta_max / current_drift
            for name, param in self.named_parameters():
                init_p = self._adapter_init_params[name].to(param.device)
                deviation = param.data - init_p
                param.data = init_p + scale * deviation


# ============================================================
# MultiSensorBackbone: Core backbone with per-sensor encoding
# ============================================================

class MultiSensorBackbone(nn.Module):
    """Multimodal backbone that encodes each sensor independently,
    then fuses via attention.

    Input:  (batch, time_steps, n_sensors)
    Output: (batch, latent_dim), List[(batch, latent_dim)] per-sensor

    KEY TIUL FIX: This backbone supports weight-invariant fusion through
    Stage B pretraining (Dirichlet weights + sensor dropout). The RUL head
    is trained to be invariant to which fusion weights are applied, preventing
    the latent scale distortion that broke SHELL.
    """

    def __init__(
        self,
        n_sensors: int = 14,
        time_steps: int = 30,
        sensor_latent_dim: int = 32,
    ):
        super().__init__()
        self.n_sensors = n_sensors
        self.time_steps = time_steps
        self.sensor_latent_dim = sensor_latent_dim

        # Per-sensor encoders (independent, not shared)
        self.sensor_encoders = nn.ModuleList([
            SensorEncoder(time_steps, sensor_latent_dim)
            for _ in range(n_sensors)
        ])

        # Cross-sensor attention for fusion
        self.cross_attention = CrossSensorAttention(
            latent_dim=sensor_latent_dim,
            n_sensors=n_sensors,
        )

        # Fusion head (weight-invariant due to Stage B pretraining)
        self.fusion_head = nn.Sequential(
            nn.Linear(sensor_latent_dim, 64),
            nn.GELU(),
        )

    def encode_sensors(
        self,
        x: torch.Tensor,
        reliability_weights: Optional[torch.Tensor] = None,
    ) -> tuple:
        """Encode each sensor independently and fuse via attention.

        Args:
            x: (batch, time_steps, n_sensors)
            reliability_weights: optional (n_sensors,) for weighted fusion

        Returns:
            z_fused: (batch, 64) — fused latent representation
            z_per_sensor: list of (batch, sensor_latent_dim) per sensor
        """
        # Encode each sensor
        z_list = []
        for i, encoder in enumerate(self.sensor_encoders):
            z_i = encoder(x[:, :, i])  # (batch, sensor_latent_dim)
            z_list.append(z_i)

        # Cross-sensor attention
        z_attn = self.cross_attention(z_list)  # (batch, sensor_latent_dim)

        # Apply reliability weights if provided (TIUL REWEIGHT action effect)
        # When weights are provided, blend attention-based and weighted fusion
        if reliability_weights is not None:
            w = reliability_weights.to(z_attn.device)
            w = w / (w.sum() + 1e-8)

            # Weighted combination of sensor latents
            z_weighted = torch.zeros_like(z_attn)
            for i, z_i in enumerate(z_list):
                z_weighted = z_weighted + w[i] * z_i

            # Blend: attention provides context, weights adjust for reliability
            z_fused = 0.5 * z_attn + 0.5 * z_weighted
        else:
            z_fused = z_attn

        # Fusion head — weight-invariant due to Stage B training
        z_fused = self.fusion_head(z_fused)

        return z_fused, z_list


# ============================================================
# CrossModalReconstructor: Lightweight reconstructor for missing sensor latents
# ============================================================

class CrossModalReconstructor(nn.Module):
    """Lightweight cross-modal reconstructor for missing sensor latents.

    Trained offline using masked autoencoder objective:
    given a subset of sensor latents (the "healthy" ones),
    reconstruct the masked sensor's latent representation.

    Architecture: simple MLP (no generative model to keep cost low)

    Input:  (batch, n_sensors * latent_dim) — concatenated latents (masked = zeros)
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

        self.encoder_in = nn.Sequential(
            nn.Linear(n_sensors * latent_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.GELU(),
        )

        self.decoder_out = nn.Linear(hidden_dim // 2, latent_dim)

        nn.init.xavier_uniform_(self.encoder_in[0].weight, gain=1.0)
        nn.init.zeros_(self.encoder_in[0].bias)
        nn.init.xavier_uniform_(self.encoder_in[2].weight, gain=1.0)
        nn.init.zeros_(self.encoder_in[2].bias)
        nn.init.xavier_uniform_(self.decoder_out.weight, gain=nn.init.calculate_gain('linear'))
        nn.init.zeros_(self.decoder_out.bias)

    def forward(
        self,
        z_per_sensor: List[torch.Tensor],
        mask: torch.Tensor,
    ) -> torch.Tensor:
        """Reconstruct the representation for masked sensors.

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

        z_cat = torch.cat(masked_z, dim=-1)  # (batch, n_sensors * latent_dim)
        h = self.encoder_in(z_cat)
        z_recon = self.decoder_out(h)
        return z_recon


# ============================================================
# RULHead: Predict Remaining Useful Life
# ============================================================

class RULHead(nn.Module):
    """Regression head for RUL prediction.

    Weight-invariant by design: trained with Dirichlet-sampled weights
    and sensor dropout during Stage B pretraining, ensuring that the head
    produces correct predictions regardless of which fusion weights are used.
    """

    def __init__(self, input_dim: int = 64, hidden_dim: int = 32):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, 16),
            nn.GELU(),
            nn.Linear(16, 1),
        )
        nn.init.xavier_uniform_(self.net[0].weight, gain=1.0)
        nn.init.zeros_(self.net[0].bias)
        nn.init.xavier_uniform_(self.net[3].weight, gain=1.0)
        nn.init.zeros_(self.net[3].bias)
        # Final layer initialized to small values for stable regression start
        nn.init.normal_(self.net[5].weight, std=0.01)
        nn.init.zeros_(self.net[5].bias)

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        return self.net(z)


# ============================================================
# TIULModel: Complete model assembly
# ============================================================

class TIULModel(nn.Module):
    """Complete TIUL model: backbone + adapter + RUL head + reconstructor.

    Design principles:
    - Backbone (frozen after pretraining): MultiSensorBackbone
    - Adapter (updated by TIUL): lightweight bottleneck on fused representation
    - RUL Head (frozen after pretraining): weight-invariant due to Stage B
    - CrossModalReconstructor (trained offline): latent recovery

    The adapter is the only component updated during test-time adaptation.
    Stage B corruption-augmented pretraining ensures the RUL head is
    weight-invariant — a critical fix from the SHELL audit.
    """

    def __init__(
        self,
        n_sensors: int = 14,
        time_steps: int = 30,
        sensor_latent_dim: int = 32,
        adapter_dim: int = 8,
        fusion_dim: int = 64,
    ):
        super().__init__()
        self.n_sensors = n_sensors
        self.time_steps = time_steps
        self.sensor_latent_dim = sensor_latent_dim
        self.adapter_dim = adapter_dim
        self.fusion_dim = fusion_dim

        # Backbone (frozen during test-time adaptation)
        self.backbone = MultiSensorBackbone(
            n_sensors=n_sensors,
            time_steps=time_steps,
            sensor_latent_dim=sensor_latent_dim,
        )

        # Lightweight adapter (updated by TIUL)
        self.adapter = Adapter(
            sensor_latent_dim=fusion_dim,
            adapter_dim=adapter_dim,
        )
        self.adapter.initialize_params()
        self.adapter.initialize_params()

        # RUL prediction head (frozen during test-time adaptation)
        self.rul_head = RULHead(input_dim=fusion_dim, hidden_dim=32)

        # Cross-modal reconstructor for missing sensor latents
        self.reconstructor = CrossModalReconstructor(
            n_sensors=n_sensors,
            latent_dim=sensor_latent_dim,
        )

        # Track initial adapter params for bounded-update projection
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
        """Compute Frobenius distance of adapter from initialization."""
        if self._adapter_init_params is None:
            return 0.0
        total_drift = 0.0
        for name, param in self.adapter.named_parameters():
            init_p = self._adapter_init_params[name].to(param.device)
            total_drift += (param.data - init_p).norm(p='fro').item() ** 2
        return total_drift ** 0.5

    def project_adapter(self, delta_max: float):
        """Project adapter to satisfy ||phi - phi_0||_F <= delta_max."""
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
        """Forward pass.

        Args:
            x: (batch, time_steps, n_sensors)
            reliability_weights: (n_sensors,) optional

        Returns:
            rul_pred: (batch, 1)
            z_fused: (batch, fusion_dim) — pre-adapter
            z_adapted: (batch, fusion_dim) — post-adapter
            z_per_sensor: list of per-sensor latents
        """
        # Backbone forward (frozen)
        z_fused, z_per_sensor = self.backbone.encode_sensors(
            x,
            reliability_weights=reliability_weights,
        )

        # Adapter (residual connection)
        z_adapted = z_fused + self.adapter(z_fused)

        # RUL prediction (frozen head)
        rul_pred = self.rul_head(z_adapted)

        return rul_pred, z_fused, z_adapted, z_per_sensor

    def extract_features(self, x: torch.Tensor) -> dict:
        """Extract intermediate features for analysis."""
        with torch.no_grad():
            rul_pred, z_fused, z_adapted, z_per_sensor = self(x)

        return {
            'rul_pred': rul_pred.squeeze(),
            'z_fused': z_fused,
            'z_adapted': z_adapted,
            'z_per_sensor': z_per_sensor,
            'adapter_drift': self.adapter_drift(),
        }


if __name__ == '__main__':
    # Quick sanity test
    model = TIULModel(n_sensors=14, time_steps=30, sensor_latent_dim=32, adapter_dim=8)
    x = torch.randn(4, 30, 14)
    rul_pred, z_fused, z_adapted, z_per_sensor = model(x)

    print(f"RUL pred shape: {rul_pred.shape}")
    print(f"z_fused shape: {z_fused.shape}")
    print(f"z_adapted shape: {z_adapted.shape}")
    print(f"z_per_sensor: list of {len(z_per_sensor)} tensors, each shape {z_per_sensor[0].shape}")
    print(f"Adapter param count: {sum(p.numel() for p in model.adapter.parameters())}")
    print(f"Total model param count: {sum(p.numel() for p in model.parameters())}")