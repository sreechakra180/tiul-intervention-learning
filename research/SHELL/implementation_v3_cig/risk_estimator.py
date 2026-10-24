"""
Action-Conditioned Risk Estimator for Counterfactual Intervention Governor (CIG)
===============================================================================
Estimates expected absolute prediction error:
    R_hat(a) = E[ |y_pred(a) - y_true| ]
Given input features:
    phi(x, h, a, pred(a), pred(0))
Uses a lightweight calibrated neural regressor with non-negative Softplus output.
"""

import os
import torch
import torch.nn as nn
import numpy as np
from typing import Dict, Tuple, Optional

class ActionRiskMLP(nn.Module):
    def __init__(self, in_features: int = 18, hidden_dim: int = 64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_features, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.LayerNorm(hidden_dim // 2),
            nn.GELU(),
            nn.Linear(hidden_dim // 2, 1),
            nn.Softplus() # Risks are strictly non-negative
        )
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)

class CalibratedActionRiskEstimator:
    def __init__(self, in_features: int = 18, hidden_dim: int = 64, device: str = 'cpu'):
        self.model = ActionRiskMLP(in_features, hidden_dim).to(device)
        self.device = device
        self.scaler_mean = None
        self.scaler_std = None
        
    def fit(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: Optional[np.ndarray] = None,
        y_val: Optional[np.ndarray] = None,
        epochs: int = 40,
        batch_size: int = 128,
        lr: float = 1e-3
    ) -> Dict[str, float]:
        """
        Trains risk estimator with Huber Loss on actual absolute errors.
        """
        # Standardize features
        self.scaler_mean = np.mean(X_train, axis=0)
        self.scaler_std = np.std(X_train, axis=0) + 1e-6
        
        X_norm = (X_train - self.scaler_mean) / self.scaler_std
        y_t = torch.tensor(y_train, dtype=torch.float32).unsqueeze(-1)
        X_t = torch.tensor(X_norm, dtype=torch.float32)
        
        dataset = torch.utils.data.TensorDataset(X_t, y_t)
        loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)
        
        optimizer = torch.optim.AdamW(self.model.parameters(), lr=lr, weight_decay=1e-4)
        criterion = nn.HuberLoss(delta=2.0)
        
        self.model.train()
        for epoch in range(epochs):
            total_loss = 0.0
            n_b = 0
            for bx, by in loader:
                bx, by = bx.to(self.device), by.to(self.device)
                optimizer.zero_grad()
                pred = self.model(bx)
                loss = criterion(pred, by)
                loss.backward()
                optimizer.step()
                total_loss += loss.item()
                n_b += 1
                
        metrics = {'train_loss': total_loss / n_b}
        
        if X_val is not None and y_val is not None:
            self.model.eval()
            with torch.no_grad():
                X_v_norm = (X_val - self.scaler_mean) / self.scaler_std
                preds_val = self.model(torch.tensor(X_v_norm, dtype=torch.float32).to(self.device)).cpu().numpy().flatten()
                val_mae = float(np.mean(np.abs(preds_val - y_val)))
                val_rmse = float(np.sqrt(np.mean((preds_val - y_val)**2)))
                corr = float(np.corrcoef(preds_val, y_val)[0, 1]) if np.std(preds_val) > 1e-6 else 0.0
                metrics.update({
                    'val_mae': val_mae,
                    'val_rmse': val_rmse,
                    'pearson_corr': corr
                })
                
        return metrics
        
    @torch.no_grad()
    def predict_risk(self, feature_vector: np.ndarray) -> float:
        """
        Inference: estimates risk R_hat(a) for a single candidate action.
        """
        self.model.eval()
        if feature_vector.ndim == 1:
            feature_vector = feature_vector[None, :]
        feat_norm = (feature_vector - self.scaler_mean) / self.scaler_std
        feat_t = torch.tensor(feat_norm, dtype=torch.float32, device=self.device)
        risk = self.model(feat_t).item()
        return float(risk)
        
    def save(self, filepath: str):
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'scaler_mean': self.scaler_mean,
            'scaler_std': self.scaler_std
        }, filepath)
        
    def load(self, filepath: str):
        ckpt = torch.load(filepath, map_location=self.device)
        self.model.load_state_dict(ckpt['model_state_dict'])
        self.scaler_mean = ckpt['scaler_mean']
        self.scaler_std = ckpt['scaler_std']
        self.model.eval()
