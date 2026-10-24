"""
Train Corruption-Augmented & Sensor-Dropout Backbone (Stage B)
=============================================================
Trains a robust backbone on disjoint training engines (Units 1-80):
- Sensor dropout: randomly zeroes 1-2 sensor channels per batch
- Random Gaussian noise injection
- Validation tracking strictly on disjoint validation engines (Units 81-100)
- Checkpoint saved to checkpoints/robust_backbone_FD001_seed42.pt
"""

import os
import sys
import argparse
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from pathlib import Path

base_dir = Path(__file__).parent.parent
sys.path.insert(0, str(base_dir / 'implementation'))
sys.path.insert(0, str(base_dir / 'implementation_v2'))

from data_v2 import load_data_partitioned, create_sequences_by_engine, ACTIVE_SENSORS
from models.backbone import SHELLModel

def train_robust(epochs: int = 35, batch_size: int = 128, lr: float = 1e-3, seed: int = 42):
    torch.manual_seed(seed)
    np.random.seed(seed)
    device = 'cpu'
    
    data = load_data_partitioned(str(base_dir / 'implementation' / 'data' / 'raw'), train_units_max=80)
    
    # Create train and val sequences
    X_train, y_train, _ = create_sequences_by_engine(data['train_df'], ACTIVE_SENSORS, window=30, step=1)
    X_val, y_val, _ = create_sequences_by_engine(data['val_df'], ACTIVE_SENSORS, window=30, step=2)
    
    print(f"[Stage B Training] Train sequences: {X_train.shape}, Val sequences: {X_val.shape}")
    
    train_dataset = TensorDataset(torch.tensor(X_train, dtype=torch.float32), torch.tensor(y_train, dtype=torch.float32))
    val_dataset = TensorDataset(torch.tensor(X_val, dtype=torch.float32), torch.tensor(y_val, dtype=torch.float32))
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    
    model = SHELLModel(n_sensors=14, time_steps=30, sensor_latent_dim=32, adapter_dim=16).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.MSELoss()
    
    best_val_rmse = float('inf')
    best_state = None
    
    save_dir = base_dir / 'implementation_v2' / 'checkpoints'
    os.makedirs(save_dir, exist_ok=True)
    
    n_sensors = 14
    train_sensors_np = data['train_df'][ACTIVE_SENSORS].values
    train_std_per_sensor = np.std(train_sensors_np, axis=0) + 1e-6
    
    for epoch in range(epochs):
        model.train()
        total_loss = 0.0
        n_batches = 0
        
        for X_b, y_b in train_loader:
            # Corruption augmentation:
            # 1. Random sensor dropout (zero out 1 or 2 channels with 30% prob)
            if np.random.rand() < 0.35:
                n_drop = np.random.randint(1, 3)
                drop_idx = np.random.choice(n_sensors, n_drop, replace=False)
                X_b = X_b.clone()
                X_b[:, :, drop_idx] = 0.0
                
            # 2. Random Gaussian noise injection (with 30% prob)
            if np.random.rand() < 0.35:
                noise_idx = np.random.choice(n_sensors, 1)[0]
                X_b = X_b.clone()
                noise = torch.randn_like(X_b[:, :, noise_idx]) * float(train_std_per_sensor[noise_idx]) * 1.5
                X_b[:, :, noise_idx] += noise
                
            y_b = y_b.unsqueeze(-1)
            
            optimizer.zero_grad()
            pred, _, _, _ = model(X_b)
            loss = criterion(pred, y_b)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            
            total_loss += loss.item()
            n_batches += 1
            
        # Validation evaluation
        model.eval()
        val_preds, val_trues = [], []
        with torch.no_grad():
            for X_vb, y_vb in val_loader:
                pred_v, _, _, _ = model(X_vb)
                val_preds.append(pred_v.squeeze().numpy())
                val_trues.append(y_vb.numpy())
                
        val_preds = np.concatenate(val_preds)
        val_trues = np.concatenate(val_trues)
        val_rmse = float(np.sqrt(np.mean((val_preds - val_trues)**2)))
        
        if val_rmse < best_val_rmse:
            best_val_rmse = val_rmse
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            
        if (epoch + 1) % 5 == 0 or epoch == epochs - 1:
            print(f"  Epoch {epoch+1:2d}/{epochs} | Train Loss: {total_loss/n_batches:.2f} | Val RMSE (Units 81-100): {val_rmse:.2f} (Best: {best_val_rmse:.2f})")
            
    # Save best checkpoint
    save_path = save_dir / 'robust_backbone_FD001_seed42.pt'
    torch.save({
        'model_state_dict': best_state,
        'n_sensors': 14,
        'time_steps': 30,
        'sensor_cols': ACTIVE_SENSORS,
        'best_val_rmse': best_val_rmse,
        'seed': seed
    }, save_path)
    print(f"[Stage B Complete] Best checkpoint saved to {save_path} with Val RMSE: {best_val_rmse:.2f}")

if __name__ == '__main__':
    train_robust(epochs=30)
