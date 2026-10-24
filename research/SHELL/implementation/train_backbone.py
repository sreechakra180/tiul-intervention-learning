"""
SHELL Pretraining Script
=========================
Train the backbone model on clean C-MAPSS data.
After pretraining, the backbone and RUL head are frozen.
Only the adapter is updated during test-time evaluation.

Usage:
    python train_backbone.py --data_dir ./data/raw --subset FD001 --seed 42
    python train_backbone.py --data_dir ./data/raw --subset FD001 --epochs 50
"""

import os
import sys
import argparse
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset, random_split
from pathlib import Path
import json

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from data.download_cmapss import load_and_preprocess, create_sequences
from models.backbone import SHELLModel, CrossModalReconstructor


def set_seed(seed: int):
    """Set all random seeds for reproducibility."""
    import random
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def train_epoch(
    model: SHELLModel,
    dataloader: DataLoader,
    optimizer: torch.optim.Optimizer,
    device: str,
    epoch: int,
) -> dict:
    """Train for one epoch. Returns metrics dict."""
    model.train()
    # Enable all parameters during pretraining
    for param in model.parameters():
        param.requires_grad = True
    
    total_loss = 0.0
    total_rmse = 0.0
    n_batches = 0
    
    criterion = nn.MSELoss()
    
    for X_batch, y_batch in dataloader:
        X_batch = X_batch.to(device)
        y_batch = y_batch.to(device).unsqueeze(-1)
        
        rul_pred, _, _, _ = model(X_batch)
        loss = criterion(rul_pred, y_batch)
        
        optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        
        rmse = torch.sqrt(loss).item()
        total_loss += loss.item()
        total_rmse += rmse
        n_batches += 1
    
    return {
        'loss': total_loss / n_batches,
        'rmse': total_rmse / n_batches,
    }


@torch.no_grad()
def evaluate(
    model: SHELLModel,
    dataloader: DataLoader,
    device: str,
) -> dict:
    """Evaluate model. Returns metrics dict."""
    model.eval()
    
    all_pred, all_true = [], []
    
    for X_batch, y_batch in dataloader:
        X_batch = X_batch.to(device)
        rul_pred, _, _, _ = model(X_batch)
        all_pred.append(rul_pred.squeeze().cpu().numpy())
        all_true.append(y_batch.numpy())
    
    pred = np.concatenate(all_pred)
    true = np.concatenate(all_true)
    
    rmse = np.sqrt(np.mean((pred - true) ** 2))
    mae = np.mean(np.abs(pred - true))
    
    # NASA scoring function (asymmetric, penalizes late predictions more)
    d = pred - true
    nasa_score = np.where(d < 0,
        np.sum(np.exp(-d / 13) - 1),
        np.sum(np.exp(d / 10) - 1)
    ).sum()
    
    return {
        'rmse': float(rmse),
        'mae': float(mae),
        'nasa_score': float(nasa_score),
    }


def train_reconstructor(
    model: SHELLModel,
    reconstructor: CrossModalReconstructor,
    dataloader: DataLoader,
    device: str,
    epochs: int = 10,
) -> None:
    """
    Train the cross-modal reconstructor using masked autoencoder objective.
    
    For each batch:
    1. Extract per-sensor latents from frozen backbone
    2. Randomly mask 1-3 sensors
    3. Train reconstructor to reconstruct masked sensors' latents
    
    The backbone is frozen during this phase.
    """
    model.eval()
    for param in model.backbone.parameters():
        param.requires_grad = False
    
    reconstructor.train()
    optimizer = torch.optim.Adam(reconstructor.parameters(), lr=1e-3)
    criterion = nn.MSELoss()
    
    n_sensors = model.n_sensors
    
    for epoch in range(epochs):
        total_loss = 0.0
        n_batches = 0
        
        for X_batch, _ in dataloader:
            X_batch = X_batch.to(device)
            
            with torch.no_grad():
                _, _, _, z_per_sensor = model(X_batch)
            
            # Random masking: mask 1 to n_sensors//3 sensors
            rng = np.random.RandomState(epoch + n_batches)
            n_mask = rng.randint(1, max(2, n_sensors // 3))
            mask_idx = rng.choice(n_sensors, n_mask, replace=False)
            
            mask = np.ones(n_sensors, dtype=np.float32)
            mask[mask_idx] = 0.0
            mask_tensor = torch.tensor(mask, device=device)
            
            # Get ground truth for masked sensors
            z_targets = [z_per_sensor[i].detach() for i in mask_idx]
            z_target_mean = torch.stack(z_targets).mean(dim=0)
            
            # Reconstruct
            z_recon = reconstructor(z_per_sensor, mask_tensor)
            
            loss = criterion(z_recon, z_target_mean)
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            n_batches += 1
        
        if (epoch + 1) % 5 == 0:
            print(f"  Reconstructor epoch {epoch+1}/{epochs}: loss={total_loss/n_batches:.4f}")


def main():
    parser = argparse.ArgumentParser(description='Pretrain SHELL backbone on C-MAPSS')
    parser.add_argument('--data_dir', type=str, default='./data/raw')
    parser.add_argument('--subset', type=str, default='FD001')
    parser.add_argument('--window', type=int, default=30)
    parser.add_argument('--batch_size', type=int, default=256)
    parser.add_argument('--epochs', type=int, default=50)
    parser.add_argument('--recon_epochs', type=int, default=20)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--save_dir', type=str, default='./checkpoints')
    parser.add_argument('--latent_dim', type=int, default=32)
    parser.add_argument('--adapter_dim', type=int, default=16)
    parser.add_argument('--val_fraction', type=float, default=0.1)
    args = parser.parse_args()
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    if device == 'cpu':
        torch.set_num_threads(os.cpu_count() or 4)

    set_seed(args.seed)
    os.makedirs(args.save_dir, exist_ok=True)
    
    print(f"[Pretraining] Device: {device} (threads={torch.get_num_threads()}), Subset: {args.subset}, Seed: {args.seed}", flush=True)
    
    # Load and preprocess data
    data = load_and_preprocess(args.data_dir, args.subset)
    sensor_cols = data['sensor_cols']
    n_sensors = len(sensor_cols)
    
    X_train, y_train = create_sequences(
        data['train_df'], sensor_cols, window=args.window)
    
    print(f"  Training sequences: X={X_train.shape}, y={y_train.shape}")
    
    # Create datasets
    X_tensor = torch.tensor(X_train, dtype=torch.float32)
    y_tensor = torch.tensor(y_train, dtype=torch.float32)
    dataset = TensorDataset(X_tensor, y_tensor)
    
    # Train/val split
    n_val = int(len(dataset) * args.val_fraction)
    n_train = len(dataset) - n_val
    train_set, val_set = random_split(
        dataset, [n_train, n_val],
        generator=torch.Generator().manual_seed(args.seed)
    )
    
    train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True,
                               num_workers=0, pin_memory=(device == 'cuda'))
    val_loader = DataLoader(val_set, batch_size=args.batch_size, shuffle=False,
                             num_workers=0)
    
    # Initialize model
    model = SHELLModel(
        n_sensors=n_sensors,
        time_steps=args.window,
        sensor_latent_dim=args.latent_dim,
        adapter_dim=args.adapter_dim,
    ).to(device)
    
    reconstructor = CrossModalReconstructor(
        n_sensors=n_sensors,
        latent_dim=args.latent_dim,
    ).to(device)
    
    total_params = sum(p.numel() for p in model.parameters())
    adapter_params = sum(p.numel() for p in model.adapter.parameters())
    print(f"  Model parameters: {total_params:,} total, {adapter_params:,} adapter")
    
    # Optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.epochs, eta_min=1e-5)
    
    # Training loop
    best_val_rmse = float('inf')
    best_model_path = os.path.join(args.save_dir, f'shell_backbone_{args.subset}_seed{args.seed}.pt')
    train_history = []
    
    print(f"\n[Pretraining backbone]")
    for epoch in range(args.epochs):
        train_metrics = train_epoch(model, train_loader, optimizer, device, epoch)
        val_metrics = evaluate(model, val_loader, device)
        scheduler.step()
        
        train_history.append({
            'epoch': epoch + 1,
            'train_loss': train_metrics['loss'],
            'train_rmse': train_metrics['rmse'],
            'val_rmse': val_metrics['rmse'],
            'val_nasa': val_metrics['nasa_score'],
        })
        
        if (epoch + 1) % 2 == 0 or epoch == 0 or (epoch + 1) == args.epochs:
            print(f"  Epoch {epoch+1:3d}/{args.epochs}: "
                  f"train_loss={train_metrics['loss']:.4f}, "
                  f"val_rmse={val_metrics['rmse']:.2f}, "
                  f"val_nasa={val_metrics['nasa_score']:.1f}", flush=True)
        
        if val_metrics['rmse'] < best_val_rmse:
            best_val_rmse = val_metrics['rmse']
            torch.save({
                'epoch': epoch + 1,
                'model_state_dict': model.state_dict(),
                'val_rmse': best_val_rmse,
                'n_sensors': n_sensors,
                'time_steps': args.window,
                'latent_dim': args.latent_dim,
                'adapter_dim': args.adapter_dim,
                'sensor_cols': sensor_cols,
                'args': vars(args),
            }, best_model_path)
    
    print(f"\n  Best val RMSE: {best_val_rmse:.2f}")
    print(f"  Backbone saved to: {best_model_path}")
    
    # Train reconstructor
    print(f"\n[Training cross-modal reconstructor]")
    model.load_state_dict(
        torch.load(best_model_path, map_location=device)['model_state_dict'])
    
    train_reconstructor(model, reconstructor, train_loader, device, epochs=args.recon_epochs)
    
    recon_path = os.path.join(args.save_dir, f'shell_reconstructor_{args.subset}_seed{args.seed}.pt')
    torch.save(reconstructor.state_dict(), recon_path)
    print(f"  Reconstructor saved to: {recon_path}")
    
    # Save training history
    history_path = os.path.join(args.save_dir, f'train_history_{args.subset}_seed{args.seed}.json')
    with open(history_path, 'w') as f:
        json.dump(train_history, f, indent=2)
    
    print(f"\n[Pretraining complete]")
    print(f"  Final validation RMSE: {best_val_rmse:.2f}")
    print(f"  Model ready for SHELL evaluation.")


if __name__ == '__main__':
    main()
