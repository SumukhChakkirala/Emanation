#!/usr/bin/env python3
"""
Train 1D CNN for pitch estimation from log-PSD.

Usage:
    python programs/train_pitch_cnn.py --data data/pitch_dataset.npz --epochs 50 --batch 64
"""
import argparse
import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, random_split


class PSDDataset(Dataset):
    """PyTorch Dataset for PSD-based pitch estimation."""
    
    def __init__(self, npz_path):
        data = np.load(npz_path)
        self.psds = torch.tensor(data['psds'], dtype=torch.float32)
        self.labels = torch.tensor(data['labels'], dtype=torch.float32)
        self.snrs = torch.tensor(data['snrs'], dtype=torch.float32)
        
        # Normalize PSDs (per-sample z-score)
        self.psds_mean = self.psds.mean(dim=1, keepdim=True)
        self.psds_std = self.psds.std(dim=1, keepdim=True) + 1e-6
        self.psds = (self.psds - self.psds_mean) / self.psds_std
        
        # Normalize labels (for regression stability)
        self.label_mean = self.labels.mean()
        self.label_std = self.labels.std() + 1e-6
        self.labels_norm = (self.labels - self.label_mean) / self.label_std
        
        print(f"Dataset loaded: {len(self)} samples, PSD dim: {self.psds.shape[1]}")
        print(f"Label mean: {self.label_mean:.1f} Hz, std: {self.label_std:.1f} Hz")
    
    def __len__(self):
        return len(self.labels)
    
    def __getitem__(self, idx):
        # Return (psd, snr, normalized_label)
        psd = self.psds[idx].unsqueeze(0)  # (1, Nfreq)
        snr = self.snrs[idx].unsqueeze(0)  # (1,)
        label = self.labels_norm[idx]       # scalar (normalized)
        return psd, snr, label
    
    def denormalize_label(self, y_norm):
        """Convert normalized prediction back to Hz."""
        return y_norm * self.label_std + self.label_mean


class PitchCNN(nn.Module):
    """1D CNN for pitch estimation from log-PSD."""
    
    def __init__(self, n_freq, use_snr=True):
        super().__init__()
        self.use_snr = use_snr
        
        # 1D CNN backbone
        self.backbone = nn.Sequential(
            nn.Conv1d(1, 64, kernel_size=7, padding=3),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.MaxPool1d(2),
            
            nn.Conv1d(64, 128, kernel_size=5, padding=2),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.MaxPool1d(2),
            
            nn.Conv1d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.AdaptiveAvgPool1d(1),
        )
        
        # Head
        head_input_dim = 256 + (1 if use_snr else 0)
        self.head = nn.Sequential(
            nn.Flatten(),
            nn.Linear(head_input_dim, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, 1),
        )
    
    def forward(self, psd, snr=None):
        x = self.backbone(psd)  # (B, 256, 1)
        x = x.squeeze(-1)       # (B, 256)
        
        if self.use_snr and snr is not None:
            x = torch.cat([x, snr], dim=1)  # (B, 257)
        
        out = self.head(x).squeeze(-1)  # (B,)
        return out


def train_epoch(model, loader, optimizer, criterion, device):
    model.train()
    total_loss = 0
    for psd, snr, label in loader:
        psd, snr, label = psd.to(device), snr.to(device), label.to(device)
        
        optimizer.zero_grad()
        pred = model(psd, snr)
        loss = criterion(pred, label)
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item() * len(label)
    
    return total_loss / len(loader.dataset)


def eval_epoch(model, loader, criterion, device, dataset):
    model.eval()
    total_loss = 0
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for psd, snr, label in loader:
            psd, snr, label = psd.to(device), snr.to(device), label.to(device)
            pred = model(psd, snr)
            loss = criterion(pred, label)
            total_loss += loss.item() * len(label)
            
            # Denormalize for metrics
            pred_hz = dataset.denormalize_label(pred.cpu())
            label_hz = dataset.denormalize_label(label.cpu())
            all_preds.append(pred_hz)
            all_labels.append(label_hz)
    
    all_preds = torch.cat(all_preds)
    all_labels = torch.cat(all_labels)
    
    # Compute metrics
    mae = torch.abs(all_preds - all_labels).mean().item()
    within_5hz = (torch.abs(all_preds - all_labels) < 5).float().mean().item() * 100
    within_10hz = (torch.abs(all_preds - all_labels) < 10).float().mean().item() * 100
    within_50hz = (torch.abs(all_preds - all_labels) < 50).float().mean().item() * 100
    
    return total_loss / len(loader.dataset), mae, within_5hz, within_10hz, within_50hz


def main():
    parser = argparse.ArgumentParser(description='Train 1D CNN for pitch estimation')
    parser.add_argument('--data', type=str, default='data/pitch_dataset.npz',
                        help='Path to dataset')
    parser.add_argument('--epochs', type=int, default=50,
                        help='Number of epochs')
    parser.add_argument('--batch', type=int, default=64,
                        help='Batch size')
    parser.add_argument('--lr', type=float, default=1e-3,
                        help='Learning rate')
    parser.add_argument('--val-split', type=float, default=0.15,
                        help='Validation split ratio')
    parser.add_argument('--save-dir', type=str, default='models',
                        help='Directory to save model')
    parser.add_argument('--no-snr', action='store_true',
                        help='Do not use SNR as input')
    parser.add_argument('--device', type=str, default='cuda',
                        help='Device (cuda or cpu)')
    
    args = parser.parse_args()
    
    # Device
    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Load dataset
    dataset = PSDDataset(args.data)
    
    # Train/val split
    val_size = int(len(dataset) * args.val_split)
    train_size = len(dataset) - val_size
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])
    
    train_loader = DataLoader(train_dataset, batch_size=args.batch, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=args.batch, shuffle=False, num_workers=0)
    
    print(f"Train: {train_size}, Val: {val_size}")
    
    # Model
    n_freq = dataset.psds.shape[1]
    model = PitchCNN(n_freq, use_snr=not args.no_snr).to(device)
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # Optimizer & loss
    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=5, factor=0.5)
    criterion = nn.L1Loss()  # MAE loss
    
    # Training loop
    best_val_mae = float('inf')
    os.makedirs(args.save_dir, exist_ok=True)
    
    print("\nTraining...")
    print("-" * 80)
    
    for epoch in range(args.epochs):
        train_loss = train_epoch(model, train_loader, optimizer, criterion, device)
        val_loss, val_mae, w5, w10, w50 = eval_epoch(model, val_loader, criterion, device, dataset)
        
        scheduler.step(val_loss)
        
        print(f"Epoch {epoch+1:3d} | Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f} | "
              f"MAE: {val_mae:.1f} Hz | <5Hz: {w5:.1f}% | <10Hz: {w10:.1f}% | <50Hz: {w50:.1f}%")
        
        # Save best model
        if val_mae < best_val_mae:
            best_val_mae = val_mae
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_mae': val_mae,
                'label_mean': dataset.label_mean.item(),
                'label_std': dataset.label_std.item(),
            }, os.path.join(args.save_dir, 'best_pitch_cnn.pt'))
    
    print("-" * 80)
    print(f"Best Val MAE: {best_val_mae:.1f} Hz")
    print(f"Model saved to {os.path.join(args.save_dir, 'best_pitch_cnn.pt')}")


if __name__ == '__main__':
    main()
