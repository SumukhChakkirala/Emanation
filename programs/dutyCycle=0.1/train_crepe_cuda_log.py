"""
Train CREPE model for pitch estimation on RF signals - CUDA OPTIMIZED VERSION

Optimizations:
1. cuDNN benchmark enabled for faster convolutions
2. Automatic Mixed Precision (AMP) training with GradScaler
3. Improved learning rate scheduler (OneCycleLR or CosineAnnealingWarmRestarts)
4. Enhanced early stopping with delta threshold
5. DataLoader optimizations (num_workers, pin_memory, prefetch)
6. Non-blocking GPU transfers

Architecture unchanged from original CREPE paper.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch.cuda.amp import autocast, GradScaler
import numpy as np
import pickle
import os
import argparse
from typing import Tuple, List, Dict
import matplotlib.pyplot as plt
from tqdm import tqdm
import time


# =============================================================================
# Constants
# =============================================================================

CREPE_FS = 16000
CREPE_FRAME_LENGTH = 1024
CREPE_N_BINS = 360
CREPE_CENTS_PER_BIN = 20
CENTS_OFFSET = 1997.3794084376191


def cents_to_hz(cents: float, fref: float = 10.0) -> float:
    """Convert cents to Hz."""
    return fref * (2.0 ** (cents / 1200.0))


def crepe_bin_to_hz(bin_idx: int) -> float:
    """Convert CREPE bin to Hz."""
    cents = CENTS_OFFSET + bin_idx * CREPE_CENTS_PER_BIN
    return cents_to_hz(cents)


def hz_to_cents(freq_hz: float, fref: float = 10.0) -> float:
    """Convert Hz to cents."""
    return 1200.0 * np.log2(freq_hz / fref)


def hz_to_crepe_bin(freq_hz: float) -> int:
    """Convert Hz to CREPE bin."""
    cents = hz_to_cents(freq_hz)
    bin_idx = int(round((cents - CENTS_OFFSET) / CREPE_CENTS_PER_BIN))
    return np.clip(bin_idx, 0, CREPE_N_BINS - 1)


# =============================================================================
# Dataset - Optimized with precompute
# =============================================================================

class CREPEDataset(Dataset):
    """
    CREPE dataset for RF signals - OPTIMIZED VERSION
    
    Precomputes magnitude and labels for faster training.
    """
    
    def __init__(
        self,
        iq_dict: Dict[str, np.ndarray],
        bin_list: List[int] = None,
        snr_range: Tuple[int, int] = None,
        target_length: int = CREPE_FRAME_LENGTH,
        gaussian_sigma: float = 1.25,
        precompute: bool = True,
    ):
        self.target_length = target_length
        self.gaussian_sigma = gaussian_sigma
        self.precompute = precompute
        
        # Filter and store samples
        self.keys = []
        self.bin_indices = []
        self._signals = []
        
        for key, signal in iq_dict.items():
            parts = key.split('_')
            bin_idx = int(parts[1])
            snr = int(parts[3])
            
            if bin_list is not None and bin_idx not in bin_list:
                continue
            if snr_range is not None and (snr < snr_range[0] or snr >= snr_range[1]):
                continue
            
            self.keys.append(key)
            self.bin_indices.append(bin_idx)
            
            if self.precompute:
                # Preprocess: magnitude, pad/truncate, normalize
                sig = np.abs(signal).astype(np.float32)
                if len(sig) < self.target_length:
                    sig = np.pad(sig, (0, self.target_length - len(sig)))
                else:
                    sig = sig[:self.target_length]
                sig = sig / (np.max(np.abs(sig)) + 1e-8)
                self._signals.append(sig)
            else:
                self._signals.append(signal)
        
        # Stack into arrays for faster indexing
        if self.precompute and len(self._signals) > 0:
            self._signals = np.stack(self._signals).astype(np.float32)
            self._labels = np.stack([self._create_gaussian_label(b) for b in self.bin_indices]).astype(np.float32)
        
        print(f"✓ Created dataset with {len(self._signals)} samples (precompute={self.precompute})")
        if len(self._signals) > 0:
            unique_bins = sorted(list(set(self.bin_indices)))
            print(f"  Unique bins: {len(unique_bins)} (range: {min(unique_bins)} to {max(unique_bins)})")
            unique_freqs = [crepe_bin_to_hz(b) for b in unique_bins]
            print(f"  Frequency range: {min(unique_freqs):.1f} Hz to {max(unique_freqs):.1f} Hz")
    
    def __len__(self):
        return len(self._signals)
    
    def __getitem__(self, idx):
        if self.precompute:
            signal = torch.from_numpy(self._signals[idx]).unsqueeze(0)
            label = torch.from_numpy(self._labels[idx])
            return signal, label
        
        # Fallback: process on-the-fly
        signal = np.abs(self._signals[idx]).astype(np.float32)
        if len(signal) < self.target_length:
            signal = np.pad(signal, (0, self.target_length - len(signal)))
        else:
            signal = signal[:self.target_length]
        signal = signal / (np.max(np.abs(signal)) + 1e-8)
        label = self._create_gaussian_label(self.bin_indices[idx])
        return torch.FloatTensor(signal).unsqueeze(0), torch.FloatTensor(label)
    
    def _create_gaussian_label(self, true_bin: int) -> np.ndarray:
        """Create Gaussian-smoothed label as per CREPE paper."""
        label = np.zeros(CREPE_N_BINS, dtype=np.float32)
        for i in range(CREPE_N_BINS):
            label[i] = np.exp(-((i - true_bin) ** 2) / (2 * self.gaussian_sigma ** 2))
        return label


# =============================================================================
# Model - UNCHANGED ARCHITECTURE
# =============================================================================

class CREPE(nn.Module):
    """
    CREPE model - exact architecture from paper (UNCHANGED).
    """
    
    def __init__(self, dropout: float = 0.25):
        super(CREPE, self).__init__()
        
        # Layer 1: conv (1024 filters)
        self.conv1 = nn.Conv1d(1, 1024, kernel_size=512, stride=4, padding=254)
        self.bn1 = nn.BatchNorm1d(1024)
        self.drop1 = nn.Dropout(dropout)
        self.pool1 = nn.MaxPool1d(2, stride=2)
        
        # Layer 2: conv (128 filters)
        self.conv2 = nn.Conv1d(1024, 128, kernel_size=64, stride=1, padding=32)
        self.bn2 = nn.BatchNorm1d(128)
        self.drop2 = nn.Dropout(dropout)
        self.pool2 = nn.MaxPool1d(2, stride=2)
        
        # Layer 3: conv (128 filters)
        self.conv3 = nn.Conv1d(128, 128, kernel_size=64, stride=1, padding=32)
        self.bn3 = nn.BatchNorm1d(128)
        self.drop3 = nn.Dropout(dropout)
        self.pool3 = nn.MaxPool1d(2, stride=2)
        
        # Layer 4: conv (128 filters)
        self.conv4 = nn.Conv1d(128, 128, kernel_size=64, stride=1, padding=32)
        self.bn4 = nn.BatchNorm1d(128)
        self.drop4 = nn.Dropout(dropout)
        self.pool4 = nn.MaxPool1d(2, stride=2)
        
        # Layer 5: conv (256 filters)
        self.conv5 = nn.Conv1d(128, 256, kernel_size=64, stride=1, padding=32)
        self.bn5 = nn.BatchNorm1d(256)
        self.drop5 = nn.Dropout(dropout)
        self.pool5 = nn.MaxPool1d(2, stride=2)
        
        # Layer 6: conv (512 filters)
        self.conv6 = nn.Conv1d(256, 512, kernel_size=64, stride=1, padding=32)
        self.bn6 = nn.BatchNorm1d(512)
        self.drop6 = nn.Dropout(dropout)
        self.pool6 = nn.MaxPool1d(2, stride=2)
        
        # Calculate output size
        with torch.no_grad():
            dummy_input = torch.randn(1, 1, 1024)
            x = self.pool1(self.drop1(F.relu(self.bn1(self.conv1(dummy_input)))))
            x = self.pool2(self.drop2(F.relu(self.bn2(self.conv2(x)))))
            x = self.pool3(self.drop3(F.relu(self.bn3(self.conv3(x)))))
            x = self.pool4(self.drop4(F.relu(self.bn4(self.conv4(x)))))
            x = self.pool5(self.drop5(F.relu(self.bn5(self.conv5(x)))))
            x = self.pool6(self.drop6(F.relu(self.bn6(self.conv6(x)))))
            flattened_size = x.view(x.size(0), -1).size(1)
        
        self.fc = nn.Linear(flattened_size, 360)
    
    def forward(self, x):
        x = self.pool1(self.drop1(F.relu(self.bn1(self.conv1(x)))))
        x = self.pool2(self.drop2(F.relu(self.bn2(self.conv2(x)))))
        x = self.pool3(self.drop3(F.relu(self.bn3(self.conv3(x)))))
        x = self.pool4(self.drop4(F.relu(self.bn4(self.conv4(x)))))
        x = self.pool5(self.drop5(F.relu(self.bn5(self.conv5(x)))))
        x = self.pool6(self.drop6(F.relu(self.bn6(self.conv6(x)))))
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x


# =============================================================================
# Early Stopping with Delta Threshold
# =============================================================================

class EarlyStopping:
    """
    Improved early stopping with:
    - Min delta threshold (ignore tiny improvements)
    - Patience counter
    - Best model checkpoint
    """
    
    def __init__(self, patience: int = 10, min_delta: float = 0.1, mode: str = 'max'):
        self.patience = patience
        self.min_delta = min_delta
        self.mode = mode
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.best_epoch = 0
    
    def __call__(self, score: float, epoch: int) -> bool:
        if self.best_score is None:
            self.best_score = score
            self.best_epoch = epoch
            return True  # Save model
        
        if self.mode == 'max':
            improved = score > self.best_score + self.min_delta
        else:
            improved = score < self.best_score - self.min_delta
        
        if improved:
            self.best_score = score
            self.best_epoch = epoch
            self.counter = 0
            return True  # Save model
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
            return False  # Don't save


# =============================================================================
# Evaluation Metrics
# =============================================================================

def evaluate_predictions(predictions: np.ndarray, targets: np.ndarray) -> Tuple[float, float, float, float]:
    """Evaluate pitch predictions using CREPE metrics."""
    bin_indices = np.arange(360)
    pred_probs = 1 / (1 + np.exp(-predictions))
    pred_bins = np.sum(pred_probs * bin_indices, axis=1) / np.sum(pred_probs, axis=1)
    
    pred_freqs = np.array([crepe_bin_to_hz(b) for b in pred_bins])
    true_freqs = np.array([crepe_bin_to_hz(b) for b in targets])
    
    errors_cents = np.abs(1200 * np.log2(pred_freqs / (true_freqs + 1e-8)))
    
    rpa_50 = np.mean(errors_cents <= 50) * 100
    rpa_25 = np.mean(errors_cents <= 25) * 100
    
    chroma_errors = np.minimum(errors_cents % 1200, 1200 - (errors_cents % 1200))
    rca = np.mean(chroma_errors <= 50) * 100
    
    mean_error = np.mean(errors_cents)
    
    return rpa_50, rpa_25, rca, mean_error


def evaluate(model, dataloader, criterion, device, use_amp: bool = True):
    """Evaluate model with AMP support."""
    model.eval()
    total_loss = 0
    all_predictions = []
    all_targets = []
    
    with torch.no_grad():
        for inputs, labels in dataloader:
            inputs = inputs.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)
            
            with autocast(enabled=use_amp):
                outputs = model(inputs)
                loss = criterion(outputs, labels)
            
            total_loss += loss.item()
            all_predictions.append(outputs.float().cpu().numpy())
            true_bins = torch.argmax(labels, dim=1).cpu().numpy()
            all_targets.append(true_bins)
    
    avg_loss = total_loss / len(dataloader)
    all_predictions = np.concatenate(all_predictions, axis=0)
    all_targets = np.concatenate(all_targets, axis=0)
    
    rpa_50, rpa_25, rca, mean_error = evaluate_predictions(all_predictions, all_targets)
    
    return avg_loss, rpa_50, rpa_25, rca, mean_error


# =============================================================================
# Training with AMP
# =============================================================================

def train_epoch(model, dataloader, criterion, optimizer, device, epoch, scaler, use_amp: bool = True):
    """Train for one epoch with AMP."""
    model.train()
    total_loss = 0
    
    pbar = tqdm(dataloader, desc=f'Epoch {epoch}')
    for batch_idx, (inputs, labels) in enumerate(pbar):
        inputs = inputs.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)
        
        optimizer.zero_grad(set_to_none=True)  # Faster than zero_grad()
        
        with autocast(enabled=use_amp):
            outputs = model(inputs)
            loss = criterion(outputs, labels)
        
        if use_amp:
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
        
        total_loss += loss.item()
        pbar.set_postfix({'loss': f'{loss.item():.6f}'})
    
    return total_loss / len(dataloader)


def main():
    parser = argparse.ArgumentParser(description='Train CREPE model - CUDA Optimized')
    parser.add_argument('--snr_min', type=int, default=0, help='Minimum SNR (default: -20)')
    parser.add_argument('--snr_max', type=int, default=20, help='Maximum SNR (default: 20)')
    parser.add_argument('--model_suffix', type=str, default='dc0.1_snr_neg20_20', help='Suffix for model files')
    parser.add_argument('--batch_size', type=int, default=128, help='Batch size (default: 128)')
    parser.add_argument('--epochs', type=int, default=100, help='Max epochs (default: 100)')
    parser.add_argument('--lr', type=float, default=0.001, help='Max learning rate (default: 0.001)')
    parser.add_argument('--patience', type=int, default=15, help='Early stopping patience (default: 15)')
    parser.add_argument('--scheduler', type=str, default='onecycle', choices=['onecycle', 'cosine', 'plateau'], 
                        help='LR scheduler type')
    args = parser.parse_args()
    
    # Check CUDA availability
    if not torch.cuda.is_available():
        print("⚠️  CUDA not available! Training will be slow on CPU.")
        use_amp = False
        device = torch.device('cpu')
    else:
        use_amp = True
        device = torch.device('cuda')
        # Enable cuDNN autotuner for faster convolutions
        torch.backends.cudnn.benchmark = True
        torch.backends.cudnn.enabled = True
        print(f"✓ CUDA enabled: {torch.cuda.get_device_name(0)}")
        print(f"✓ cuDNN version: {torch.backends.cudnn.version()}")
    
    # Configuration
    config = {
        'data_path': r'IQData\iq_dict_continuous_freq_SNR0_20_logarithmic.pkl',
        'batch_size': args.batch_size,
        'epochs': args.epochs,
        'lr': args.lr,
        'weight_decay': 1e-4,
        'dropout': 0.35,
        'gaussian_sigma': 1.25,
        'device': str(device),
        'save_dir': os.path.join(os.path.dirname(__file__), 'models'),
        'patience': args.patience,
        'min_delta': 0.1,  # Minimum improvement threshold (in % RPA)
        'snr_range': (args.snr_min, args.snr_max + 1),
        'model_suffix': args.model_suffix,
        'scheduler': args.scheduler,
        'num_workers': min(8, max(1, (os.cpu_count() or 2) - 1)),
        'use_amp': use_amp,
    }
    
    print("=" * 80)
    print("Training CREPE Model - CUDA OPTIMIZED")
    print("=" * 80)
    print(f"\nConfiguration:")
    for k, v in config.items():
        print(f"  {k}: {v}")
    
    # Load dataset
    print(f"\n📂 Loading data from {config['data_path']}...")
    start_time = time.time()
    with open(config['data_path'], 'rb') as f:
        iq_dict = pickle.load(f)
    print(f"✓ Loaded {len(iq_dict)} samples in {time.time() - start_time:.1f}s")
    
    # Extract bins
    all_bins = sorted(list(set([int(k.split('_')[1]) for k in iq_dict.keys()])))
    print(f"✓ Found {len(all_bins)} unique bins (range: {min(all_bins)}-{max(all_bins)})")
    
    # Split bins (60/20/20)
    train_bins = [b for i, b in enumerate(all_bins) if i % 5 not in [0, 1]]
    val_bins = [b for i, b in enumerate(all_bins) if i % 5 == 0]
    test_bins = [b for i, b in enumerate(all_bins) if i % 5 == 1]
    
    print(f"\n📊 Train/Val/Test split (60/20/20):")
    print(f"  Train bins: {len(train_bins)}")
    print(f"  Val bins: {len(val_bins)}")
    print(f"  Test bins: {len(test_bins)}")
    
    # Create datasets with precompute
    print(f"\n📊 Using SNR range: {config['snr_range'][0]} to {config['snr_range'][1] - 1} dB")
    
    train_dataset = CREPEDataset(
        iq_dict=iq_dict,
        bin_list=train_bins,
        snr_range=config['snr_range'],
        gaussian_sigma=config['gaussian_sigma'],
        precompute=True
    )
    
    val_dataset = CREPEDataset(
        iq_dict=iq_dict,
        bin_list=val_bins,
        snr_range=config['snr_range'],
        gaussian_sigma=config['gaussian_sigma'],
        precompute=True
    )
    
    # DataLoader with optimizations
    pin_memory = device.type == 'cuda'
    num_workers = config['num_workers']
    
    train_loader = DataLoader(
        train_dataset, 
        batch_size=config['batch_size'],
        shuffle=True, 
        num_workers=num_workers,
        pin_memory=pin_memory,
        persistent_workers=(num_workers > 0),
        prefetch_factor=2 if num_workers > 0 else None,
        drop_last=True
    )
    
    val_loader = DataLoader(
        val_dataset, 
        batch_size=config['batch_size'],
        shuffle=False, 
        num_workers=num_workers,
        pin_memory=pin_memory,
        persistent_workers=(num_workers > 0),
        prefetch_factor=2 if num_workers > 0 else None
    )
    
    # Create model
    print(f"\n🏗️  Creating CREPE model...")
    model = CREPE(dropout=config['dropout'])
    model = model.to(device)
    
    # Try torch.compile (PyTorch 2.0+)
    if hasattr(torch, 'compile'):
        try:
            model = torch.compile(model, mode='reduce-overhead')
            print("✓ torch.compile enabled (reduce-overhead mode)")
        except Exception as e:
            print(f"⚠️  torch.compile failed: {e}")
    
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"✓ Model created: {n_params:,} trainable parameters")
    
    # Loss, optimizer, scaler
    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=config['lr'], weight_decay=config['weight_decay'])
    scaler = GradScaler(enabled=use_amp)
    
    # Learning rate scheduler
    total_steps = len(train_loader) * config['epochs']
    
    if config['scheduler'] == 'onecycle':
        # OneCycleLR: ramps up then down, good for fast convergence
        scheduler = torch.optim.lr_scheduler.OneCycleLR(
            optimizer,
            max_lr=config['lr'],
            total_steps=total_steps,
            pct_start=0.1,
            anneal_strategy='cos',
            div_factor=25,
            final_div_factor=1000
        )
        step_per_batch = True
        print(f"✓ Using OneCycleLR scheduler (max_lr={config['lr']})")
    elif config['scheduler'] == 'cosine':
        # CosineAnnealingWarmRestarts: periodic restarts
        scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
            optimizer,
            T_0=10,
            T_mult=2,
            eta_min=1e-6
        )
        step_per_batch = False
        print(f"✓ Using CosineAnnealingWarmRestarts scheduler")
    else:
        # ReduceLROnPlateau: reduce when validation loss plateaus
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='max', factor=0.5, patience=5, min_lr=1e-6, verbose=True
        )
        step_per_batch = False
        print(f"✓ Using ReduceLROnPlateau scheduler")
    
    # Early stopping
    early_stopping = EarlyStopping(
        patience=config['patience'],
        min_delta=config['min_delta'],
        mode='max'
    )
    
    # Training loop
    print(f"\n🚀 Starting training for up to {config['epochs']} epochs...")
    print("=" * 80)
    
    history = {'train_loss': [], 'val_loss': [], 'rpa_50': [], 'rpa_25': [], 'rca': [], 'lr': []}
    os.makedirs(config['save_dir'], exist_ok=True)
    
    epoch_times = []
    
    for epoch in range(1, config['epochs'] + 1):
        epoch_start = time.time()
        
        # Train
        model.train()
        total_loss = 0
        pbar = tqdm(train_loader, desc=f'Epoch {epoch}')
        
        for batch_idx, (inputs, labels) in enumerate(pbar):
            inputs = inputs.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)
            
            optimizer.zero_grad(set_to_none=True)
            
            with autocast(enabled=use_amp):
                outputs = model(inputs)
                loss = criterion(outputs, labels)
            
            if use_amp:
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                scaler.step(optimizer)
                scaler.update()
            else:
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()
            
            if step_per_batch:
                scheduler.step()
            
            total_loss += loss.item()
            pbar.set_postfix({'loss': f'{loss.item():.6f}', 'lr': f'{optimizer.param_groups[0]["lr"]:.2e}'})
        
        train_loss = total_loss / len(train_loader)
        
        # Validate
        val_loss, rpa_50, rpa_25, rca, mean_error = evaluate(model, val_loader, criterion, device, use_amp)
        
        # Step scheduler (epoch-based)
        if not step_per_batch:
            if config['scheduler'] == 'plateau':
                scheduler.step(rpa_50)
            else:
                scheduler.step()
        
        current_lr = optimizer.param_groups[0]['lr']
        epoch_time = time.time() - epoch_start
        epoch_times.append(epoch_time)
        
        # Record history
        history['train_loss'].append(train_loss)
        history['val_loss'].append(val_loss)
        history['rpa_50'].append(rpa_50)
        history['rpa_25'].append(rpa_25)
        history['rca'].append(rca)
        history['lr'].append(current_lr)
        
        # Print results
        print(f"\n┌─────────────────────────────────────────┐")
        print(f"│ Epoch {epoch}/{config['epochs']} ({epoch_time:.1f}s)")
        print(f"├─────────────────────────────────────────┤")
        print(f"│ Train Loss:     {train_loss:.6f}")
        print(f"│ Val Loss:       {val_loss:.6f}")
        print(f"│ Learning Rate:  {current_lr:.2e}")
        print(f"├─────────────────────────────────────────┤")
        print(f"│ RPA (50 cents): {rpa_50:6.2f}%")
        print(f"│ RPA (25 cents): {rpa_25:6.2f}%")
        print(f"│ RCA:            {rca:6.2f}%")
        print(f"│ Mean Error:     {mean_error:6.1f} cents")
        print(f"└─────────────────────────────────────────┘")
        
        # Early stopping check
        should_save = early_stopping(rpa_50, epoch)
        
        if should_save:
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'rpa_50': rpa_50,
                'rpa_25': rpa_25,
                'rca': rca,
                'config': config,
            }, os.path.join(config['save_dir'], f"crepe_best_{config['model_suffix']}.pth"))
            print(f"✓ Saved best model (RPA: {rpa_50:.2f}%)")
        else:
            print(f"⏳ No improvement for {early_stopping.counter}/{config['patience']} epochs "
                  f"(best: {early_stopping.best_score:.2f}% at epoch {early_stopping.best_epoch})")
        
        if early_stopping.early_stop:
            print(f"\n⚠️  Early stopping triggered after {epoch} epochs")
            print(f"   Best RPA: {early_stopping.best_score:.2f}% at epoch {early_stopping.best_epoch}")
            break
    
    # Save final model
    torch.save(model.state_dict(), os.path.join(config['save_dir'], f"crepe_final_log_{config['model_suffix']}.pth"))
    with open(os.path.join(config['save_dir'], f"training_history_{config['model_suffix']}.pkl"), 'wb') as f:
        pickle.dump(history, f)
    
    # Plot training curves
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    epochs = range(1, len(history['train_loss']) + 1)
    
    axes[0, 0].plot(epochs, history['train_loss'], label='Train')
    axes[0, 0].plot(epochs, history['val_loss'], label='Val')
    axes[0, 0].set_xlabel('Epoch')
    axes[0, 0].set_ylabel('Loss')
    axes[0, 0].set_title('Training Loss')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)
    
    axes[0, 1].plot(epochs, history['rpa_50'], label='RPA 50c')
    axes[0, 1].plot(epochs, history['rpa_25'], label='RPA 25c')
    axes[0, 1].set_xlabel('Epoch')
    axes[0, 1].set_ylabel('Accuracy (%)')
    axes[0, 1].set_title('Raw Pitch Accuracy')
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)
    
    axes[1, 0].plot(epochs, history['lr'])
    axes[1, 0].set_xlabel('Epoch')
    axes[1, 0].set_ylabel('Learning Rate')
    axes[1, 0].set_title('Learning Rate Schedule')
    axes[1, 0].set_yscale('log')
    axes[1, 0].grid(True, alpha=0.3)
    
    axes[1, 1].text(0.5, 0.6, f'Best RPA: {early_stopping.best_score:.2f}%', 
                   ha='center', va='center', fontsize=20, weight='bold')
    axes[1, 1].text(0.5, 0.4, f'Epoch: {early_stopping.best_epoch}', 
                   ha='center', va='center', fontsize=14)
    axes[1, 1].text(0.5, 0.2, f'Avg epoch time: {np.mean(epoch_times):.1f}s', 
                   ha='center', va='center', fontsize=12)
    axes[1, 1].axis('off')
    
    plt.tight_layout()
    plt.savefig(os.path.join(config['save_dir'], f"training_curves_{config['model_suffix']}.png"), dpi=150)
    plt.close()
    
    print("\n" + "=" * 80)
    print("✓ Training complete!")
    print("=" * 80)
    print(f"\nBest validation RPA: {early_stopping.best_score:.2f}% at epoch {early_stopping.best_epoch}")
    print(f"Average epoch time: {np.mean(epoch_times):.1f}s")
    print(f"Models saved to: {config['save_dir']}")


if __name__ == "__main__":
    main()
