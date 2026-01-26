"""
Train CREPE architecture on RF emanation data (instead of music).

This script:
- Loads RF IQ data from DiracCombPlots.py output (iq_dict pickle)
- Implements exact CREPE architecture (6 conv layers + dense)
- Trains on 1040-sample signals (center-cropped to 1024 for CREPE)
- Uses fundamental frequency F_h = 220 kHz as target pitch
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import pickle
import os
from typing import Tuple, Optional

# =============================================================================
# CREPE Model Architecture (Exact Reimplementation)
# =============================================================================

import torch
import torch.nn as nn
import torch.nn.functional as F

class CREPE(nn.Module):
    """
    Exact implementation of the CREPE architecture from arXiv:1802.06182.
    """
    def __init__(self, capacity_multiplier: int = 32, dropout: float = 0.25):
        super(CREPE, self).__init__()
        
        c = capacity_multiplier  # 32 for 'full', 4 for 'tiny'

        # Layer 1: [1, 1024] -> [1024 filters, 128 width]
        self.conv1 = nn.Conv1d(1, 32 * c, kernel_size=512, stride=4, padding=254)
        self.bn1 = nn.BatchNorm1d(32 * c)

        # Layers 2-4: Filter count 128 (for c=32)
        self.conv2 = nn.Conv1d(32 * c, 4 * c, kernel_size=64, padding=32)
        self.bn2 = nn.BatchNorm1d(4 * c)

        self.conv3 = nn.Conv1d(4 * c, 4 * c, kernel_size=64, padding=32)
        self.bn3 = nn.BatchNorm1d(4 * c)

        self.conv4 = nn.Conv1d(4 * c, 4 * c, kernel_size=64, padding=32)
        self.bn4 = nn.BatchNorm1d(4 * c)

        # Layer 5: Filter count 256 (for c=32)
        self.conv5 = nn.Conv1d(4 * c, 8 * c, kernel_size=64, padding=32)
        self.bn5 = nn.BatchNorm1d(8 * c)

        # Layer 6: Filter count 512 (for c=32)
        self.conv6 = nn.Conv1d(8 * c, 16 * c, kernel_size=64, padding=32)
        self.bn6 = nn.BatchNorm1d(16 * c)

        # Dense layers: 2048 -> 2048 -> 360 (as shown in Figure 1)
        self.fc1 = nn.Linear(16 * c * 4, 2048)
        self.dropout = nn.Dropout(dropout)
        self.fc2 = nn.Linear(2048, 360)

    def forward(self, x):
        # Input should be (Batch, 1, 1024)
        if x.dim() == 2:
            x = x.unsqueeze(1)

        # Conv blocks (no dropout here, only after fc1)
        x = F.max_pool1d(F.relu(self.bn1(self.conv1(x))), 2)
        x = F.max_pool1d(F.relu(self.bn2(self.conv2(x))), 2)
        x = F.max_pool1d(F.relu(self.bn3(self.conv3(x))), 2)
        x = F.max_pool1d(F.relu(self.bn4(self.conv4(x))), 2)
        x = F.max_pool1d(F.relu(self.bn5(self.conv5(x))), 2)
        x = F.max_pool1d(F.relu(self.bn6(self.conv6(x))), 2)

        # Flatten and dense layers
        x = x.flatten(1)
        x = F.relu(self.fc1(x))
        x = self.dropout(x)  # Dropout only here
        x = self.fc2(x)
        
        return x  # Return raw logits 

# =============================================================================
# Dataset for RF Emanation Data
# =============================================================================

class RFEmanationDataset(Dataset):
    """
    Dataset wrapper for RF IQ data with known fundamental frequency
    """
    def __init__(self, iq_dict: dict, F_h: float, Fs: float, target_length: int = 1024, n_augment: int = 200):
        """
        Args:
            iq_dict: Dict mapping SNR keys to IQ arrays
            F_h: Fundamental frequency in Hz (e.g., 220e3)
            Fs: Sampling rate in Hz (e.g., 25e6)
            target_length: Target frame length (1024 for CREPE)
            n_augment: Number of augmented samples to create per original sample
        """
        self.samples = []
        self.targets = []
        self.snr_labels = []
        
        # Filter to SNR range [15, 20]
        import re
        def parse_snr(key):
            match = re.search(r'[-+]?\d+', str(key))
            return int(match.group()) if match else None
        
        filtered_dict = {}
        for key, val in iq_dict.items():
            snr = parse_snr(key)
            if snr is not None and 15 <= snr <= 20:
                filtered_dict[key] = val
        
        if len(filtered_dict) == 0:
            raise ValueError("No samples found with SNR between 15 and 20!")
        
        print(f"✓ Filtered to {len(filtered_dict)} SNR levels in range [15, 20]: {list(filtered_dict.keys())}")
        
        # Convert F_h to CREPE pitch bin
        # CREPE bins: 360 bins, 20 cents each, starting at ~32.7 Hz (C1)
        # Bin 0 corresponds to 10 * 2^(1997.38/1200) ≈ 32.7 Hz
        # F_h = 220 kHz -> cents = 1200 * log2(220000/10) ≈ 16858 cents
        # Bin index = (cents - 1997.38) / 20
        cents = 1200.0 * np.log2(F_h / 10.0)
        bin_index = int((cents - 1997.3794084376191) / 20.0)
        
        # Clamp to valid range [0, 359]
        bin_index = np.clip(bin_index, 0, 359)
        
        print(f"F_h = {F_h/1e3:.1f} kHz -> {cents:.1f} cents -> CREPE bin {bin_index}")
        
        # Create soft Gaussian target (as done in CREPE paper)
        target = self._create_gaussian_target(bin_index, sigma=25)
        
        # Process each SNR level with augmentation
        for key, iq in filtered_dict.items():
            # Extract magnitude
            if np.iscomplexobj(iq):
                base_signal = np.abs(iq).astype(np.float32)
            else:
                base_signal = np.asarray(iq, dtype=np.float32)
            
            # Create n_augment versions of this signal
            for aug_idx in range(n_augment):
                # Apply augmentations:
                # 1. Random time shift (circular)
                shift = np.random.randint(-128, 129)
                signal = np.roll(base_signal, shift)
                
                # 2. Random crop position (instead of always center)
                if len(signal) >= target_length:
                    max_start = len(signal) - target_length
                    start = np.random.randint(0, max_start + 1)
                    signal = signal[start:start+target_length]
                else:
                    pad = target_length - len(signal)
                    signal = np.pad(signal, (pad//2, pad - pad//2), mode='constant')
                
                # 3. Add very small Gaussian noise (to create variation without changing SNR much)
                noise_scale = 0.001 * signal.std()
                signal = signal + np.random.normal(0, noise_scale, signal.shape).astype(np.float32)
                
                # 4. Random amplitude scaling (±5%)
                scale = np.random.uniform(0.95, 1.05)
                signal = signal * scale
                
                # Normalize (zero mean, unit variance)
                signal = (signal - signal.mean()) / (signal.std() + 1e-8)
                
                self.samples.append(signal)
                self.targets.append(target)
                self.snr_labels.append(f"{key}_aug{aug_idx}")
        
        print(f"✓ Created {len(self.samples)} augmented samples ({n_augment} per original)")
    
    def _create_gaussian_target(self, center_bin: int, sigma: float = 25.0, n_bins: int = 360) -> np.ndarray:
        """
        Create Gaussian soft target centered at center_bin
        
        Args:
            center_bin: Center bin index
            sigma: Standard deviation in bins
            n_bins: Total number of bins (360)
        Returns:
            target: (360,) array with Gaussian distribution
        """
        bins = np.arange(n_bins)
        target = np.exp(-0.5 * ((bins - center_bin) / sigma) ** 2)
        target = target / target.sum()  # Normalize to sum to 1
        return target.astype(np.float32)
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        return self.samples[idx], self.targets[idx]


# =============================================================================
# Training Functions
# =============================================================================

def train_epoch(model: nn.Module, loader: DataLoader, optimizer, criterion, device: str):
    """Train for one epoch"""
    model.train()
    total_loss = 0.0
    
    for batch_idx, (x, y) in enumerate(loader):
        x = x.to(device)
        y = y.to(device)
        
        optimizer.zero_grad()
        logits = model(x)
        loss = criterion(logits, y)
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item() * x.size(0)
        
        if batch_idx % 10 == 0:
            print(f"  Batch {batch_idx}/{len(loader)}, Loss: {loss.item():.6f}")
    
    return total_loss / len(loader.dataset)


def evaluate(model: nn.Module, loader: DataLoader, criterion, device: str):
    """Evaluate model"""
    model.eval()
    total_loss = 0.0
    
    with torch.no_grad():
        for x, y in loader:
            x = x.to(device)
            y = y.to(device)
            logits = model(x)
            loss = criterion(logits, y)
            total_loss += loss.item() * x.size(0)
    
    return total_loss / len(loader.dataset)


# =============================================================================
# Main Training Script
# =============================================================================

def main():
    print("=" * 80)
    print("Training CREPE Architecture on RF Emanation Data")
    print("=" * 80)
    
    # Configuration
    config = {
        'data_path': './IQData/iq_dict_SNR_20_toMinus40_dc_0_ptsecsdata_1_Fh_220_kHz.pkl',
        'F_h': 220e3,  # Fundamental frequency in Hz
        'Fs': 25e6,    # Sampling rate in Hz
        'input_length': 1024,  # CREPE input length
        'capacity': 32,  # 'full' model capacity multiplier
        'batch_size': 16,
        'epochs': 50,
        'lr': 1e-4,
        'device': 'cuda' if torch.cuda.is_available() else 'cpu',
        'save_dir': './models_rf/',
    }
    
    print("\nConfiguration:")
    for k, v in config.items():
        print(f"  {k}: {v}")
    
    os.makedirs(config['save_dir'], exist_ok=True)
    
    # Load data
    print(f"\n📂 Loading data from {config['data_path']}...")
    with open(config['data_path'], 'rb') as f:
        iq_dict = pickle.load(f)
    
    print(f"✓ Loaded {len(iq_dict)} SNR levels")
    for key in list(iq_dict.keys())[:3]:
        print(f"  {key}: {iq_dict[key].shape}")
    
    # Create dataset
    print("\n📊 Creating dataset...")
    dataset = RFEmanationDataset(
        iq_dict=iq_dict,
        F_h=config['F_h'],
        Fs=config['Fs'],
        target_length=config['input_length'],
        n_augment=1024  # Creates 200 augmented samples per original
    )
    
    print(f"✓ Dataset created: {len(dataset)} samples")
    
    # Split into train/val
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(
        dataset, [train_size, val_size],
        generator=torch.Generator().manual_seed(42)
    )
    
    train_loader = DataLoader(train_dataset, batch_size=config['batch_size'], shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=config['batch_size'], shuffle=False)
    
    print(f"✓ Train: {train_size}, Val: {val_size}")
    
    # Create model
    print(f"\n🏗️  Creating CREPE model (capacity={config['capacity']})...")
    model = CREPE(
        capacity_multiplier=config['capacity'],
        dropout=0.25
    )
    model = model.to(config['device'])
    
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"✓ Model created: {n_params:,} trainable parameters")
    
    # Optimizer and loss
    optimizer = torch.optim.Adam(model.parameters(), lr=config['lr'])
    criterion = nn.BCEWithLogitsLoss()
    
    # Training loop
    print(f"\n🚀 Starting training for {config['epochs']} epochs...")
    print("=" * 80)
    
    best_val_loss = float('inf')
    
    for epoch in range(1, config['epochs'] + 1):
        print(f"\nEpoch {epoch}/{config['epochs']}")
        print("-" * 80)
        
        train_loss = train_epoch(model, train_loader, optimizer, criterion, config['device'])
        val_loss = evaluate(model, val_loader, criterion, config['device'])
        
        print(f"\n  Train Loss: {train_loss:.6f}")
        print(f"  Val Loss:   {val_loss:.6f}")
        
        # Save best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            save_path = os.path.join(config['save_dir'], 'crepe_rf_best.pth')
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'train_loss': train_loss,
                'val_loss': val_loss,
                'config': config,
            }, save_path)
            print(f"  ✓ Saved best model to {save_path}")
        
        # Save checkpoint every 10 epochs
        if epoch % 10 == 0:
            checkpoint_path = os.path.join(config['save_dir'], f'crepe_rf_epoch{epoch}.pth')
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'train_loss': train_loss,
                'val_loss': val_loss,
                'config': config,
            }, checkpoint_path)
            print(f"  ✓ Saved checkpoint to {checkpoint_path}")
    
    print("\n" + "=" * 80)
    print("✅ Training complete!")
    print(f"Best validation loss: {best_val_loss:.6f}")
    print(f"Models saved to: {config['save_dir']}")


if __name__ == "__main__":
    main()
