"""
Train CREPE model for pitch estimation on RF signals - FIXED VERSION

Key fixes:
1. Correct model architecture with proper dimension calculation
2. Proper train/val split - split by bins with overlap to enable interpolation
3. Better data loading - handle complex RF signals correctly
4. Correct label generation - Gaussian smoothing as per paper
5. Proper evaluation metrics - RPA, RCA as defined in paper
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import numpy as np
import pickle
import os
from typing import Tuple, List, Dict
import matplotlib.pyplot as plt
from tqdm import tqdm


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
# Dataset
# =============================================================================

class CREPEDataset(Dataset):
    """
    CREPE dataset for RF signals - FIXED VERSION
    
    Key fixes:
    - Handles complex RF signals (magnitude only for now)
    - Proper Gaussian label smoothing (sigma in bins, not Hz)
    - Efficient caching of labels
    """
    
    def __init__(
        self,
        iq_dict: Dict[str, np.ndarray],
        bin_list: List[int] = None,
        snr_range: Tuple[int, int] = None,
        target_length: int = CREPE_FRAME_LENGTH,
        gaussian_sigma: float = 1.25,  # In bins (25 cents / 20 cents per bin)
    ):
        """
        Args:
            iq_dict: Dictionary mapping keys to complex IQ signals
            bin_list: List of bins to include (None = all)
            snr_range: (min_snr, max_snr) to include (None = all)
            target_length: Target length for signals
            gaussian_sigma: Gaussian blur sigma in BINS (not cents!)
        """
        self.target_length = target_length
        self.gaussian_sigma = gaussian_sigma
        
        # Filter samples based on bin and SNR
        self.samples = []
        self.labels = []
        
        for key, signal in iq_dict.items():
            # Parse key: "BIN_XXX_SNR_YY_AUG_ZZ"
            parts = key.split('_')
            bin_idx = int(parts[1])
            snr = int(parts[3])
            
            # Filter by bin
            if bin_list is not None and bin_idx not in bin_list:
                continue
            
            # Filter by SNR
            if snr_range is not None:
                if snr < snr_range[0] or snr > snr_range[1]:
                    continue
            
            self.samples.append((key, signal))
            self.labels.append(bin_idx)
        
        print(f"✓ Created dataset with {len(self.samples)} samples")
        if len(self.samples) > 0:
            unique_bins = sorted(list(set(self.labels)))
            print(f"  Unique bins: {len(unique_bins)} "
                  f"(range: {min(unique_bins)} to {max(unique_bins)})")
            unique_freqs = [crepe_bin_to_hz(b) for b in unique_bins]
            print(f"  Frequency range: {min(unique_freqs):.1f} Hz to {max(unique_freqs):.1f} Hz")
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        key, signal = self.samples[idx]
        bin_idx = self.labels[idx]
        
        # Take magnitude of complex signal
        signal = np.abs(signal)
        
        # Pad or truncate to target length
        if len(signal) < self.target_length:
            signal = np.pad(signal, (0, self.target_length - len(signal)))
        else:
            signal = signal[:self.target_length]
        
        # Normalize
        signal = signal / (np.max(np.abs(signal)) + 1e-8)
        
        # Create Gaussian-smoothed label
        label = self._create_gaussian_label(bin_idx)
        
        return torch.FloatTensor(signal).unsqueeze(0), torch.FloatTensor(label)
    
    def _create_gaussian_label(self, true_bin: int) -> np.ndarray:
        """
        Create Gaussian-smoothed label as per CREPE paper.
        
        From paper: "the target is Gaussian-blurred in frequency such that 
        the energy surrounding a ground truth frequency decays with a 
        standard deviation of 25 cents"
        
        25 cents / 20 cents per bin = 1.25 bins sigma
        """
        label = np.zeros(CREPE_N_BINS, dtype=np.float32)
        
        # Gaussian centered at true_bin with sigma in bins
        for i in range(CREPE_N_BINS):
            label[i] = np.exp(-((i - true_bin) ** 2) / (2 * self.gaussian_sigma ** 2))
        
        # Normalize (though paper doesn't explicitly do this)
        # label = label / (label.sum() + 1e-8)
        
        return label


# =============================================================================
# Model - FIXED ARCHITECTURE
# =============================================================================

class CREPE(nn.Module):
    """
    CREPE model - exact architecture from paper with FIXED dimensions.
    
    From paper:
    - 6 convolutional layers
    - Input: 1024 samples @ 16kHz
    - Output: 360-dimensional pitch vector
    - Batch norm + dropout(0.25) in conv layers
    """
    
    def __init__(self, capacity_multiplier: int = 32, dropout: float = 0.25):
        """
        Args:
            capacity_multiplier: Channel multiplier (paper uses 32, 48, 64, 96)
            dropout: Dropout rate (paper uses 0.25)
        """
        super(CREPE, self).__init__()
        
        cap = capacity_multiplier
        
        # Layer 1: conv (1024) -> pool -> (128 after pool)
        self.conv1 = nn.Conv1d(1, cap, kernel_size=512, stride=4, padding=254)
        self.bn1 = nn.BatchNorm1d(cap)
        self.drop1 = nn.Dropout(dropout)
        self.pool1 = nn.MaxPool1d(2, stride=2)
        
        # Layer 2: conv -> pool -> (64 after pool)
        self.conv2 = nn.Conv1d(cap, cap, kernel_size=64, stride=1, padding=32)
        self.bn2 = nn.BatchNorm1d(cap)
        self.drop2 = nn.Dropout(dropout)
        self.pool2 = nn.MaxPool1d(2, stride=2)
        
        # Layer 3: conv -> pool -> (32 after pool)
        self.conv3 = nn.Conv1d(cap, cap * 2, kernel_size=64, stride=1, padding=32)
        self.bn3 = nn.BatchNorm1d(cap * 2)
        self.drop3 = nn.Dropout(dropout)
        self.pool3 = nn.MaxPool1d(2, stride=2)
        
        # Layer 4: conv -> pool -> (16 after pool)
        self.conv4 = nn.Conv1d(cap * 2, cap * 2, kernel_size=64, stride=1, padding=32)
        self.bn4 = nn.BatchNorm1d(cap * 2)
        self.drop4 = nn.Dropout(dropout)
        self.pool4 = nn.MaxPool1d(2, stride=2)
        
        # Layer 5: conv -> pool -> (8 after pool)
        self.conv5 = nn.Conv1d(cap * 2, cap * 4, kernel_size=64, stride=1, padding=32)
        self.bn5 = nn.BatchNorm1d(cap * 4)
        self.drop5 = nn.Dropout(dropout)
        self.pool5 = nn.MaxPool1d(2, stride=2)
        
        # Layer 6: conv -> pool -> (4 after pool)
        self.conv6 = nn.Conv1d(cap * 4, cap * 4, kernel_size=64, stride=1, padding=32)
        self.bn6 = nn.BatchNorm1d(cap * 4)
        self.drop6 = nn.Dropout(dropout)
        self.pool6 = nn.MaxPool1d(2, stride=2)
        
        # After 6 pools of stride 2: 1024 -> 512 -> 256 -> 128 -> 64 -> 32 -> 16
        # But with stride=4 in conv1: 1024 -> 256 -> 128 -> 64 -> 32 -> 16 -> 8 -> 4
        # Actual calculation needed - let's compute it properly
        
        # Calculate the actual output size by doing a forward pass
        with torch.no_grad():
            dummy_input = torch.randn(1, 1, 1024)
            x = self.pool1(self.drop1(F.relu(self.bn1(self.conv1(dummy_input)))))
            x = self.pool2(self.drop2(F.relu(self.bn2(self.conv2(x)))))
            x = self.pool3(self.drop3(F.relu(self.bn3(self.conv3(x)))))
            x = self.pool4(self.drop4(F.relu(self.bn4(self.conv4(x)))))
            x = self.pool5(self.drop5(F.relu(self.bn5(self.conv5(x)))))
            x = self.pool6(self.drop6(F.relu(self.bn6(self.conv6(x)))))
            flattened_size = x.view(x.size(0), -1).size(1)
        
        # FC layer with correct input size
        self.fc = nn.Linear(flattened_size, 360)
    
    def forward(self, x):
        # Input: (batch, 1, 1024)
        
        x = self.pool1(self.drop1(F.relu(self.bn1(self.conv1(x)))))
        x = self.pool2(self.drop2(F.relu(self.bn2(self.conv2(x)))))
        x = self.pool3(self.drop3(F.relu(self.bn3(self.conv3(x)))))
        x = self.pool4(self.drop4(F.relu(self.bn4(self.conv4(x)))))
        x = self.pool5(self.drop5(F.relu(self.bn5(self.conv5(x)))))
        x = self.pool6(self.drop6(F.relu(self.bn6(self.conv6(x)))))
        
        # Flatten
        x = x.view(x.size(0), -1)
        
        # FC to 360 bins (logits, sigmoid applied in loss)
        x = self.fc(x)
        
        return x


# =============================================================================
# Evaluation Metrics
# =============================================================================

def evaluate_predictions(predictions: np.ndarray, targets: np.ndarray) -> Tuple[float, float, float, float]:
    """
    Evaluate pitch predictions using CREPE metrics.
    
    Args:
        predictions: (N, 360) predicted distributions
        targets: (N,) true bin indices
    
    Returns:
        rpa_50: Raw Pitch Accuracy @ 50 cents
        rpa_25: Raw Pitch Accuracy @ 25 cents  
        rca: Raw Chroma Accuracy @ 50 cents
        mean_error: Mean pitch error in cents
    """
    # Get predicted bins (weighted average as per paper)
    bin_indices = np.arange(360)
    
    # Apply sigmoid to logits
    pred_probs = 1 / (1 + np.exp(-predictions))
    
    # Weighted average
    pred_bins = np.sum(pred_probs * bin_indices, axis=1) / np.sum(pred_probs, axis=1)
    
    # Convert to frequencies
    pred_freqs = np.array([crepe_bin_to_hz(b) for b in pred_bins])
    true_freqs = np.array([crepe_bin_to_hz(b) for b in targets])
    
    # Compute errors in cents
    errors_cents = np.abs(1200 * np.log2(pred_freqs / (true_freqs + 1e-8)))
    
    # RPA @ 50 cents (quarter tone)
    rpa_50 = np.mean(errors_cents <= 50) * 100
    
    # RPA @ 25 cents
    rpa_25 = np.mean(errors_cents <= 25) * 100
    
    # RCA @ 50 cents (chroma = pitch class, mod 12 semitones = 1200 cents)
    chroma_errors = np.minimum(errors_cents % 1200, 1200 - (errors_cents % 1200))
    rca = np.mean(chroma_errors <= 50) * 100
    
    # Mean error
    mean_error = np.mean(errors_cents)
    
    return rpa_50, rpa_25, rca, mean_error


def evaluate(model, dataloader, criterion, device):
    """Evaluate model on a dataset."""
    model.eval()
    total_loss = 0
    all_predictions = []
    all_targets = []
    
    with torch.no_grad():
        for inputs, labels in dataloader:
            inputs = inputs.to(device)
            labels = labels.to(device)
            
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            
            total_loss += loss.item()
            all_predictions.append(outputs.cpu().numpy())
            
            # Get true bin from Gaussian label
            true_bins = torch.argmax(labels, dim=1).cpu().numpy()
            all_targets.append(true_bins)
    
    avg_loss = total_loss / len(dataloader)
    all_predictions = np.concatenate(all_predictions, axis=0)
    all_targets = np.concatenate(all_targets, axis=0)
    
    rpa_50, rpa_25, rca, mean_error = evaluate_predictions(all_predictions, all_targets)
    
    return avg_loss, rpa_50, rpa_25, rca, mean_error


# =============================================================================
# Training
# =============================================================================

def train_epoch(model, dataloader, criterion, optimizer, device, epoch):
    """Train for one epoch."""
    model.train()
    total_loss = 0
    
    pbar = tqdm(dataloader, desc=f'Epoch {epoch}')
    for batch_idx, (inputs, labels) in enumerate(pbar):
        inputs = inputs.to(device)
        labels = labels.to(device)
        
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
        pbar.set_postfix({'loss': f'{loss.item():.6f}'})
    
    return total_loss / len(dataloader)


def main():
    # Configuration
    config = {
        'data_path': './IQData/iq_dict_crepe_dirac_comb.pkl',
        'batch_size': 32,
        'epochs': 30,
        'lr': 0.0002,  # As per paper
        'capacity': 32,  # Model capacity
        'dropout': 0.25,  # As per paper
        'gaussian_sigma': 1.25,  # 25 cents / 20 cents per bin
        'device': 'cuda' if torch.cuda.is_available() else 'cpu',
        'save_dir': './models_crepe/',
    }
    
    print("=" * 80)
    print("Training CREPE Model - Fixed Version")
    print("=" * 80)
    print(f"\nConfiguration:")
    for k, v in config.items():
        print(f"  {k}: {v}")
    
    # Load data
    print(f"\n📂 Loading data from {config['data_path']}...")
    with open(config['data_path'], 'rb') as f:
        iq_dict = pickle.load(f)
    print(f"✓ Loaded {len(iq_dict)} samples")
    
    # Extract all bins from dataset
    all_bins = sorted(list(set([int(k.split('_')[1]) for k in iq_dict.keys()])))
    print(f"✓ Found {len(all_bins)} unique bins (range: {min(all_bins)}-{max(all_bins)})")
    
    # CRITICAL FIX: Split bins with OVERLAP
    # Use 80% of bins for training, 20% for validation
    # But ensure there's overlap (every 5th bin goes to validation)
    train_bins = [b for i, b in enumerate(all_bins) if i % 5 != 0]
    val_bins = [b for i, b in enumerate(all_bins) if i % 5 == 0]
    
    print(f"\n📊 Train/Val split:")
    print(f"  Train bins: {len(train_bins)} (every 4 out of 5 bins)")
    print(f"  Val bins: {len(val_bins)} (every 5th bin)")
    print(f"  Overlap: Validation bins are interspersed with training bins")
    
    # Create datasets - use high SNR for training
    train_dataset = CREPEDataset(
        iq_dict=iq_dict,
        bin_list=train_bins,
        snr_range=(0, 20),  # High SNR for training
        gaussian_sigma=config['gaussian_sigma']
    )
    
    val_dataset = CREPEDataset(
        iq_dict=iq_dict,
        bin_list=val_bins,
        snr_range=(0, 20),  # Same SNR range
        gaussian_sigma=config['gaussian_sigma']
    )
    
    train_loader = DataLoader(train_dataset, batch_size=config['batch_size'], 
                             shuffle=True, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=config['batch_size'], 
                           shuffle=False, num_workers=0)
    
    # Create model
    print(f"\n🏗️  Creating CREPE model (capacity={config['capacity']})...")
    model = CREPE(capacity_multiplier=config['capacity'], dropout=config['dropout'])
    device = torch.device(config['device'])
    model = model.to(device)
    
    # Count parameters
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"✓ Model created: {n_params:,} trainable parameters")
    
    # Loss and optimizer (as per paper)
    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=config['lr'])
    
    # Training loop
    print(f"\n🚀 Starting training for {config['epochs']} epochs...")
    print("=" * 80)
    
    best_rpa = 0
    history = {'train_loss': [], 'val_loss': [], 'rpa_50': [], 'rpa_25': [], 'rca': []}
    
    os.makedirs(config['save_dir'], exist_ok=True)
    
    for epoch in range(1, config['epochs'] + 1):
        # Train
        train_loss = train_epoch(model, train_loader, criterion, optimizer, device, epoch)
        
        # Validate
        val_loss, rpa_50, rpa_25, rca, mean_error = evaluate(model, val_loader, criterion, device)
        
        history['train_loss'].append(train_loss)
        history['val_loss'].append(val_loss)
        history['rpa_50'].append(rpa_50)
        history['rpa_25'].append(rpa_25)
        history['rca'].append(rca)
        
        # Print results
        print(f"\n┌─────────────────────────────────────────┐")
        print(f"│ Epoch {epoch}/{config['epochs']}")
        print(f"├─────────────────────────────────────────┤")
        print(f"│ Train Loss:     {train_loss:.6f}")
        print(f"│ Val Loss:       {val_loss:.6f}")
        print(f"├─────────────────────────────────────────┤")
        print(f"│ RPA (50 cents): {rpa_50:6.2f}%")
        print(f"│ RPA (25 cents): {rpa_25:6.2f}%")
        print(f"│ RCA:            {rca:6.2f}%")
        print(f"│ Mean Error:     {mean_error:6.1f} cents")
        print(f"└─────────────────────────────────────────┘\n")
        
        # Save best model
        if rpa_50 > best_rpa:
            best_rpa = rpa_50
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'rpa_50': rpa_50,
                'rpa_25': rpa_25,
                'rca': rca,
                'config': config,
            }, os.path.join(config['save_dir'], 'crepe_best.pth'))
            print(f"✓ Saved best model (RPA: {rpa_50:.2f}%)")
    
    # Save final model and history
    torch.save(model.state_dict(), os.path.join(config['save_dir'], 'crepe_final.pth'))
    with open(os.path.join(config['save_dir'], 'training_history.pkl'), 'wb') as f:
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
    
    axes[1, 0].plot(epochs, history['rca'])
    axes[1, 0].set_xlabel('Epoch')
    axes[1, 0].set_ylabel('Accuracy (%)')
    axes[1, 0].set_title('Raw Chroma Accuracy')
    axes[1, 0].grid(True, alpha=0.3)
    
    axes[1, 1].text(0.5, 0.5, f'Best RPA: {best_rpa:.2f}%', 
                   ha='center', va='center', fontsize=20, weight='bold')
    axes[1, 1].axis('off')
    
    plt.tight_layout()
    plt.savefig(os.path.join(config['save_dir'], 'training_curves.png'), dpi=150)
    plt.close()
    
    print("\n" + "=" * 80)
    print("✓ Training complete!")
    print("=" * 80)
    print(f"\nBest validation RPA: {best_rpa:.2f}%")
    print(f"Models saved to: {config['save_dir']}")


if __name__ == "__main__":
    main()