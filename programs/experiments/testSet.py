"""
Test CREPE model on test set - STANDALONE VERSION
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import numpy as np
import pickle
import os
from typing import Tuple, List, Dict
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
    """CREPE dataset for RF signals."""
    
    def __init__(
        self,
        iq_dict: Dict[str, np.ndarray],
        bin_list: List[int] = None,
        snr_range: Tuple[int, int] = None,
        target_length: int = CREPE_FRAME_LENGTH,
        gaussian_sigma: float = 1.25,
    ):
        self.target_length = target_length
        self.gaussian_sigma = gaussian_sigma
        
        # Filter samples based on bin and SNR
        self.samples = []
        self.labels = []
        
        for key, signal in iq_dict.items():
            parts = key.split('_')
            bin_idx = int(parts[1])
            snr = int(parts[3])
            
            if bin_list is not None and bin_idx not in bin_list:
                continue
            
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
        """Create Gaussian-smoothed label."""
        label = np.zeros(CREPE_N_BINS, dtype=np.float32)
        
        for i in range(CREPE_N_BINS):
            label[i] = np.exp(-((i - true_bin) ** 2) / (2 * self.gaussian_sigma ** 2))
        
        return label


# =============================================================================
# Model
# =============================================================================

class CREPE(nn.Module):
    """CREPE model - exact architecture from paper."""
    
    def __init__(self, dropout: float = 0.25):
        super(CREPE, self).__init__()
        
        self.conv1 = nn.Conv1d(1, 1024, kernel_size=512, stride=4, padding=254)
        self.bn1 = nn.BatchNorm1d(1024)
        self.drop1 = nn.Dropout(dropout)
        self.pool1 = nn.MaxPool1d(2, stride=2)
        
        self.conv2 = nn.Conv1d(1024, 128, kernel_size=64, stride=1, padding=32)
        self.bn2 = nn.BatchNorm1d(128)
        self.drop2 = nn.Dropout(dropout)
        self.pool2 = nn.MaxPool1d(2, stride=2)
        
        self.conv3 = nn.Conv1d(128, 128, kernel_size=64, stride=1, padding=32)
        self.bn3 = nn.BatchNorm1d(128)
        self.drop3 = nn.Dropout(dropout)
        self.pool3 = nn.MaxPool1d(2, stride=2)
        
        self.conv4 = nn.Conv1d(128, 128, kernel_size=64, stride=1, padding=32)
        self.bn4 = nn.BatchNorm1d(128)
        self.drop4 = nn.Dropout(dropout)
        self.pool4 = nn.MaxPool1d(2, stride=2)
        
        self.conv5 = nn.Conv1d(128, 256, kernel_size=64, stride=1, padding=32)
        self.bn5 = nn.BatchNorm1d(256)
        self.drop5 = nn.Dropout(dropout)
        self.pool5 = nn.MaxPool1d(2, stride=2)
        
        self.conv6 = nn.Conv1d(256, 512, kernel_size=64, stride=1, padding=32)
        self.bn6 = nn.BatchNorm1d(512)
        self.drop6 = nn.Dropout(dropout)
        self.pool6 = nn.MaxPool1d(2, stride=2)
        
        # Calculate the actual output size
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
# Evaluation Metrics
# =============================================================================

def evaluate_predictions(predictions: np.ndarray, targets: np.ndarray) -> Tuple[float, float, float, float]:
    """Evaluate pitch predictions using CREPE metrics."""
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
    
    # RPA @ 50 cents
    rpa_50 = np.mean(errors_cents <= 50) * 100
    
    # RPA @ 25 cents
    rpa_25 = np.mean(errors_cents <= 25) * 100
    
    # RCA @ 50 cents
    chroma_errors = np.minimum(errors_cents % 1200, 1200 - (errors_cents % 1200))
    rca = np.mean(chroma_errors <= 50) * 100
    
    # Mean error
    mean_error = np.mean(errors_cents)
    
    return rpa_50, rpa_25, rca, mean_error


# =============================================================================
# Test Function
# =============================================================================

def evaluate_test_set(config, test_dataset, device='cuda'):
    """Evaluate the best model on the test set."""
    
    best_model_path = os.path.join(config['save_dir'], 'crepe_best.pth')
    
    print(f"\n🔄 Loading best model from {best_model_path}...")
    
    # FIX THE UNPICKLING ERROR
    torch.serialization.add_safe_globals([np.core.multiarray.scalar])
    
    checkpoint = torch.load(best_model_path, weights_only=False, map_location=device)
    
    # Create model and load weights
    model = CREPE(dropout=config['dropout']).to(device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    
    print(f"✓ Loaded model from epoch {checkpoint['epoch']}")
    print(f"  Best validation RPA: {checkpoint['rpa_50']:.2f}%")
    
    # Create test dataloader
    test_loader = DataLoader(
        test_dataset,
        batch_size=config['batch_size'],
        shuffle=False,
        num_workers=0
    )
    
    # Evaluate
    all_preds = []
    all_targets = []
    test_loss = 0.0
    criterion = nn.BCEWithLogitsLoss()
    
    print(f"\n📊 Evaluating on test set ({len(test_dataset)} samples)...")
    
    with torch.no_grad():
        for batch_idx, (x, y) in enumerate(tqdm(test_loader, desc="Test")):
            x, y = x.to(device), y.to(device)
            
            logits = model(x)
            loss = criterion(logits, y)
            test_loss += loss.item()
            
            all_preds.append(logits.cpu().numpy())
            true_bins = torch.argmax(y, dim=1).cpu().numpy()
            all_targets.append(true_bins)
    
    all_preds = np.concatenate(all_preds, axis=0)
    all_targets = np.concatenate(all_targets, axis=0)
    test_loss /= len(test_loader)
    
    # Calculate metrics
    rpa_50, rpa_25, rca, mean_error = evaluate_predictions(all_preds, all_targets)
    
    # Print results
    print("\n" + "="*80)
    print("📊 FINAL TEST SET EVALUATION")
    print("="*80)
    print(f"\n┌─────────────────────────────────────┐")
    print(f"│ Test Loss:          {test_loss:.6f}      │")
    print(f"├─────────────────────────────────────┤")
    print(f"│ RPA (50 cents):     {rpa_50:6.2f}%       │")
    print(f"│ RPA (25 cents):     {rpa_25:6.2f}%       │")
    print(f"│ RCA (exact bin):    {rca:6.2f}%       │")
    print(f"│ Mean Error:         {mean_error:6.1f} cents     │")
    print(f"└─────────────────────────────────────┘")
    
    return {
        'test_loss': test_loss,
        'rpa_50': rpa_50,
        'rpa_25': rpa_25,
        'rca': rca,
        'mean_error': mean_error,
        'predictions': all_preds,
        'targets': all_targets
    }


# =============================================================================
# Main
# =============================================================================

if __name__ == "__main__":
    print("=" * 80)
    print("CREPE Model - Test Set Evaluation")
    print("=" * 80)
    
    config = {
        'data_path': './IQData/iq_dict_crepe_dirac_comb.pkl',
        'batch_size': 32,
        'dropout': 0.25,
        'gaussian_sigma': 1.25,
        'device': 'cuda' if torch.cuda.is_available() else 'cpu',
        'save_dir': './models_crepe/',
    }
    
    print(f"\nConfiguration:")
    for k, v in config.items():
        print(f"  {k}: {v}")
    
    # Load dataset
    print(f"\n📂 Loading data from {config['data_path']}...")
    with open(config['data_path'], 'rb') as f:
        iq_dict = pickle.load(f)
    print(f"✓ Loaded {len(iq_dict)} samples")
    
    # Get test bins (same split as training)
    all_bins = sorted(list(set([int(k.split('_')[1]) for k in iq_dict.keys()])))
    test_bins = [b for i, b in enumerate(all_bins) if i % 5 == 1]  # 20% test
    
    print(f"✓ Test bins: {len(test_bins)}")
    
    # Create test dataset
    test_dataset = CREPEDataset(
        iq_dict=iq_dict,
        bin_list=test_bins,
        snr_range=(15, 21),
        gaussian_sigma=config['gaussian_sigma']
    )
    
    # Evaluate
    results = evaluate_test_set(config, test_dataset, device=config['device'])
    
    # Save results
    results_path = os.path.join(config['save_dir'], 'test_results_standalone.pkl')
    with open(results_path, 'wb') as f:
        pickle.dump(results, f)
    
    print(f"\n💾 Results saved to: {results_path}")
    print("\n" + "="*80)
    print("✓ Test evaluation complete!")
    print("="*80)