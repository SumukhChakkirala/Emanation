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
import argparse
from typing import Tuple, List, Dict, Optional
import matplotlib.pyplot as plt
from tqdm import tqdm
from sklearn.model_selection import train_test_split 

# =============================================================================
# Constants
# =============================================================================

CREPE_FS = 16000
CREPE_FRAME_LENGTH = 1024
CREPE_N_BINS = 360
CREPE_CENTS_PER_BIN = 20
DEFAULT_RF_FMIN = 32.7
DEFAULT_RF_FMAX = 2069.0
CASE_RF_RANGES = {
    'A': (80e3, 1e6),
    'B': (3.2e3, 100e3),
    'C': (32.0, 4e3),
}


def _infer_case_from_path(data_path: str) -> Optional[str]:
    lower_path = os.path.basename(data_path).lower()
    for case in ('a', 'b', 'c'):
        if f"case{case}" in lower_path:
            return case.upper()
    return None


def _resolve_rf_range(data_path: str, case: str, rf_fmin: Optional[float], rf_fmax: Optional[float]) -> Tuple[float, float, str]:
    if rf_fmin is not None and rf_fmax is not None:
        return float(rf_fmin), float(rf_fmax), 'manual'

    case_u = case.upper() if case else ''
    if case_u in CASE_RF_RANGES:
        lo, hi = CASE_RF_RANGES[case_u]
        return float(lo), float(hi), f'case-{case_u}'

    inferred_case = _infer_case_from_path(data_path)
    if inferred_case in CASE_RF_RANGES:
        lo, hi = CASE_RF_RANGES[inferred_case]
        return float(lo), float(hi), f'path-inferred-{inferred_case}'

    return float(DEFAULT_RF_FMIN), float(DEFAULT_RF_FMAX), 'default-crepe'


def freq_to_model_bin(freq_hz: float, fmin_hz: float, fmax_hz: float) -> float:
    freq = float(np.clip(freq_hz, fmin_hz, fmax_hz))
    lo = np.log2(float(fmin_hz))
    hi = np.log2(float(fmax_hz))
    pos = (np.log2(freq) - lo) / (hi - lo + 1e-12)
    return float(np.clip(pos * (CREPE_N_BINS - 1), 0.0, CREPE_N_BINS - 1.0))


def model_bin_to_freq(bin_idx: float, fmin_hz: float, fmax_hz: float) -> float:
    pos = float(np.clip(bin_idx, 0.0, CREPE_N_BINS - 1.0)) / float(CREPE_N_BINS - 1)
    lo = np.log2(float(fmin_hz))
    hi = np.log2(float(fmax_hz))
    return float(2.0 ** (lo + pos * (hi - lo)))

def parse_key(key: str) -> Tuple[int, int, float]:
    parts = key.split('_')
    bin_idx = int(parts[1])
    snr = int(parts[3])
    if 'FH' in parts:
        fh_idx = parts.index('FH') + 1
        f_h = float(parts[fh_idx])
    else:
        f_h = None
    return bin_idx, snr, f_h


def _fftshift_frame(
    signal_in: np.ndarray,
    target_length: int = CREPE_FRAME_LENGTH,
) -> np.ndarray:
    x = np.asarray(signal_in)
    if np.iscomplexobj(x):
        x = np.abs(x)
    x = x.astype(np.float32)

    if len(x) < target_length:
        pad_total = target_length - len(x)
        pad_left = pad_total // 2
        pad_right = pad_total - pad_left
        frame = np.pad(x, (pad_left, pad_right), mode='constant')
    elif len(x) > target_length:
        start = (len(x) - target_length) // 2
        frame = x[start:start + target_length]
    else:
        frame = x

    frame = np.fft.fftshift(frame).astype(np.float32)
    frame /= (np.max(np.abs(frame)) + 1e-8)
    return frame
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
        gaussian_sigma: float = 25.0,
        rf_fmin: float = DEFAULT_RF_FMIN,
        rf_fmax: float = DEFAULT_RF_FMAX,
        input_feature: str = 'raw',
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
        self.gaussian_sigma_bins = max(float(gaussian_sigma) / float(CREPE_CENTS_PER_BIN), 1e-6)
        self.rf_fmin = float(rf_fmin)
        self.rf_fmax = float(rf_fmax)
        self.input_feature = str(input_feature).lower()
        if self.input_feature not in ('raw', 'fftshift'):
            raise ValueError(f"Unsupported input_feature: {input_feature}. Use 'raw' or 'fftshift'.")
        
        # Filter samples based on bin and SNR
        self.samples = []
        self.fh = []
        
        for key, signal in iq_dict.items():
            # Parse key: "BIN_XXX_SNR_YY_AUG_ZZ"
            # parts = key.split('_')
            # bin_idx = int(parts[1])
            # snr = int(parts[3])
            bin_idx, snr, fh = parse_key(key)
            
            if bin_list is not None and bin_idx not in bin_list:
                continue

            if snr_range is not None:
                if snr < snr_range[0] or snr > snr_range[1]:
                    continue

            if fh is None:
                fh = model_bin_to_freq(bin_idx, self.rf_fmin, self.rf_fmax)

            self.samples.append((key, signal))
            self.fh.append(fh)
        
        print(f"[OK] Created dataset with {len(self.samples)} samples")
        # if len(self.samples) > 0:
        #     unique_bins = sorted(list(set(self.labels)))
        #     print(f"  Unique bins: {len(unique_bins)} "
        #           f"(range: {min(unique_bins)} to {max(unique_bins)})")
        #     unique_freqs = [crepe_bin_to_hz(b) for b in unique_bins]
        #     print(f"  Frequency range: {min(unique_freqs):.1f} Hz to {max(unique_freqs):.1f} Hz")
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        key, signal = self.samples[idx]
        fh = self.fh[idx]

        signal = np.asarray(signal)
        if self.input_feature == 'fftshift':
            signal = _fftshift_frame(
                signal_in=signal,
                target_length=self.target_length,
            )
        else:
            if np.iscomplexobj(signal):
                signal = np.abs(signal)
            signal = signal.astype(np.float32)

            if len(signal) < self.target_length:
                signal = np.pad(signal, (0, self.target_length - len(signal)))
            else:
                signal = signal[:self.target_length]
            
            # Normalize
            signal = signal / (np.max(np.abs(signal)) + 1e-8)
        
        # Create Gaussian-smoothed label
        label = self._create_gaussian_label(fh)
        
        return torch.FloatTensor(signal).unsqueeze(0), torch.FloatTensor(label)
    
    def _create_gaussian_label(self, fh: float) -> np.ndarray:
        """
        Create Gaussian-smoothed label as per CREPE paper (Eq. 3).
        
        From paper: "the target is Gaussian-blurred in frequency such that 
        the energy surrounding a ground truth frequency decays with a 
        standard deviation of 25 cents"
        
        Paper formula: y_i = exp(-(c_i - c_true)^2 / (2 * 25^2))
        In bins: sigma = 25 cents / 20 cents per bin = 1.25 bins
        """
        label = np.zeros(CREPE_N_BINS, dtype=np.float64)
        true_bin = freq_to_model_bin(fh, self.rf_fmin, self.rf_fmax)
        for i in range(CREPE_N_BINS):
            label[i] = np.exp(-((float(i) - true_bin) ** 2) / (2 * (self.gaussian_sigma_bins ** 2)))
        
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
    
    def __init__(self, dropout: float = 0.25):
        """
        Args:
            dropout: Dropout rate (paper uses 0.25)
        """
        super(CREPE, self).__init__()
        
        # Standard CREPE Architecture filter counts
        # L1: 1024 filters, L2-L4: 128 filters, L5: 256 filters, L6: 512 filters
        
        # Layer 1: conv (1024) -> pool -> (1024 filters)
        self.conv1 = nn.Conv1d(1, 1024, kernel_size=512, stride=4, padding=254)
        self.bn1 = nn.BatchNorm1d(1024)
        self.drop1 = nn.Dropout(dropout)
        self.pool1 = nn.MaxPool1d(2, stride=2)
        
        # Layer 2: conv -> pool -> (128 filters)
        self.conv2 = nn.Conv1d(1024, 128, kernel_size=64, stride=1, padding=32)
        self.bn2 = nn.BatchNorm1d(128)
        self.drop2 = nn.Dropout(dropout)
        self.pool2 = nn.MaxPool1d(2, stride=2)
        
        # Layer 3: conv -> pool -> (128 filters)
        self.conv3 = nn.Conv1d(128, 128, kernel_size=64, stride=1, padding=32)
        self.bn3 = nn.BatchNorm1d(128)
        self.drop3 = nn.Dropout(dropout)
        self.pool3 = nn.MaxPool1d(2, stride=2)
        
        # Layer 4: conv -> pool -> (128 filters)
        self.conv4 = nn.Conv1d(128, 128, kernel_size=64, stride=1, padding=32)
        self.bn4 = nn.BatchNorm1d(128)
        self.drop4 = nn.Dropout(dropout)
        self.pool4 = nn.MaxPool1d(2, stride=2)
        
        # Layer 5: conv -> pool -> (256 filters)
        self.conv5 = nn.Conv1d(128, 256, kernel_size=64, stride=1, padding=32)
        self.bn5 = nn.BatchNorm1d(256)
        self.drop5 = nn.Dropout(dropout)
        self.pool5 = nn.MaxPool1d(2, stride=2)
        
        # Layer 6: conv -> pool -> (512 filters)
        self.conv6 = nn.Conv1d(256, 512, kernel_size=64, stride=1, padding=32)
        self.bn6 = nn.BatchNorm1d(512)
        self.drop6 = nn.Dropout(dropout)
        self.pool6 = nn.MaxPool1d(2, stride=2)
        
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

def evaluate_predictions(
    predictions: np.ndarray,
    targets: np.ndarray,
    rf_fmin: float,
    rf_fmax: float,
) -> Tuple[float, float, float, float]:
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
    pred_freqs = np.array([model_bin_to_freq(b, rf_fmin, rf_fmax) for b in pred_bins])
    true_freqs = np.array([model_bin_to_freq(b, rf_fmin, rf_fmax) for b in targets])
    
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


def evaluate(model, dataloader, criterion, device, rf_fmin: float, rf_fmax: float):
    """Evaluate model on a dataset."""
    model.eval()
    total_loss = 0
    total_samples = 0
    sum_error_cents = 0.0
    count_rpa_50 = 0
    count_rpa_25 = 0
    count_rca_50 = 0

    bin_indices = np.arange(CREPE_N_BINS, dtype=np.float64)
    
    with torch.no_grad():
        for inputs, labels in dataloader:
            inputs = inputs.to(device)
            labels = labels.to(device)
            
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            
            total_loss += loss.item()
            logits_np = outputs.cpu().numpy().astype(np.float64)
            true_bins = torch.argmax(labels, dim=1).cpu().numpy().astype(np.float64)

            pred_probs = 1.0 / (1.0 + np.exp(-logits_np))
            pred_bins = np.sum(pred_probs * bin_indices[None, :], axis=1) / (np.sum(pred_probs, axis=1) + 1e-8)

            pred_freqs = np.array([model_bin_to_freq(b, rf_fmin, rf_fmax) for b in pred_bins], dtype=np.float64)
            true_freqs = np.array([model_bin_to_freq(b, rf_fmin, rf_fmax) for b in true_bins], dtype=np.float64)
            errors_cents = np.abs(1200.0 * np.log2((pred_freqs + 1e-8) / (true_freqs + 1e-8)))
            chroma_errors = np.minimum(errors_cents % 1200.0, 1200.0 - (errors_cents % 1200.0))

            batch_n = int(errors_cents.shape[0])
            total_samples += batch_n
            sum_error_cents += float(np.sum(errors_cents))
            count_rpa_50 += int(np.sum(errors_cents <= 50.0))
            count_rpa_25 += int(np.sum(errors_cents <= 25.0))
            count_rca_50 += int(np.sum(chroma_errors <= 50.0))
    
    avg_loss = total_loss / len(dataloader)

    if total_samples > 0:
        rpa_50 = 100.0 * count_rpa_50 / total_samples
        rpa_25 = 100.0 * count_rpa_25 / total_samples
        rca = 100.0 * count_rca_50 / total_samples
        mean_error = sum_error_cents / total_samples
    else:
        rpa_50 = 0.0
        rpa_25 = 0.0
        rca = 0.0
        mean_error = 0.0

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
        
        # Gradient clipping to prevent exploding gradients
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        
        optimizer.step()
        
        total_loss += loss.item()
        pbar.set_postfix({'loss': f'{loss.item():.6f}'})
    
    return total_loss / len(dataloader)


def main():
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Train CREPE model with configurable SNR range')
    parser.add_argument('--data_path', type=str, default='C:\\Users\\User1\\Downloads\\Emanation\\Results\\iq_dict_caseA_25MHz(4-4-26).pkl',
                        help='Path to training dataset pickle')
    parser.add_argument('--save_dir', type=str, default='./models_crepe/',
                        help='Directory to save checkpoints/results')
    parser.add_argument('--case', type=str, default='AUTO', choices=['AUTO', 'A', 'B', 'C'])
    parser.add_argument('--rf_fmin', type=float, default=None)
    parser.add_argument('--rf_fmax', type=float, default=None)
    parser.add_argument('--snr_min', type=int, default=-6, help='Minimum SNR (default: -10)')
    parser.add_argument('--snr_max', type=int, default=-10, help='Maximum SNR (default: 20)')
    parser.add_argument('--model_suffix', type=str, default='snr_15_20(18-3)', help='Suffix for model files')
    parser.add_argument('--input_feature', type=str, default='raw', choices=['raw', 'fftshift'],
                        help='Input feature for CNN: raw (existing behavior) or fftshift (test-2 mode)')
    parser.add_argument('--batch_size', type=int, default=64, help='Batch size')
    parser.add_argument('--epochs', type=int, default=30, help='Max training epochs')
    parser.add_argument('--lr', type=float, default=1e-4, help='Learning rate')
    parser.add_argument('--weight_decay', type=float, default=1e-4, help='Weight decay')
    parser.add_argument('--dropout', type=float, default=0.35, help='Dropout rate')
    parser.add_argument('--gaussian_sigma', type=float, default=25.0,
                        help='Gaussian sigma for label smoothing (in cents domain used by label function)')
    parser.add_argument('--patience', type=int, default=5, help='Early stopping patience')
    parser.add_argument('--lr_patience', type=int, default=2, help='LR scheduler patience')
    args = parser.parse_args()

    case_arg = '' if args.case == 'AUTO' else args.case
    rf_fmin, rf_fmax, rf_range_source = _resolve_rf_range(args.data_path, case_arg, args.rf_fmin, args.rf_fmax)
    
    # Configuration block - EDIT THESE VALUES DIRECTLY
    config = {
        'data_path': args.data_path,
        'batch_size': args.batch_size,
        'epochs': args.epochs,
        'lr': args.lr,
        # 'capacity' removed - using standard CREPE architecture
        'weight_decay': args.weight_decay,
        'dropout': args.dropout,
        'gaussian_sigma': args.gaussian_sigma,
        # we can change this hyperparameter to see if it improves performance for CNN model, but 25 cents is the standard choice for CREPE
        'device': 'cuda' if torch.cuda.is_available() else 'cpu',
        'save_dir': args.save_dir,
        'patience': args.patience,
        'lr_patience': args.lr_patience,
        'case': case_arg if case_arg else 'AUTO',
        'rf_fmin': rf_fmin,
        'rf_fmax': rf_fmax,
        'rf_range_source': rf_range_source,
        'snr_range': (args.snr_min, args.snr_max),
        'model_suffix': args.model_suffix,
        'input_feature': args.input_feature,
    }
    
    print("=" * 80)
    print("Training CREPE Model - Standard Architecture")
    print("=" * 80)
    print(f"\nConfiguration:")
    for k, v in config.items():
        print(f"  {k}: {v}")
    
    # Load dataset
    print(f"\n📂 Loading data from {config['data_path']}...")
    with open(config['data_path'], 'rb') as f:
        iq_dict = pickle.load(f)
    print(f"[OK] Loaded {len(iq_dict)} samples")
    
    # Extract all pitch bins from dataset
    all_bins = sorted(list(set([int(k.split('_')[1]) for k in iq_dict.keys()])))
    print(f"[OK] Found {len(all_bins)} unique bins (range: {min(all_bins)}-{max(all_bins)})")
    
    # # Split bins with OVERLAP - 60/20/20 split
    # # Interspersed to ensure coverage across frequency range
    # train_bins = [b for i, b in enumerate(all_bins) if i % 5 not in [0, 1]]  # 60%
    # val_bins = [b for i, b in enumerate(all_bins) if i % 5 == 0]              # 20%
    # test_bins = [b for i, b in enumerate(all_bins) if i % 5 == 1]             # 20%

    #input frames
    print(f"loaded {len(iq_dict)} samples")
    all_frames = list(iq_dict.keys())
    print(f"found {len(all_frames)} frames")

    # Use 60/20/20 split for Case C, standard 5/5/90 split otherwise
    if case_arg.upper() == 'C':
        train_temp, test_frames = train_test_split(all_frames, test_size=0.2, random_state=42)
        train_frames, val_frames = train_test_split(train_temp, test_size=0.25, random_state=42)
        split_name = "60/20/20"
    else:
        train_frames, temp_frames = train_test_split(all_frames, test_size=0.8, random_state=42)
        val_frames, test_frames = train_test_split(temp_frames, test_size=7/8, random_state=42)
        split_name = "5/5/90"

    print(f"\n📊 Train/Val/Test split (Random {split_name}):")
    print(f"  Train frames: {len(train_frames)} ({100*len(train_frames)/len(all_frames):.0f}%)")
    print(f"  Val frames: {len(val_frames)} ({100*len(val_frames)/len(all_frames):.0f}%)")
    print(f"  Test frames: {len(test_frames)} ({100*len(test_frames)/len(all_frames):.0f}%)")
    print(f"  Random seed: 42")

    train_iq_dict = {k: iq_dict[k] for k in train_frames}
    val_iq_dict = {k: iq_dict[k] for k in val_frames}
    test_iq_dict = {k: iq_dict[k] for k in test_frames}
    # Create datasets - use configured SNR range
    print(f"\n📊 Using SNR range: {config['snr_range'][0]} to {config['snr_range'][1]} dB")
    # Dataset objects
    train_dataset = CREPEDataset(
        iq_dict=train_iq_dict,
        #bin_list=train_frames,
        snr_range=config['snr_range'],  # Use configured SNR range
        gaussian_sigma=config['gaussian_sigma'],
        rf_fmin=config['rf_fmin'],
        rf_fmax=config['rf_fmax'],
        input_feature=config['input_feature'],
    )
    
    val_dataset = CREPEDataset(
        iq_dict=val_iq_dict,
        #bin_list=val_frames,
        snr_range=config['snr_range'],  # Same SNR range
        gaussian_sigma=config['gaussian_sigma'],
        rf_fmin=config['rf_fmin'],
        rf_fmax=config['rf_fmax'],
        input_feature=config['input_feature'],
    )
    
    train_loader = DataLoader(train_dataset, batch_size=config['batch_size'], 
                             shuffle=True, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=config['batch_size'], 
                           shuffle=True, num_workers=0)
    # Create model
    print(f"\n🏗️  Creating CREPE model (Standard Architecture)...")
    model = CREPE(dropout=config['dropout'])
    device = torch.device(config['device'])
    model = model.to(device)
    
    # Count parameters
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"[OK] Model created: {n_params:,} trainable parameters")
    
    # Loss and optimizer (as per paper)
    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=config['lr'])
    
    # Learning rate scheduler - reduce LR when validation loss plateaus
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=config['lr_patience'], 
        verbose=True, min_lr=1e-6
    )
    
    # Training loop
    print(f"\n🚀 Starting training for {config['epochs']} epochs...")
    print("=" * 80)
    
    # Previous metric tracking (kept for reference):
    # best_rpa = 0
    best_val_loss = float('inf')
    best_epoch = 0
    patience_counter = 0
    history = {'train_loss': [], 'val_loss': [], 'rpa_50': [], 'rpa_25': [], 'rca': []}
    
    os.makedirs(config['save_dir'], exist_ok=True)
    
    for epoch in range(1, config['epochs'] + 1):
        # Train
        train_loss = train_epoch(model, train_loader, criterion, optimizer, device, epoch)
        
        # Validate
        val_loss, rpa_50, rpa_25, rca, mean_error = evaluate(
            model,
            val_loader,
            criterion,
            device,
            config['rf_fmin'],
            config['rf_fmax'],
        )

        history['train_loss'].append(train_loss)
        history['val_loss'].append(val_loss)
        history['rpa_50'].append(rpa_50)
        history['rpa_25'].append(rpa_25)
        history['rca'].append(rca)
        
        # Learning rate scheduler step
        scheduler.step(val_loss)
        current_lr = optimizer.param_groups[0]['lr']
        
        # Print results
        print(f"\n┌─────────────────────────────────────────┐")
        print(f"│ Epoch {epoch}/{config['epochs']}")
        print(f"├─────────────────────────────────────────┤")
        print(f"│ Train Loss:     {train_loss:.6f}")
        print(f"│ Val Loss:       {val_loss:.6f}")
        print(f"│ Learning Rate:  {current_lr:.6f}")
        print(f"├─────────────────────────────────────────┤")
        print(f"│ RPA (50 cents): {rpa_50:6.2f}%")
        print(f"│ RPA (25 cents): {rpa_25:6.2f}%")
        print(f"│ RCA:            {rca:6.2f}%")
        print(f"│ Mean Error:     {mean_error:6.1f} cents")
        print(f"└─────────────────────────────────────────┘\n")
        
        # Previous RPA-based checkpointing (disabled):
        # if rpa_50 > best_rpa:
        #     best_rpa = rpa_50
        #     best_epoch = epoch
        #     patience_counter = 0
        #     torch.save({
        #         'epoch': epoch,
        #         'model_state_dict': model.state_dict(),
        #         'optimizer_state_dict': optimizer.state_dict(),
        #         'rpa_50': rpa_50,
        #         'rpa_25': rpa_25,
        #         'rca': rca,
        #         'config': config,
        #     }, os.path.join(config['save_dir'], f"crepe_best_{config['model_suffix']}.pth"))
        #     print(f"✓ Saved best model (RPA: {rpa_50:.2f}%)")
        # else:
        #     patience_counter += 1
        #     print(f"⏳ No improvement for {patience_counter} epochs (best: {best_rpa:.2f}% at epoch {best_epoch})")
        #
        #     if patience_counter >= config['patience']:
        #         print(f"\n⚠️  Early stopping triggered after {epoch} epochs")
        #         print(f"   Best RPA: {best_rpa:.2f}% at epoch {best_epoch}")
        #         break

        # New loss-based checkpointing
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_epoch = epoch
            patience_counter = 0
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_loss': val_loss,
                'config': config,
            }, os.path.join(config['save_dir'], f"crepe_best_{config['model_suffix']}.pth"))
            print(f"[OK] Saved best model (Val Loss: {val_loss:.6f})")
        else:
            patience_counter += 1
            print(f"⏳ No improvement for {patience_counter} epochs (best val loss: {best_val_loss:.6f} at epoch {best_epoch})")

            if patience_counter >= config['patience']:
                print(f"\n⚠️  Early stopping triggered after {epoch} epochs")
                print(f"   Best val loss: {best_val_loss:.6f} at epoch {best_epoch}")
                break
    
    # Save final model and history
    torch.save(model.state_dict(), os.path.join(config['save_dir'], f"crepe_final_{config['model_suffix']}.pth"))
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
    
    if len(history['rpa_50']) == len(history['train_loss']) and len(history['rpa_25']) == len(history['train_loss']):
        axes[0, 1].plot(epochs, history['rpa_50'], label='RPA 50c')
        axes[0, 1].plot(epochs, history['rpa_25'], label='RPA 25c')
        axes[0, 1].legend()
    else:
        axes[0, 1].text(0.5, 0.5, 'RPA metrics disabled', ha='center', va='center')
    axes[0, 1].set_xlabel('Epoch')
    axes[0, 1].set_ylabel('Accuracy (%)')
    axes[0, 1].set_title('Raw Pitch Accuracy')
    axes[0, 1].grid(True, alpha=0.3)

    if len(history['rca']) == len(history['train_loss']):
        axes[1, 0].plot(epochs, history['rca'])
    else:
        axes[1, 0].text(0.5, 0.5, 'RCA metric disabled', ha='center', va='center')
    axes[1, 0].set_xlabel('Epoch')
    axes[1, 0].set_ylabel('Accuracy (%)')
    axes[1, 0].set_title('Raw Chroma Accuracy')
    axes[1, 0].grid(True, alpha=0.3)
    
    axes[1, 1].text(0.5, 0.5, f'Best Val Loss: {best_val_loss:.6f}', 
                   ha='center', va='center', fontsize=20, weight='bold')
    axes[1, 1].axis('off')
    
    plt.tight_layout()
    plt.savefig(os.path.join(config['save_dir'], f"training_curves_{config['model_suffix']}.png"), dpi=150)
    plt.close()
    
    print("\n" + "=" * 80)
    print("[OK] Training complete!")
    print("=" * 80)
    print(f"\nBest validation loss: {best_val_loss:.6f}")
    print(f"Models saved to: {config['save_dir']}")


if __name__ == "__main__":
    main()