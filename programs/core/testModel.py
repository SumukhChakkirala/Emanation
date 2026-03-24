"""
Test CREPE continuous-frequency model on test split.
"""

import argparse
import os
import pickle
from typing import Dict, Tuple, Optional

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm


CREPE_FRAME_LENGTH = 1024
CREPE_N_BINS = 360
CREPE_CENTS_PER_BIN = 20
DEFAULT_RF_FMIN = 32.7
DEFAULT_RF_FMAX = 2069.0
CASE_RF_RANGES = {
    'A': (800e3, 1e6),
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
    f_h = None
    if 'FH' in parts:
        fh_idx = parts.index('FH') + 1
        f_h = float(parts[fh_idx])
    return bin_idx, snr, f_h


class CREPEDataset(Dataset):
    def __init__(
        self,
        iq_dict: Dict[str, np.ndarray],
        snr_range: Tuple[int, int],
        gaussian_sigma_cents: float,
        rf_fmin: float,
        rf_fmax: float,
        target_length: int = CREPE_FRAME_LENGTH,
    ):
        self.target_length = target_length
        self.gaussian_sigma_cents = gaussian_sigma_cents
        self.gaussian_sigma_bins = max(float(gaussian_sigma_cents) / float(CREPE_CENTS_PER_BIN), 1e-6)
        self.rf_fmin = float(rf_fmin)
        self.rf_fmax = float(rf_fmax)
        self.samples = []
        self.fh_values = []

        for key, signal in iq_dict.items():
            _, snr, f_h = parse_key(key)
            if snr < snr_range[0] or snr > snr_range[1]:
                continue
            if f_h is None:
                continue
            self.samples.append((key, signal))
            self.fh_values.append(f_h)

        print(f"✓ Created dataset with {len(self.samples)} samples")
        if self.samples:
            print(f"  f_h range: {min(self.fh_values):.1f} Hz to {max(self.fh_values):.1f} Hz")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        _, signal = self.samples[idx]
        f_h = self.fh_values[idx]

        signal = np.abs(signal)
        if len(signal) < self.target_length:
            signal = np.pad(signal, (0, self.target_length - len(signal)))
        else:
            signal = signal[:self.target_length]

        signal = signal / (np.max(np.abs(signal)) + 1e-8)
        label = self._create_gaussian_label(f_h)

        return torch.FloatTensor(signal).unsqueeze(0), torch.FloatTensor(label), torch.tensor(f_h, dtype=torch.float32)

    def _create_gaussian_label(self, f_h: float) -> np.ndarray:
        label = np.zeros(CREPE_N_BINS, dtype=np.float32)
        true_bin = freq_to_model_bin(f_h, self.rf_fmin, self.rf_fmax)
        for i in range(CREPE_N_BINS):
            label[i] = np.exp(-((float(i) - true_bin) ** 2) / (2.0 * (self.gaussian_sigma_bins ** 2)))
        return label


class CREPE(nn.Module):
    def __init__(self, dropout: float = 0.25):
        super().__init__()

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
        return self.fc(x)


def evaluate_continuous(config, test_dataset, device):
    model_path = os.path.join(config['save_dir'], f"crepe_best_{config['model_suffix']}.pth")
    print(f"\n🔄 Loading model: {model_path}")

    try:
        torch.serialization.add_safe_globals([np.core.multiarray.scalar])
    except AttributeError:
        pass

    checkpoint = torch.load(model_path, weights_only=False, map_location=device)
    model = CREPE(dropout=config['dropout']).to(device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()

    if 'val_loss' in checkpoint:
        print(f"✓ Loaded epoch {checkpoint.get('epoch', 'N/A')} (val loss: {checkpoint['val_loss']:.6f})")
    else:
        print(f"✓ Loaded epoch {checkpoint.get('epoch', 'N/A')}")

    test_loader = DataLoader(test_dataset, batch_size=config['batch_size'], shuffle=False, num_workers=0)
    criterion = nn.BCEWithLogitsLoss()

    total_loss = 0.0
    all_logits = []
    all_true_fh = []

    print(f"📊 Evaluating {len(test_dataset)} samples...")
    with torch.no_grad():
        for x, y, f_h in tqdm(test_loader, desc='Test'):
            x = x.to(device)
            y = y.to(device)

            logits = model(x)
            loss = criterion(logits, y)

            total_loss += loss.item()
            all_logits.append(logits.cpu().numpy())
            all_true_fh.append(f_h.cpu().numpy())

    total_loss /= len(test_loader)
    all_logits = np.concatenate(all_logits, axis=0)
    all_true_fh = np.concatenate(all_true_fh, axis=0)

    bin_indices = np.arange(CREPE_N_BINS)
    probs = 1.0 / (1.0 + np.exp(-all_logits))
    pred_bins = np.sum(probs * bin_indices, axis=1) / (np.sum(probs, axis=1) + 1e-8)
    pred_freqs = np.array([model_bin_to_freq(b, config['rf_fmin'], config['rf_fmax']) for b in pred_bins])

    errors_cents = np.abs(1200.0 * np.log2((pred_freqs + 1e-8) / (all_true_fh + 1e-8)))
    mean_error_cents = float(np.mean(errors_cents))
    rpa_50 = float(np.mean(errors_cents <= 50.0) * 100.0)
    rpa_25 = float(np.mean(errors_cents <= 25.0) * 100.0)

    print("\n" + "=" * 80)
    print("📊 CONTINUOUS TEST EVALUATION")
    print("=" * 80)
    print(f"Test Loss: {total_loss:.6f}")
    print(f"RF mapping range: [{config['rf_fmin']:.1f}, {config['rf_fmax']:.1f}] Hz ({config['rf_range_source']})")
    print(f"Mean Error: {mean_error_cents:.2f} cents")
    print(f"RPA @50c: {rpa_50:.2f}%")
    print(f"RPA @25c: {rpa_25:.2f}%")

    results = {
        'test_loss': total_loss,
        'mean_error_cents': mean_error_cents,
        'rpa_50': rpa_50,
        'rpa_25': rpa_25,
        'true_fh': all_true_fh,
        'pred_freqs': pred_freqs,
        'predictions': all_logits,
    }

    os.makedirs(config['save_dir'], exist_ok=True)
    results_path = os.path.join(config['save_dir'], f"test_results_continuous_{config['model_suffix']}.pkl")
    with open(results_path, 'wb') as f:
        pickle.dump(results, f)
    print(f"💾 Saved results: {results_path}")

    plt.figure(figsize=(10, 8))
    plt.scatter(all_true_fh, pred_freqs, s=4, alpha=0.25, color='royalblue', label='Predictions')
    min_f = float(np.min(all_true_fh))
    max_f = float(np.max(all_true_fh))
    plt.plot([min_f, max_f], [min_f, max_f], 'r--', linewidth=2, label='Ideal (y=x)')
    plt.xlabel('True Frequency (Hz)')
    plt.ylabel('Predicted Frequency (Hz)')
    plt.title(f"Continuous Test: True vs Predicted ({config['model_suffix']})")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plot_path = os.path.join(config['save_dir'], f"scatter_true_vs_pred_continuous_{config['model_suffix']}.png")
    plt.savefig(plot_path, dpi=150)
    plt.close()
    print(f"🖼️ Saved scatter plot: {plot_path}")


def main():
    parser = argparse.ArgumentParser(description='Test continuous CREPE model')
    parser.add_argument('--data_path', type=str, default=r'C:\Users\User1\Downloads\Emanation\IQData\iq_dict_continuous_freq_SNR0_20_continous.pkl')
    parser.add_argument('--save_dir', type=str, default='./models_crepe/')
    parser.add_argument('--model_suffix', type=str, default='snr_neg20_20(20-3)')
    parser.add_argument('--case', type=str, default='AUTO', choices=['AUTO', 'A', 'B', 'C'])
    parser.add_argument('--rf_fmin', type=float, default=None)
    parser.add_argument('--rf_fmax', type=float, default=None)
    parser.add_argument('--snr_min', type=int, default=-10)
    parser.add_argument('--snr_max', type=int, default=20)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--dropout', type=float, default=0.25)
    parser.add_argument('--gaussian_sigma_cents', type=float, default=25.0)
    args = parser.parse_args()

    case_arg = '' if args.case == 'AUTO' else args.case
    rf_fmin, rf_fmax, rf_range_source = _resolve_rf_range(args.data_path, case_arg, args.rf_fmin, args.rf_fmax)

    config = {
        'data_path': args.data_path,
        'save_dir': args.save_dir,
        'model_suffix': args.model_suffix,
        'case': case_arg if case_arg else 'AUTO',
        'rf_fmin': rf_fmin,
        'rf_fmax': rf_fmax,
        'rf_range_source': rf_range_source,
        'snr_min': args.snr_min,
        'snr_max': args.snr_max,
        'batch_size': args.batch_size,
        'dropout': args.dropout,
        'gaussian_sigma_cents': args.gaussian_sigma_cents,
        'device': 'cuda' if torch.cuda.is_available() else 'cpu',
    }

    print("=" * 80)
    print("CREPE Continuous Model - Test Set Evaluation")
    print("=" * 80)
    for k, v in config.items():
        print(f"{k}: {v}")

    with open(config['data_path'], 'rb') as f:
        iq_dict = pickle.load(f)
    print(f"\n✓ Loaded {len(iq_dict)} total samples")

    all_frames = list(iq_dict.keys())
    _, temp_frames = train_test_split(all_frames, test_size=0.6, random_state=42)
    _, test_frames = train_test_split(temp_frames, test_size=2/3, random_state=42)
    test_iq_dict = {k: iq_dict[k] for k in test_frames}
    print(f"✓ Test frames: {len(test_frames)} ({100 * len(test_frames) / len(all_frames):.0f}%) [40/20/40 split]")

    test_dataset = CREPEDataset(
        iq_dict=test_iq_dict,
        snr_range=(config['snr_min'], config['snr_max']),
        gaussian_sigma_cents=config['gaussian_sigma_cents'],
        rf_fmin=config['rf_fmin'],
        rf_fmax=config['rf_fmax'],
    )

    evaluate_continuous(config, test_dataset, device=config['device'])

    print("\n" + "=" * 80)
    print("✓ Continuous test evaluation complete")
    print("=" * 80)


if __name__ == '__main__':
    main()
