"""
Evaluate continuous CREPE model per SNR and plot metrics from saved results.
"""

import argparse
import os
import pickle
import sys
from pickle import UnpicklingError
from typing import Dict, List

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader
from tqdm import tqdm

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'Venkatesh'))
from testSet_continuous import CREPE, CREPEDataset, CREPE_N_BINS, crepe_bin_to_hz


def load_model(config: Dict, device: str) -> torch.nn.Module:
    model_path = os.path.join(config['save_dir'], f"crepe_best_{config['model_suffix']}.pth")
    print(f"Loading model: {model_path}")

    try:
        torch.serialization.add_safe_globals([np.core.multiarray.scalar])
    except AttributeError:
        pass

    checkpoint = torch.load(model_path, weights_only=False, map_location=device)
    model = CREPE(dropout=config['dropout']).to(device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()

    epoch = checkpoint.get('epoch', 'N/A')
    val_loss = checkpoint.get('val_loss', None)
    if val_loss is None:
        print(f"Loaded checkpoint from epoch {epoch}")
    else:
        print(f"Loaded checkpoint from epoch {epoch} (val loss: {val_loss:.6f})")

    return model


def build_test_split(iq_dict: Dict) -> Dict:
    all_frames = list(iq_dict.keys())
    _, temp_frames = train_test_split(all_frames, test_size=0.6, random_state=42)
    _, test_frames = train_test_split(temp_frames, test_size=2 / 3, random_state=42)
    return {k: iq_dict[k] for k in test_frames}


def evaluate_single_snr(model: torch.nn.Module, test_iq_dict: Dict, snr: int, config: Dict, device: str) -> Dict:
    dataset = CREPEDataset(
        iq_dict=test_iq_dict,
        snr_range=(snr, snr),
        gaussian_sigma_cents=config['gaussian_sigma_cents'],
    )

    if len(dataset) == 0:
        return {
            'snr': snr,
            'num_samples': 0,
            'test_loss': float('nan'),
            'mean_error_cents': float('nan'),
            'rpa_50': float('nan'),
            'errors_cents': np.array([]),
        }

    loader = DataLoader(dataset, batch_size=config['batch_size'], shuffle=False, num_workers=0)
    criterion = nn.BCEWithLogitsLoss()

    total_loss = 0.0
    all_logits: List[np.ndarray] = []
    all_true_fh: List[np.ndarray] = []

    with torch.no_grad():
        for x, y, f_h in tqdm(loader, desc=f'SNR {snr:>3d}'):
            x = x.to(device)
            y = y.to(device)

            logits = model(x)
            loss = criterion(logits, y)

            total_loss += loss.item()
            all_logits.append(logits.cpu().numpy())
            all_true_fh.append(f_h.cpu().numpy())

    total_loss /= len(loader)

    logits_np = np.concatenate(all_logits, axis=0)
    true_fh = np.concatenate(all_true_fh, axis=0)

    probs = 1.0 / (1.0 + np.exp(-logits_np))
    bin_indices = np.arange(CREPE_N_BINS)
    pred_bins = np.sum(probs * bin_indices, axis=1) / (np.sum(probs, axis=1) + 1e-8)
    pred_freqs = np.array([crepe_bin_to_hz(b) for b in pred_bins])

    errors_cents = np.abs(1200.0 * np.log2((pred_freqs + 1e-8) / (true_fh + 1e-8)))
    mean_error_cents = float(np.mean(errors_cents))
    rpa_50 = float(np.mean(errors_cents <= 50.0) * 100.0)

    return {
        'snr': snr,
        'num_samples': int(len(dataset)),
        'test_loss': float(total_loss),
        'mean_error_cents': mean_error_cents,
        'rpa_50': rpa_50,
        'errors_cents': errors_cents,
    }


def run_forward_pass_each_snr(config: Dict) -> Dict:
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    try:
        with open(config['data_path'], 'rb') as f:
            iq_dict = pickle.load(f)
    except (UnpicklingError, EOFError) as exc:
        raise RuntimeError(
            f"Failed to load dataset '{config['data_path']}'. File appears truncated/corrupt. "
            "Use a known-good file such as 'IQData/iq_dict_continuous_freq_SNR0_20.pkl' "
            "or regenerate this pickle."
        ) from exc

    print(f"Loaded total samples: {len(iq_dict)}")
    test_iq_dict = build_test_split(iq_dict)
    print(f"Test split samples: {len(test_iq_dict)}")

    model = load_model(config, device)

    snr_values = list(range(config['snr_min'], config['snr_max'] + 1))

    results_dict = {
        'config': config,
        'device': device,
        'snr_values': [],
        'rpa_50': [],
        'mean_error_cents': [],
        'test_loss': [],
        'num_samples': [],
        'per_snr': {},
    }

    for snr in snr_values:
        metrics = evaluate_single_snr(model, test_iq_dict, snr, config, device)

        results_dict['snr_values'].append(snr)
        results_dict['rpa_50'].append(metrics['rpa_50'])
        results_dict['mean_error_cents'].append(metrics['mean_error_cents'])
        results_dict['test_loss'].append(metrics['test_loss'])
        results_dict['num_samples'].append(metrics['num_samples'])
        results_dict['per_snr'][snr] = metrics

        print(
            f"SNR {snr:>3d} dB | samples={metrics['num_samples']:>5d} | "
            f"RPA@50={metrics['rpa_50']:.2f}% | mean_error={metrics['mean_error_cents']:.2f} cents"
        )

    os.makedirs(config['save_dir'], exist_ok=True)
    results_path = os.path.join(
        config['save_dir'],
        f"per_snr_results_continuous_{config['model_suffix']}.pkl",
    )
    with open(results_path, 'wb') as f:
        pickle.dump(results_dict, f)

    print(f"Saved results dict: {results_path}")
    return results_dict


def plot_from_results_dict(results_dict: Dict, save_dir: str, model_suffix: str) -> None:
    snr = np.array(results_dict['snr_values'])
    rpa_50 = np.array(results_dict['rpa_50'], dtype=float)
    mean_error = np.array(results_dict['mean_error_cents'], dtype=float)

    valid_rpa = ~np.isnan(rpa_50)
    valid_err = ~np.isnan(mean_error)

    plt.figure(figsize=(10, 6))
    plt.plot(snr[valid_rpa], rpa_50[valid_rpa], marker='o', linewidth=2)
    plt.xlabel('SNR (dB)')
    plt.ylabel('RPA @ 50 cents (%)')
    plt.title(f'RPA@50 vs SNR ({model_suffix})')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    rpa_plot_path = os.path.join(save_dir, f"rpa50_vs_snr_{model_suffix}.png")
    plt.savefig(rpa_plot_path, dpi=150)
    plt.close()

    fig, axes = plt.subplots(2, 1, figsize=(10, 9), sharex=True)

    axes[0].plot(snr[valid_rpa], rpa_50[valid_rpa], marker='o', linewidth=2, color='royalblue')
    axes[0].set_ylabel('RPA @ 50 cents (%)')
    axes[0].set_title(f'Per-SNR Metrics ({model_suffix})')
    axes[0].grid(True, alpha=0.3)

    axes[1].plot(snr[valid_err], mean_error[valid_err], marker='s', linewidth=2, color='darkorange')
    axes[1].set_xlabel('SNR (dB)')
    axes[1].set_ylabel('Mean Error (cents)')
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    combined_plot_path = os.path.join(save_dir, f"metrics_vs_snr_{model_suffix}.png")
    plt.savefig(combined_plot_path, dpi=150)
    plt.close()

    print(f"Saved plot: {rpa_plot_path}")
    print(f"Saved plot: {combined_plot_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description='Forward pass each SNR and plot RPA/mean error')
    parser.add_argument('--data_path', type=str, default=r'IQData\iq_dict_continuous_freq_SNRneg10_20-20-3-26.pkl')
    parser.add_argument('--save_dir', type=str, default='./models_crepe/')
    parser.add_argument('--model_suffix', type=str, default='snr_neg10_20(20-3)')
    parser.add_argument('--snr_min', type=int, default=-10)
    parser.add_argument('--snr_max', type=int, default=20)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--dropout', type=float, default=0.25)
    parser.add_argument('--gaussian_sigma_cents', type=float, default=25.0)
    args = parser.parse_args()

    config = {
        'data_path': args.data_path,
        'save_dir': args.save_dir,
        'model_suffix': args.model_suffix,
        'snr_min': args.snr_min,
        'snr_max': args.snr_max,
        'batch_size': args.batch_size,
        'dropout': args.dropout,
        'gaussian_sigma_cents': args.gaussian_sigma_cents,
    }

    results_dict = run_forward_pass_each_snr(config)
    plot_from_results_dict(results_dict, config['save_dir'], config['model_suffix'])


if __name__ == '__main__':
    main()
