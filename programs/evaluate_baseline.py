#!/usr/bin/env python3
"""
Evaluate classical baseline (EstimatePeaks + EstimateHarmonic) on test set.
Compares against DL model predictions.

Usage:
    python programs/evaluate_baseline.py --data data/pitch_dataset.npz --model models/best_pitch_cnn.pt
"""
import argparse
import os
import sys
import numpy as np
import yaml
import torch

# Add programs folder to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from EstimatePeaks_search import EstimatePeaks, Noisespread
from EstimateHarmonic_search import EstimateHarmonic
from train_pitch_cnn import PitchCNN, PSDDataset


def load_config():
    """Load YAML config for classical pipeline."""
    yaml_path = os.path.join(os.path.dirname(__file__), '..', 'synapse_emanation_search.yaml')
    if os.path.exists(yaml_path):
        with open(yaml_path, 'r') as f:
            return yaml.safe_load(f)
    else:
        # Default config
        return {
            'EstimatePeaks': {
                'num_slice_ns': 10,
                'p1': 84,
                'p2': 16,
                'ns_estimate_percentile': 50,
                'pct_samp_NF': 10,
                'NF_prctile': 50,
            },
            'EmanationDetection': {
                'gb_thresh_hh': 0.6,
                'ntimes_ns': 2,
                'Maxpeaks': 200,
                'min_peaks_detect': 6,
            },
            'EstimateHarmonic': {
                'num_steps_coarse': 6000,
                'num_steps_finesearch': 6000,
                'Err_thresh_dict': {500: 10, 1000: 2, 50000: 2, 100000000000: 2},
                'p_hh_1': 0.5,
                'p_hh_2': 0.5,
                'wt_meas_pred_hh': 0.5,
            }
        }


def run_classical_pipeline(psd_db, f_range, fs, config, debug=False):
    """
    Run classical EstimatePeaks + EstimateHarmonic pipeline.
    
    Returns:
        estimated_f0: Estimated fundamental frequency (Hz), or None if failed
    """
    # Frequency bin size should be calculated from the actual spacing
    if len(f_range) > 1:
        Fb = f_range[1] - f_range[0]  # Frequency bin size
    else:
        Fb = fs / len(psd_db)  # Fallback
    
    if debug:
        print(f"\nDEBUG: PSD shape: {psd_db.shape}")
        print(f"DEBUG: Freq range: {f_range[0]:.2f} to {f_range[-1]:.2f} Hz")
        print(f"DEBUG: Freq bin size: {Fb:.2f} Hz")
        print(f"DEBUG: PSD range: {psd_db.min():.2f} to {psd_db.max():.2f} dB")
    
    try:
        # EstimatePeaks
        result = EstimatePeaks(
            psd_db, f_range, Fb,
            min_peaks_detect=config['EmanationDetection']['min_peaks_detect'],
            gb_thresh=config['EmanationDetection']['gb_thresh_hh'],
            ntimes_ns=config['EmanationDetection']['ntimes_ns'],
            Maxpeaks=config['EmanationDetection']['Maxpeaks_hh'],
            config_dict=config
        )
        
        if len(result) < 3 or len(result[0]) == 0:
            if debug:
                print("DEBUG: EstimatePeaks returned no peaks")
            return None
        
        SNR_pos, f_est_pos, locpeaks = result
        
        if debug:
            print(f"DEBUG: Found {len(f_est_pos)} peaks")
        
        # Only use positive frequencies
        pos_idx = f_est_pos > 0
        if not np.any(pos_idx):
            if debug:
                print("DEBUG: No positive frequency peaks found")
            return None
        
        SNR_pos = np.atleast_1d(SNR_pos)[pos_idx]
        f_est_pos = np.atleast_1d(f_est_pos)[pos_idx]
        
        if debug:
            print(f"DEBUG: {len(f_est_pos)} positive frequency peaks")
        
        if len(f_est_pos) < config['EmanationDetection']['min_peaks_detect']:
            if debug:
                print(f"DEBUG: Not enough peaks ({len(f_est_pos)} < {config['EmanationDetection']['min_peaks_detect']})")
            return None
        
        # EstimateHarmonic
        numpeaks_crossthresh = config['EmanationDetection']['numpeaks_crossthresh']
        high_ff_search = True  # Search for high fundamental frequencies
        
        result_harm = EstimateHarmonic(
            SNR_pos, f_est_pos,
            numpeaks_crossthresh,
            high_ff_search,
            config
        )
        
        if result_harm is None or len(result_harm) < 1:
            if debug:
                print("DEBUG: EstimateHarmonic failed")
            return None
        
        fundamental_harmonic = result_harm[0]
        if debug:
            print(f"DEBUG: Estimated f0: {fundamental_harmonic:.2f} Hz")
        return fundamental_harmonic
        
    except Exception as e:
        if debug:
            print(f"DEBUG: Exception in classical pipeline: {e}")
        return None


def evaluate_dl_model(model_path, dataset, device):
    """
    Evaluate trained DL model on dataset.
    
    Returns:
        predictions: Array of predicted frequencies (Hz)
    """
    # Load model
    checkpoint = torch.load(model_path, map_location=device)
    n_freq = dataset.psds.shape[1]
    model = PitchCNN(n_freq, use_snr=True).to(device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    
    # Override dataset normalization with saved values
    label_mean = checkpoint['label_mean']
    label_std = checkpoint['label_std']
    
    predictions = []
    
    with torch.no_grad():
        for i in range(len(dataset)):
            psd, snr, _ = dataset[i]
            psd = psd.unsqueeze(0).to(device)
            snr = snr.unsqueeze(0).to(device)
            
            pred_norm = model(psd, snr).cpu().item()
            pred_hz = pred_norm * label_std + label_mean
            predictions.append(pred_hz)
    
    return np.array(predictions)


def main():
    parser = argparse.ArgumentParser(description='Evaluate classical baseline vs DL model')
    parser.add_argument('--data', type=str, default='data/pitch_dataset.npz',
                        help='Path to dataset')
    parser.add_argument('--model', type=str, default='models/best_pitch_cnn.pt',
                        help='Path to trained DL model')
    parser.add_argument('--num-eval', type=int, default=500,
                        help='Number of samples to evaluate (classical is slow)')
    parser.add_argument('--device', type=str, default='cuda',
                        help='Device for DL model')
    
    args = parser.parse_args()
    
    # Load data
    data = np.load(args.data)
    psds = data['psds']
    labels = data['labels']
    f_range = data['f_range']
    fs = float(data['fs'])
    snrs = data['snrs']
    
    print(f"Loaded {len(labels)} samples")
    
    # Load config
    config = load_config()
    
    # Evaluate subset (classical is slow)
    n_eval = min(args.num_eval, len(labels))
    indices = np.random.choice(len(labels), n_eval, replace=False)
    
    print(f"\nEvaluating {n_eval} samples...")
    print("-" * 80)
    
    # Classical baseline
    print("Running classical pipeline (EstimatePeaks + EstimateHarmonic)...")
    print(f"First running with debug on sample 0...")
    
    # Debug first sample
    psd_debug = psds[indices[0]]
    label_debug = labels[indices[0]]
    print(f"  True f0: {label_debug:.2f} Hz")
    _ = run_classical_pipeline(psd_debug, f_range, fs, config, debug=True)
    
    print("\nProcessing all samples...")
    classical_preds = []
    classical_success = 0
    
    for i, idx in enumerate(indices):
        psd_db = psds[idx]
        pred = run_classical_pipeline(psd_db, f_range, fs, config, debug=False)
        
        if pred is not None:
            classical_preds.append(pred)
            classical_success += 1
        else:
            classical_preds.append(np.nan)
        
        if (i + 1) % 50 == 0:
            print(f"  Processed {i+1}/{n_eval}")
    
    classical_preds = np.array(classical_preds)
    classical_labels = labels[indices]
    classical_snrs = snrs[indices]
    
    # Compute classical metrics (only on successful predictions)
    valid_mask = ~np.isnan(classical_preds)
    if np.any(valid_mask):
        classical_errors = np.abs(classical_preds[valid_mask] - classical_labels[valid_mask])
        classical_mae = np.mean(classical_errors)
        classical_w5 = np.mean(classical_errors < 5) * 100
        classical_w10 = np.mean(classical_errors < 10) * 100
        classical_w50 = np.mean(classical_errors < 50) * 100
    else:
        classical_mae = float('nan')
        classical_w5 = classical_w10 = classical_w50 = 0
    
    print(f"\nClassical Pipeline Results:")
    print(f"  Success rate: {classical_success}/{n_eval} ({100*classical_success/n_eval:.1f}%)")
    print(f"  MAE: {classical_mae:.1f} Hz")
    print(f"  <5 Hz: {classical_w5:.1f}%")
    print(f"  <10 Hz: {classical_w10:.1f}%")
    print(f"  <50 Hz: {classical_w50:.1f}%")
    
    # DL model
    if os.path.exists(args.model):
        print("\nRunning DL model...")
        device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
        
        # Create subset dataset
        class SubsetDataset:
            def __init__(self, psds, snrs, labels, indices):
                self.psds = torch.tensor(psds[indices], dtype=torch.float32)
                self.snrs = torch.tensor(snrs[indices], dtype=torch.float32)
                self.labels = torch.tensor(labels[indices], dtype=torch.float32)
                
                # Normalize PSDs
                self.psds_mean = self.psds.mean(dim=1, keepdim=True)
                self.psds_std = self.psds.std(dim=1, keepdim=True) + 1e-6
                self.psds = (self.psds - self.psds_mean) / self.psds_std
            
            def __len__(self):
                return len(self.labels)
            
            def __getitem__(self, idx):
                psd = self.psds[idx].unsqueeze(0)
                snr = self.snrs[idx].unsqueeze(0)
                label = self.labels[idx]
                return psd, snr, label
        
        subset_dataset = SubsetDataset(psds, snrs, labels, indices)
        dl_preds = evaluate_dl_model(args.model, subset_dataset, device)
        
        # Compute DL metrics
        dl_errors = np.abs(dl_preds - classical_labels)
        dl_mae = np.mean(dl_errors)
        dl_w5 = np.mean(dl_errors < 5) * 100
        dl_w10 = np.mean(dl_errors < 10) * 100
        dl_w50 = np.mean(dl_errors < 50) * 100
        
        print(f"\nDL Model Results:")
        print(f"  MAE: {dl_mae:.1f} Hz")
        print(f"  <5 Hz: {dl_w5:.1f}%")
        print(f"  <10 Hz: {dl_w10:.1f}%")
        print(f"  <50 Hz: {dl_w50:.1f}%")
        
        # Comparison
        print("\n" + "=" * 80)
        print("COMPARISON SUMMARY")
        print("=" * 80)
        print(f"{'Metric':<20} {'Classical':<15} {'DL Model':<15} {'Winner':<10}")
        print("-" * 80)
        print(f"{'MAE (Hz)':<20} {classical_mae:<15.1f} {dl_mae:<15.1f} {'DL' if dl_mae < classical_mae else 'Classical':<10}")
        print(f"{'<5 Hz (%)':<20} {classical_w5:<15.1f} {dl_w5:<15.1f} {'DL' if dl_w5 > classical_w5 else 'Classical':<10}")
        print(f"{'<10 Hz (%)':<20} {classical_w10:<15.1f} {dl_w10:<15.1f} {'DL' if dl_w10 > classical_w10 else 'Classical':<10}")
        print(f"{'<50 Hz (%)':<20} {classical_w50:<15.1f} {dl_w50:<15.1f} {'DL' if dl_w50 > classical_w50 else 'Classical':<10}")
        print(f"{'Success Rate (%)':<20} {100*classical_success/n_eval:<15.1f} {'100.0':<15} {'DL':<10}")
    else:
        print(f"\nDL model not found at {args.model}. Train first with train_pitch_cnn.py")


if __name__ == '__main__':
    main()
