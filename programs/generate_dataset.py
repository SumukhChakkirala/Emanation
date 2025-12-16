#!/usr/bin/env python3
"""
Generate labeled dataset for DL pitch estimation.
Creates synthetic IQ signals with known fundamental frequencies, computes PSDs, and saves for training.

Usage:
    python programs/generate_dataset.py --out-file data/pitch_dataset.npz --num-samples 10000
"""
import argparse
import os
import sys
import numpy as np
from scipy.signal.windows import kaiser

# Add programs folder to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from EstimatePeaks_search import WelchPSDEstimate


def generate_dirac_comb_iq(F_h, duty_cycle, time_duration, Fs, snr_db):
    """
    Generate IQ signal for a rectangular pulse train (Dirac comb in frequency domain).
    
    Args:
        F_h: Fundamental frequency (Hz)
        duty_cycle: Duty cycle of pulse (0 to 1)
        time_duration: Duration of signal (seconds)
        Fs: Sampling rate (Hz)
        snr_db: SNR in dB
    
    Returns:
        iq: Complex IQ signal with noise
    """
    Ts = 1 / Fs
    T_h = 1 / F_h  # Period
    T_pulse = T_h * duty_cycle  # Pulse width
    
    num_samples = int(time_duration * Fs)
    t = np.arange(num_samples) * Ts
    
    # Create rectangular pulse train
    phase = (t % T_h)
    signal = np.where(phase < T_pulse, 1.0, 0.0)
    
    # Compute signal power
    signal_power = np.mean(signal ** 2)
    
    # Add complex Gaussian noise for target SNR
    snr_linear = 10 ** (snr_db / 10)
    noise_power = signal_power / snr_linear if snr_linear > 0 else signal_power * 100
    noise_std = np.sqrt(noise_power / 2)  # Split between real/imag
    
    noise = noise_std * (np.random.randn(num_samples) + 1j * np.random.randn(num_samples))
    iq = signal.astype(np.complex128) + noise
    
    return iq


def compute_log_psd(iq, fs, dur_ensemble=0.001, perc_overlap=75, kaiser_beta=10, target_bins=2048):
    """
    Compute log-PSD using WelchPSDEstimate, downsampled to fixed size.
    
    Args:
        iq: Complex IQ signal
        fs: Sampling rate
        dur_ensemble: Duration per FFT window (seconds)
        perc_overlap: Overlap percentage
        kaiser_beta: Kaiser window beta
        target_bins: Target number of frequency bins (downsample if larger)
    
    Returns:
        log_psd: Log-magnitude PSD (dB), shape (target_bins,)
        f_range: Frequency axis (Hz)
    """
    # Compute power signal
    iq_feature = np.real(iq * np.conj(iq))
    iq_feature = iq_feature - np.mean(iq_feature)
    
    # Minimal config dict (WelchPSDEstimate doesn't use it much)
    config_dict = {}
    
    psd = WelchPSDEstimate(iq_feature, fs, dur_ensemble, perc_overlap, kaiser_beta, config_dict)
    
    # Convert to dB
    log_psd = 10.0 * np.log10(np.maximum(psd, 1e-30))
    
    # Frequency axis (original)
    n_freq = len(psd)
    f_range_orig = np.linspace(-fs/2, fs/2, n_freq)
    
    # Downsample to fixed size for consistent CNN input
    if n_freq > target_bins:
        # Average pooling to downsample
        factor = n_freq // target_bins
        # Trim to make divisible
        trim_len = factor * target_bins
        log_psd = log_psd[:trim_len].reshape(target_bins, factor).mean(axis=1)
        f_range = f_range_orig[:trim_len:factor][:target_bins]
    else:
        f_range = f_range_orig
    
    return log_psd.astype(np.float32), f_range.astype(np.float32)


def generate_dataset(num_samples, fs, time_duration, dur_ensemble, perc_overlap, kaiser_beta,
                     f_h_range, duty_cycle_range, snr_range, seed=42):
    """
    Generate full dataset with varied parameters.
    
    Args:
        num_samples: Number of samples to generate
        fs: Sampling rate (Hz)
        time_duration: Duration per sample (seconds)
        dur_ensemble: PSD window duration
        perc_overlap: PSD overlap
        kaiser_beta: Kaiser beta
        f_h_range: (min, max) fundamental frequency (Hz)
        duty_cycle_range: (min, max) duty cycle
        snr_range: (min, max) SNR (dB)
        seed: Random seed
    
    Returns:
        psds: Array of log-PSDs, shape (N, Nfreq)
        labels: Array of fundamental frequencies (Hz), shape (N,)
        meta: Dict with SNRs, duty_cycles, etc.
    """
    np.random.seed(seed)
    
    psds = []
    labels = []
    snrs = []
    duty_cycles = []
    
    print(f"Generating {num_samples} samples...")
    
    for i in range(num_samples):
        # Random parameters
        f_h = np.random.uniform(f_h_range[0], f_h_range[1])
        duty_cycle = np.random.uniform(duty_cycle_range[0], duty_cycle_range[1])
        snr_db = np.random.uniform(snr_range[0], snr_range[1])
        
        # Generate IQ
        iq = generate_dirac_comb_iq(f_h, duty_cycle, time_duration, fs, snr_db)
        
        # Compute PSD
        log_psd, f_range = compute_log_psd(iq, fs, dur_ensemble, perc_overlap, kaiser_beta)
        
        psds.append(log_psd)
        labels.append(f_h)
        snrs.append(snr_db)
        duty_cycles.append(duty_cycle)
        
        if (i + 1) % 100 == 0:
            print(f"  Generated {i+1}/{num_samples}")
    
    psds = np.stack(psds)
    labels = np.array(labels, dtype=np.float32)
    snrs = np.array(snrs, dtype=np.float32)
    duty_cycles = np.array(duty_cycles, dtype=np.float32)
    
    return psds, labels, f_range, {'snrs': snrs, 'duty_cycles': duty_cycles}


def main():
    parser = argparse.ArgumentParser(description='Generate dataset for DL pitch estimation')
    parser.add_argument('--out-file', type=str, default='data/pitch_dataset.npz',
                        help='Output file path')
    parser.add_argument('--num-samples', type=int, default=5000,
                        help='Number of samples to generate')
    parser.add_argument('--fs', type=float, default=25e6,
                        help='Sampling rate (Hz)')
    parser.add_argument('--time-duration', type=float, default=0.1,
                        help='Duration per sample (seconds)')
    parser.add_argument('--dur-ensemble', type=float, default=0.001,
                        help='PSD window duration (seconds)')
    parser.add_argument('--perc-overlap', type=float, default=75,
                        help='PSD overlap percentage')
    parser.add_argument('--kaiser-beta', type=float, default=10,
                        help='Kaiser window beta')
    parser.add_argument('--f-h-min', type=float, default=100e3,
                        help='Min fundamental frequency (Hz)')
    parser.add_argument('--f-h-max', type=float, default=500e3,
                        help='Max fundamental frequency (Hz)')
    parser.add_argument('--duty-min', type=float, default=0.05,
                        help='Min duty cycle')
    parser.add_argument('--duty-max', type=float, default=0.20,
                        help='Max duty cycle')
    parser.add_argument('--snr-min', type=float, default=-20,
                        help='Min SNR (dB)')
    parser.add_argument('--snr-max', type=float, default=20,
                        help='Max SNR (dB)')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed')
    
    args = parser.parse_args()
    
    # Create output directory
    os.makedirs(os.path.dirname(args.out_file) or '.', exist_ok=True)
    
    # Generate dataset
    psds, labels, f_range, meta = generate_dataset(
        num_samples=args.num_samples,
        fs=args.fs,
        time_duration=args.time_duration,
        dur_ensemble=args.dur_ensemble,
        perc_overlap=args.perc_overlap,
        kaiser_beta=args.kaiser_beta,
        f_h_range=(args.f_h_min, args.f_h_max),
        duty_cycle_range=(args.duty_min, args.duty_max),
        snr_range=(args.snr_min, args.snr_max),
        seed=args.seed
    )
    
    # Save
    np.savez_compressed(
        args.out_file,
        psds=psds,
        labels=labels,
        f_range=f_range,
        snrs=meta['snrs'],
        duty_cycles=meta['duty_cycles'],
        fs=args.fs,
        dur_ensemble=args.dur_ensemble
    )
    
    print(f"\nSaved dataset to {args.out_file}")
    print(f"  Samples: {len(labels)}")
    print(f"  PSD shape: {psds.shape}")
    print(f"  Label range: {labels.min():.1f} - {labels.max():.1f} Hz")
    print(f"  SNR range: {meta['snrs'].min():.1f} - {meta['snrs'].max():.1f} dB")


if __name__ == '__main__':
    main()
