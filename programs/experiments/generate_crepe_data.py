"""
Generate CREPE-compatible synthetic RF signal data with varied pitches.

This script replicates the CREPE training data characteristics:
- 16 kHz sampling rate
- 1024 samples per frame (64 ms)
- Varied fundamental frequencies (32.7 Hz to 1975.5 Hz)
- Dense pitch coverage across all 360 bins
- Multiple SNR levels for robustness

Key fixes:
1. Dense pitch coverage - generate samples for ALL 360 bins
2. Proper train/val split - ensure overlap so model can interpolate
3. More augmentation - multiple samples per pitch/SNR combination
4. RF signal generation - complex IQ signals with realistic noise
"""

import numpy as np
import os
import pickle
from typing import List, Tuple, Dict
import matplotlib.pyplot as plt


# =============================================================================
# CREPE Constants (from paper)
# =============================================================================

CREPE_FS = 16000           # 16 kHz sampling rate
CREPE_FRAME_LENGTH = 1024  # 1024 samples = 64 ms at 16 kHz
CREPE_N_BINS = 360         # 360 pitch bins
CREPE_CENTS_PER_BIN = 20   # 20 cents per bin
CREPE_FMIN = 32.70         # C1 (~32.7 Hz) - minimum frequency
CREPE_FMAX = 1975.53       # B7 (~1975 Hz) - maximum frequency


def hz_to_cents(freq_hz: float, fref: float = 10.0) -> float:
    """Convert frequency in Hz to cents (relative to fref)."""
    return 1200.0 * np.log2(freq_hz / fref)


def cents_to_hz(cents: float, fref: float = 10.0) -> float:
    """Convert cents to frequency in Hz."""
    return fref * (2.0 ** (cents / 1200.0))


def hz_to_crepe_bin(freq_hz: float) -> int:
    """Convert frequency in Hz to CREPE bin index (0-359)."""
    cents = hz_to_cents(freq_hz)
    # CREPE bin 0 corresponds to ~32.7 Hz (C1)
    CENTS_OFFSET = 1997.3794084376191
    bin_idx = int(round((cents - CENTS_OFFSET) / CREPE_CENTS_PER_BIN))
    return np.clip(bin_idx, 0, CREPE_N_BINS - 1)


def crepe_bin_to_hz(bin_idx: int) -> float:
    """Convert CREPE bin index to frequency in Hz."""
    CENTS_OFFSET = 1997.3794084376191
    cents = CENTS_OFFSET + bin_idx * CREPE_CENTS_PER_BIN
    return cents_to_hz(cents)


# =============================================================================
# RF Signal Generation
# =============================================================================

def generate_harmonic_rf_signal(
    F_h: float, 
    Fs: float, 
    duration: float, 
    n_harmonics: int = 8,
    harmonic_decay: float = 1.5,
    phase_noise: bool = True
) -> np.ndarray:
    """
    Generate a complex RF signal with harmonic structure at fundamental frequency F_h.
    
    This creates a more realistic signal than pure sinusoids by including:
    - Multiple harmonics with 1/f^decay amplitude
    - Random phase offsets per harmonic
    - Optional phase noise (jitter)
    
    Args:
        F_h: Fundamental frequency in Hz
        Fs: Sampling rate in Hz
        duration: Duration in seconds
        n_harmonics: Number of harmonics to include
        harmonic_decay: Harmonic amplitude decay rate (higher = faster decay)
        phase_noise: Add realistic phase jitter
    
    Returns:
        Complex RF signal (I + jQ)
    """
    n_samples = int(duration * Fs)
    t = np.arange(n_samples) / Fs
    
    # Start with zeros
    signal_I = np.zeros(n_samples, dtype=np.float32)
    signal_Q = np.zeros(n_samples, dtype=np.float32)
    
    for h in range(1, n_harmonics + 1):
        freq = F_h * h
        
        # Only include harmonics below Nyquist
        if freq < Fs / 2:
            # Harmonic amplitude decreases with harmonic number
            amplitude = 1.0 / (h ** harmonic_decay)
            
            # Random phase offset for each harmonic
            phase_I = np.random.uniform(0, 2 * np.pi)
            phase_Q = np.random.uniform(0, 2 * np.pi)
            
            # Add phase noise if requested
            if phase_noise and h == 1:  # Only on fundamental
                phase_jitter = np.random.normal(0, 0.01, n_samples)
                phase_I_vec = 2 * np.pi * freq * t + phase_I + phase_jitter
                phase_Q_vec = 2 * np.pi * freq * t + phase_Q + phase_jitter
            else:
                phase_I_vec = 2 * np.pi * freq * t + phase_I
                phase_Q_vec = 2 * np.pi * freq * t + phase_Q
            
            signal_I += amplitude * np.cos(phase_I_vec)
            signal_Q += amplitude * np.sin(phase_Q_vec)
    
    # Normalize to unit power
    power = np.mean(signal_I**2 + signal_Q**2)
    if power > 0:
        signal_I /= np.sqrt(power)
        signal_Q /= np.sqrt(power)
    
    # Create complex signal
    complex_signal = signal_I + 1j * signal_Q
    
    return complex_signal.astype(np.complex64)


def add_complex_noise(signal: np.ndarray, snr_db: float, seed: int = None) -> np.ndarray:
    """
    Add complex Gaussian noise to achieve specified SNR.
    
    Args:
        signal: Input complex signal
        snr_db: Target SNR in dB
        seed: Random seed for reproducibility
    
    Returns:
        noisy_signal: Complex signal with AWGN added
    """
    if seed is not None:
        np.random.seed(seed)
    
    # Signal power (complex)
    signal_power = np.mean(np.abs(signal)**2)
    
    # Noise power for target SNR
    noise_power = signal_power / (10 ** (snr_db / 10))
    
    # Generate complex Gaussian noise
    noise_std = np.sqrt(noise_power / 2)  # Split between I and Q
    noise_I = np.random.normal(0, noise_std, len(signal))
    noise_Q = np.random.normal(0, noise_std, len(signal))
    noise = noise_I + 1j * noise_Q
    
    return signal + noise


# =============================================================================
# Dataset Generation - FIXED VERSION
# =============================================================================

def generate_crepe_dataset_dense(
    output_path: str,
    bins_to_generate: List[int] = None,
    snr_list: List[int] = None,
    samples_per_bin_snr: int = 20,
    n_harmonics: int = 8,
    seed: int = 42
) -> dict:
    """
    Generate a CREPE-compatible dataset with DENSE pitch coverage.
    
    KEY FIX: Instead of sparse pitches, we generate samples for many/all CREPE bins
    so the model learns the full pitch space and can interpolate.
    
    Args:
        output_path: Path to save the pickle file
        bins_to_generate: List of CREPE bin indices to generate (None = all 360)
        snr_list: List of SNR values in dB
        samples_per_bin_snr: Number of augmented samples per (bin, SNR) pair
        n_harmonics: Number of harmonics in signal
        seed: Random seed
    
    Returns:
        iq_dict: Dictionary mapping keys to IQ arrays
    """
    np.random.seed(seed)
    
    if snr_list is None:
        # Wider SNR range for robustness
        snr_list = [20, 15, 10, 5, 0, -5, -10, -15, -20]
    
    if bins_to_generate is None:
        # Generate ALL 360 bins for complete coverage
        bins_to_generate = list(range(CREPE_N_BINS))
    
    print("=" * 80)
    print("Generating CREPE-Compatible Dataset (DENSE PITCH COVERAGE)")
    print("=" * 80)
    print(f"\nParameters:")
    print(f"  Sampling rate: {CREPE_FS} Hz")
    print(f"  Frame length: {CREPE_FRAME_LENGTH} samples ({CREPE_FRAME_LENGTH/CREPE_FS*1000:.1f} ms)")
    print(f"  Number of bins: {len(bins_to_generate)} (out of {CREPE_N_BINS})")
    print(f"  SNR range: {max(snr_list)} dB to {min(snr_list)} dB ({len(snr_list)} levels)")
    print(f"  Samples per (bin, SNR): {samples_per_bin_snr}")
    print(f"  Total samples: {len(bins_to_generate) * len(snr_list) * samples_per_bin_snr:,}")
    
    duration = CREPE_FRAME_LENGTH / CREPE_FS
    
    iq_dict = {}
    sample_count = 0
    
    for bin_idx in bins_to_generate:
        # Get frequency for this bin
        F_h = crepe_bin_to_hz(bin_idx)
        
        # Verify this is in valid range
        if F_h < CREPE_FMIN or F_h > CREPE_FMAX:
            continue
        
        for snr in snr_list:
            for aug_idx in range(samples_per_bin_snr):
                # Generate clean signal with varying characteristics
                # Vary harmonics and decay for diversity
                n_harm = np.random.randint(5, n_harmonics + 1)
                decay = np.random.uniform(1.2, 2.0)
                phase_noise = np.random.rand() > 0.3  # 70% with phase noise
                
                clean_signal = generate_harmonic_rf_signal(
                    F_h, CREPE_FS, duration, 
                    n_harmonics=n_harm,
                    harmonic_decay=decay,
                    phase_noise=phase_noise
                )
                
                # Add noise with deterministic seed for reproducibility
                noise_seed = bin_idx * 100000 + snr * 1000 + aug_idx
                noisy_signal = add_complex_noise(clean_signal, snr, seed=noise_seed)
                
                # Key format: "BIN_XXX_SNR_YY_AUG_ZZ"
                key = f"BIN_{bin_idx:03d}_SNR_{snr:+03d}_AUG_{aug_idx:03d}"
                iq_dict[key] = noisy_signal
                sample_count += 1
        
        if (bin_idx + 1) % 50 == 0 or bin_idx == 0:
            print(f"  Generated bin {bin_idx}/{bins_to_generate[-1]}: "
                  f"F_h = {F_h:.2f} Hz ({sample_count:,} samples so far)")
    
    # Save to pickle
    os.makedirs(os.path.dirname(output_path) if os.path.dirname(output_path) else '.', exist_ok=True)
    with open(output_path, 'wb') as f:
        pickle.dump(iq_dict, f)
    
    print(f"\n✓ Saved {len(iq_dict):,} samples to {output_path}")
    
    # Statistics
    bins = sorted(list(set([int(k.split('_')[1]) for k in iq_dict.keys()])))
    freqs = [crepe_bin_to_hz(b) for b in bins]
    
    print(f"\nDataset statistics:")
    print(f"  Unique bins: {len(bins)}")
    print(f"  Frequency range: {min(freqs):.1f} Hz - {max(freqs):.1f} Hz")
    print(f"  Bin range: {min(bins)} - {max(bins)}")
    print(f"  Samples per bin: {len(snr_list) * samples_per_bin_snr}")
    
    return iq_dict


def visualize_dataset(iq_dict: dict, n_samples: int = 6):
    """Visualize samples from the dataset."""
    import re
    
    keys = list(iq_dict.keys())
    np.random.shuffle(keys)
    keys = keys[:n_samples]
    
    fig, axes = plt.subplots(n_samples, 2, figsize=(14, 3 * n_samples))
    if n_samples == 1:
        axes = axes.reshape(1, -1)
    
    for i, key in enumerate(keys):
        # Parse key
        bin_match = re.search(r'BIN_(\d+)', key)
        snr_match = re.search(r'SNR_([+-]?\d+)', key)
        
        bin_idx = int(bin_match.group(1)) if bin_match else 0
        snr = int(snr_match.group(1)) if snr_match else 0
        F_h = crepe_bin_to_hz(bin_idx)
        
        signal = iq_dict[key]
        
        # Time domain - magnitude
        t = np.arange(len(signal)) / CREPE_FS * 1000  # ms
        axes[i, 0].plot(t, np.abs(signal), linewidth=0.5)
        axes[i, 0].set_xlabel('Time (ms)')
        axes[i, 0].set_ylabel('Magnitude')
        axes[i, 0].set_title(f'Bin {bin_idx}: F_h = {F_h:.1f} Hz, SNR = {snr} dB')
        axes[i, 0].grid(True, alpha=0.3)
        
        # Frequency domain
        fft = np.fft.fftshift(np.fft.fft(signal))
        freqs = np.fft.fftshift(np.fft.fftfreq(len(signal), 1/CREPE_FS))
        axes[i, 1].plot(freqs, 20 * np.log10(np.abs(fft) + 1e-10), linewidth=0.5)
        
        # Mark fundamental and harmonics
        for h in range(1, 6):
            if F_h * h < CREPE_FS / 2:
                axes[i, 1].axvline(x=F_h * h, color='r', linestyle='--', 
                                  alpha=0.5, linewidth=0.8)
        
        axes[i, 1].set_xlabel('Frequency (Hz)')
        axes[i, 1].set_ylabel('Magnitude (dB)')
        axes[i, 1].set_xlim([0, 2500])
        axes[i, 1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('dataset_visualization.png', dpi=150, bbox_inches='tight')
    plt.close()
    print("✓ Saved visualization to dataset_visualization.png")


# =============================================================================
# Main
# =============================================================================

if __name__ == "__main__":
    OUTPUT_DIR = './IQData/'
    OUTPUT_FILE = 'iq_dict_crepe_dense.pkl'
    OUTPUT_PATH = os.path.join(OUTPUT_DIR, OUTPUT_FILE)
    
    # Strategy: Generate dense coverage
    # Option 1: ALL 360 bins (takes longer but best results)
    # Option 2: Every 2nd bin = 180 bins (faster, still good)
    # Option 3: Every 3rd bin = 120 bins (balance speed/coverage)
    
    # For demonstration, use every 2nd bin (180 bins total)
    # Change to range(360) for full coverage
    bins_to_generate = list(range(0, 360, 2))  # Every other bin
    
    iq_dict = generate_crepe_dataset_dense(
        output_path=OUTPUT_PATH,
        bins_to_generate=bins_to_generate,  # 180 bins
        snr_list=[20, 15, 10, 5, 0, -5, -10, -15, -20],  # 9 SNR levels
        samples_per_bin_snr=10,  # 10 augmentations per (bin, SNR)
        n_harmonics=8,
        seed=42
    )
    
    # Total: 180 bins × 9 SNRs × 10 augmentations = 16,200 samples
    
    visualize_dataset(iq_dict, n_samples=6)
    
    print("\n" + "=" * 80)
    print("Dataset generation complete!")
    print("=" * 80)
    print(f"\nNext steps:")
    print(f"  1. Run train_crepe.py to train the model")
    print(f"  2. Evaluate with RPA/RCA metrics")