"""
Generate CREPE-compatible synthetic RF signal data with varied pitches.

This script uses the Dirac comb + rectangular pulse approach from DiracCombPlots.py:
- Creates a rectangular pulse of width T = duty_cycle * T_h
- Creates a Dirac comb with period T_h = 1/F_h
- Convolves them to get a pulse train (real-valued baseband signal)
- Adds complex Gaussian noise (I + jQ) at various SNR levels

Key characteristics:
- 16 kHz sampling rate (CREPE standard)
- 1024 samples per frame (64 ms)
- Varied fundamental frequencies (32.7 Hz to 1975.5 Hz)
- Dense pitch coverage across all 360 bins
- Multiple SNR levels for robustness (matching DiracCombPlots.py: 20 dB to -40 dB)
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
# RF Signal Generation (Dirac Comb + Rectangular Pulse approach)
# =============================================================================

def generate_dirac_comb_signal(
    F_h: float, 
    Fs: float, 
    duration: float, 
    duty_cycle: float = 0.1
) -> np.ndarray:
    """
    Generate a pulse train signal using Dirac comb convolved with rectangular pulse.
    
    This follows the approach from DiracCombPlots.py:
    1. Create a rectangular pulse of width T = duty_cycle * T_h
    2. Create a Dirac comb with period T_h = 1/F_h
    3. Convolve them to get the pulse train
    
    Args:
        F_h: Fundamental frequency in Hz (pulse repetition rate)
        Fs: Sampling rate in Hz
        duration: Duration in seconds
        duty_cycle: Duty cycle of the pulse (0 to 1)
    
    Returns:
        Real-valued pulse train signal (baseband)
    """
    Ts = 1 / Fs  # Sampling period
    T_h = 1 / F_h  # Period of the pulse train
    T = T_h * duty_cycle  # Duration of each pulse
    
    n_samples = int(duration * Fs)
    
    # Create time vector for the full signal
    t_dc = np.linspace(0, duration, n_samples)
    
    # Create one period of rectangular pulse (centered)
    n_samples_period = int(T_h / Ts)
    t_rp = np.linspace(0, T_h, n_samples_period)
    y_rp = np.zeros(len(t_rp), dtype=np.float32)
    
    # Center the pulse in the period
    ps_idx = np.abs(t_rp - (T_h / 2 - T / 2)).argmin()  # pulse start index
    pe_idx = np.abs(t_rp - (T / 2 + T_h / 2)).argmin()  # pulse end index
    y_rp[ps_idx:pe_idx] = 1.0
    
    # Create Dirac comb
    dir_c = np.zeros(n_samples, dtype=np.float32)
    
    # Place impulses at each period
    impulse_times = np.arange(T_h, duration, T_h)
    impulse_indices = np.rint(impulse_times / Ts).astype(int)
    # Ensure indices are within bounds
    impulse_indices = impulse_indices[impulse_indices < n_samples]
    dir_c[impulse_indices] = 1.0
    
    # Convolve rectangular pulse with Dirac comb to get pulse train
    rect_tr = np.convolve(y_rp, dir_c, 'same')
    
    # Normalize to unit power
    power = np.mean(rect_tr ** 2)
    if power > 0:
        rect_tr = rect_tr / np.sqrt(power)
    
    return rect_tr.astype(np.float32)


def generate_harmonic_rf_signal(
    F_h: float, 
    Fs: float, 
    duration: float, 
    duty_cycle: float = 0.1,
    n_harmonics: int = 8,
    harmonic_decay: float = 1.5,
    phase_noise: bool = True
) -> np.ndarray:
    """
    Generate a signal using Dirac comb approach (like DiracCombPlots.py).
    
    This creates a pulse train by convolving a rectangular pulse with a Dirac comb,
    which naturally produces harmonics at multiples of F_h.
    
    Args:
        F_h: Fundamental frequency in Hz
        Fs: Sampling rate in Hz
        duration: Duration in seconds
        duty_cycle: Duty cycle of the rectangular pulse (0 to 1)
        n_harmonics: Not used (kept for API compatibility)
        harmonic_decay: Not used (kept for API compatibility)
        phase_noise: Not used (kept for API compatibility)
    
    Returns:
        Real-valued pulse train signal (baseband)
    """
    # Use the Dirac comb approach
    return generate_dirac_comb_signal(F_h, Fs, duration, duty_cycle)


def add_complex_noise(signal: np.ndarray, snr_db: float, seed: int = None) -> np.ndarray:
    """
    Add complex Gaussian noise to achieve specified SNR.
    
    This follows the approach from DiracCombPlots.py:
    - Compute signal variance
    - Generate complex noise (I + jQ) based on target SNR
    - Add noise to signal
    
    Args:
        signal: Input signal (real or complex)
        snr_db: Target SNR in dB
        seed: Random seed for reproducibility
    
    Returns:
        noisy_signal: Signal with complex AWGN added
    """
    if seed is not None:
        np.random.seed(seed)
    
    # Signal variance (following DiracCombPlots approach)
    var_y = np.var(np.real(signal))
    
    # Noise variance for target SNR: var_s = 0.5 * (var_y / 10^(SNR/10))
    var_s = 0.5 * (var_y / (10 ** (snr_db / 10)))
    
    # Generate complex Gaussian noise (I + jQ)
    if seed is not None:
        np.random.seed(seed)
    w_s_I = np.random.normal(loc=0, scale=np.sqrt(var_s), size=len(signal))
    if seed is not None:
        np.random.seed(seed + 1)
    w_s_Q = np.random.normal(loc=0, scale=np.sqrt(var_s), size=len(signal))
    w_s = w_s_I + 1j * w_s_Q
    
    # Add noise to signal (real signal + complex noise -> complex signal)
    noisy_signal = signal + w_s
    
    return noisy_signal.astype(np.complex64)


# =============================================================================
# Dataset Generation - FIXED VERSION
# =============================================================================

def generate_crepe_dataset_dense(
    output_path: str,
    bins_to_generate: List[int] = None,
    snr_list: List[int] = None,
    samples_per_bin_snr: int = 20,
    duty_cycle: float = 0.1,
    seed: int = 42
) -> dict:
    """
    Generate a CREPE-compatible dataset with DENSE pitch coverage.
    
    Uses the Dirac comb + rectangular pulse approach from DiracCombPlots.py.
    
    Args:
        output_path: Path to save the pickle file
        bins_to_generate: List of CREPE bin indices to generate (None = all 360)
        snr_list: List of SNR values in dB
        samples_per_bin_snr: Number of augmented samples per (bin, SNR) pair
        duty_cycle: Duty cycle of the rectangular pulse (0 to 1)
        seed: Random seed
    
    Returns:
        iq_dict: Dictionary mapping keys to IQ arrays
    """
    np.random.seed(seed)
    
    if snr_list is None:
        # SNR range: 15 to 20 dB (high SNR for cleaner signals)
        snr_list = list(range(15, 21))  # [15, 16, 17, 18, 19, 20]
    
    if bins_to_generate is None:
        # Generate ALL 360 bins for complete coverage
        bins_to_generate = list(range(CREPE_N_BINS))
    
    print("=" * 80)
    print("Generating CREPE-Compatible Dataset (Dirac Comb Approach)")
    print("=" * 80)
    print(f"\nParameters:")
    print(f"  Sampling rate: {CREPE_FS} Hz")
    print(f"  Frame length: {CREPE_FRAME_LENGTH} samples ({CREPE_FRAME_LENGTH/CREPE_FS*1000:.1f} ms)")
    print(f"  Duty cycle: {duty_cycle*100:.0f}%")
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
                # Vary duty cycle slightly for augmentation
                dc = duty_cycle * np.random.uniform(0.8, 1.2)
                dc = np.clip(dc, 0.05, 0.5)  # Keep duty cycle reasonable
                
                # Generate clean pulse train signal using Dirac comb approach
                clean_signal = generate_dirac_comb_signal(
                    F_h, CREPE_FS, duration, 
                    duty_cycle=dc
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
    OUTPUT_FILE = 'iq_dict_crepe_dirac_comb.pkl'
    OUTPUT_PATH = os.path.join(OUTPUT_DIR, OUTPUT_FILE)
    
    # Strategy: Generate dense coverage using Dirac comb approach
    # Option 1: ALL 360 bins (takes longer but best results)
    # Option 2: Every 2nd bin = 180 bins (faster, still good)
    # Option 3: Every 3rd bin = 120 bins (balance speed/coverage)
    
    # For demonstration, use every 2nd bin (180 bins total)
    # Change to range(360) for full coverage
    bins_to_generate = list(range(0, 360))  # Every other bin
    
    # Duty cycle (same as DiracCombPlots.py default)
    duty_cycle = 0.1
    
    iq_dict = generate_crepe_dataset_dense(
        output_path=OUTPUT_PATH,
        bins_to_generate=bins_to_generate,  # 360 bins
        snr_list=list(range(15, 21)),  # SNR range: 15 to 20 dB
        samples_per_bin_snr=10,  # 10 augmentations per (bin, SNR)
        duty_cycle=duty_cycle,
        seed=42
    )
    
    # Total: 180 bins × 6 SNRs × 10 augmentations = 10,800 samples
    
    visualize_dataset(iq_dict, n_samples=6)
    
    print("\n" + "=" * 80)
    print("Dataset generation complete!")
    print("=" * 80)
    print(f"\nNext steps:")
    print(f"  1. Run train_crepe.py to train the model")
    print(f"  2. Evaluate with RPA/RCA metrics")