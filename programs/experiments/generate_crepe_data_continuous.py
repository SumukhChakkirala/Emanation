"""
Generate CREPE-compatible dataset with CONTINUOUS frequency coverage.

Unlike the bin-based approach, this uses integer F_h values from 32 to 1976 Hz.
This better simulates real-world scenarios where frequencies are continuous.

Structure:
- F_h: Integer frequencies from 32 to 1976 Hz (~1945 frequencies)
- SNR: 5 values from 15 to 20 dB
- Augmentations: 10 per (F_h, SNR) pair

Total samples: 1945 × 5 × 10 = 97,250 samples
"""

import numpy as np
import os
import pickle

# Import from existing code
from generate_crepe_data import (
    CREPE_FS,
    CREPE_FRAME_LENGTH,
    generate_dirac_comb_signal,
    add_complex_noise,
    hz_to_crepe_bin
)


def generate_continuous_dataset(
    output_path: str,
    f_min: int = 32,
    f_max: int = 1976,
    snr_list: list = None,
    n_augments: int = 10,
    duty_cycle: float = 0.5,
    seed: int = 42
) -> dict:
    """
    Generate dataset with continuous (integer) F_h values.
    
    Args:
        output_path: Path to save the pickle file
        f_min: Minimum frequency in Hz (default: 32)
        f_max: Maximum frequency in Hz (default: 1976)
        snr_list: List of SNR values (default: [15, 16, 17, 18, 19, 20])
        n_augments: Number of augmentations per (F_h, SNR) pair
        duty_cycle: Duty cycle for pulse generation
        seed: Random seed
    
    Returns:
        iq_dict: Dictionary mapping keys to IQ arrays
    """
    np.random.seed(seed)
    
    if snr_list is None:
        snr_list = [15, 16, 17, 18, 19, 20]  # 5 SNR values
    
    # Generate list of integer frequencies
    freq_list = list(range(f_min, f_max + 1))
    n_freqs = len(freq_list)
    
    print("=" * 70)
    print("Generating Continuous Frequency Dataset")
    print("=" * 70)
    print(f"\nParameters:")
    print(f"  Sampling rate: {CREPE_FS} Hz")
    print(f"  Frame length: {CREPE_FRAME_LENGTH} samples")
    print(f"  F_h range: {f_min} Hz to {f_max} Hz ({n_freqs} frequencies)")
    print(f"  SNR values: {snr_list} ({len(snr_list)} levels)")
    print(f"  Augmentations: {n_augments}")
    print(f"  Duty cycle: {duty_cycle*100:.0f}%")
    print(f"  Total samples: {n_freqs * len(snr_list) * n_augments:,}")
    print("=" * 70)
    
    duration = CREPE_FRAME_LENGTH / CREPE_FS
    iq_dict = {}
    sample_count = 0
    
    # Loop 1: For each integer F_h
    for f_h in freq_list:
        bin_idx = hz_to_crepe_bin(f_h)  # Get corresponding bin for reference
        
        # Loop 2: For each SNR
        for snr in snr_list:
            
            # Loop 3: For each augmentation
            for aug_idx in range(n_augments):
                # Vary duty cycle slightly for augmentation
                dc = duty_cycle * np.random.uniform(0.8, 1.2)
                dc = np.clip(dc, 0.05, 0.5)
                
                # Generate clean signal
                clean_signal = generate_dirac_comb_signal(
                    f_h, CREPE_FS, duration, duty_cycle=dc
                )
                
                # Add noise
                noise_seed = f_h * 10000 + snr * 100 + aug_idx
                noisy_signal = add_complex_noise(clean_signal, snr, seed=noise_seed)
                
                # Key format: "FH_XXXX_BIN_XXX_SNR_XX_AUG_XX"
                key = f"FH_{f_h:04d}_BIN_{bin_idx:03d}_SNR_{snr:02d}_AUG_{aug_idx:02d}"
                iq_dict[key] = noisy_signal
                sample_count += 1
        
        # Progress update every 100 frequencies
        if (f_h - f_min + 1) % 200 == 0 or f_h == f_min:
            pct = (f_h - f_min + 1) / n_freqs * 100
            print(f"  F_h = {f_h:4d} Hz (bin {bin_idx:3d}) | {sample_count:,} samples | {pct:.1f}%")
    
    # Save to pickle
    os.makedirs(os.path.dirname(output_path) if os.path.dirname(output_path) else '.', exist_ok=True)
    with open(output_path, 'wb') as f:
        pickle.dump(iq_dict, f)
    
    print(f"\n✓ Saved {len(iq_dict):,} samples to {output_path}")
    
    # Statistics
    freqs = sorted(list(set([int(k.split('_')[1]) for k in iq_dict.keys()])))
    print(f"\nDataset statistics:")
    print(f"  Unique frequencies: {len(freqs)}")
    print(f"  Frequency range: {min(freqs)} Hz - {max(freqs)} Hz")
    print(f"  Samples per frequency: {len(snr_list) * n_augments}")
    
    return iq_dict


if __name__ == "__main__":
    OUTPUT_DIR = './IQData/'
    OUTPUT_FILE = 'iq_dict_continuous_freq.pkl'
    OUTPUT_PATH = os.path.join(OUTPUT_DIR, OUTPUT_FILE)
    
    iq_dict = generate_continuous_dataset(
        output_path=OUTPUT_PATH,
        f_min=32,
        f_max=1976,
        snr_list=[15, 16, 17, 18, 19, 20],  # 5 SNR values (15-20 dB)
        n_augments=10,
        duty_cycle=0.1,
        seed=42
    )
    
    # Total: 1945 freqs × 6 SNRs × 10 augments = 116,700 samples
    
    print("\n" + "=" * 70)
    print("Dataset generation complete!")
    print("=" * 70)
