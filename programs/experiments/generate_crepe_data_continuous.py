"""
Generate CREPE-compatible dataset with TRULY CONTINUOUS frequency coverage.

f_h is randomly generated between fmin and fmax as continuous values (1 decimal place).
This better simulates real-world scenarios where frequencies are truly continuous.

Structure:
- Loop 1: n_input_frames iterations (each picks a random f_h, uniformly distributed)
- Loop 2: iterate over SNR values
- Seed changes with each iteration index for reproducibility

Total samples: n_input_frames × len(snr_list)
"""

import numpy as np
import pickle
import os
from tqdm import tqdm

# Single main seed for reproducible signal parameters (Fh, duty cycle)
main_seed = 1234
rng = np.random.default_rng(seed=main_seed)

# Separate unseeded generator for random noise (different each run)
noise_rng = np.random.default_rng()

# Import from existing code
from generate_crepe_data import (
    CREPE_FS,
    CREPE_FRAME_LENGTH,
    generate_dirac_comb_signal,
    add_complex_noise,
    hz_to_crepe_bin
)


# def generate_random_fh(f_min: float, f_max: float, seed: int) -> float:
#     """
#     Generate a random f_h value between f_min and f_max with 1 decimal place.
#     Uniformly distributed across the frequency range.
#     
#     Args:
#         f_min: Minimum frequency in Hz
#         f_max: Maximum frequency in Hz
#         seed: Random seed for reproducibility
#     
#     Returns:
#         f_h: Random frequency rounded to 1 decimal place
#     """
#     np.random.seed(seed)
#     f_h = np.random.uniform(f_min, f_max)
#     # Round to 1 decimal place
#     f_h = round(f_h, 1)
#     return f_h


def generate_continuous_dataset(
    output_path: str,
    f_min: float = 32.0,
    f_max: float = 1976.0,
    snr_list: list = None,
    n_input_frames: int = 10000,
    duty_cycle: float = 0.5
) -> dict:
    """
    Generate dataset with truly continuous (random) f_h values.
    Frequencies are uniformly distributed between f_min and f_max.
    
    Args:
        output_path: Path to save the pickle file
        f_min: Minimum frequency in Hz (default: 32.0)
        f_max: Maximum frequency in Hz (default: 1976.0)
        snr_list: List of SNR values (default: [0, 1, 2, ..., 20])
        n_input_frames: Number of input frames (each gets a random f_h)
        duty_cycle: Duty cycle for pulse generation
    
    Returns:
        iq_dict: Dictionary mapping keys to IQ arrays
    """
    if snr_list is None:
        snr_list = list(range(0, 21))  # Default SNR values from 0 to 20 dB
    
    n_snr = len(snr_list)
    total_samples = n_input_frames * n_snr
    
    # Pre-generate all Fh values upfront using main rng
    Fh_values = rng.uniform(f_min, f_max, n_input_frames)
    Fh_values = np.round(Fh_values, 1)  # Round to 1 decimal place
    
    print("=" * 70)
    print("Generating TRULY Continuous Frequency Dataset")
    print("=" * 70)
    print(f"\nParameters:")
    print(f"  Sampling rate: {CREPE_FS} Hz")
    print(f"  Frame length: {CREPE_FRAME_LENGTH} samples")
    print(f"  F_h range: {f_min} Hz to {f_max} Hz (continuous, 1 decimal place)")
    print(f"  Distribution: Uniform")
    print(f"  SNR values: {snr_list} ({n_snr} levels)")
    print(f"  Input frames: {n_input_frames:,}")
    print(f"  Duty cycle: {duty_cycle*100:.0f}%")
    print(f"  Main seed: {main_seed}")
    print(f"  Total samples: {total_samples:,}")
    print("=" * 70)
    
    duration = CREPE_FRAME_LENGTH / CREPE_FS
    iq_dict = {}
    sample_count = 0
    
    # Loop 1: For each input frame (use pre-generated f_h values)
    for frame_idx in tqdm(range(n_input_frames), desc="Generating frames"):
        
        # Use pre-generated f_h value
        f_h = Fh_values[frame_idx]
        
        # Convert to CREPE bin for the label
        bin_idx = hz_to_crepe_bin(f_h)
        
        # Generate clean signal
        clean_signal = generate_dirac_comb_signal(
            F_h=f_h,
            Fs=CREPE_FS,
            duration=duration,
            duty_cycle=duty_cycle
        )
        
        # Loop 2: For each SNR value
        for snr_idx, snr in enumerate(snr_list):
            # Add noise using unseeded noise generator
            noisy_signal = add_complex_noise(clean_signal, snr, noise_rng)
            
            # Key format: BIN_XXX_SNR_XX_IDX_XXXXX_FH_XXXX.X
            # BIN and SNR are in positions expected by training code
            key = f"BIN_{bin_idx:03d}_SNR_{snr:02d}_IDX_{frame_idx:05d}_FH_{f_h:07.1f}"
            
            iq_dict[key] = noisy_signal
            sample_count += 1
    
    # Save to pickle
    os.makedirs(os.path.dirname(output_path) if os.path.dirname(output_path) else '.', exist_ok=True)
    with open(output_path, 'wb') as f:
        pickle.dump(iq_dict, f)
    
    print(f"\n✓ Saved {len(iq_dict):,} samples to {output_path}")
    
    # Statistics - extract f_h values from keys
    fh_values = []
    for k in iq_dict.keys():
        # Extract FH value from key
        parts = k.split('_')
        fh_idx = parts.index('FH') + 1
        fh_values.append(float(parts[fh_idx]))
    
    unique_fh = len(set(fh_values))
    print(f"\nDataset statistics:")
    print(f"  Unique f_h values: {unique_fh}")
    print(f"  f_h range: {min(fh_values):.1f} Hz - {max(fh_values):.1f} Hz")
    print(f"  Input frames: {n_input_frames:,}")
    print(f"  SNR levels: {n_snr}")
    print(f"  Total samples: {len(iq_dict):,}")
    
    return iq_dict


if __name__ == "__main__":
    OUTPUT_DIR = './IQData/'
    OUTPUT_FILE = 'iq_dict_continuous_freq_SNR0_20(25-2-26).pkl'
    
    # Configuration
    config = {
        'output_path': os.path.join(OUTPUT_DIR, OUTPUT_FILE),
        'f_min': 32.7,
        'f_max': 2067.0,
        'snr_list': list(range(0, 21)),  # 21 SNR levels (0 to 20 dB)
        'n_input_frames': 33333,               # 10,000 random frequencies
        'duty_cycle': 0.5
    }
    # Total: 33,333 frames × 21 SNRs = 699,993 samples (rounded to 700k for easier tracking)
    # Total: 10,000 frames × 6 SNRs = 60,000 samples
    
    iq_dict = generate_continuous_dataset(**config)