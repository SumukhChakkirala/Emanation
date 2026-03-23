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
from scipy import signal
import yaml

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


def _load_emanation_config(cfg_path: str = 'synapse_emanation_search.yaml') -> dict:
    if os.path.exists(cfg_path):
        with open(cfg_path, 'r') as file:
            return yaml.safe_load(file)
    return {'EmanationDetection': {'kaiser_beta_hh': 10.0, 'numtaps': 129}}


def _extract_1024_frame_from_25mhz(
    iq_signal: np.ndarray,
    fs_raw: float,
    case: str,
    cfg: dict,
    target_length: int = CREPE_FRAME_LENGTH,
    welch_nperseg: int = 1000,
) -> np.ndarray:
    """Convert long IQ snapshot to fixed 1024-point real frame for CNN input."""
    _, frame, _ = _extract_psd_and_ifft_from_25mhz(
        iq_signal=iq_signal,
        fs_raw=fs_raw,
        case=case,
        cfg=cfg,
        target_length=target_length,
        welch_nperseg=welch_nperseg,
    )
    return frame


def _extract_psd_and_ifft_from_25mhz(
    iq_signal: np.ndarray,
    fs_raw: float,
    case: str,
    cfg: dict,
    target_length: int = CREPE_FRAME_LENGTH,
    welch_nperseg: int = 1000,
):
    """Convert long IQ snapshot to PSD (1000 bins) and fixed 1024-point IFFT real frame."""
    ed_cfg = cfg.get('EmanationDetection', {}) if cfg is not None else {}
    kaiser_beta_hh = float(ed_cfg.get('kaiser_beta_hh', 10.0))
    numtaps = int(ed_cfg.get('numtaps', 129))

    x = np.asarray(iq_signal, dtype=np.complex64)
    fs_eff = float(fs_raw)

    case_u = case.upper()
    if case_u == 'B':
        cutoff_hz = 1e6
        dec_order = 25
        taps = signal.firwin(numtaps, cutoff_hz, window=('kaiser', kaiser_beta_hh), fs=fs_eff)
        x = signal.lfilter(taps, 1, x)
        x = signal.decimate(x, dec_order, ftype='fir', zero_phase=True)
        fs_eff = fs_eff / dec_order
    elif case_u == 'C':
        cutoff_hz = 40e3
        dec_order = int(fs_eff / cutoff_hz)  # 25e6/40e3 = 625
        dec_order = max(dec_order, 1)
        taps = signal.firwin(numtaps, cutoff_hz, window=('kaiser', kaiser_beta_hh), fs=fs_eff)
        x = signal.lfilter(taps, 1, x)
        x = signal.decimate(x, dec_order, ftype='fir', zero_phase=True)
        fs_eff = fs_eff / dec_order

    x_feature = np.real(np.multiply(x, np.conj(x)))
    x_feature = x_feature - np.mean(x_feature)

    nperseg = min(int(welch_nperseg), len(x_feature))
    if nperseg < 8:
        return np.zeros(target_length, dtype=np.float32)

    window = signal.windows.kaiser(nperseg, beta=kaiser_beta_hh)
    _, psd = signal.welch(
        x_feature,
        fs=fs_eff,
        window=window,
        nperseg=nperseg,
        noverlap=0,
        nfft=nperseg,
        return_onesided=False,
        detrend=False,
        scaling='density'
    )

    psd = np.asarray(psd, dtype=np.float64)
    psd[~np.isfinite(psd)] = 0.0
    psd = np.maximum(psd, 1e-20)
    psd = np.fft.fftshift(psd)

    if len(psd) != 1000:
        src = np.linspace(0.0, 1.0, len(psd), endpoint=True)
        dst = np.linspace(0.0, 1.0, 1000, endpoint=True)
        psd = np.interp(dst, src, psd)

    pad_total = max(target_length - len(psd), 0)
    pad_left = pad_total // 2
    pad_right = pad_total - pad_left
    psd_padded = np.pad(psd, (pad_left, pad_right), mode='constant')

    frame_td = np.fft.ifft(np.fft.ifftshift(psd_padded))
    frame = np.real(frame_td[:target_length]).astype(np.float32)
    frame /= (np.max(np.abs(frame)) + 1e-8)

    return psd.astype(np.float64), frame, fs_eff


def generate_25mhz_case_dataset(
    output_path: str,
    case: str,
    f_min: float,
    f_max: float,
    snr_list: list = None,
    n_input_frames: int = 10000,
    duty_cycle: float = 0.5,
    fs_raw: float = 25e6,
    capture_duration_s: float = 0.1,
    cfg_path: str = 'synapse_emanation_search.yaml',
) -> dict:
    """Generate 25 MHz Case A/B/C dataset with model-ready 1024-point inputs."""
    if snr_list is None:
        snr_list = list(range(-20, 21))

    cfg = _load_emanation_config(cfg_path)

    n_snr = len(snr_list)
    total_samples = n_input_frames * n_snr
    fh_values = rng.uniform(f_min, f_max, n_input_frames)
    fh_values = np.round(fh_values, 1)

    print("=" * 80)
    print(f"Generating 25MHz Case {case.upper()} dataset")
    print("=" * 80)
    print(f"  fs_raw: {fs_raw/1e6:.1f} MHz")
    print(f"  capture_duration_s: {capture_duration_s}")
    print(f"  f_h range: {f_min} Hz to {f_max} Hz")
    print(f"  SNR levels: {snr_list}")
    print(f"  input frames: {n_input_frames:,}")
    print(f"  total samples: {total_samples:,}")

    iq_dict = {}
    for frame_idx in tqdm(range(n_input_frames), desc=f"Case-{case.upper()} frames"):
        f_h = float(fh_values[frame_idx])
        bin_idx = hz_to_crepe_bin(f_h)

        clean_signal = generate_dirac_comb_signal(
            F_h=f_h,
            Fs=fs_raw,
            duration=capture_duration_s,
            duty_cycle=duty_cycle
        )

        for snr in snr_list:
            noisy_signal = add_complex_noise(clean_signal, snr, noise_rng)
            frame = _extract_1024_frame_from_25mhz(
                iq_signal=noisy_signal,
                fs_raw=fs_raw,
                case=case,
                cfg=cfg,
                target_length=CREPE_FRAME_LENGTH,
                welch_nperseg=1000
            )

            key = f"BIN_{bin_idx:03d}_SNR_{snr:+03d}_IDX_{frame_idx:05d}_FH_{f_h:09.1f}_CASE_{case.upper()}"
            iq_dict[key] = frame

    os.makedirs(os.path.dirname(output_path) if os.path.dirname(output_path) else '.', exist_ok=True)
    with open(output_path, 'wb') as file:
        pickle.dump(iq_dict, file)

    print(f"\n✓ Saved {len(iq_dict):,} samples to {output_path}")
    return iq_dict


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
def save_dataset_info(output_path: str, iq_dict: dict, case: str, params: dict) -> None:
    """Save dataset metadata and statistics to a text file."""
    info_path = output_path.replace('.pkl', '_info.txt')
    
    # Extract f_h values from keys
    fh_values = []
    snr_values = []
    for k in iq_dict.keys():
        parts = k.split('_')
        try:
            fh_idx = parts.index('FH') + 1
            snr_idx = parts.index('SNR') + 1
            fh_values.append(float(parts[fh_idx]))
            snr_values.append(int(parts[snr_idx]))
        except (ValueError, IndexError):
            pass
    
    unique_fh = len(set(fh_values))
    unique_snr = sorted(set(snr_values))
    
    with open(info_path, 'w') as f:
        f.write("=" * 80 + "\n")
        f.write(f"DATASET: Case {case.upper()} - 25 MHz\n")
        f.write("=" * 80 + "\n\n")
        
        f.write("GENERATION PARAMETERS:\n")
        f.write(f"  Sampling rate (fs_raw): {params['fs_raw']/1e6:.1f} MHz\n")
        f.write(f"  Capture duration: {params['capture_duration_s']} s\n")
        f.write(f"  Frequency range: {params['f_min']/1e3:.1f} kHz - {params['f_max']/1e3:.1f} kHz\n")
        f.write(f"  Duty cycle: {params['duty_cycle']*100:.0f}%\n")
        f.write(f"  Number of input frames: {params['n_input_frames']:,}\n")
        f.write(f"  SNR levels: {unique_snr}\n")
        f.write(f"  Main seed: {main_seed}\n\n")
        
        f.write("DATASET STATISTICS:\n")
        f.write(f"  Total samples: {len(iq_dict):,}\n")
        f.write(f"  Unique f_h values: {unique_fh}\n")
        f.write(f"  f_h range (actual): {min(fh_values):.1f} Hz - {max(fh_values):.1f} Hz\n")
        f.write(f"  Unique SNR values: {len(unique_snr)}\n")
        f.write(f"  SNR range: {min(unique_snr)} dB - {max(unique_snr)} dB\n\n")
        
        f.write("PROCESSING PIPELINE:\n")
        if case.upper() == 'A':
            f.write("  1. Generate Dirac comb signal at f_h (25 MHz sampling)\n")
            f.write("  2. Add complex Gaussian noise at specified SNR\n")
            f.write("  3. Feature extraction: |x|^2 - mean\n")
            f.write("  4. Welch PSD estimation (nperseg=1000)\n")
            f.write("  5. IFFT + pad/truncate to 1024 samples\n")
            f.write("  6. Normalize to [-1, 1]\n")
        elif case.upper() == 'B':
            f.write("  1. Generate Dirac comb signal at f_h (25 MHz sampling)\n")
            f.write("  2. Add complex Gaussian noise at specified SNR\n")
            f.write("  3. LPF @ 1 MHz (Kaiser window, beta=10)\n")
            f.write("  4. Decimate by 25x (effective fs = 1 MHz)\n")
            f.write("  5. Feature extraction: |x|^2 - mean\n")
            f.write("  6. Welch PSD estimation (nperseg=1000)\n")
            f.write("  7. IFFT + pad/truncate to 1024 samples\n")
            f.write("  8. Normalize to [-1, 1]\n")
        else:  # Case C
            f.write("  1. Generate Dirac comb signal at f_h (25 MHz sampling)\n")
            f.write("  2. Add complex Gaussian noise at specified SNR\n")
            f.write("  3. LPF @ 40 kHz (Kaiser window, beta=10)\n")
            f.write("  4. Decimate by 625x (effective fs = 40 kHz)\n")
            f.write("  5. Feature extraction: |x|^2 - mean\n")
            f.write("  6. Welch PSD estimation (nperseg=1000)\n")
            f.write("  7. IFFT + pad/truncate to 1024 samples\n")
            f.write("  8. Normalize to [-1, 1]\n")
            f.write("  9. Note: CREPE bins saturate above ~2069 Hz\n")
        
        f.write(f"\nOUTPUT:\n")
        f.write(f"  Pickle file: {output_path}\n")
        f.write(f"  Info file: {info_path}\n")
        f.write("=" * 80 + "\n")
    
    print(f"✓ Dataset info saved to {info_path}")

# ...existing code...

if __name__ == "__main__":
    OUTPUT_DIR = './results/'
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # Case A: detect fh in [800 kHz, 1 MHz]
    # params_a = {
    #     'fs_raw': 25e6,
    #     'capture_duration_s': 0.1,
    #     'f_min': 800e3,
    #     'f_max': 1e6,
    #     'n_input_frames': 33333,
    #     'duty_cycle': 0.5
    # }
    # snr_list_a = list(range(-10, 21))
    # dict_a = generate_25mhz_case_dataset(
    #     output_path=os.path.join(OUTPUT_DIR, 'iq_dict_caseA_25MHz(23-3-26).pkl'),
    #     case='A',
    #     snr_list=snr_list_a,
    #     **params_a
    # )
    # save_dataset_info(os.path.join(OUTPUT_DIR, 'iq_dict_caseA_25MHz(23-3-26).pkl'), dict_a, 'A', params_a)

    # Case B: detect fh in [3.2 kHz, 100 kHz]
    params_b = {
        'fs_raw': 25e6,
        'capture_duration_s': 0.01,
        'f_min': 3.2e3,
        'f_max': 100e3,
        'n_input_frames': 33333,
        'duty_cycle': 0.5
    }
    snr_list_b = list(range(-10, 21))
    dict_b = generate_25mhz_case_dataset(
        output_path=os.path.join(OUTPUT_DIR, 'iq_dict_caseB_25MHz(23-3-26).pkl'),
        case='B',
        snr_list=snr_list_b,
        **params_b
    )
    save_dataset_info(os.path.join(OUTPUT_DIR, 'iq_dict_caseB_25MHz(23-3-26).pkl'), dict_b, 'B', params_b)

    # Case C: detect lower fundamentals, strong decimation path
    params_c = {
        'fs_raw': 25e6,
        'capture_duration_s': 0.01,
        'f_min': 32.0,
        'f_max': 4e3,
        'n_input_frames': 33333,
        'duty_cycle': 0.5
    }
    snr_list_c = list(range(-10, 21))
    dict_c = generate_25mhz_case_dataset(
        output_path=os.path.join(OUTPUT_DIR, 'iq_dict_caseC_25MHz(23-3-26).pkl'),
        case='C',
        snr_list=snr_list_c,
        **params_c
    )
    save_dataset_info(os.path.join(OUTPUT_DIR, 'iq_dict_caseC_25MHz(23-3-26).pkl'), dict_c, 'C', params_c)