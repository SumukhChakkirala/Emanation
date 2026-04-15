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
import argparse
from tqdm import tqdm
from scipy import signal
import yaml

try:
    import cupy as cp  # type: ignore[import-not-found]
    from cupyx.scipy import signal as csignal  # type: ignore[import-not-found]
    GPU_AVAILABLE = True
except Exception:
    cp = None
    csignal = None
    GPU_AVAILABLE = False

# Single main seed for reproducible signal parameters (Fh, duty cycle)
main_seed = 1234
rng = np.random.default_rng(seed=main_seed)

# Separate unseeded generator for random noise (different each run)
noise_rng = np.random.default_rng()

_CPU_TAPS_CACHE = {}
_GPU_TAPS_CACHE = {}

# CREPE bin constants
CREPE_N_BINS = 360
CREPE_CENTS_PER_BIN = 20


def freq_to_model_bin(freq_hz: float, fmin_hz: float, fmax_hz: float) -> int:
    """
    Map RF frequency to CREPE bin using case-specific frequency range.
    Uses logarithmic (musical) scale like CREPE paper.
    """
    freq = float(np.clip(freq_hz, fmin_hz, fmax_hz))
    lo = np.log2(float(fmin_hz))
    hi = np.log2(float(fmax_hz))
    pos = (np.log2(freq) - lo) / (hi - lo + 1e-12)
    bin_idx = int(np.clip(pos * (CREPE_N_BINS - 1), 0.0, CREPE_N_BINS - 1.0))
    return bin_idx

# Import from existing code
from generate_crepe_data import (
    CREPE_FS,
    CREPE_FRAME_LENGTH,
    generate_dirac_comb_signal,
    add_complex_noise,
)


def _load_emanation_config(cfg_path: str = 'synapse_emanation_search.yaml') -> dict:
    if os.path.exists(cfg_path):
        with open(cfg_path, 'r') as file:
            return yaml.safe_load(file)
    return {'EmanationDetection': {'kaiser_beta_hh': 10.0, 'numtaps': 129}}


def _case_decimation_params(case: str, fs_raw: float):
    case_u = case.upper()
    if case_u == 'B':
        return 1e6, 25
    if case_u == 'C':
        cutoff_hz = 40e3
        dec_order = int(round(fs_raw / cutoff_hz))
        return cutoff_hz, max(dec_order, 1)
    return None, 1


def _get_cpu_taps(fs_raw: float, cutoff_hz: float, numtaps: int, kaiser_beta_hh: float):
    key = (float(fs_raw), float(cutoff_hz), int(numtaps), float(kaiser_beta_hh))
    taps = _CPU_TAPS_CACHE.get(key)
    if taps is None:
        taps = signal.firwin(
            int(numtaps),
            float(cutoff_hz),
            window=('kaiser', float(kaiser_beta_hh)),
            fs=float(fs_raw)
        ).astype(np.float32)
        _CPU_TAPS_CACHE[key] = taps
    return taps


def _get_gpu_taps(cpu_taps: np.ndarray):
    if not GPU_AVAILABLE:
        return None
    key = (cpu_taps.shape[0], str(cpu_taps.dtype), float(cpu_taps[0]), float(cpu_taps[-1]))
    taps = _GPU_TAPS_CACHE.get(key)
    if taps is None:
        taps = cp.asarray(cpu_taps)
        _GPU_TAPS_CACHE[key] = taps
    return taps


def _apply_case_filter_and_decimate(
    iq_signal: np.ndarray,
    fs_raw: float,
    case: str,
    numtaps: int,
    kaiser_beta_hh: float,
    use_gpu: bool = False,
):
    x = np.asarray(iq_signal, dtype=np.complex64)
    cutoff_hz, dec_order = _case_decimation_params(case, fs_raw)
    if cutoff_hz is None:
        return x, float(fs_raw)

    taps_cpu = _get_cpu_taps(fs_raw, cutoff_hz, numtaps, kaiser_beta_hh)

    if use_gpu and GPU_AVAILABLE:
        x_gpu = cp.asarray(x)
        taps_gpu = _get_gpu_taps(taps_cpu)
        x_dec = csignal.upfirdn(taps_gpu, x_gpu, up=1, down=int(dec_order))
        x_out = cp.asnumpy(x_dec)
    else:
        x_out = signal.upfirdn(taps_cpu, x, up=1, down=int(dec_order))

    return np.asarray(x_out, dtype=np.complex64), float(fs_raw) / float(dec_order)


def _extract_1024_frame_from_25mhz(
    iq_signal: np.ndarray,
    fs_raw: float,
    case: str,
    cfg: dict,
    target_length: int = CREPE_FRAME_LENGTH,
    welch_nperseg: int = 1000,
    use_gpu: bool = False,
) -> np.ndarray:
    """Convert long IQ snapshot to fixed 1024-point real frame for CNN input."""
    _, frame, _ = _extract_psd_and_ifft_from_25mhz(
        iq_signal=iq_signal,
        fs_raw=fs_raw,
        case=case,
        cfg=cfg,
        target_length=target_length,
        welch_nperseg=welch_nperseg,
        use_gpu=use_gpu,
    )
    return frame


def _extract_psd_and_ifft_from_25mhz(
    iq_signal: np.ndarray,
    fs_raw: float,
    case: str,
    cfg: dict,
    target_length: int = CREPE_FRAME_LENGTH,
    welch_nperseg: int = 1000,
    use_gpu: bool = False,
):
    """Convert long IQ snapshot to PSD (1000 bins) and fixed 1024-point IFFT real frame."""
    ed_cfg = cfg.get('EmanationDetection', {}) if cfg is not None else {}
    kaiser_beta_hh = float(ed_cfg.get('kaiser_beta_hh', 10.0))
    numtaps = int(ed_cfg.get('numtaps', 129))

    x, fs_eff = _apply_case_filter_and_decimate(
        iq_signal=iq_signal,
        fs_raw=fs_raw,
        case=case,
        numtaps=numtaps,
        kaiser_beta_hh=kaiser_beta_hh,
        use_gpu=use_gpu,
    )

    x_feature = np.real(np.multiply(x, np.conj(x)))
    x_feature = x_feature - np.mean(x_feature)

    nperseg = min(int(welch_nperseg), len(x_feature))
    if nperseg < 8:
        return np.zeros(1000, dtype=np.float64), np.zeros(target_length, dtype=np.float32), fs_eff

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
    capture_duration_s: float = 0.01,
    cfg_path: str = 'synapse_emanation_search.yaml',
    use_gpu: bool = False,
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
    print(f"  backend: {'GPU(CuPy)' if (use_gpu and GPU_AVAILABLE) else 'CPU(SciPy)'}")

    iq_dict = {}
    for frame_idx in tqdm(range(n_input_frames), desc=f"Case-{case.upper()} frames"):
        f_h = float(fh_values[frame_idx])
        bin_idx = freq_to_model_bin(f_h, f_min, f_max)

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
                welch_nperseg=1000,
                use_gpu=use_gpu,
            )

            key = f"BIN_{bin_idx:03d}_SNR_{snr:+03d}_IDX_{frame_idx:05d}_FH_{f_h:09.1f}_CASE_{case.upper()}"
            iq_dict[key] = frame

    os.makedirs(os.path.dirname(output_path) if os.path.dirname(output_path) else '.', exist_ok=True)
    with open(output_path, 'wb') as file:
        pickle.dump(iq_dict, file)

    print(f"\n[OK] Saved {len(iq_dict):,} samples to {output_path}")
    return iq_dict


def run_gpu_quality_check(
    case: str = 'B',
    trials: int = 3,
    snr_db: float = 20.0,
    fs_raw: float = 25e6,
    cfg_path: str = 'synapse_emanation_search.yaml',
):
    """Quick parity check between CPU and GPU processing paths."""
    if not GPU_AVAILABLE:
        print("GPU quality check skipped: CuPy is not available.")
        return None

    cfg = _load_emanation_config(cfg_path)
    local_rng = np.random.default_rng(2026)
    local_noise_rng = np.random.default_rng(2027)

    if case.upper() == 'A':
        f_min, f_max, duration = 80e3, 1e6, 0.01
    elif case.upper() == 'B':
        f_min, f_max, duration = 3.2e3, 100e3, 0.01
    else:
        f_min, f_max, duration = 32.0, 4e3, 0.01

    mse_list = []
    max_abs_list = []

    for _ in range(int(trials)):
        f_h = float(np.round(local_rng.uniform(f_min, f_max), 1))
        clean_signal = generate_dirac_comb_signal(
            F_h=f_h,
            Fs=fs_raw,
            duration=duration,
            duty_cycle=0.5
        )
        noisy_signal = add_complex_noise(clean_signal, snr_db, local_noise_rng)

        _, frame_cpu, _ = _extract_psd_and_ifft_from_25mhz(
            iq_signal=noisy_signal,
            fs_raw=fs_raw,
            case=case,
            cfg=cfg,
            target_length=CREPE_FRAME_LENGTH,
            welch_nperseg=1000,
            use_gpu=False,
        )
        _, frame_gpu, _ = _extract_psd_and_ifft_from_25mhz(
            iq_signal=noisy_signal,
            fs_raw=fs_raw,
            case=case,
            cfg=cfg,
            target_length=CREPE_FRAME_LENGTH,
            welch_nperseg=1000,
            use_gpu=True,
        )

        diff = frame_cpu.astype(np.float64) - frame_gpu.astype(np.float64)
        mse_list.append(float(np.mean(diff ** 2)))
        max_abs_list.append(float(np.max(np.abs(diff))))

    metrics = {
        'case': case.upper(),
        'trials': int(trials),
        'mse_mean': float(np.mean(mse_list)),
        'max_abs_mean': float(np.mean(max_abs_list)),
        'max_abs_worst': float(np.max(max_abs_list)),
    }

    print("GPU quality check:")
    print(metrics)
    return metrics


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
        bin_idx = freq_to_model_bin(f_h, f_min, f_max)
        
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
    
    print(f"\n[OK] Saved {len(iq_dict):,} samples to {output_path}")
    
    # Statistics - extract f_h values and bins from keys
    fh_values = []
    bin_values = []
    for k in iq_dict.keys():
        # Extract FH value from key
        parts = k.split('_')
        fh_idx = parts.index('FH') + 1
        fh_values.append(float(parts[fh_idx]))
        # Extract BIN value from key
        bin_idx = parts.index('BIN') + 1
        bin_values.append(int(parts[bin_idx]))
    
    unique_fh = len(set(fh_values))
    unique_bins = len(set(bin_values))
    print(f"\nDataset statistics:")
    print(f"  Unique f_h values: {unique_fh}")
    print(f"  f_h range: {min(fh_values):.1f} Hz - {max(fh_values):.1f} Hz")
    print(f"  Unique bins: {unique_bins}")
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
        if 'use_gpu' in params:
            f.write(f"  Backend: {'GPU(CuPy)' if params['use_gpu'] else 'CPU(SciPy)'}\n\n")
        
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
    
    print(f"[OK] Dataset info saved to {info_path}")

# ...existing code...

def _case_defaults(case: str, n_input_frames: int, use_gpu: bool, date_tag: str = ''):
    case_u = case.upper()
    if case_u == 'A':
        params = {
            'fs_raw': 25e6,
            'capture_duration_s': 0.01,
            'f_min': 80e3,
            'f_max': 1e6,
            'n_input_frames': n_input_frames,
            'duty_cycle': 0.5,
            'use_gpu': use_gpu,
        }
        filename = f'iq_dict_caseA_25MHz_{date_tag}.pkl' if date_tag else 'iq_dict_caseA_25MHz(4-4-26).pkl'
    elif case_u == 'B':
        params = {
            'fs_raw': 25e6,
            'capture_duration_s': 0.01,
            'f_min': 3.2e3,
            'f_max': 100e3,
            'n_input_frames': n_input_frames,
            'duty_cycle': 0.5,
            'use_gpu': use_gpu,
        }
        filename = f'iq_dict_caseB_25MHz_{date_tag}.pkl' if date_tag else 'iq_dict_caseB_25MHz(4-4-26).pkl'
    elif case_u == 'C':
        params = {
            'fs_raw': 25e6,
            'capture_duration_s': 0.01,
            'f_min': 32.0,
            'f_max': 4e3,
            'n_input_frames': n_input_frames,
            'duty_cycle': 0.5,
            'use_gpu': use_gpu,
        }
        filename = f'iq_dict_caseC_25MHz_{date_tag}.pkl' if date_tag else 'iq_dict_caseC_25MHz(4-4-26).pkl'
    else:
        raise ValueError(f"Unsupported case: {case}")

    return params, list(range(-10, 21)), filename


def _run_single_case(case: str, output_dir: str, n_input_frames: int, use_gpu: bool, date_tag: str = ''):
    params, snr_list, filename = _case_defaults(case, n_input_frames, use_gpu, date_tag)
    output_path = os.path.join(output_dir, filename)
    iq_dict = generate_25mhz_case_dataset(
        output_path=output_path,
        case=case,
        snr_list=snr_list,
        **params,
    )
    save_dataset_info(output_path, iq_dict, case, params)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Generate 25MHz Case A/B/C datasets')
    parser.add_argument('--case', type=str, default='ALL', choices=['A', 'B', 'C', 'ALL'])
    parser.add_argument('--output_dir', type=str, default='./results/')
    parser.add_argument('--n_input_frames', type=int, default=33333)
    parser.add_argument('--date_tag', type=str, default='')
    parser.add_argument('--backend', type=str, default='auto', choices=['auto', 'cpu', 'gpu'])
    parser.add_argument('--quality_check', action='store_true')
    args = parser.parse_args()

    output_dir = args.output_dir
    os.makedirs(output_dir, exist_ok=True)

    if args.backend == 'gpu':
        if not GPU_AVAILABLE:
            raise RuntimeError('GPU backend requested but CuPy is not available.')
        use_gpu = True
    elif args.backend == 'cpu':
        use_gpu = False
    else:
        use_gpu = GPU_AVAILABLE

    print(f"CuPy available: {GPU_AVAILABLE}")
    print(f"Selected backend: {'GPU(CuPy)' if use_gpu else 'CPU(SciPy)'}")

    if args.quality_check and use_gpu:
        run_gpu_quality_check(case='B', trials=3, snr_db=20.0)
        run_gpu_quality_check(case='C', trials=3, snr_db=20.0)

    if args.case == 'ALL':
        for case_name in ['A', 'B', 'C']:
            _run_single_case(case_name, output_dir, args.n_input_frames, use_gpu, args.date_tag)
    else:
        _run_single_case(args.case, output_dir, args.n_input_frames, use_gpu, args.date_tag)