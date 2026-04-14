"""
Generate CREPE-compatible dataset following the Unified Approach (notes, Apr 2026).

Architecture (per sample):
    raw y[n] at fs_native=25MHz, N=250,000 samples
        -> channel pipeline: ETU70 -> CFO -> phase offset
        -> LPF + decimate by D  ->  ND = int(N/D) samples at fs_eff
        -> zero-pad Nz zeros    ->  ND + Nz  (satisfies (ND+Nz) % Nc == 0)
        -> Welch PSD, nperseg=Nc=1024, noverlap=Nc//2, Ns slices -> 1024-point output
        -> AWGN (per SNR in outer loop)

Four cases from notes with fixed [f1, f2] bounds:
    factor = sqrt(1 + (beta/pi)^2)
    fs_i   = Nc * f1_i / factor    (target effective sample rate for case i)
    D_i    = round(fs_native / fs_i)

    [f1, f2] are fixed by notes; fs_i and D_i are derived from beta.

Zero-padding Nz: smallest Nz in [0, Nc) s.t. (ND+Nz) % Nc == 0
"""

import numpy as np
import pickle
import os
import argparse
from tqdm import tqdm
from scipy import signal
import yaml

try:
    import cupy as cp
    from cupyx.scipy import signal as csignal
    GPU_AVAILABLE = True
except Exception:
    cp = None
    csignal = None
    GPU_AVAILABLE = False

# ─────────────────────────────────────────────────────────────────────────────
# Clock / channel parameters (unchanged from original)
# ─────────────────────────────────────────────────────────────────────────────
f_c       = 1e9
clockrate = 100e6

clockeffects_dict = {}
clockeffects_dict['XOFreq']               = 10e6
clockeffects_dict['XO_standardDeviation'] = 1e-4
clockeffects_dict['XO_maxdeviation']      = 5
clockeffects_dict['LOScalingFactor']      = f_c       / clockeffects_dict['XOFreq']
clockeffects_dict['TimetickScalingFactor']= clockrate / clockeffects_dict['XOFreq']
clockeffects_dict['CFO_standardDeviation']= clockeffects_dict['XO_standardDeviation'] * clockeffects_dict['LOScalingFactor']
clockeffects_dict['SRO_standardDeviation']= clockeffects_dict['XO_standardDeviation'] * clockeffects_dict['TimetickScalingFactor']
clockeffects_dict['CFO_maxdeviation']     = clockeffects_dict['XO_maxdeviation']       * clockeffects_dict['LOScalingFactor']
clockeffects_dict['SRO_maxdeviation']     = clockeffects_dict['XO_maxdeviation']       * clockeffects_dict['TimetickScalingFactor']

# ETU70 fading model parameters (unchanged)
_ETU70_delays_ns = [0, 50, 120, 200, 230, 500, 1600, 2300, 5000]
_ETU70_mags_dB   = [-1, -1, -1,   0,   0,   0,   -3,   -5,   -7]
_ETU70_mags      = [10 ** (m / 20.0) for m in _ETU70_mags_dB]
_ETU70_fD        = 70
_NUM_SINUSOIDS   = 8
_NTAPS           = 8
_LOS             = False
_KFACTOR         = 4

from artifacts import CFOArtifact, phaseOffset

# ─────────────────────────────────────────────────────────────────────────────
# Unified approach — case parameters from notes
# Derived from:  factor = sqrt(1 + (beta/pi)^2)
#                fs_i   = Nc * f1_i / factor
#                D_i    = round(25e6 / fs_i)
# ─────────────────────────────────────────────────────────────────────────────
CREPE_N_BINS        = 360
CREPE_CENTS_PER_BIN = 20
CREPE_FRAME_LENGTH  = 1024          # Nc
FS_NATIVE           = 25e6
CAPTURE_DURATION_S  = 0.01
N_RAW               = int(FS_NATIVE * CAPTURE_DURATION_S)   # 250,000
DUTY_CYCLE          = 0.5
BETA_KAISER         = 10.0
NUMTAPS             = 129
N_HARMONICS         = 5


def _kaiser_factor(beta: float) -> float:
    return float(np.sqrt(1.0 + (float(beta) / np.pi) ** 2))


KAISER_FACTOR = _kaiser_factor(BETA_KAISER)

# Case index -> (f1_hz, f2_hz) fixed by notes
CASE_BOUNDS = {
    1: dict(f1=80e3,  f2=2.5e6),
    2: dict(f1=30.0,  f2=1e3),
    3: dict(f1=1e3,   f2=33e3),
    4: dict(f1=33e3,  f2=100e3),
}


def _build_case_params() -> dict:
    """Build per-case fs and decimation from fixed [f1, f2] and beta."""
    params = {}
    for case_idx, bounds in CASE_BOUNDS.items():
        f1 = float(bounds['f1'])
        f2 = float(bounds['f2'])
        fs_target = (CREPE_FRAME_LENGTH * f1) / KAISER_FACTOR
        D = max(1, int(round(FS_NATIVE / fs_target)))
        fs_eff = FS_NATIVE / D

        params[case_idx] = dict(
            f1=f1,
            f2=f2,
            D=D,
            fs_eff=fs_eff,
            fs_target=fs_target,
            f2_limit=fs_eff / (2.0 * N_HARMONICS),
        )
    return params

CASE_PARAMS = _build_case_params()

def _compute_nz(ND: int, Nc: int = CREPE_FRAME_LENGTH) -> int:
    """Smallest Nz in [0, Nc) such that (ND + Nz) % Nc == 0."""
    return (Nc - ND % Nc) % Nc

# Pre-verify Nz values (sanity check at import time)
_NZ = {}
for _ci, _cp_dict in CASE_PARAMS.items():
    _ND = int(N_RAW / _cp_dict['D'])
    _Nz = _compute_nz(_ND)
    assert _Nz < CREPE_FRAME_LENGTH, f"Case {_ci}: Nz={_Nz} >= Nc"
    _NZ[_ci] = _Nz

# ─────────────────────────────────────────────────────────────────────────────
# Bin mapping
# ─────────────────────────────────────────────────────────────────────────────
def freq_to_model_bin(freq_hz: float, fmin_hz: float, fmax_hz: float) -> int:
    freq = float(np.clip(freq_hz, fmin_hz, fmax_hz))
    lo   = np.log2(float(fmin_hz))
    hi   = np.log2(float(fmax_hz))
    pos  = (np.log2(freq) - lo) / (hi - lo + 1e-12)
    return int(np.clip(round(pos * (CREPE_N_BINS - 1)), 0, CREPE_N_BINS - 1))

# ─────────────────────────────────────────────────────────────────────────────
# Channel pipeline (unchanged from original)
# ─────────────────────────────────────────────────────────────────────────────
def _fractional_delay(signal_in: np.ndarray, delay_samples: float) -> np.ndarray:
    n = signal_in.shape[0]
    idx      = np.arange(n, dtype=np.float64) - float(delay_samples)
    out_real = np.interp(idx, np.arange(n), np.real(signal_in), left=0.0, right=0.0)
    out_imag = np.interp(idx, np.arange(n), np.imag(signal_in), left=0.0, right=0.0)
    return (out_real + 1j * out_imag).astype(np.complex64)

def _etu70_fading_numpy(samples: np.ndarray, samp_rate: float, seed: int) -> np.ndarray:
    x   = np.asarray(samples, dtype=np.complex64)
    n   = x.shape[0]
    t   = np.arange(n, dtype=np.float64) / float(samp_rate)
    delays_samples = [d * 1e-9 * float(samp_rate) for d in _ETU70_delays_ns]
    rng_local = np.random.default_rng(int(seed))
    y   = np.zeros(n, dtype=np.complex64)
    for delay_samp, mag in zip(delays_samples, _ETU70_mags):
        x_delayed = _fractional_delay(x, delay_samp)
        phase0    = rng_local.uniform(0.0, 2.0 * np.pi)
        doppler   = np.exp(1j * (2.0 * np.pi * _ETU70_fD * t + phase0)).astype(np.complex64)
        y        += (mag * x_delayed * doppler).astype(np.complex64)
    return y

def apply_channel_pipeline(
    clean_signal: np.ndarray,
    samp_rate:    float,
    seed:         int,
) -> np.ndarray:
    """clean -> ETU70 fading -> CFO -> phase offset  (AWGN added per-SNR later)."""
    XO_val_len = len(clean_signal) + 10
    np.random.seed(seed)
    ferr_bias_XO = np.random.uniform(
        -clockeffects_dict['XO_maxdeviation'] + clockeffects_dict['XO_standardDeviation'],
         clockeffects_dict['XO_maxdeviation'] - clockeffects_dict['XO_standardDeviation'],
    )
    XO_val    = np.zeros((XO_val_len,))
    XO_val[0] = clockeffects_dict['XO_standardDeviation'] * np.random.randn() + ferr_bias_XO
    while (XO_val[0] >  clockeffects_dict['XO_maxdeviation']) or \
          (XO_val[0] < -clockeffects_dict['XO_maxdeviation']):
        XO_val[0] = clockeffects_dict['XO_standardDeviation'] * np.random.randn() + ferr_bias_XO
    for i in range(1, XO_val_len):
        XO_val[i] = clockeffects_dict['XO_standardDeviation'] * np.random.randn() + XO_val[i - 1]
        while (XO_val[i] >  clockeffects_dict['XO_maxdeviation']) or \
              (XO_val[i] < -clockeffects_dict['XO_maxdeviation']):
            XO_val[i] = clockeffects_dict['XO_standardDeviation'] * np.random.randn() + XO_val[i - 1]

    samples_faded = _etu70_fading_numpy(clean_signal, samp_rate, seed)
    samples_cfo   = CFOArtifact(samples_faded, XO_val, clockeffects_dict, samp_rate)
    samples_phase = phaseOffset(samples_cfo, seed)
    return np.asarray(samples_phase, dtype=np.complex64)

# ─────────────────────────────────────────────────────────────────────────────
# FIR filter tap cache
# ─────────────────────────────────────────────────────────────────────────────
_CPU_TAPS_CACHE = {}
_GPU_TAPS_CACHE = {}

def _get_cpu_taps(fs_raw: float, cutoff_hz: float) -> np.ndarray:
    key  = (float(fs_raw), float(cutoff_hz))
    taps = _CPU_TAPS_CACHE.get(key)
    if taps is None:
        taps = signal.firwin(
            NUMTAPS, float(cutoff_hz),
            window=('kaiser', BETA_KAISER), fs=float(fs_raw),
        ).astype(np.float32)
        _CPU_TAPS_CACHE[key] = taps
    return taps

def _get_gpu_taps(cpu_taps: np.ndarray):
    if not GPU_AVAILABLE:
        return None
    key  = id(cpu_taps)
    taps = _GPU_TAPS_CACHE.get(key)
    if taps is None:
        taps = cp.asarray(cpu_taps)
        _GPU_TAPS_CACHE[key] = taps
    return taps

# ─────────────────────────────────────────────────────────────────────────────
# Decimation
# ─────────────────────────────────────────────────────────────────────────────
def _decimate(x: np.ndarray, fs_raw: float, D: int, use_gpu: bool = False) -> np.ndarray:
    """
    LPF at fs_raw/(2D) then downsample by D.
    Case 1 (D=1): returns x unchanged.
    """
    if D == 1:
        return np.asarray(x, dtype=np.complex64)

    cutoff_hz = fs_raw / (2.0 * D)
    taps_cpu  = _get_cpu_taps(fs_raw, cutoff_hz)

    if use_gpu and GPU_AVAILABLE:
        x_gpu  = cp.asarray(x)
        out    = csignal.upfirdn(_get_gpu_taps(taps_cpu), x_gpu, up=1, down=D)
        return cp.asnumpy(out).astype(np.complex64)
    else:
        return signal.upfirdn(taps_cpu, x, up=1, down=D).astype(np.complex64)

# ─────────────────────────────────────────────────────────────────────────────
# Feature extraction: Welch PSD -> 1024-point frame
#
# Pipeline (Image 1 + Image 2 notes):
#   1. Decimate noisy IQ by D  ->  x_dec  (length ND = int(N_RAW/D))
#   2. Feature: s[n] = |x_dec[n]|^2 - mean   (real-valued, DC removed)
#   3. Zero-pad Nz zeros at the end  ->  length ND+Nz
#   4. Welch PSD: nperseg=Nc=1024, noverlap=Nc//2, Kaiser window beta=10
#      Ns = (ND+Nz - noverlap) // (Nc - noverlap)  segments (scipy handles this)
#      Output: 1024-point one-sided PSD at fs_eff
#   5. Normalize to [0,1]
# ─────────────────────────────────────────────────────────────────────────────
def extract_welch_frame(
    noisy_signal: np.ndarray,
    case_idx:     int,
    use_gpu:      bool = False,
) -> np.ndarray:
    cp_dict  = CASE_PARAMS[case_idx]
    D        = cp_dict['D']
    fs_eff   = cp_dict['fs_eff']
    Nc       = CREPE_FRAME_LENGTH      # 1024
    noverlap = Nc // 2                 # 50% overlap (Welch with overlap, per notes)

    # Step 1 — decimate
    x_dec = _decimate(noisy_signal, FS_NATIVE, D, use_gpu=use_gpu)

    ND = len(x_dec)

    # Step 2 — squared magnitude, remove DC
    s = np.real(x_dec * np.conj(x_dec)).astype(np.float64)
    s -= s.mean()

    # Step 3 — zero-pad so (ND+Nz) % Nc == 0  (for non-overlapping alignment)
    Nz    = _compute_nz(ND, Nc)
    s_pad = np.concatenate([s, np.zeros(Nz, dtype=np.float64)])

    # Step 4 — Welch PSD
    nperseg = min(Nc, len(s_pad))
    if nperseg < 4:
        return np.zeros(Nc, dtype=np.float32)

    win = signal.windows.kaiser(nperseg, beta=BETA_KAISER)
    _noverlap = min(noverlap, nperseg - 1)

    _, psd = signal.welch(
        s_pad,
        fs=fs_eff,
        window=win,
        nperseg=nperseg,
        noverlap=_noverlap,
        nfft=Nc,                    # output exactly Nc frequency bins
        return_onesided=True,       # one-sided; Nc//2+1 bins
        detrend=False,
        scaling='density',
    )

    # psd has Nc//2+1 = 513 bins; interpolate to Nc=1024 for CREPE compatibility
    psd = np.asarray(psd, dtype=np.float64)
    psd[~np.isfinite(psd)] = 0.0
    psd = np.maximum(psd, 1e-20)

    src = np.linspace(0.0, 1.0, len(psd), endpoint=True)
    dst = np.linspace(0.0, 1.0, Nc, endpoint=True)
    frame = np.interp(dst, src, psd).astype(np.float32)

    # Step 5 — normalize to [0, 1]
    mx = frame.max()
    if mx > 1e-12:
        frame /= mx

    return frame

# ─────────────────────────────────────────────────────────────────────────────
# Signal / noise generators  (unchanged interface)
# ─────────────────────────────────────────────────────────────────────────────
from generate_crepe_data import (
    generate_dirac_comb_signal,
    add_complex_noise,
)

# ─────────────────────────────────────────────────────────────────────────────
# Main dataset generator
# ─────────────────────────────────────────────────────────────────────────────
main_seed = 1234
rng       = np.random.default_rng(seed=main_seed)

def generate_unified_case_dataset(
    output_path:    str,
    case_idx:       int,
    snr_list:       list  = None,
    n_input_frames: int   = 10000,
    use_gpu:        bool  = False,
) -> dict:
    """
    Generate dataset for one case of the unified approach.

    Args:
        output_path:    Path for output .pkl file.
        case_idx:       1, 2, 3, or 4 (matches notes).
        snr_list:       List of SNR values in dB.
        n_input_frames: Number of unique f_h draws (random, uniform in [f1,f2]).
        use_gpu:        Use CuPy for filtering if available.

    Returns:
        dict mapping key -> 1024-point float32 Welch PSD frame.
    """
    if snr_list is None:
        snr_list = list(range(-10, 21))

    cp_dict = CASE_PARAMS[case_idx]
    f1, f2  = cp_dict['f1'], cp_dict['f2']
    D       = cp_dict['D']
    fs_eff  = cp_dict['fs_eff']
    ND      = int(N_RAW / D)
    Nz      = _compute_nz(ND)
    Ns_welch_est = (ND + Nz) // CREPE_FRAME_LENGTH   # non-overlap slices (informational)

    total_samples = n_input_frames * len(snr_list)

    # Continuous f_h values, uniform in [f1, f2], 1 decimal place
    fh_values = np.round(rng.uniform(f1, f2, n_input_frames), 1)

    print("=" * 80)
    print(f"Generating Unified Case {case_idx} dataset")
    print("=" * 80)
    print(f"  [f1, f2]:          [{f1:.1f}, {f2:.1f}] Hz")
    print(f"  factor:            {KAISER_FACTOR:.6f} (=sqrt(1+(beta/pi)^2), beta={BETA_KAISER})")
    print(f"  fs_target:         {cp_dict['fs_target']:.1f} Hz  (= Nc*f1/factor)")
    print(f"  fs_native:         {FS_NATIVE/1e6:.1f} MHz")
    print(f"  Decimation D:      {D}")
    print(f"  fs_eff:            {fs_eff:.1f} Hz")
    print(f"  f2 upper limit:    {cp_dict['f2_limit']:.1f} Hz  (= fs_eff/(2*{N_HARMONICS}))")
    print(f"  N_raw:             {N_RAW:,}  (= fs_native × {CAPTURE_DURATION_S} s)")
    print(f"  ND = int(N/D):     {ND}")
    print(f"  Nz (zero-pad):     {Nz}   [(ND+Nz)%Nc = {(ND+Nz)%CREPE_FRAME_LENGTH}]")
    print(f"  Welch segments:    {Ns_welch_est} (non-overlap) / ~{2*Ns_welch_est-1} (50% overlap)")
    print(f"  SNR levels:        {snr_list}")
    print(f"  input frames:      {n_input_frames:,}")
    print(f"  total samples:     {total_samples:,}")
    print(f"  channel pipeline:  ETU70 -> CFO -> phase offset -> AWGN")
    print(f"  feature:           |x|^2 - mean -> Welch PSD (nperseg=1024, 50% overlap)")
    print(f"  backend:           {'GPU(CuPy)' if (use_gpu and GPU_AVAILABLE) else 'CPU(SciPy)'}")
    print()

    iq_dict = {}
    seed    = 1680000

    for frame_idx in tqdm(range(n_input_frames), desc=f"Case-{case_idx} frames"):
        f_h     = float(fh_values[frame_idx])
        bin_idx = freq_to_model_bin(f_h, f1, f2)

        # Generate clean signal at native 25 MHz
        clean_signal = generate_dirac_comb_signal(
            F_h=f_h, Fs=FS_NATIVE,
            duration=CAPTURE_DURATION_S, duty_cycle=DUTY_CYCLE,
        )

        # Apply channel artifacts once per frame (expensive)
        signal_no_awgn = apply_channel_pipeline(
            clean_signal=clean_signal,
            samp_rate=FS_NATIVE,
            seed=seed,
        )

        for snr in snr_list:
            noisy_signal = add_complex_noise(
                signal_no_awgn, snr, np.random.default_rng(seed)
            )

            frame = extract_welch_frame(
                noisy_signal=noisy_signal,
                case_idx=case_idx,
                use_gpu=use_gpu,
            )

            key = (
                f"BIN_{bin_idx:03d}_SNR_{snr:+03d}_IDX_{frame_idx:05d}"
                f"_FH_{f_h:09.1f}_CASE_{case_idx}"
            )
            iq_dict[key] = frame
            seed += 1

    os.makedirs(
        os.path.dirname(output_path) if os.path.dirname(output_path) else '.',
        exist_ok=True,
    )
    with open(output_path, 'wb') as fh:
        pickle.dump(iq_dict, fh)

    print(f"\n[OK] Saved {len(iq_dict):,} samples -> {output_path}")
    return iq_dict

# ─────────────────────────────────────────────────────────────────────────────
# Metadata helper
# ─────────────────────────────────────────────────────────────────────────────
def save_dataset_info(output_path: str, iq_dict: dict, case_idx: int) -> None:
    cp_dict  = CASE_PARAMS[case_idx]
    info_path = output_path.replace('.pkl', '_info.txt')
    snr_values = []
    for k in iq_dict:
        parts = k.split('_')
        try:
            snr_values.append(int(parts[parts.index('SNR') + 1]))
        except (ValueError, IndexError):
            pass

    ND = int(N_RAW / cp_dict['D'])
    Nz = _compute_nz(ND)

    with open(info_path, 'w') as fh:
        fh.write("=" * 80 + "\n")
        fh.write(f"DATASET: Unified Case {case_idx}\n")
        fh.write("=" * 80 + "\n\n")
        fh.write("CASE PARAMETERS (from notes, beta=10, Nc=1024):\n")
        fh.write(f"  [f1, f2]       = [{cp_dict['f1']:.1f}, {cp_dict['f2']:.1f}] Hz\n")
        fh.write(f"  factor         = sqrt(1 + (beta/pi)^2) = {KAISER_FACTOR:.6f}\n")
        fh.write(f"  fs_native      = {FS_NATIVE/1e6:.1f} MHz\n")
        fh.write(f"  fs_i formula   = Nc * f1 / factor = {cp_dict['fs_target']:.1f} Hz\n")
        fh.write(f"  D              = {cp_dict['D']}\n")
        fh.write(f"  fs_eff         = {cp_dict['fs_eff']:.1f} Hz\n")
        fh.write(f"  f2 <= fs/(2Nh) = {cp_dict['f2_limit']:.1f} Hz  (Nh={N_HARMONICS})\n")
        fh.write(f"  N_raw          = {N_RAW}\n")
        fh.write(f"  ND = int(N/D)  = {ND}\n")
        fh.write(f"  Nz             = {Nz}  [(ND+Nz) % Nc = {(ND+Nz) % CREPE_FRAME_LENGTH}]\n\n")
        fh.write("CHANNEL PIPELINE:\n")
        fh.write("  clean signal -> ETU70 fading -> CFO -> phase offset -> AWGN\n\n")
        fh.write("FEATURE PIPELINE:\n")
        fh.write(f"  1. Decimate by D={cp_dict['D']} (LPF cutoff={FS_NATIVE/(2*cp_dict['D']):.1f} Hz, Kaiser beta=10)\n")
        fh.write(f"  2. s[n] = |x_dec[n]|^2 - mean(|x_dec|^2)  [DC removed]\n")
        fh.write(f"  3. Zero-pad {Nz} zeros  [total length = {ND+Nz}]\n")
        fh.write(f"  4. Welch PSD (nperseg=1024, noverlap=512, Kaiser beta=10, nfft=1024)\n")
        fh.write(f"  5. Interpolate to 1024 bins\n")
        fh.write(f"  6. Normalize to [0, 1]\n\n")
        fh.write(f"SNR levels: {sorted(set(snr_values))}\n")
        fh.write(f"Total samples: {len(iq_dict):,}\n")
        fh.write(f"Output pickle: {output_path}\n")
        fh.write("=" * 80 + "\n")

    print(f"[OK] Info saved -> {info_path}")

# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────
def _run_single_case(case_idx: int, output_dir: str, n_input_frames: int,
                     use_gpu: bool, date_tag: str = '') -> None:
    tag      = f'_{date_tag}' if date_tag else '_unified'
    filename = f'iq_dict_case{case_idx}{tag}.pkl'
    out_path = os.path.join(output_dir, filename)
    iq_dict  = generate_unified_case_dataset(
        output_path=out_path,
        case_idx=case_idx,
        n_input_frames=n_input_frames,
        use_gpu=use_gpu,
    )
    save_dataset_info(out_path, iq_dict, case_idx)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Generate Unified Case 1-4 datasets')
    parser.add_argument('--case',           type=str, default='ALL',
                        choices=['1', '2', '3', '4', 'ALL'])
    parser.add_argument('--output_dir',     type=str, default='./results/')
    parser.add_argument('--n_input_frames', type=int, default=33333)
    parser.add_argument('--date_tag',       type=str, default='')
    parser.add_argument('--backend',        type=str, default='auto',
                        choices=['auto', 'cpu', 'gpu'])
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    if args.backend == 'gpu':
        use_gpu = GPU_AVAILABLE
        if not GPU_AVAILABLE:
            print('Warning: GPU requested but CuPy unavailable. Falling back to CPU.')
    elif args.backend == 'cpu':
        use_gpu = False
    else:
        use_gpu = GPU_AVAILABLE

    print(f"CuPy available : {GPU_AVAILABLE}")
    print(f"Backend        : {'GPU(CuPy)' if use_gpu else 'CPU(SciPy)'}")

    cases_to_run = [1, 2, 3, 4] if args.case == 'ALL' else [int(args.case)]
    for c in cases_to_run:
        _run_single_case(c, args.output_dir, args.n_input_frames, use_gpu, args.date_tag)
