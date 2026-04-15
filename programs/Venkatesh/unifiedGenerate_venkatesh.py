"""
Generate CREPE-compatible dataset following the Unified Approach (notes, Apr 2026).

TWO MODES (select via --no_decimation flag):

  DEFAULT - decimated mode:
    raw y[n] at fs_native=25MHz, N=250,000 samples
        -> channel pipeline: ETU70 -> CFO -> phase offset
        -> LPF + decimate by D  ->  ND = int(N/D) samples at fs_eff
        -> zero-pad Nz zeros    ->  ND + Nz  (satisfies (ND+Nz) % Nc == 0)
        -> Welch PSD, nperseg=Nc=1024, noverlap=Nc//2
           (ND+Nz)/Nc non-overlapping slices -> 1024-point output
        -> AWGN (per SNR in outer loop)

  OR - no-decimation mode  (--no_decimation):
    raw y[n] at fs_native=25MHz, N=250,000 samples
        -> channel pipeline: ETU70 -> CFO -> phase offset
        -> NO filtering / decimation  (D=1 forced, fs_eff=fs_native=25 MHz)
        -> zero-pad Nz=880 zeros  ->  N+Nz=250,880  [(N+Nz)%Nc=0]
        -> Welch PSD, nperseg=Nc=1024, noverlap=Nc//2
           (N+Nz)/Nc = 245 non-overlapping Welch averaging slices -> 1024-point output
        -> AWGN (per SNR in outer loop)
    f_h range still taken from the chosen case (1-4) so bin mapping stays consistent.

Four cases, hardcoded from notes (beta=10, Nc=1024, constant=3.2 approx):
    fs_i  = Nc * f1_i / 3.2       (target effective sample rate for case i)
    D_i   = round(fs_native / fs_i)

    Case 1:  [f1, f2] = [80 kHz,  2.5 MHz],  D=1,    fs_eff=25.000 MHz
    Case 2:  [f1, f2] = [30 Hz,   1 kHz],    D=2604, fs_eff=9600.6 Hz
    Case 3:  [f1, f2] = [1 kHz,   33 kHz],   D=78,   fs_eff=320.5 kHz
    Case 4:  [f1, f2] = [33 kHz,  100 kHz],  D=2,    fs_eff=12.500 MHz

Zero-padding Nz: smallest Nz in [0, Nc) s.t. (ND+Nz) % Nc == 0

CLI examples:
    # Decimated mode, all cases
    python generate_unified_dataset.py --case ALL

    # No-decimation OR mode, Case 3 f_h range
    python generate_unified_dataset.py --case 3 --no_decimation

    # No-decimation, all cases, GPU backend
    python generate_unified_dataset.py --case ALL --no_decimation --backend gpu
"""

import numpy as np
import pickle
import os
import argparse
from tqdm import tqdm
from scipy import signal

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
_LOS             = False
_KFACTOR         = 4

from artifacts import CFOArtifact, phaseOffset

# ─────────────────────────────────────────────────────────────────────────────
# Global constants
# ─────────────────────────────────────────────────────────────────────────────
CREPE_N_BINS       = 360
CREPE_FRAME_LENGTH = 1024          # Nc
FS_NATIVE          = 25e6
CAPTURE_DURATION_S = 0.01
N_RAW              = int(FS_NATIVE * CAPTURE_DURATION_S)   # 250,000
DUTY_CYCLE         = 0.5
BETA_KAISER        = 10.0
NUMTAPS            = 129

# ─────────────────────────────────────────────────────────────────────────────
# Case parameters — f1, f2 hardcoded from notes; D, fs_eff calculated
# fs_i = Nc * f1_i / KAISER_CONSTANT  (beta=10),  D = round(fs_native / fs_i)
# ─────────────────────────────────────────────────────────────────────────────
KAISER_CONSTANT = np.sqrt(1+(BETA_KAISER/np.pi)**2)#3.336  # Empirical constant for beta=10 Kaiser window

# Req: f_h >= f1_i == BETA_KAISER*fs_i/CREPE_FRAME_LENGTH  ->  fs_i <= CREPE_FRAME_LENGTH * f_h / BETA_KAISER 
def _compute_case_params(f1: float, f2: float) -> dict:
    """Compute D and fs_eff from f1 using Kaiser constant."""
    fs_i = CREPE_FRAME_LENGTH * f1 / KAISER_CONSTANT
    D = round(FS_NATIVE / fs_i)
    #fs_eff = FS_NATIVE / D ## This is incorrect, if round function input is say  higher than 0.5+integer, then 
    ## we could have a situation where the F_h value is lower than resolution and performance degrades.
    # THerefore return fs_i instead of fs_eff.
    return dict(f1=f1, f2=f2, D=D, fs_eff=fs_i)

CASE_PARAMS = {
    1: _compute_case_params(80e3,   2.5e6),
    2: _compute_case_params(30.0,   1e3),
    3: _compute_case_params(1e3,    33e3),
    4: _compute_case_params(33e3,   100e3),
}


def _compute_nz(ND: int, Nc: int = CREPE_FRAME_LENGTH) -> int:
    """Smallest Nz in [0, Nc) such that (ND + Nz) % Nc == 0."""
    return (Nc - ND % Nc) % Nc


# Sanity-check all decimated-mode Nz at import time
_NZ = {}
for _ci, _cpd in CASE_PARAMS.items():
    _ND = int(N_RAW / _cpd['D'])
    _Nz = _compute_nz(_ND)
    assert _Nz < CREPE_FRAME_LENGTH, f"Case {_ci}: Nz={_Nz} >= Nc"
    _NZ[_ci] = _Nz

# ─────────────────────────────────────────────────────────────────────────────
# No-decimation (OR) mode — pre-computed constants
#
# Notes: "OR: if no decimation -> N to Nc, (N+Nz)%Nc=0,
#         (N+Nz)/Nc = number of slices for Welch PSD averaging"
#
# D=1 forced, fs_eff=25 MHz, full N_RAW=250,000 samples used directly.
# Nz=880  ->  total=250,880  ->  (N+Nz)/Nc = 245 non-overlapping slices.
# ─────────────────────────────────────────────────────────────────────────────
_NO_DEC_ND    = N_RAW                        # 250,000
_NO_DEC_NZ    = _compute_nz(_NO_DEC_ND)     # 880
_NO_DEC_TOTAL = _NO_DEC_ND + _NO_DEC_NZ     # 250,880
_NO_DEC_NS    = _NO_DEC_TOTAL // CREPE_FRAME_LENGTH   # 245 slices (no overlap)
assert _NO_DEC_TOTAL % CREPE_FRAME_LENGTH == 0, "No-dec zero-pad sanity check failed"

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
# Channel pipeline
# ─────────────────────────────────────────────────────────────────────────────
def _fractional_delay(signal_in: np.ndarray, delay_samples: float) -> np.ndarray:
    n        = signal_in.shape[0]
    idx      = np.arange(n, dtype=np.float64) - float(delay_samples)
    out_real = np.interp(idx, np.arange(n), np.real(signal_in), left=0.0, right=0.0)
    out_imag = np.interp(idx, np.arange(n), np.imag(signal_in), left=0.0, right=0.0)
    return (out_real + 1j * out_imag).astype(np.complex64)


def _etu70_fading_numpy(samples: np.ndarray, samp_rate: float, rng=None, seed=None) -> np.ndarray:
    x              = np.asarray(samples, dtype=np.complex64)
    n              = x.shape[0]
    t              = np.arange(n, dtype=np.float64) / float(samp_rate)
    delays_samples = [d * 1e-9 * float(samp_rate) for d in _ETU70_delays_ns]
    if rng is None:
        rng = np.random.default_rng(int(seed) if seed is not None else None)
    y              = np.zeros(n, dtype=np.complex64)
    for delay_samp, mag in zip(delays_samples, _ETU70_mags):
        x_delayed = _fractional_delay(x, delay_samp)
        phase0    = rng.uniform(0.0, 2.0 * np.pi)
        doppler   = np.exp(1j * (2.0 * np.pi * _ETU70_fD * t + phase0)).astype(np.complex64)
        y        += (mag * x_delayed * doppler).astype(np.complex64)
    return y


def apply_channel_pipeline(
    clean_signal: np.ndarray,
    samp_rate:    float,
    rng:          np.random.Generator = None,
    seed:         int = None,
) -> np.ndarray:
    """clean -> ETU70 -> CFO -> phase offset.  AWGN is added per-SNR later."""
    XO_val_len   = len(clean_signal) + 10
    if rng is None:
        rng = np.random.default_rng(int(seed) if seed is not None else None)

    ferr_bias_XO = rng.uniform(
        -clockeffects_dict['XO_maxdeviation'] + clockeffects_dict['XO_standardDeviation'],
         clockeffects_dict['XO_maxdeviation'] - clockeffects_dict['XO_standardDeviation'],
    )
    XO_val    = np.zeros((XO_val_len,))
    XO_val[0] = clockeffects_dict['XO_standardDeviation'] * rng.standard_normal() + ferr_bias_XO
    while (XO_val[0] >  clockeffects_dict['XO_maxdeviation']) or \
          (XO_val[0] < -clockeffects_dict['XO_maxdeviation']):
        XO_val[0] = clockeffects_dict['XO_standardDeviation'] * rng.standard_normal() + ferr_bias_XO
    for i in range(1, XO_val_len):
        XO_val[i] = clockeffects_dict['XO_standardDeviation'] * rng.standard_normal() + XO_val[i - 1]
        while (XO_val[i] >  clockeffects_dict['XO_maxdeviation']) or \
              (XO_val[i] < -clockeffects_dict['XO_maxdeviation']):
            XO_val[i] = clockeffects_dict['XO_standardDeviation'] * rng.standard_normal() + XO_val[i - 1]

    samples_faded = _etu70_fading_numpy(clean_signal, samp_rate, rng=rng)
    samples_cfo   = CFOArtifact(samples_faded, XO_val, clockeffects_dict, samp_rate)
    samples_phase = phaseOffset(samples_cfo, rng=rng)
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
def _decimate(x: np.ndarray, fs_raw: float, D: int, use_gpu: bool = False, fs_eff: float = None) -> np.ndarray:
    """LPF at fs_raw/(2D) + downsample by D.  D=1 returns x unchanged."""
    if D == 1:
        return np.asarray(x, dtype=np.complex64)
    cutoff_hz = fs_eff/2 if fs_eff is not None else fs_raw / (2.0 * D)
    ## changed above line online with previous comment on fs_eff compuation in caseparams
    taps_cpu  = _get_cpu_taps(fs_raw, cutoff_hz)
    if use_gpu and GPU_AVAILABLE:
        out = csignal.upfirdn(_get_gpu_taps(taps_cpu), cp.asarray(x), up=1, down=D)
        return cp.asnumpy(out).astype(np.complex64)
    return signal.upfirdn(taps_cpu, x, up=1, down=D).astype(np.complex64)

# ─────────────────────────────────────────────────────────────────────────────
# Core Welch feature extraction (shared by both modes)
#
# Steps:
#   1. s[n] = |x_in[n]|^2 - mean   (DC removed)
#   2. zero-pad Nz zeros -> (ND+Nz), divisible by Nc
#   3. Welch PSD: nperseg=1024, noverlap=512, Kaiser beta=10, nfft=1024
#      -> 513 one-sided bins
#   4. Interpolate 513 -> 1024 bins
#   5. Normalize to [0, 1]
# ─────────────────────────────────────────────────────────────────────────────
def _welch_1024(x_in: np.ndarray, fs_eff: float, Nz: int) -> np.ndarray:
    """
    Compute 1024-point Welch PSD feature from a complex IQ array.

    Args:
        x_in:   Complex IQ (already decimated, or raw for no-dec mode).
        fs_eff: Effective sample rate of x_in (Hz).
        Nz:     Zero-padding length (pre-computed so (len(x_in)+Nz) % Nc == 0).

    Returns:
        1024-point float32 PSD frame normalized to [0, 1].
    """
    Nc       = CREPE_FRAME_LENGTH   # 1024
    noverlap = Nc // 2              # 512  (50% overlap, per notes)

    # Step 1
    s  = np.real(x_in * np.conj(x_in)).astype(np.float64) # feature extraction
    s -= s.mean() # DC removal

    # Step 2
    s_pad = np.concatenate([s, np.zeros(Nz, dtype=np.float64)]) # zero padding

    # Step 3
    nperseg  = min(Nc, len(s_pad))
    if nperseg < 4:
        return np.zeros(Nc, dtype=np.float32)
    _noverlap = min(noverlap, nperseg - 1)

    _, psd = signal.welch(
        s_pad,
        fs=fs_eff,
        window=signal.windows.kaiser(nperseg, beta=BETA_KAISER),
        nperseg=nperseg,
        noverlap=_noverlap,
        nfft=Nc,               # -> 513 one-sided bins
        return_onesided=False,
        detrend=False,
        scaling='density',
    )
    # We need two sided PSD, that is how we have been using it to train.
    # In future we can potentially generate 2048 samples and then use only half of it for training.
    # this is because the signal is real and we can get the same information from one sided PSD, 
  

    # Step 4
    psd = np.asarray(psd, dtype=np.float64)
    psd[~np.isfinite(psd)] = 0.0
    psd   = np.maximum(psd, 1e-20)
    # src   = np.linspace(0.0, 1.0, len(psd), endpoint=True)
    # dst   = np.linspace(0.0, 1.0, Nc, endpoint=True)
    frame = psd#np.interp(dst, src, psd).astype(np.float32)
    # since we are taking two sided PSD we do not need the interpolation
    # Step 5
    # we are doing normalization here to keep range of values within 0 to 1 for the values in a frame.
    mx = frame.max()
    if mx > 1e-12:
        frame /= mx
    return frame


def extract_welch_frame(
    noisy_signal:  np.ndarray,
    case_idx:      int,
    no_decimation: bool = False,
    use_gpu:       bool = False,
) -> np.ndarray:
    """
    Full feature extraction for one noisy IQ capture.

    Args:
        noisy_signal:  Raw 25 MHz complex IQ, length N_RAW=250,000.
        case_idx:      Case 1-4 (determines D and fs_eff in decimated mode).
        no_decimation: If True, skip LPF/decimation (OR branch from notes).
        use_gpu:       Use CuPy for decimation if available.
        fs_eff:        Effective sample rate (used if fs_eff is None).
    Returns:
        1024-point float32 normalized Welch PSD frame.
    """
    if no_decimation:
        # OR branch: no filtering, full N_RAW samples at native 25 MHz
        x_in   = np.asarray(noisy_signal, dtype=np.complex64)
        fs_eff = FS_NATIVE
        Nz     = _NO_DEC_NZ    # 880, pre-computed
    else:
        # Default: LPF + decimate
        cpd    = CASE_PARAMS[case_idx]
        fs_eff = cpd['fs_eff']
        x_in   = _decimate(noisy_signal, FS_NATIVE, cpd['D'], use_gpu=use_gpu, fs_eff=fs_eff)
        
        Nz     = _compute_nz(len(x_in))

    return _welch_1024(x_in, fs_eff, Nz)

# ─────────────────────────────────────────────────────────────────────────────
# Signal / noise helpers (unchanged interface)
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
    snr_list:       list = None,
    n_input_frames: int  = 10000,
    no_decimation:  bool = False,
    use_gpu:        bool = False,
) -> dict:
    """
    Generate dataset for one case of the unified approach.

    Args:
        output_path:    .pkl output path.
        case_idx:       1-4 (defines f_h range).
        snr_list:       SNR values in dB.
        n_input_frames: Number of unique f_h draws.
        no_decimation:  If True, OR mode — no LPF/decimation.
        use_gpu:        Use CuPy for decimation filtering.

    Returns:
        dict: key -> 1024-point float32 Welch PSD frame.
    """
    if snr_list is None:
        snr_list = list(range(-10, 21))

    cpd    = CASE_PARAMS[case_idx]
    f1, f2 = cpd['f1'], cpd['f2']

    if no_decimation:
        D          = 1
        fs_eff     = FS_NATIVE
        ND         = _NO_DEC_ND   # 250,000
        Nz         = _NO_DEC_NZ   # 880
        Ns_no_ovlp = _NO_DEC_NS   # 245
        mode_label = "no-decimation (OR)"
        mode_tag   = "NODEC"
    else:
        D          = cpd['D']
        fs_eff     = cpd['fs_eff']
        ND         = int(N_RAW / D)
        Nz         = _compute_nz(ND)
        Ns_no_ovlp = (ND + Nz) / CREPE_FRAME_LENGTH ##since we already padded zeros to make overal number multiple of 1024,
        ##we can directly compute the number without the // which does a floor division. 
        mode_label = "withdecimation"
        mode_tag   = "WITHDEC"

    Ns_50ovlp     = max(1, 2 * Ns_no_ovlp - 1)
    total_samples = n_input_frames * len(snr_list)

    fh_values = np.round(rng.uniform(f1, f2, n_input_frames), 1)

    print("=" * 80)
    print(f"Generating Unified Case {case_idx} dataset  [{mode_label}]")
    print("=" * 80)
    print(f"  [f1, f2]:          [{f1:.1f}, {f2:.1f}] Hz")
    print(f"  fs_native:         {FS_NATIVE/1e6:.1f} MHz")
    print(f"  Decimation D:      {D}{'  (no filtering)' if no_decimation else ''}")
    print(f"  fs_eff:            {fs_eff:.1f} Hz")
    print(f"  N_raw:             {N_RAW:,}  (= fs_native x {CAPTURE_DURATION_S} s)")
    print(f"  ND (after dec):    {ND:,}")
    print(f"  Nz (zero-pad):     {Nz}   [(ND+Nz)%Nc = {(ND+Nz) % CREPE_FRAME_LENGTH}]")
    print(f"  (ND+Nz)/Nc:        {Ns_no_ovlp}  non-overlapping Welch slices")
    print(f"  Welch ~slices:     ~{Ns_50ovlp}  at 50% overlap")
    print(f"  SNR levels:        {snr_list}")
    print(f"  input frames:      {n_input_frames:,}")
    print(f"  total samples:     {total_samples:,}")
    print(f"  channel pipeline:  ETU70 -> CFO -> phase offset -> AWGN")
    print(f"  feature:           |x|^2 - mean -> Welch PSD (nperseg=1024, 50% overlap)")
    print(f"  backend:           {'GPU(CuPy)' if (use_gpu and GPU_AVAILABLE) else 'CPU(SciPy)'}")
    print()

    iq_dict      = {}
    frame_seeds  = rng.integers(0, 2**32, size=n_input_frames, dtype=np.uint32)

    for frame_idx in tqdm(range(n_input_frames), desc=f"Case-{case_idx} [{mode_tag}] frames"):
        f_h     = float(fh_values[frame_idx])
        bin_idx = freq_to_model_bin(f_h, f1, f2)
        frame_rng = np.random.default_rng(int(frame_seeds[frame_idx]))

        clean_signal = generate_dirac_comb_signal(
            F_h=f_h, Fs=FS_NATIVE,
            duration=CAPTURE_DURATION_S, duty_cycle=DUTY_CYCLE,
        )

        signal_no_awgn = apply_channel_pipeline(
            clean_signal=clean_signal,
            samp_rate=FS_NATIVE,
            rng=frame_rng,
        )

        for snr in snr_list:
            noise_rng = np.random.default_rng(int(frame_rng.integers(0, 2**32, dtype=np.uint32)))
            noisy_signal = add_complex_noise(signal_no_awgn, snr, noise_rng)

            frame = extract_welch_frame(
                noisy_signal=noisy_signal,
                case_idx=case_idx,
                no_decimation=no_decimation,
            )

            key = (
                f"BIN_{bin_idx:03d}_SNR_{snr:+03d}_IDX_{frame_idx:05d}"
                f"_FH_{f_h:09.1f}_CASE_{case_idx}_{mode_tag}"
            )
            iq_dict[key] = frame

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
def save_dataset_info(
    output_path:   str,
    iq_dict:       dict,
    case_idx:      int,
    no_decimation: bool = False,
) -> None:
    cpd       = CASE_PARAMS[case_idx]
    info_path = output_path.replace('.pkl', '_info.txt')
    snr_values = []
    for k in iq_dict:
        parts = k.split('_')
        try:
            snr_values.append(int(parts[parts.index('SNR') + 1]))
        except (ValueError, IndexError):
            pass

    if no_decimation:
        D, fs_eff, ND, Nz = 1, FS_NATIVE, _NO_DEC_ND, _NO_DEC_NZ
    else:
        D, fs_eff = cpd['D'], cpd['fs_eff']
        ND = int(N_RAW / D)
        Nz = _compute_nz(ND)

    with open(info_path, 'w') as fh:
        fh.write("=" * 80 + "\n")
        fh.write(f"DATASET: Unified Case {case_idx}"
                 f"{'  [no-decimation / OR mode]' if no_decimation else ''}\n")
        fh.write("=" * 80 + "\n\n")
        fh.write("CASE PARAMETERS (notes, beta=10, Nc=1024, constant=3.2):\n")
        fh.write(f"  [f1, f2]       = [{cpd['f1']:.1f}, {cpd['f2']:.1f}] Hz\n")
        fh.write(f"  fs_native      = {FS_NATIVE/1e6:.1f} MHz\n")
        if not no_decimation:
            fh.write(f"  fs_i formula   = Nc * f1 / 3.2 = {CREPE_FRAME_LENGTH * cpd['f1'] / 3.2:.1f} Hz\n")
        fh.write(f"  D              = {D}{'  (no-decimation OR mode)' if no_decimation else ''}\n")
        fh.write(f"  fs_eff         = {fs_eff:.1f} Hz\n")
        fh.write(f"  N_raw          = {N_RAW}\n")
        fh.write(f"  ND             = {ND}\n")
        fh.write(f"  Nz             = {Nz}  [(ND+Nz) % Nc = {(ND+Nz) % CREPE_FRAME_LENGTH}]\n")
        fh.write(f"  (ND+Nz)/Nc     = {(ND+Nz)//CREPE_FRAME_LENGTH}  non-overlapping Welch slices\n\n")
        fh.write("CHANNEL PIPELINE:\n")
        fh.write("  clean signal -> ETU70 fading -> CFO -> phase offset -> AWGN\n\n")
        fh.write("FEATURE PIPELINE:\n")
        if no_decimation:
            fh.write("  1. [NO decimation - OR mode, raw 25 MHz samples used directly]\n")
        else:
            fh.write(f"  1. Decimate by D={D} (LPF cutoff={FS_NATIVE/(2*D):.1f} Hz, Kaiser beta=10)\n")
        fh.write("  2. s[n] = |x[n]|^2 - mean  [DC removed]\n")
        fh.write(f"  3. Zero-pad {Nz} zeros  [total length = {ND+Nz}]\n")
        fh.write("  4. Welch PSD (nperseg=1024, noverlap=512, Kaiser beta=10, nfft=1024)\n")
        fh.write("  5. Interpolate 513 -> 1024 bins\n")
        fh.write("  6. Normalize to [0, 1]\n\n")
        fh.write(f"SNR levels: {sorted(set(snr_values))}\n")
        fh.write(f"Total samples: {len(iq_dict):,}\n")
        fh.write(f"Output pickle: {output_path}\n")
        fh.write("=" * 80 + "\n")

    print(f"[OK] Info saved -> {info_path}")

# ─────────────────────────────────────────────────────────────────────────────
# CLI helpers
# ─────────────────────────────────────────────────────────────────────────────
def _run_single_case(
    case_idx:       int,
    output_dir:     str,
    n_input_frames: int,
    use_gpu:        bool,
    no_decimation:  bool = False,
    date_tag:       str  = '',
) -> None:
    suffix   = '_nodec' if no_decimation else '_unified'
    tag      = f'_{date_tag}' if date_tag else suffix
    filename = f'iq_dict_case{case_idx}{tag}.pkl'
    out_path = os.path.join(output_dir, filename)
    iq_dict  = generate_unified_case_dataset(
        output_path=out_path,
        case_idx=case_idx,
        n_input_frames=n_input_frames,
        no_decimation=no_decimation,
        use_gpu=use_gpu,
    )
    save_dataset_info(out_path, iq_dict, case_idx, no_decimation=no_decimation)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description='Generate Unified Case 1-4 datasets (decimated or no-decimation OR mode)',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  Decimated mode, all cases:
    python generate_unified_dataset.py --case ALL

  No-decimation (OR) mode, Case 3 f_h range:
    python generate_unified_dataset.py --case 3 --no_decimation

  No-decimation, all cases, GPU backend:
    python generate_unified_dataset.py --case ALL --no_decimation --backend gpu

  Custom output dir, date tag:
    python generate_unified_dataset.py --case 2 --output_dir ./out/ --date_tag 15apr
        """,
    )
    parser.add_argument(
        '--case', type=str, default='ALL',
        choices=['1', '2', '3', '4', 'ALL'],
        help='Case index 1-4 or ALL. Determines [f1,f2] frequency range.',
    )
    parser.add_argument(
        '--no_decimation', action='store_true', default=False,
        help=(
            'OR mode (from notes): skip LPF/decimation entirely. '
            'Uses full N=250,000 raw 25 MHz samples directly. '
            'Zero-pads to 250,880 -> 245 non-overlapping Welch slices. '
            'f_h range still follows the chosen --case.'
        ),
    )
    parser.add_argument('--output_dir',     type=str, default='./results/')
    parser.add_argument('--n_input_frames', type=int, default=33333)
    parser.add_argument('--date_tag',       type=str, default='')
    parser.add_argument(
        '--backend', type=str, default='auto', choices=['auto', 'cpu', 'gpu'],
        help='Compute backend for decimation filtering (ignored in --no_decimation mode).',
    )
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

    print(f"CuPy available  : {GPU_AVAILABLE}")
    print(f"Backend         : {'GPU(CuPy)' if use_gpu else 'CPU(SciPy)'}")
    print(f"Mode            : {'no-decimation (OR)' if args.no_decimation else 'decimated'}")
    print()

    cases_to_run = [1, 2, 3, 4] if args.case == 'ALL' else [int(args.case)]
    for c in cases_to_run:
        _run_single_case(
            case_idx=c,
            output_dir=args.output_dir,
            n_input_frames=args.n_input_frames,
            use_gpu=use_gpu,
            no_decimation=args.no_decimation,
            date_tag=args.date_tag,
        )