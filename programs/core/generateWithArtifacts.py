"""
Generate CREPE-compatible dataset with TRULY CONTINUOUS frequency coverage.

f_h is randomly generated between fmin and fmax as continuous values (1 decimal place).
This better simulates real-world scenarios where frequencies are truly continuous.

Structure:
- Loop 1: n_input_frames iterations (each picks a random f_h, uniformly distributed)
- Loop 2: iterate over SNR values
- Seed changes with each iteration index for reproducibility

Total samples: n_input_frames × len(snr_list)

Channel pipeline (per sample):
    clean signal -> ETU70 multipath fading (GNU Radio) -> CFO -> phase offset -> AWGN
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

# ─────────────────────────────────────────────
# Clock / channel parameters (mirrors file 1)
# ─────────────────────────────────────────────
f_c         = 1e9       # Centre frequency (Hz)
clockrate   = 100e6     # ADC/DAC clock rate (Hz)

clockeffects_dict = {}
clockeffects_dict['XOFreq']               = 10e6
clockeffects_dict['XO_standardDeviation'] = 1e-4
clockeffects_dict['XO_maxdeviation']      = 5
clockeffects_dict['LOScalingFactor']      = f_c      / clockeffects_dict['XOFreq']
clockeffects_dict['TimetickScalingFactor']= clockrate / clockeffects_dict['XOFreq']
clockeffects_dict['CFO_standardDeviation']= clockeffects_dict['XO_standardDeviation'] * clockeffects_dict['LOScalingFactor']
clockeffects_dict['SRO_standardDeviation']= clockeffects_dict['XO_standardDeviation'] * clockeffects_dict['TimetickScalingFactor']
clockeffects_dict['CFO_maxdeviation']     = clockeffects_dict['XO_maxdeviation']      * clockeffects_dict['LOScalingFactor']
clockeffects_dict['SRO_maxdeviation']     = clockeffects_dict['XO_maxdeviation']      * clockeffects_dict['TimetickScalingFactor']

# ETU70 fading model parameters (mirrors file 1)
_ETU70_delays_ns = [0, 50, 120, 200, 230, 500, 1600, 2300, 5000]   # nanoseconds
_ETU70_mags_dB   = [-1, -1, -1,   0,   0,   0,   -3,   -5,   -7]
_ETU70_mags      = [10 ** (m / 20.0) for m in _ETU70_mags_dB]
_ETU70_fD        = 70          # max Doppler shift (Hz) for ETU70
_NUM_SINUSOIDS   = 8
_NTAPS           = 8
_LOS             = False       # Rayleigh (no line-of-sight)
_KFACTOR         = 4


# ─────────────────────────────────────────────
# Clock artifact functions
# Import directly from clockArtifacts.py —
# the same module used by the RML22 notebook.
# ─────────────────────────────────────────────
from artifacts import CFOArtifact, phaseOffset


# ─────────────────────────────────────────────
# Full channel pipeline
# GNU Radio implementation (NumPy/SciPy approximation kept commented out).
# ─────────────────────────────────────────────


def _fractional_delay(signal_in: np.ndarray, delay_samples: float) -> np.ndarray:
    """Apply fractional delay using linear interpolation."""
    n = signal_in.shape[0]
    idx = np.arange(n, dtype=np.float64) - float(delay_samples)
    out_real = np.interp(idx, np.arange(n), np.real(signal_in), left=0.0, right=0.0)
    out_imag = np.interp(idx, np.arange(n), np.imag(signal_in), left=0.0, right=0.0)
    return (out_real + 1j * out_imag).astype(np.complex64)


def _etu70_fading_numpy(samples: np.ndarray, samp_rate: float, seed: int) -> np.ndarray:
    """Approximate ETU70-like fading using fractional-delay taps and Doppler phase rotation."""
    x = np.asarray(samples, dtype=np.complex64)
    n = x.shape[0]
    t = np.arange(n, dtype=np.float64) / float(samp_rate)
    delays_samples = [d * 1e-9 * float(samp_rate) for d in _ETU70_delays_ns]

    rng_local = np.random.default_rng(int(seed))
    y = np.zeros(n, dtype=np.complex64)
    for delay_samp, mag in zip(delays_samples, _ETU70_mags):
        x_delayed = _fractional_delay(x, delay_samp)
        phase0 = rng_local.uniform(0.0, 2.0 * np.pi)
        doppler = np.exp(1j * (2.0 * np.pi * _ETU70_fD * t + phase0)).astype(np.complex64)
        y += (mag * x_delayed * doppler).astype(np.complex64)

    return y

def apply_channel_pipeline(
    clean_signal: np.ndarray,
    samp_rate:    float,
    seed:         int,
) -> np.ndarray:
    """
    Apply channel artifacts using NumPy/SciPy:

        clean -> ETU70 fading (NumPy) -> CFO -> phase offset
    """
    # GNU Radio import no longer needed

    # ── XO value generation (copied verbatim from the RML22 notebook) ────────
    XO_val_len = len(clean_signal) + 10   # +10 safety margin (same as notebook)
    np.random.seed(seed)

    ferr_bias_XO = np.random.uniform(
        -clockeffects_dict['XO_maxdeviation'] + clockeffects_dict['XO_standardDeviation'],
         clockeffects_dict['XO_maxdeviation'] - clockeffects_dict['XO_standardDeviation'],
    )
    XO_val = np.zeros((XO_val_len,))
    XO_val[0] = clockeffects_dict['XO_standardDeviation'] * np.random.randn() + ferr_bias_XO
    while (XO_val[0] > clockeffects_dict['XO_maxdeviation']) or \
          (XO_val[0] < -clockeffects_dict['XO_maxdeviation']):
        XO_val[0] = clockeffects_dict['XO_standardDeviation'] * np.random.randn() + ferr_bias_XO

    for i in range(1, XO_val_len):
        XO_val[i] = clockeffects_dict['XO_standardDeviation'] * np.random.randn() + XO_val[i - 1]
        while (XO_val[i] > clockeffects_dict['XO_maxdeviation']) or \
              (XO_val[i] < -clockeffects_dict['XO_maxdeviation']):
            XO_val[i] = clockeffects_dict['XO_standardDeviation'] * np.random.randn() + XO_val[i - 1]

    # ── Step 1: ETU70-style fading (NumPy/SciPy) ─────────────────────────────
    samples_SRO_Fading = _etu70_fading_numpy(clean_signal, samp_rate, seed)

    # ── GNU Radio reference path [disabled for performance] ────────────────────
    # delays = [d * 1e-9 * samp_rate for d in _ETU70_delays_ns]
    # fading_block = channels.selective_fading_model(
    #     _NUM_SINUSOIDS,
    #     _ETU70_fD / samp_rate,
    #     _LOS,
    #     _KFACTOR,
    #     seed,
    #     delays,
    #     _ETU70_mags,
    #     _NTAPS,
    # )
    # samples_SRO_src_block = blocks.vector_source_c(clean_signal.tolist(), False, 1, [])
    # snk2 = blocks.vector_sink_c()
    # tb1 = gr.top_block()
    # tb1.connect(samples_SRO_src_block, fading_block, snk2)
    # tb1.run()
    # samples_SRO_Fading = np.array(snk2.data(), dtype=np.complex64)

    # ── Step 3: CFO ──────────────────────────────────────────────────────────
    samples_SRO_Fading_CFO = CFOArtifact(samples_SRO_Fading, XO_val, clockeffects_dict, samp_rate)

    # ── Step 4: Phase offset ─────────────────────────────────────────────────
    samples_SRO_Fading_CFO_Phaseoffset = phaseOffset(samples_SRO_Fading_CFO, seed)

    # AWGN is applied per-SNR in the outer dataset loop for performance.
    return np.asarray(samples_SRO_Fading_CFO_Phaseoffset, dtype=np.complex64)



# ─────────────────────────────────────────────
# Original constants / helpers (unchanged)
# ─────────────────────────────────────────────

# Single main seed for reproducible signal parameters (Fh, duty cycle)
main_seed = 1234
rng       = np.random.default_rng(seed=main_seed)

_CPU_TAPS_CACHE = {}
_GPU_TAPS_CACHE = {}

CREPE_N_BINS      = 360
CREPE_CENTS_PER_BIN = 20


def freq_to_model_bin(freq_hz: float, fmin_hz: float, fmax_hz: float) -> int:
    freq = float(np.clip(freq_hz, fmin_hz, fmax_hz))
    lo   = np.log2(float(fmin_hz))
    hi   = np.log2(float(fmax_hz))
    pos  = (np.log2(freq) - lo) / (hi - lo + 1e-12)
    return int(np.clip(pos * (CREPE_N_BINS - 1), 0.0, CREPE_N_BINS - 1.0))


from generate_crepe_data import (
    CREPE_FS,
    CREPE_FRAME_LENGTH,
    generate_dirac_comb_signal,
    add_complex_noise,
)


def _load_emanation_config(cfg_path: str = 'synapse_emanation_search.yaml') -> dict:
    if os.path.exists(cfg_path):
        with open(cfg_path, 'r') as fh:
            return yaml.safe_load(fh)
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


def _get_cpu_taps(fs_raw, cutoff_hz, numtaps, kaiser_beta_hh):
    key  = (float(fs_raw), float(cutoff_hz), int(numtaps), float(kaiser_beta_hh))
    taps = _CPU_TAPS_CACHE.get(key)
    if taps is None:
        taps = signal.firwin(
            int(numtaps), float(cutoff_hz),
            window=('kaiser', float(kaiser_beta_hh)), fs=float(fs_raw)
        ).astype(np.float32)
        _CPU_TAPS_CACHE[key] = taps
    return taps


def _get_gpu_taps(cpu_taps):
    if not GPU_AVAILABLE:
        return None
    key  = (cpu_taps.shape[0], str(cpu_taps.dtype), float(cpu_taps[0]), float(cpu_taps[-1]))
    taps = _GPU_TAPS_CACHE.get(key)
    if taps is None:
        taps = cp.asarray(cpu_taps)
        _GPU_TAPS_CACHE[key] = taps
    return taps


def _apply_case_filter_and_decimate(iq_signal, fs_raw, case, numtaps, kaiser_beta_hh, use_gpu=False):
    x          = np.asarray(iq_signal, dtype=np.complex64)
    cutoff_hz, dec_order = _case_decimation_params(case, fs_raw)
    if cutoff_hz is None:
        return x, float(fs_raw)
    taps_cpu = _get_cpu_taps(fs_raw, cutoff_hz, numtaps, kaiser_beta_hh)
    if use_gpu and GPU_AVAILABLE:
        x_out = cp.asnumpy(csignal.upfirdn(_get_gpu_taps(taps_cpu), cp.asarray(x), up=1, down=int(dec_order)))
    else:
        x_out = signal.upfirdn(taps_cpu, x, up=1, down=int(dec_order))
    return np.asarray(x_out, dtype=np.complex64), float(fs_raw) / float(dec_order)


def _extract_1024_frame_from_25mhz(iq_signal, fs_raw, case, cfg,
                                    target_length=CREPE_FRAME_LENGTH,
                                    welch_nperseg=1000, use_gpu=False):
    _, frame, _ = _extract_psd_and_ifft_from_25mhz(
        iq_signal=iq_signal, fs_raw=fs_raw, case=case, cfg=cfg,
        target_length=target_length, welch_nperseg=welch_nperseg, use_gpu=use_gpu,
    )
    return frame


def _extract_psd_and_ifft_from_25mhz(iq_signal, fs_raw, case, cfg,
                                       target_length=CREPE_FRAME_LENGTH,
                                       welch_nperseg=1000, use_gpu=False):
    ed_cfg        = cfg.get('EmanationDetection', {}) if cfg is not None else {}
    kaiser_beta_hh= float(ed_cfg.get('kaiser_beta_hh', 10.0))
    numtaps       = int(ed_cfg.get('numtaps', 129))

    x, fs_eff = _apply_case_filter_and_decimate(
        iq_signal=iq_signal, fs_raw=fs_raw, case=case,
        numtaps=numtaps, kaiser_beta_hh=kaiser_beta_hh, use_gpu=use_gpu,
    )

    x_feature  = np.real(np.multiply(x, np.conj(x)))
    x_feature -= np.mean(x_feature)

    nperseg = min(int(welch_nperseg), len(x_feature))
    if nperseg < 8:
        return np.zeros(1000, dtype=np.float64), np.zeros(target_length, dtype=np.float32), fs_eff

    window = signal.windows.kaiser(nperseg, beta=kaiser_beta_hh)
    _, psd = signal.welch(
        x_feature, fs=fs_eff, window=window,
        nperseg=nperseg, noverlap=0, nfft=nperseg,
        return_onesided=False, detrend=False, scaling='density',
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
    pad_left  = pad_total // 2
    psd_padded= np.pad(psd, (pad_left, pad_total - pad_left), mode='constant')

    frame_td = np.fft.ifft(np.fft.ifftshift(psd_padded))
    frame    = np.real(frame_td[:target_length]).astype(np.float32)
    frame   /= (np.max(np.abs(frame)) + 1e-8)

    return psd.astype(np.float64), frame, fs_eff


# ─────────────────────────────────────────────
# Main dataset generators  (modified to include
# the full channel pipeline)
# ─────────────────────────────────────────────

def generate_25mhz_case_dataset(
    output_path:       str,
    case:              str,
    f_min:             float,
    f_max:             float,
    snr_list:          list  = None,
    n_input_frames:    int   = 10000,
    duty_cycle:        float = 0.5,
    fs_raw:            float = 25e6,
    capture_duration_s:float = 0.01,
    cfg_path:          str   = 'synapse_emanation_search.yaml',
    use_gpu:           bool  = False,
) -> dict:
    """Generate 25 MHz Case A/B/C dataset with model-ready 1024-point inputs.

    Channel pipeline applied to every sample:
        clean -> SRO -> ETU70 fading -> CFO -> phase offset -> AWGN
    """
    if snr_list is None:
        snr_list = list(range(-20, 21))

    cfg        = _load_emanation_config(cfg_path)
    n_snr      = len(snr_list)
    total_samples = n_input_frames * n_snr
    fh_values  = np.round(rng.uniform(f_min, f_max, n_input_frames), 1)

    print("=" * 80)
    print(f"Generating 25MHz Case {case.upper()} dataset")
    print("=" * 80)
    print(f"  fs_raw:             {fs_raw/1e6:.1f} MHz")
    print(f"  capture_duration_s: {capture_duration_s}")
    print(f"  f_h range:          {f_min} Hz to {f_max} Hz")
    print(f"  SNR levels:         {snr_list}")
    print(f"  input frames:       {n_input_frames:,}")
    print(f"  total samples:      {total_samples:,}")
    print(f"  channel pipeline:   ETU70 (GNU Radio) -> CFO -> phase -> AWGN")
    print(f"  feature backend:    {'GPU(CuPy)' if (use_gpu and GPU_AVAILABLE) else 'CPU(SciPy)'}")

    iq_dict = {}
    seed    = 1680000   # mirrors file 1 starting seed

    for frame_idx in tqdm(range(n_input_frames), desc=f"Case-{case.upper()} frames"):
        f_h     = float(fh_values[frame_idx])
        bin_idx = freq_to_model_bin(f_h, f_min, f_max)

        clean_signal = generate_dirac_comb_signal(
            F_h=f_h, Fs=fs_raw,
            duration=capture_duration_s, duty_cycle=duty_cycle,
        )

        # Apply expensive channel artifacts once per frame.
        signal_no_artif_noise = apply_channel_pipeline(
            clean_signal=clean_signal,
            samp_rate=fs_raw,
            seed=seed,
        )

        for snr in snr_list:
            noisy_signal = add_complex_noise(signal_no_artif_noise, snr, np.random.default_rng(seed))

            frame = _extract_1024_frame_from_25mhz(
                iq_signal=noisy_signal, fs_raw=fs_raw, case=case, cfg=cfg,
                target_length=CREPE_FRAME_LENGTH, welch_nperseg=1000, use_gpu=use_gpu,
            )

            key = (
                f"BIN_{bin_idx:03d}_SNR_{snr:+03d}_IDX_{frame_idx:05d}"
                f"_FH_{f_h:09.1f}_CASE_{case.upper()}"
            )
            iq_dict[key] = frame
            seed += 1   # advance seed per sample, like file 1

    os.makedirs(os.path.dirname(output_path) if os.path.dirname(output_path) else '.', exist_ok=True)
    with open(output_path, 'wb') as fh:
        pickle.dump(iq_dict, fh)

    print(f"\n[OK] Saved {len(iq_dict):,} samples to {output_path}")
    return iq_dict



# ─────────────────────────────────────────────
# Metadata / info helpers (unchanged)
# ─────────────────────────────────────────────

def save_dataset_info(output_path: str, iq_dict: dict, case: str, params: dict) -> None:
    info_path  = output_path.replace('.pkl', '_info.txt')
    fh_values  = []
    snr_values = []
    for k in iq_dict:
        parts = k.split('_')
        try:
            fh_values.append( float(parts[parts.index('FH')  + 1]))
            snr_values.append(int(  parts[parts.index('SNR') + 1]))
        except (ValueError, IndexError):
            pass

    unique_snr = sorted(set(snr_values))

    with open(info_path, 'w') as fh:
        fh.write("=" * 80 + "\n")
        fh.write(f"DATASET: Case {case.upper()} - 25 MHz\n")
        fh.write("=" * 80 + "\n\n")
        fh.write("GENERATION PARAMETERS:\n")
        fh.write(f"  Sampling rate (fs_raw): {params['fs_raw']/1e6:.1f} MHz\n")
        fh.write(f"  Capture duration:       {params['capture_duration_s']} s\n")
        fh.write(f"  Frequency range:        {params['f_min']/1e3:.1f} kHz - {params['f_max']/1e3:.1f} kHz\n")
        fh.write(f"  Duty cycle:             {params['duty_cycle']*100:.0f}%\n")
        fh.write(f"  Number of input frames: {params['n_input_frames']:,}\n")
        fh.write(f"  SNR levels:             {unique_snr}\n")
        fh.write(f"  Main seed:              {main_seed}\n")
        if 'use_gpu' in params:
            fh.write(f"  Feature backend: {'GPU(CuPy)' if params['use_gpu'] else 'CPU(SciPy)'}\n")
        fh.write("\nCHANNEL PIPELINE:\n")
        fh.write("  clean signal\n")
        fh.write("    -> ETU70 multipath fading (GNU Radio)\n")
        fh.write("    -> CFO  (centre-frequency offset)\n")
        fh.write("    -> phase offset (uniform random)\n")
        fh.write("    -> AWGN\n\n")
        fh.write("PROCESSING PIPELINE (feature extraction):\n")
        if case.upper() == 'A':
            fh.write("  1. Feature extraction: |x|^2 - mean\n")
            fh.write("  2. Welch PSD (nperseg=1000)\n")
            fh.write("  3. IFFT + pad/truncate to 1024\n")
            fh.write("  4. Normalize to [-1,1]\n")
        elif case.upper() == 'B':
            fh.write("  1. LPF @ 1 MHz (Kaiser, beta=10) -> decimate x25\n")
            fh.write("  2. Feature extraction: |x|^2 - mean\n")
            fh.write("  3. Welch PSD (nperseg=1000)\n")
            fh.write("  4. IFFT + pad/truncate to 1024\n")
            fh.write("  5. Normalize to [-1,1]\n")
        else:
            fh.write("  1. LPF @ 40 kHz (Kaiser, beta=10) -> decimate x625\n")
            fh.write("  2. Feature extraction: |x|^2 - mean\n")
            fh.write("  3. Welch PSD (nperseg=1000)\n")
            fh.write("  4. IFFT + pad/truncate to 1024\n")
            fh.write("  5. Normalize to [-1,1]\n")
        fh.write(f"\nOUTPUT:\n  Pickle: {output_path}\n  Info:   {info_path}\n")
        fh.write("=" * 80 + "\n")

    print(f"[OK] Dataset info saved to {info_path}")




def _case_defaults(case, n_input_frames, use_gpu, date_tag=''):
    case_u = case.upper()
    defs   = {
        'A': dict(fs_raw=25e6, capture_duration_s=0.01, f_min=80e3,  f_max=1e6,   duty_cycle=0.5),
        'B': dict(fs_raw=25e6, capture_duration_s=0.01, f_min=3.2e3, f_max=100e3, duty_cycle=0.5),
        'C': dict(fs_raw=25e6, capture_duration_s=0.01, f_min=32.0,  f_max=4e3,   duty_cycle=0.5),
    }
    params = {**defs[case_u], 'n_input_frames': n_input_frames, 'use_gpu': use_gpu}
    tag    = f'_{date_tag}' if date_tag else '(4-4-26)'
    fname  = f'iq_dict_case{case_u}_25MHz{tag}.pkl'
    return params, list(range(-10, 21)), fname


def _run_single_case(case, output_dir, n_input_frames, use_gpu, date_tag=''):
    params, snr_list, filename = _case_defaults(case, n_input_frames, use_gpu, date_tag)
    output_path = os.path.join(output_dir, filename)
    iq_dict = generate_25mhz_case_dataset(
        output_path=output_path, case=case, snr_list=snr_list, **params)
    save_dataset_info(output_path, iq_dict, case, params)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Generate 25MHz Case A/B/C datasets')
    parser.add_argument('--case',           type=str, default='ALL', choices=['A','B','C','ALL'])
    parser.add_argument('--output_dir',     type=str, default='./results/')
    parser.add_argument('--n_input_frames', type=int, default=33333)
    parser.add_argument('--date_tag',       type=str, default='')
    parser.add_argument('--backend',        type=str, default='auto', choices=['auto','cpu','gpu'])
    parser.add_argument('--quality_check',  action='store_true')
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    if args.backend == 'gpu':
        if not GPU_AVAILABLE:
            print('Warning: GPU backend requested but CuPy is not available. Falling back to CPU.')
            use_gpu = False
        else:
            use_gpu = True
    elif args.backend == 'cpu':
        use_gpu = False
    else:
        use_gpu = GPU_AVAILABLE

    print(f"CuPy available: {GPU_AVAILABLE}")
    print(f"Selected backend: {'GPU(CuPy)' if use_gpu else 'CPU(SciPy)'}")

    # if args.quality_check and use_gpu:
    #     run_gpu_quality_check(case='B', trials=3, snr_db=20.0)
    #     run_gpu_quality_check(case='C', trials=3, snr_db=20.0)

    if args.case == 'ALL':
        for c in ['A', 'B', 'C']:
            _run_single_case(c, args.output_dir, args.n_input_frames, use_gpu, args.date_tag)
    else:
        _run_single_case(args.case, args.output_dir, args.n_input_frames, use_gpu, args.date_tag)