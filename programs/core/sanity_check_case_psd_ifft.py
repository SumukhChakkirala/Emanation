import argparse
import os

import matplotlib.pyplot as plt
import numpy as np

from generateData import _extract_psd_and_ifft_from_25mhz, _load_emanation_config
from generate_crepe_data import add_complex_noise, generate_dirac_comb_signal


def run_sanity_checks(
    output_dir: str,
    num_examples: int,
    snr_db: float,
    seed: int,
):
    os.makedirs(output_dir, exist_ok=True)
    cfg = _load_emanation_config('synapse_emanation_search.yaml')

    fs_raw = 25e6
    case_ranges = {
        'A': (800e3, 1e6, 0.1),
        'B': (3.2e3, 100e3, 0.01),
        'C': (32.0, 4e3, 0.1),
    }

    rng = np.random.default_rng(seed)
    noise_rng = np.random.default_rng(seed + 1000)

    fig_psd, axes_psd = plt.subplots(3, num_examples, figsize=(4 * num_examples, 10), squeeze=False)
    fig_ifft, axes_ifft = plt.subplots(3, num_examples, figsize=(4 * num_examples, 10), squeeze=False)

    all_ifft = {}

    for row, case in enumerate(['A', 'B', 'C']):
        f_min, f_max, duration = case_ranges[case]
        fh_values = np.round(rng.uniform(f_min, f_max, num_examples), 1)
        print(f"Case {case} random f_h values (Hz): {fh_values.tolist()}")

        all_ifft[case] = {}

        for col, f_h in enumerate(fh_values):
            clean_signal = generate_dirac_comb_signal(
                F_h=float(f_h),
                Fs=fs_raw,
                duration=duration,
                duty_cycle=0.5,
            )
            noisy_signal = add_complex_noise(clean_signal, snr_db, noise_rng)

            psd, ifft_frame, fs_eff = _extract_psd_and_ifft_from_25mhz(
                iq_signal=noisy_signal,
                fs_raw=fs_raw,
                case=case,
                cfg=cfg,
                target_length=1024,
                welch_nperseg=1000,
            )

            all_ifft[case][f"fh_{float(f_h):.1f}"] = ifft_frame

            freq_axis = np.linspace(-fs_eff / 2.0, fs_eff / 2.0, len(psd), endpoint=False)
            axes_psd[row, col].plot(freq_axis, 10 * np.log10(psd + 1e-20), linewidth=1.0)
            axes_psd[row, col].set_title(f"Case {case} | f_h={float(f_h):.1f} Hz")
            axes_psd[row, col].set_xlabel("Frequency (Hz)")
            axes_psd[row, col].set_ylabel("PSD (dB)")
            axes_psd[row, col].grid(alpha=0.3)

            axes_ifft[row, col].plot(ifft_frame, linewidth=1.0)
            axes_ifft[row, col].set_title(f"Case {case} | f_h={float(f_h):.1f} Hz")
            axes_ifft[row, col].set_xlabel("Sample Index")
            axes_ifft[row, col].set_ylabel("IFFT Output")
            axes_ifft[row, col].grid(alpha=0.3)

    fig_psd.suptitle(f"Sanity Check: PSD Plots (SNR={snr_db} dB)", fontsize=14)
    fig_psd.tight_layout(rect=[0, 0.03, 1, 0.97])
    psd_path = os.path.join(output_dir, "sanity_psd_cases_ABC.png")
    fig_psd.savefig(psd_path, dpi=160)

    fig_ifft.suptitle(f"Sanity Check: IFFT Outputs (SNR={snr_db} dB)", fontsize=14)
    fig_ifft.tight_layout(rect=[0, 0.03, 1, 0.97])
    ifft_plot_path = os.path.join(output_dir, "sanity_ifft_cases_ABC.png")
    fig_ifft.savefig(ifft_plot_path, dpi=160)

    ifft_npy_path = os.path.join(output_dir, "sanity_ifft_outputs_cases_ABC.npy")
    np.save(ifft_npy_path, all_ifft, allow_pickle=True)

    plt.close(fig_psd)
    plt.close(fig_ifft)

    print("\nSaved sanity-check outputs:")
    print(f"  PSD figure : {psd_path}")
    print(f"  IFFT figure: {ifft_plot_path}")
    print(f"  IFFT array : {ifft_npy_path}")


def main():
    parser = argparse.ArgumentParser(description="Sanity checks: PSD + IFFT outputs for cases A/B/C")
    parser.add_argument("--output_dir", type=str, default="./results/sanity_checks")
    parser.add_argument("--num_examples", type=int, default=4)
    parser.add_argument("--snr_db", type=float, default=-10.0)
    parser.add_argument("--seed", type=int, default=2026)
    args = parser.parse_args()

    run_sanity_checks(
        output_dir=args.output_dir,
        num_examples=args.num_examples,
        snr_db=args.snr_db,
        seed=args.seed,
    )


if __name__ == "__main__":
    main()
