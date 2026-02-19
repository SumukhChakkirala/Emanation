"""
Plot PSD for bins 333 and 338 from the continuous IQ dataset.
"""

import pickle
import os
import matplotlib.pyplot as plt
import numpy as np

# Import the CREPE constants and functions from testSet_continous.py
from testSet_continous import CREPE_FS, analyze_bin_psd

# Path to the IQ data
IQDATA_PATH = './IQData/iq_dict_continuous_freq.pkl'
SAVE_DIR = './models_crepe/'

# Load IQ data
def load_iq_dict(path):
    with open(path, 'rb') as f:
        return pickle.load(f)

def plot_psd_for_bin(iq_dict, bin_idx, snr=20, save_dir=SAVE_DIR):
    psd_result = analyze_bin_psd(iq_dict, bin_idx, snr=snr)
    if psd_result:
        plt.figure(figsize=(8, 4))
        psd_db = 10 * np.log10(psd_result['avg_psd'] + 1e-10)
        plt.plot(psd_result['freqs'], psd_db, linewidth=1.2, label=f'Bin {bin_idx}')
        plt.axvline(x=psd_result['expected_freq'], color='r', linestyle='--', label=f'Expected: {psd_result["expected_freq"]:.1f} Hz')
        for h in range(2, 5):
            harm_freq = psd_result['expected_freq'] * h
            if harm_freq < CREPE_FS / 2:
                plt.axvline(x=harm_freq, color='orange', linestyle=':', alpha=0.5)
        plt.xlabel('Frequency (Hz)')
        plt.ylabel('PSD (dB)')
        plt.title(f'PSD for Bin {bin_idx} (Expected F: {psd_result["expected_freq"]:.1f} Hz)')
        plt.xlim([0, min(2000, CREPE_FS/2)])
        plt.legend(fontsize=9)
        plt.grid(True, alpha=0.3)
        out_path = os.path.join(save_dir, f'psd_bin_{bin_idx}.png')
        plt.tight_layout()
        plt.savefig(out_path, dpi=150)
        plt.close()
        print(f"✓ Saved PSD plot for bin {bin_idx} to {out_path}")
    else:
        print(f"No data found for bin {bin_idx} at SNR={snr}.")

if __name__ == "__main__":
    iq_dict = load_iq_dict(IQDATA_PATH)
    for bin_idx in [333, 338]:
        plot_psd_for_bin(iq_dict, bin_idx, snr=20, save_dir=SAVE_DIR)
