
import numpy as np
import matplotlib.pyplot as plt
import pickle
import os
import sys
from scipy.signal.windows import kaiser

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from generate_crepe_data import crepe_bin_to_hz, CREPE_FS

def plot_psd_comparison(iq_dict, bin_idx, aug_idx, snr_list, output_path=None):
    """Plot PSD for a specific bin and augmentation across SNR values."""
    F_h = crepe_bin_to_hz(bin_idx)
    fig, ax = plt.subplots(figsize=(14, 7))
    colors = plt.cm.plasma(np.linspace(0, 1, len(snr_list)))
    
    kaiser_beta = 14 # Default from Plot_functions_search.py

    for i, snr in enumerate(snr_list):
        key = f"BIN_{bin_idx:03d}_SNR_{snr:+03d}_AUG_{aug_idx:03d}"
        if key in iq_dict:
            signal = iq_dict[key]
            
            # Kaiser windowed PSD calculation
            w = kaiser(len(signal), kaiser_beta)
            w /= np.sum(w)
            w_energy = (np.real(np.vdot(w, w))) / len(w)
            iq_w = np.multiply(signal, w)
            fft_iq = np.fft.fftshift(np.abs(np.fft.fft(iq_w)))
            psd_val = np.multiply(fft_iq, fft_iq) / (w_energy * len(w)*CREPE_FS)
            psd = 10 * np.log10(psd_val + 1e-20)

            freqs = np.fft.fftshift(np.fft.fftfreq(len(signal), 1/CREPE_FS))
            
            pos_mask = freqs >= 0
            ax.plot(freqs[pos_mask], psd[pos_mask], label=f'SNR = {snr:+3d} dB', 
                   linewidth=1.5, color=colors[i], alpha=0.85)
    
    # Mark harmonics
    for h in range(1, 8):
        harmonic = F_h * h
        if harmonic < CREPE_FS / 2:
            ax.axvline(harmonic, color='red', linestyle='--', alpha=0.3, linewidth=1.2)
            if h == 1:
                ax.text(harmonic + 20, ax.get_ylim()[1] - 8, f'F₀={F_h:.1f} Hz', 
                       rotation=0, va='top', fontsize=10, color='red', fontweight='bold')
    
    xaxis = min(CREPE_FS/2, F_h*10)
    
    ax.set_title(f'PSD Comparison Across SNR Levels\nBin {bin_idx} (F₀ = {F_h:.2f} Hz) | Aug {aug_idx}', 
                fontsize=14, fontweight='bold')
    ax.set_xlabel('Frequency (Hz)', fontsize=12)
    ax.set_ylabel('Power Spectral Density (dB)', fontsize=12)
    ax.set_xlim([0, xaxis])
    ax.grid(True, alpha=0.3, linestyle=':', linewidth=0.8)
    ax.legend(loc='upper right', fontsize=10, framealpha=0.9)
    plt.tight_layout()
    
    if output_path:
        plt.savefig(output_path, dpi=200, bbox_inches='tight')
        print(f"✓ Saved to {output_path}")
        plt.close()
    else:
        plt.show()

if __name__ == '__main__':
    # Load dataset
    dataset_path = r'programs\dutyCycle=0.1\IQData\iq_dict_crepe_dirac_comb_SNR0_20.pkl'
    #if not os.path.exists(dataset_path):
        #dataset_path = '../../IQData/iq_dict_crepe_dirac_comb.pkl'
    
    with open(dataset_path, 'rb') as f:
        iq_dict = pickle.load(f)
    
    # Plot configuration
    plot_psd_comparison(iq_dict, bin_idx=234, aug_idx=0, 
                       snr_list=[-20, -10, -5, 0, 5, 10,20],
                       output_path='psd_comparison_bin234_aug0.png')

#example

