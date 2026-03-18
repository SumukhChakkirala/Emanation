
import matplotlib.pyplot as plt
plt.figure(1)
plt.hist(all_fh, bins = 30)
plt.show()

from scipy.signal.windows import kaiser
import matplotlib.pyplot as plt
import numpy as np
import time
import imageio
kaiser_beta = 10
width = 3
aug_idx = 1
snr_val = 20
bin_idx = 100
frames = []
for snr_val in range(15, 20,1):   # example range
    key = 'BIN_' + str(f"{bin_idx:0{width}d}") + '_SNR_' + str(f"{snr_val:+d}") + '_AUG_' + str(f"{aug_idx:0{width}d}")
    
    signal = iq_dict[key]
    signal_temp = np.real(np.multiply(signal, np.conj(signal))) 
    
    w = kaiser(len(signal_temp), kaiser_beta)
    w /= np.sum(w)
    w_energy = (np.real(np.vdot(w, w))) / len(w)
    
    iq_w = np.multiply(signal_temp, w)
    iq_noFE = np.multiply(signal, w)

    fft_iqNoFE = np.fft.fftshift(np.abs(np.fft.fft(iq_noFE)))
    psd_val_noFE = 10*np.log10(np.multiply(fft_iqNoFE, fft_iqNoFE) / (w_energy * len(w)*CREPE_FS))

    fft_iq = np.fft.fftshift(np.abs(np.fft.fft(iq_w)))
    psd_val = np.multiply(fft_iq, fft_iq) / (w_energy * len(w)*CREPE_FS)
    psd = 10 * np.log10(psd_val + 1e-20)

    freqs = np.fft.fftshift(np.fft.fftfreq(len(signal_temp), 1/CREPE_FS))
    pos_mask = freqs >= 0

    plt.figure(figsize=(15, 10))
    plt.subplot(2,1,1)
    plt.plot(freqs[pos_mask], psd[pos_mask], label='withFE', linewidth=1.5, alpha=0.85)
    plt.plot(freqs[pos_mask], psd_val_noFE[pos_mask], label='withoutFE')

    plt.title(f"BIN {bin_idx}")
    plt.legend()
    plt.subplot(2,1,2)
    plt.plot(iq_dict[key], label='IQ Signal')
    # plt.show(block=False)

    # plt.pause(2)   # show for 3 seconds
    # plt.close()
    # capture frame
    plt.tight_layout()

    # draw figure
    fig = plt.gcf()
    fig.canvas.draw()

    # grab frame (works on macOS)
    frame = np.asarray(fig.canvas.buffer_rgba())
    frames.append(frame)

    plt.close()

# save video
imageio.mimsave("psd_bins.mp4", frames, fps=1)






from scipy.signal.windows import kaiser
import matplotlib.pyplot as plt
import numpy as np
import imageio.v2 as imageio

kaiser_beta = 10
width = 3
aug_idx = 1
snr_val = 20
frames = []
bin_idx = 100
for aug_idx in range(1, 150, 1):
    key = 'BIN_' + str(f"{bin_idx:0{width}d}") + '_SNR_' + str(f"{snr_val:+d}") + '_AUG_' + str(f"{aug_idx:0{width}d}")

    signal = iq_dict[key]
    signal_temp = np.real(signal * np.conj(signal))

    w = kaiser(len(signal_temp), kaiser_beta)
    w /= np.sum(w)
    w_energy = np.real(np.vdot(w, w)) / len(w)

    iq_w = signal_temp * w
    iq_noFE = signal * w

    fft_iqNoFE = np.fft.fftshift(np.abs(np.fft.fft(iq_noFE)))
    psd_val_noFE = 10 * np.log10((fft_iqNoFE**2) / (w_energy * len(w) * CREPE_FS) + 1e-20)

    fft_iq = np.fft.fftshift(np.abs(np.fft.fft(iq_w)))
    psd_val = (fft_iq**2) / (w_energy * len(w) * CREPE_FS)
    psd = 10 * np.log10(psd_val + 1e-20)

    freqs = np.fft.fftshift(np.fft.fftfreq(len(signal_temp), 1 / CREPE_FS))
    pos_mask = freqs >= 0

    fig = plt.figure(figsize=(15, 10))

    plt.subplot(2, 1, 1)
    plt.plot(freqs[pos_mask], psd[pos_mask], label='withFE', linewidth=1.5, alpha=0.85)
    plt.plot(freqs[pos_mask], psd_val_noFE[pos_mask], label='withoutFE')
    hz_val = crepe_bin_to_hz(bin_idx)
    plt.xlabel(f"Freq {hz_val} Hz.")
    plt.ylabel(f"Aug_IDX {aug_idx}")
    plt.legend()

    plt.subplot(2, 1, 2)
    plt.plot(np.real(signal), label='Real')
    plt.plot(np.imag(signal), label='Imag')
    plt.legend()

    plt.tight_layout()

    fig.canvas.draw()
    frame = np.asarray(fig.canvas.buffer_rgba())
    frames.append(frame)

    plt.close(fig)

with imageio.get_writer("psd_aug_idx.mp4", fps=1) as writer:
    for frame in frames:
        writer.append_data(frame)



#### code to unit test the Gaussian aproximation:
label = np.zeros(CREPE_N_BINS, dtype=np.float32)
        
        # Gaussian centered at true_bin with sigma in bins

c_true = 1200*np.log2(fh/CREPE_F_REF)
c_est = 0
label_sum = 0
for i in range(CREPE_N_BINS):
    c_i = CREPE_CENTS_PER_BIN*(i-1) + CENTS_OFFSET
    c_est = c_est + c_i*label[i] 
    label_sum = label_sum + label[i]
c_est = c_est / (label_sum + 1e-8)
print("C estimate is: ", c_est, "C_true is", c_true)
            
            # label[i] = np.exp(-(c_i-c_true)**2 / (2 * self.gaussian_sigma ** 2))
            # label[i] = np.exp(-((i - true_bin) ** 2) / (2 * self.gaussian_sigma ** 2))
        
        # Normalize (though paper doesn't explicitly do this)
        # label = label / (label.sum() + 1e-8)

label = np.zeros(CREPE_N_BINS, dtype=np.float64)

# Gaussian centered at true_bin with sigma in bins

label = np.zeros(CREPE_N_BINS, dtype=np.float64)
        
        # Gaussian centered at true_bin with sigma in bins
        
c_true = 1200*np.log2(fh/CREPE_F_REF)
for i in range(CREPE_N_BINS):
    c_i = CREPE_CENTS_PER_BIN*(i-1) + CENTS_OFFSET
    exp_val = ((c_i-c_true)**2) / (2 * (self.gaussian_sigma ** 2))
    print(expl_val)
    label[i] = np.exp(-exp_val)
    print("Label at bin ", i, " is ", label[i])
    # label[i] = np.exp(-((i - true_bin) ** 2) / (2 * self.gaussian_sigma ** 2))
        
        # Normalize (though paper doesn't explicitly do this)
        # label = label / (label.sum() + 1e-8)
        