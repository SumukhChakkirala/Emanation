import matplotlib
matplotlib.use('Agg')  # Non-interactive backend for saving without display
import matplotlib.pyplot as plt
import numpy as np
from scipy import signal

# Parameters
f = 1000  # Hz
fs = 1e6  # Sampling rate 1 MHz
cutoff = 2000  # Hz
numtaps = 129
beta = 10

# Generate t values from 0 to 0.005 s
t_values = np.linspace(0, 0.005, 5000)

# Generate signal: cos(2pi f t)
sig = np.cos(2 * np.pi * f * t_values)

# Design Kaiser filter
taps = signal.firwin(numtaps, cutoff, window=('kaiser', beta), fs=fs)

# Apply filter
filtered = signal.lfilter(taps, 1, sig)

# Plot
plt.figure(figsize=(10, 6))
plt.plot(t_values, sig, label='Original cos(2π f t)', alpha=0.7)
plt.plot(t_values, filtered, label='Filtered with Kaiser LPF', linewidth=2)
plt.title(r'Original and Kaiser-Filtered cos(2π f t) with f=1000 Hz, cutoff=2000 Hz')
plt.xlabel(r'$t$ (seconds)')
plt.ylabel('Amplitude')
plt.legend()
plt.grid(True)
plt.savefig('kaiser_filtered_cos.png')
plt.show()