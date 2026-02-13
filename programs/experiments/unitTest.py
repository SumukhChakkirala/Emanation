# test_crepe_functions.py
import sys
import numpy as np

# Add the path to the file
sys.path.insert(0, r'C:\Users\User1\Downloads\Emanation\programs\experiments')

# Import the module
from generate_crepe_data import (
    hz_to_cents,
    cents_to_hz,
    hz_to_crepe_bin,
    crepe_bin_to_hz,
    CREPE_CENTS_PER_BIN,
    CREPE_N_BINS
)

print("=" * 60)
print("CREPE Function Unit Tests")
print("=" * 60)
print(f"Constants: CREPE_CENTS_PER_BIN = {CREPE_CENTS_PER_BIN}, CREPE_N_BINS = {CREPE_N_BINS}")
print()

# Test 1: hz_to_cents()
print("Test 1: hz_to_cents()")
print("-" * 40)
test_freqs = [10.0, 20.0, 440.0, 880.0, 32.7]
for freq in test_freqs:
    cents = hz_to_cents(freq, fref=10.0)
    print(f"  {freq:8.1f} Hz → {cents:10.4f} cents")
print()

# Test 2: cents_to_hz()
print("Test 2: cents_to_hz()")
print("-" * 40)
test_cents = [0.0, 1200.0, 2400.0, 6551.3, 1997.38]
for cent_val in test_cents:
    freq = cents_to_hz(cent_val, fref=10.0)
    print(f"  {cent_val:10.2f} cents → {freq:10.4f} Hz")
print()

# Test 3: Round-trip Hz → Cents → Hz
print("Test 3: Round-trip Hz → Cents → Hz")
print("-" * 40)
for freq in [440.0, 880.0, 220.0, 100.0]:
    cents = hz_to_cents(freq)
    freq_back = cents_to_hz(cents)
    error = abs(freq - freq_back)
    print(f"  {freq:8.1f} Hz → {cents:10.4f} cents → {freq_back:10.4f} Hz (error: {error:.6f})")
print()

# Test 4: hz_to_crepe_bin()
print("Test 4: hz_to_crepe_bin()")
print("-" * 40)
test_freqs_bin = [32.7, 65.4, 130.8, 261.6, 440.0, 880.0]
for freq in test_freqs_bin:
    bin_idx = hz_to_crepe_bin(freq)
    print(f"  {freq:8.1f} Hz → bin {bin_idx:3d}")
print()

# Test 5: crepe_bin_to_hz()
print("Test 5: crepe_bin_to_hz()")
print("-" * 40)
test_bins = [0, 50, 100, 150, 200, 250, 300, 359]
for bin_idx in test_bins:
    freq = crepe_bin_to_hz(bin_idx)
    print(f"  bin {bin_idx:3d} → {freq:10.4f} Hz")
print()

# Test 6: Round-trip Hz → Bin → Hz
print("Test 6: Round-trip Hz → Bin → Hz")
print("-" * 40)
for freq in [440.0, 880.0, 220.0, 100.0, 261.6]:
    bin_idx = hz_to_crepe_bin(freq)
    freq_back = crepe_bin_to_hz(bin_idx)
    error = abs(freq - freq_back)
    print(f"  {freq:8.1f} Hz → bin {bin_idx:3d} → {freq_back:10.4f} Hz (error: {error:.4f} Hz)")
print()

# Test 7: Octave relationships
print("Test 7: Octave relationships (should differ by 1200 cents)")
print("-" * 40)
base_freq = 55.0  # A1
for octave in range(5):
    freq = base_freq * (2 ** octave)
    cents = hz_to_cents(freq)
    print(f"  A{octave+1}: {freq:8.1f} Hz → {cents:10.4f} cents")
print()

# Test 8: Edge cases
print("Test 8: Edge cases")
print("-" * 40)
# Bin 0 and max bin
freq_bin0 = crepe_bin_to_hz(0)
freq_bin_max = crepe_bin_to_hz(CREPE_N_BINS - 1)
print(f"  Bin 0 → {freq_bin0:.4f} Hz")
print(f"  Bin {CREPE_N_BINS - 1} → {freq_bin_max:.4f} Hz")

# Very low and high frequencies
very_low = hz_to_crepe_bin(10.0)
very_high = hz_to_crepe_bin(5000.0)
print(f"  10 Hz → bin {very_low} (clipped)")
print(f"  5000 Hz → bin {very_high} (clipped)")

print("\n" + "=" * 60)
print("Tests Complete!")
print("=" * 60)