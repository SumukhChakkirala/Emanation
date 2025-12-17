#!/usr/bin/env python3
"""
Convert existing IQ dictionary to CREPE dataset format
Loads pre-generated synthetic Dirac comb IQ data and reformats for CREPE training
"""

import numpy as np
import pickle
import argparse
import librosa

import os 

class IQToCREPEConverter:
    """Convert IQ dictionary to CREPE dataset"""
    
    def __init__(self):
        """Initialize CREPE parameters"""
        self.crepe_sr = 16000  # CREPE standard sample rate
        self.crepe_samples = 1024  # 1024 samples = 64ms @ 16kHz
        self.cent_bins = 360  # Output: 360 cent bins
        self.fmin = 10  # Hz (reference for cent calculation)
        self.fmax = 2000  # Hz
    
    def hz_to_cents(self, frequency_hz):
        """
        Convert frequency in Hz to cent bin representation
        Returns array of length 360 with Gaussian peak at true pitch
        """
        if frequency_hz <= 0:
            return np.zeros(self.cent_bins)
        
        # Convert frequency to cents (relative to 1 Hz reference)
        cents = 1200 * np.log2(frequency_hz)
        
        # Map to [0, cent_bins) range
        cent_offset = 1200 * np.log2(self.fmin)
        cents_normalized = cents + cent_offset
        
        # Create Gaussian distribution around true pitch
        bin_width = (1200 * np.log2(self.fmax / self.fmin)) / self.cent_bins
        bin_idx = cents_normalized / bin_width
        
        # Create output: Gaussian centered at true pitch
        cent_output = np.zeros(self.cent_bins)
        
        if 0 <= bin_idx < self.cent_bins:
            # Gaussian distribution (width ~25 cents)
            gaussian_width = 2.0  # bins
            for i in range(self.cent_bins):
                cent_output[i] = np.exp(-0.5 * ((i - bin_idx) / gaussian_width) ** 2)
            cent_output /= np.max(cent_output)  # Normalize
        
        return cent_output
    
    def convert_iq_dict_to_crepe(self, iq_dict, iq_sample_rate, fundamental_hz, 
                                  num_windows_per_signal=5):
        """
        Convert IQ dictionary to CREPE dataset
        
        Args:
            iq_dict: Dict with keys like "SNR_0", "SNR_-10", etc.
            iq_sample_rate: Original sample rate (e.g., 25 MHz)
            fundamental_hz: Known fundamental frequency (e.g., 220 kHz)
            num_windows_per_signal: How many 1024-sample windows to extract per IQ signal
        
        Returns:
            dataset: List of (x, y) tuples where:
                     x: (1024,) float32 audio at 16 kHz, normalized [-1, 1]
                     y: (361,) float32 [360 cent bins + 1 confidence]
            metadata: Metadata about the dataset
        """
        dataset = []
        metadata = {'samples': []}
        
        print(f"Converting IQ dictionary to CREPE format...")
        print(f"  Original sample rate: {iq_sample_rate/1e6:.1f} MHz")
        print(f"  Target sample rate: {self.crepe_sr} kHz")
        print(f"  Fundamental frequency: {fundamental_hz/1e3:.1f} kHz")
        print(f"  Windows per signal: {num_windows_per_signal}\n")
        
        # Sort SNR values numerically
        snr_keys = sorted(iq_dict.keys(), key=lambda x: float(x.split('_')[1]))
        
        for snr_key in snr_keys:
            iq_signal = np.asarray(iq_dict[snr_key])
            snr_value = float(snr_key.split('_')[1])
            
            print(f"Processing {snr_key} (shape {iq_signal.shape})...", end='', flush=True)
            
            # Take magnitude if complex
            if np.iscomplexobj(iq_signal):
                signal_mag = np.abs(iq_signal)
            else:
                signal_mag = iq_signal
            
            # Resample to 16 kHz
            signal_resampled = librosa.resample(
                signal_mag, 
                orig_sr=int(iq_sample_rate), 
                target_sr=self.crepe_sr
            )
            
            # Extract multiple windows from this signal
            if len(signal_resampled) >= self.crepe_samples:
                max_start = len(signal_resampled) - self.crepe_samples + 1
                
                if num_windows_per_signal > 1:
                    # Evenly space windows across the signal
                    window_indices = np.linspace(0, max_start-1, num_windows_per_signal, dtype=int)
                else:
                    window_indices = [0]
                
                for win_idx in window_indices:
                    x = signal_resampled[win_idx:win_idx + self.crepe_samples].copy()
                    
                    # Normalize to [-1, 1]
                    x_max = np.max(np.abs(x))
                    if x_max > 0:
                        x = x / x_max * 0.95
                    x = x.astype(np.float32)
                    
                    # Ground truth pitch after resampling
                    # Fundamental frequency is downsampled: F_h * (16kHz / 25MHz)
                    pitch_hz = fundamental_hz * (self.crepe_sr / iq_sample_rate)
                    
                    # Generate cent output (360 bins + 1 confidence)
                    y_cents = self.hz_to_cents(pitch_hz)
                    y = np.concatenate([y_cents, [1.0]]).astype(np.float32)  # Shape: (361,)
                    
                    dataset.append((x, y))
                    metadata['samples'].append({
                        'snr_db': snr_value,
                        'pitch_hz': pitch_hz,
                        'source': snr_key
                    })
            else:
                print(f" WARNING: Signal too short ({len(signal_resampled)} < {self.crepe_samples})")
                continue
            
            print(f" ✓ {len(window_indices)} windows")
        
        print(f"\nDataset created!")
        print(f"  Total samples: {len(dataset)}")
        print(f"  Sample 0: x={dataset[0][0].shape}, y={dataset[0][1].shape}")
        print(f"  x range: [{dataset[0][0].min():.3f}, {dataset[0][0].max():.3f}]")
        print(f"  y range: [{dataset[0][1].min():.3f}, {dataset[0][1].max():.3f}]")
        
        return dataset, metadata
    
    def save_dataset(self, dataset, metadata, output_file):
        """Save to pickle file"""
        output_dir = os.path.dirname(output_file) if os.path.dirname(output_file) else '.'
        os.makedirs(output_dir, exist_ok=True)
        
        with open(output_file, 'wb') as f:
            pickle.dump({
                'dataset': dataset,
                'metadata': metadata,
                'format_info': {
                    'input_samples': self.crepe_samples,
                    'input_sample_rate': self.crepe_sr,
                    'output_bins': self.cent_bins,
                    'description': 'Dataset of (audio, pitch_distribution) tuples for CREPE training'
                }
            }, f)
        
        print(f"\n✓ Saved to: {output_file}")
        
        # Calculate memory usage
        mem_mb = len(dataset) * (self.crepe_samples * 4 + 361 * 4) / (1024**2)
        print(f"  Memory usage: ~{mem_mb:.1f} MB")


def main():
    parser = argparse.ArgumentParser(
        description='Convert existing IQ dictionary pickle to CREPE dataset format'
    )
    parser.add_argument('iq_pkl', 
                       help='Input IQ pickle file (e.g., IQData/iq_dict_SNR_20_toMinus40_dc_5_ptsecsdata_1_Fh_220_kHz.pkl)')
    parser.add_argument('--output', type=str, default='crepe_dataset.pkl',
                       help='Output CREPE dataset pickle (default: crepe_dataset.pkl)')
    parser.add_argument('--windows-per-snr', type=int, default=5,
                       help='Number of 1024-sample windows to extract per SNR level (default: 5)')
    parser.add_argument('--fs', type=float, default=25e6,
                       help='Original sample rate in Hz (default: 25 MHz)')
    parser.add_argument('--fundamental', type=float, default=220e3,
                       help='Fundamental frequency in Hz (default: 220 kHz)')
    
    args = parser.parse_args()
    
    # Load IQ dictionary
    print(f"Loading {args.iq_pkl}...")
    with open(args.iq_pkl, 'rb') as f:
        iq_dict = pickle.load(f)
    print(f"Loaded {len(iq_dict)} IQ signals\n")
    
    # Convert
    converter = IQToCREPEConverter()
    dataset, metadata = converter.convert_iq_dict_to_crepe(
        iq_dict,
        iq_sample_rate=args.fs,
        fundamental_hz=args.fundamental,
        num_windows_per_signal=args.windows_per_snr
    )
    
    # Save
    converter.save_dataset(dataset, metadata, args.output)


if __name__ == '__main__':
    main()