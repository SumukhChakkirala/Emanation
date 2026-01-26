"""
CREPE: A Convolutional Representation for Pitch Estimation
PyTorch implementation based on Kim et al. (2018)

Paper: https://arxiv.org/abs/1802.06182
Original TensorFlow implementation: https://github.com/marl/crepe

Architecture from Section 2.2 of the paper:
- Input: 1024 samples at 16kHz (64ms)
- 6 convolutional layers with 2D convolutions
- Capacity multiplier scales number of filters
- Output: 360 pitch classes (20 cents each, spanning C1 to B7)
"""

import math
import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Optional
import urllib.request
import urllib.error


# =============================================================================
# Helper Functions for Pitch Processing
# =============================================================================

def create_frames(audio, hop_length=160, frame_length=1024, center=True):
    """
    Create overlapping frames from audio signal
    
    Parameters
    ----------
    audio : np.ndarray
        Audio signal of shape (n_samples,)
    hop_length : int
        Hop size in samples (default: 160 = 10ms at 16kHz)
    frame_length : int
        Frame length in samples (default: 1024 = 64ms at 16kHz)
    center : bool
        If True, pad audio so frames are centered
    
    Returns
    -------
    np.ndarray
        Frames of shape (n_frames, frame_length)
    """
    audio = np.asarray(audio, dtype=np.float32)
    
    if center:
        # Pad audio to center frames
        audio = np.pad(audio, frame_length // 2, mode='constant', constant_values=0)
    
    # Number of frames
    n_frames = 1 + (len(audio) - frame_length) // hop_length
    
    if n_frames < 1:
        # If audio is too short, pad it
        audio = np.pad(audio, (0, frame_length - len(audio)), mode='constant')
        n_frames = 1
    
    # Create frames using stride tricks
    from numpy.lib.stride_tricks import as_strided
    frames = as_strided(
        audio,
        shape=(n_frames, frame_length),
        strides=(hop_length * audio.itemsize, audio.itemsize)
    )
    return frames.copy()


def normalize_frames(frames):
    """
    Normalize frames to zero mean and unit variance (per frame)
    
    Parameters
    ----------
    frames : np.ndarray
        Frames of shape (n_frames, frame_length)
    
    Returns
    -------
    np.ndarray
        Normalized frames
    """
    frames = frames.astype(np.float32)
    frames -= np.mean(frames, axis=1, keepdims=True)
    std = np.std(frames, axis=1, keepdims=True)
    std = np.clip(std, 1e-8, None)  # Avoid division by zero
    frames /= std
    return frames


def cents_to_frequency(cents):
    """
    Convert cents to frequency in Hz
    
    CREPE uses 360 bins spanning C1 (32.70 Hz) to B7 (3951.07 Hz)
    Each bin is 20 cents apart
    
    Parameters
    ----------
    cents : np.ndarray
        Pitch in cents
    
    Returns
    -------
    np.ndarray
        Frequency in Hz
    """
    # Reference: 10 Hz is 0 cents in CREPE's system
    # f = 10 * 2^(cents/1200)
    return 10.0 * (2.0 ** (cents / 1200.0))


def frequency_to_cents(frequency):
    """
    Convert frequency in Hz to cents
    
    Parameters
    ----------
    frequency : np.ndarray
        Frequency in Hz
    
    Returns
    -------
    np.ndarray
        Pitch in cents
    """
    # cents = 1200 * log2(f/10)
    return 1200.0 * np.log2(frequency / 10.0 + 1e-8)


def activation_to_cents(activation, center=None):
    """
    Convert activation to cents using weighted average around peak
    
    From CREPE paper: uses a weighted average of the bins
    around the peak to refine the pitch estimate
    
    Parameters
    ----------
    activation : np.ndarray
        Activation matrix of shape (n_frames, 360) or (360,)
    center : int or None
        Center bin for weighted average (if None, uses argmax)
    
    Returns
    -------
    np.ndarray or float
        Pitch in cents
    """
    # CREPE pitch range: C1 (32.70 Hz) to B7 (3951.07 Hz)
    # 360 bins at 20 cents each
    # C1 in cents relative to 10 Hz: 1200 * log2(32.70/10) ≈ 1997.38
    CENTS_PER_BIN = 20
    N_BINS = 360
    
    # Create cents mapping for all 360 bins
    # Bin 0 = C1 ≈ 1997.38 cents, Bin 359 = B7 ≈ 9177.38 cents
    cents_mapping = np.linspace(0, (N_BINS - 1) * CENTS_PER_BIN, N_BINS) + 1997.3794084376191
    
    activation = np.asarray(activation)
    
    if activation.ndim == 1:
        # Single frame
        if center is None:
            center = int(np.argmax(activation))
        
        # Weighted average of 9 bins around center (4 on each side)
        start = max(0, center - 4)
        end = min(N_BINS, center + 5)
        
        salience = activation[start:end]
        cents_vals = cents_mapping[start:end]
        
        weight_sum = np.sum(salience)
        if weight_sum <= 0:
            return np.nan
        
        return float(np.sum(salience * cents_vals) / weight_sum)
    else:
        # Multiple frames - process each
        return np.array([
            activation_to_cents(activation[i, :])
            for i in range(activation.shape[0])
        ])


# =============================================================================
# CREPE Model Architecture
# =============================================================================

class TFSameConv2d(nn.Module):
    """
    Conv2d layer that EXACTLY replicates TensorFlow's 'SAME' padding behavior
    
    TensorFlow's SAME padding with stride:
    - output_size = ceil(input_size / stride)
    - pad_total = max(0, (output_size - 1) * stride + kernel_size - input_size)
    - pad_before = pad_total // 2
    - pad_after = pad_total - pad_before
    - Convolution is applied WITH stride (not stride=1 then slice)
    
    This is crucial for exact replication of CREPE's behavior.
    """
    
    def __init__(self, in_channels, out_channels, kernel_size, stride):
        super(TFSameConv2d, self).__init__()
        self.kernel_size = kernel_size  # (height, width)
        self.stride = stride  # (stride_h, stride_w)
        
        # Conv2d with stride applied directly - padding computed dynamically
        self.conv = nn.Conv2d(
            in_channels, out_channels, kernel_size,
            stride=stride, padding=0, bias=True
        )
    
    def forward(self, x):
        # x shape: (batch, channels, height, width)
        input_h, input_w = x.shape[2], x.shape[3]
        
        # Calculate TF 'SAME' padding for height dimension
        output_h = math.ceil(input_h / self.stride[0])
        pad_h_total = max(0, (output_h - 1) * self.stride[0] + self.kernel_size[0] - input_h)
        pad_h_before = pad_h_total // 2
        pad_h_after = pad_h_total - pad_h_before
        
        # Width dimension (always 1 in CREPE, so no padding needed)
        output_w = math.ceil(input_w / self.stride[1])
        pad_w_total = max(0, (output_w - 1) * self.stride[1] + self.kernel_size[1] - input_w)
        pad_w_before = pad_w_total // 2
        pad_w_after = pad_w_total - pad_w_before
        
        # Apply padding: F.pad uses (left, right, top, bottom) order
        x = F.pad(x, (pad_w_before, pad_w_after, pad_h_before, pad_h_after))
        
        # Apply strided convolution (stride is in the conv layer itself)
        x = self.conv(x)
        
        return x


class CREPE(nn.Module):
    """
    CREPE: Convolutional Representation for Pitch Estimation
    
    From Kim et al. (2018): "CREPE: A Convolutional Representation for Pitch Estimation"
    
    Architecture (Table 1 in paper):
    - Input: 1024 samples (64ms at 16kHz), reshaped to (batch, 1, 1024, 1)
    - 6 conv blocks, each with: Conv2D -> ReLU -> BatchNorm -> MaxPool -> Dropout
    - Final dense layer with 360 outputs (pitch bins)
    - Sigmoid activation for pitch salience
    
    Capacity multiplier scales the number of filters:
    - 'tiny':   4 (total params ~1M)
    - 'small':  8 (total params ~4M)
    - 'medium': 16 (total params ~10M)
    - 'large':  24 (total params ~20M)
    - 'full':   32 (total params ~34M)
    
    Parameters
    ----------
    model_capacity : str
        One of 'tiny', 'small', 'medium', 'large', 'full'
    """
    
    def __init__(self, model_capacity='full'):
        super(CREPE, self).__init__()
        
        # Capacity multiplier from paper
        capacity_multiplier = {
            'tiny': 4,
            'small': 8,
            'medium': 16,
            'large': 24,
            'full': 32
        }[model_capacity]
        
        # Architecture from Table 1 in the paper
        # Base filter counts: [32, 4, 4, 4, 8, 16] * capacity_multiplier
        # For 'full' (32x): [1024, 128, 128, 128, 256, 512]
        filters = [n * capacity_multiplier for n in [32, 4, 4, 4, 8, 16]]
        
        # Kernel sizes (height x width): all are (width, 1) where width varies
        widths = [512, 64, 64, 64, 64, 64]
        
        # Strides: first layer has stride 4, rest have stride 1
        strides = [(4, 1), (1, 1), (1, 1), (1, 1), (1, 1), (1, 1)]
        
        # Build convolutional layers
        # EXACT order from original CREPE: Conv → BatchNorm → ReLU → MaxPool → Dropout
        self.conv_layers = nn.ModuleList()
        in_channels = 1
        
        for i, (f, w, s) in enumerate(zip(filters, widths, strides)):
            block = nn.Sequential(
                TFSameConv2d(in_channels, f, kernel_size=(w, 1), stride=s),
                # TensorFlow BatchNorm defaults: epsilon=0.001, momentum=0.99
                # PyTorch momentum = 1 - TF momentum, so 1 - 0.99 = 0.01
                nn.BatchNorm2d(f, eps=0.001, momentum=0.01),
                nn.ReLU(),
                nn.MaxPool2d(kernel_size=(2, 1), stride=(2, 1)),
                nn.Dropout(0.25)
            )
            self.conv_layers.append(block)
            in_channels = f
        
        # After 6 conv blocks with pooling:
        # Input: 1024 -> /4 (stride) -> 256 -> /2 (pool) -> 128
        # -> /2 -> 64 -> /2 -> 32 -> /2 -> 16 -> /2 -> 8 -> /2 -> 4
        # Final shape: (batch, 512, 4, 1) for 'full' capacity
        # Flatten: 512 * 4 = 2048
        
        self.fc_size = filters[-1] * 4  # 512 * 4 = 2048 for 'full'
        self.classifier = nn.Linear(self.fc_size, 360)
    
    def forward(self, x):
        """
        Forward pass
        
        Parameters
        ----------
        x : torch.Tensor
            Input tensor of shape (batch, 1024)
        
        Returns
        -------
        torch.Tensor
            Output tensor of shape (batch, 360) with pitch salience
        """
        # Reshape input: (batch, 1024) -> (batch, 1, 1024, 1)
        if x.dim() == 2:
            x = x.unsqueeze(1).unsqueeze(-1)
        elif x.dim() == 3:
            x = x.unsqueeze(-1)
        
        # Apply conv blocks
        for conv_block in self.conv_layers:
            x = conv_block(x)
        
        # Flatten: (batch, channels, height, 1) -> (batch, channels * height)
        x = x.view(x.size(0), -1)
        
        # Classifier with sigmoid
        x = self.classifier(x)
        x = torch.sigmoid(x)
        
        return x


# =============================================================================
# Pitch Prediction Function
# =============================================================================

def predict_pitch(audio, sr=16000, model=None, device='cpu',
                  step_size=10, center=True, batch_size=32):
    """
    Predict pitch from audio using CREPE
    
    Parameters
    ----------
    audio : np.ndarray
        Audio signal (mono)
    sr : int
        Sample rate (will resample to 16kHz if different)
    model : CREPE
        CREPE model (if None, creates a new 'full' model)
    device : str
        'cpu' or 'cuda'
    step_size : int
        Step size in milliseconds (default: 10ms for ~84% overlap)
    center : bool
        Center frames
    batch_size : int
        Batch size for inference
    
    Returns
    -------
    Tuple of (time, frequency, confidence, activation)
        time : np.ndarray - timestamps in seconds
        frequency : np.ndarray - pitch in Hz
        confidence : np.ndarray - confidence scores (max activation per frame)
        activation : np.ndarray - raw activation matrix (n_frames, 360)
    """
    # Create model if not provided
    if model is None:
        model = CREPE('full')
    
    model = model.to(device)
    model.eval()
    
    # Ensure audio is float32
    audio = np.asarray(audio, dtype=np.float32)
    
    # Resample to 16kHz if necessary
    if sr != 16000:
        from scipy import signal
        audio = signal.resample_poly(audio, 16000, sr).astype(np.float32)
    
    # Create overlapping frames
    hop_length = int(16000 * step_size / 1000)  # 10ms = 160 samples
    frames = create_frames(audio, hop_length=hop_length, center=center)
    frames = normalize_frames(frames)
    
    # Run inference in batches
    activations = []
    with torch.no_grad():
        for i in range(0, len(frames), batch_size):
            batch = frames[i:i + batch_size]
            batch_tensor = torch.from_numpy(batch).float().to(device)
            activation = model(batch_tensor)
            activations.append(activation.cpu().numpy())
    
    if len(activations) == 0:
        return np.array([]), np.array([]), np.array([]), np.zeros((0, 360))
    
    activation = np.vstack(activations)
    
    # Get confidence (max activation per frame)
    confidence = activation.max(axis=1)
    
    # Convert to frequency
    cents = activation_to_cents(activation)
    frequency = cents_to_frequency(cents)
    frequency[np.isnan(frequency)] = 0
    
    # Create timestamps
    time = np.arange(len(confidence)) * step_size / 1000.0
    
    return time, frequency, confidence, activation


# =============================================================================
# Weight Loading (placeholder for TensorFlow weight conversion)
# =============================================================================

def load_tf_weights(model, tf_model_path):
    """
    Load TensorFlow CREPE weights into PyTorch model
    
    Parameters
    ----------
    model : CREPE
        PyTorch CREPE model
    tf_model_path : str
        Path to TensorFlow model weights (.h5 file)
    """
    try:
        import h5py
    except ImportError:
        raise ImportError("h5py is required to load TensorFlow weights")
    
    def _pick(w, *names):
        for n in names:
            if n in w:
                return w[n]
        return None

    def _read_layer_weights(h5, layer_name):
        if layer_name not in h5:
            return {}
        g = h5[layer_name]
        if layer_name in g and hasattr(g[layer_name], 'keys'):
            g = g[layer_name]
        weights = {}
        for k in g.keys():
            obj = g[k]
            if hasattr(obj, 'shape'):
                weights[k] = np.array(obj)
        return weights

    print(f"Loading TensorFlow weights from {tf_model_path}...")

    with h5py.File(tf_model_path, 'r') as h5:
        conv_names = [f'conv{i}' for i in range(1, 7)]
        bn_names = [f'conv{i}-BN' for i in range(1, 7)]

        for i in range(6):
            conv_w = _read_layer_weights(h5, conv_names[i])
            bn_w = _read_layer_weights(h5, bn_names[i])

            kernel = _pick(conv_w, 'kernel:0', 'kernel')
            bias = _pick(conv_w, 'bias:0', 'bias')
            if kernel is None:
                raise ValueError(f"Missing conv kernel for layer '{conv_names[i]}'")

            # TF: (kh, kw, in_ch, out_ch) -> PyTorch: (out_ch, in_ch, kh, kw)
            kernel_pt = np.transpose(kernel, (3, 2, 0, 1))
            pt_conv = model.conv_layers[i][0].conv
            pt_conv.weight.data.copy_(torch.from_numpy(kernel_pt))
            if bias is not None:
                pt_conv.bias.data.copy_(torch.from_numpy(bias))

            gamma = _pick(bn_w, 'gamma:0', 'gamma')
            beta = _pick(bn_w, 'beta:0', 'beta')
            moving_mean = _pick(bn_w, 'moving_mean:0', 'moving_mean')
            moving_var = _pick(bn_w, 'moving_variance:0', 'moving_variance')

            pt_bn = model.conv_layers[i][1]
            pt_bn.weight.data.copy_(torch.from_numpy(gamma))
            pt_bn.bias.data.copy_(torch.from_numpy(beta))
            pt_bn.running_mean.data.copy_(torch.from_numpy(moving_mean))
            pt_bn.running_var.data.copy_(torch.from_numpy(moving_var))

        # Dense classifier
        dense_w = _read_layer_weights(h5, 'classifier')
        kernel = _pick(dense_w, 'kernel:0', 'kernel')
        bias = _pick(dense_w, 'bias:0', 'bias')
        model.classifier.weight.data.copy_(torch.from_numpy(kernel.T))
        model.classifier.bias.data.copy_(torch.from_numpy(bias))

    model.eval()
    print("✅ TensorFlow weights loaded")
    return model


def get_crepe_weights_path(model_capacity='full'):
    """
    Find CREPE weights from the installed crepe package or local cache
    
    Parameters
    ----------
    model_capacity : str
        Model capacity: 'tiny', 'small', 'medium', 'large', or 'full'
    
    Returns
    -------
    str
        Path to weights file (.h5 from crepe package)
    """
    # First check for cached PyTorch weights
    cache_dir = os.path.join(os.path.dirname(__file__), '..', 'crepe_weights')
    pth_path = os.path.join(cache_dir, f"crepe_{model_capacity}.pth")
    if os.path.exists(pth_path):
        return pth_path
    
    # Check for .h5 file from crepe package
    # We need to import the real crepe package, not this file
    import importlib.util
    import sys
    
    # Find the real crepe package (not this file)
    for path in sys.path:
        if 'site-packages' in path:
            crepe_init = os.path.join(path, 'crepe', '__init__.py')
            if os.path.exists(crepe_init):
                h5_path = os.path.join(path, 'crepe', f"model-{model_capacity}.h5")
                if os.path.exists(h5_path):
                    print(f"✅ Found CREPE weights at: {h5_path}")
                    return h5_path
    
    # Check common locations
    possible_paths = [
        f"model-{model_capacity}.h5",
        f"crepe_weights/model-{model_capacity}.h5",
        f"crepe_weights/crepe_{model_capacity}.pth",
        os.path.join(os.path.dirname(__file__), f"model-{model_capacity}.h5"),
        os.path.join(os.path.dirname(__file__), '..', 'crepe_weights', f"model-{model_capacity}.h5"),
    ]
    
    for path in possible_paths:
        if os.path.exists(path):
            return path
    
    raise FileNotFoundError(
        f"CREPE weights not found. Install crepe package: pip install crepe"
    )


def load_crepe_model(model_capacity='full', device='cpu'):
    """
    Load CREPE model with pre-trained weights
    
    Parameters
    ----------
    model_capacity : str
        Model capacity: 'tiny', 'small', 'medium', 'large', or 'full'
    device : str
        'cpu' or 'cuda'
    
    Returns
    -------
    CREPE
        Model with loaded weights
    """
    model = CREPE(model_capacity)
    
    # Check for cached PyTorch weights first
    cache_dir = os.path.join(os.path.dirname(__file__), '..', 'crepe_weights')
    os.makedirs(cache_dir, exist_ok=True)
    pth_path = os.path.join(cache_dir, f"crepe_{model_capacity}.pth")
    
    if os.path.exists(pth_path):
        print(f"📂 Loading cached PyTorch weights from {pth_path}")
        return load_pytorch_weights(model, pth_path, device)
    
    # Find and load .h5 weights from crepe package
    weights_path = get_crepe_weights_path(model_capacity)
    
    if weights_path.endswith('.h5'):
        print(f"📂 Loading TensorFlow weights from {weights_path}")
        model = load_tf_weights(model, weights_path)
        
        # Cache as PyTorch format for faster loading next time
        torch.save(model.state_dict(), pth_path)
        print(f"💾 Cached PyTorch weights to {pth_path}")
    else:
        model = load_pytorch_weights(model, weights_path, device)
    
    return model.to(device)


def load_pytorch_weights(model, path, device='cpu'):
    """Load PyTorch weights from .pth file"""
    if not os.path.exists(path):
        raise FileNotFoundError(f"Weights not found: {path}")
    print(f"Loading weights from {path}...")
    state = torch.load(path, map_location=device, weights_only=True)
    model.load_state_dict(state)
    model.to(device).eval()
    print("✅ Weights loaded successfully")
    return model


# =============================================================================
# Main: Test with pickle dataset
# =============================================================================

if __name__ == "__main__":
    import pickle
    
    print("=" * 70)
    print("CREPE: Convolutional Representation for Pitch Estimation")
    print("PyTorch Implementation (Kim et al., 2018)")
    print("=" * 70)
    
    # Create model
    model = CREPE('full')
    n_params = sum(p.numel() for p in model.parameters())
    print(f"\n✅ Model created: 'full' capacity")
    print(f"   Parameters: {n_params:,}")
    
    # Test model with dummy input to verify architecture
    print("\n🔧 Verifying architecture...")
    dummy_input = torch.randn(1, 1024)
    with torch.no_grad():
        model.eval()
        dummy_output = model(dummy_input)
    print(f"   Input shape:  {dummy_input.shape}")
    print(f"   Output shape: {dummy_output.shape}")
    assert dummy_output.shape == (1, 360), f"Expected (1, 360), got {dummy_output.shape}"
    print("   ✓ Architecture verified!")
    
    # Load weights using load_crepe_model
    weights_loaded = False
    try:
        print("\n📥 Loading CREPE weights...")
        model = load_crepe_model('full', device='cpu')
        weights_loaded = True
        print("✅ Model ready with pre-trained weights!")
    except Exception as e:
        print(f"⚠️  Failed to load weights: {e}")
        print("   Using random initialization.")
        print("\n   To get weights, install: pip install crepe h5py")

    
    # Load dataset from pickle file
    dataset_path = r'IQData\iq_dict_SNR_20_toMinus40_dc_0_ptsecsdata_1_Fh_220_kHz.pkl'
    print(f"\n📂 Loading dataset from {dataset_path}...")
    
    try:
        # Custom unpickler to handle NumPy version mismatch
        class NumpyUnpickler(pickle.Unpickler):
            def find_class(self, module, name):
                # Redirect numpy._core to numpy.core for older NumPy versions
                if module.startswith('numpy._core'):
                    module = module.replace('numpy._core', 'numpy.core')
                return super().find_class(module, name)
        
        with open(dataset_path, 'rb') as f:
            dataset = NumpyUnpickler(f).load()
        
        print(f"   Dataset loaded successfully!")
        print(f"   Type: {type(dataset)}")
        
        if isinstance(dataset, dict):
            print(f"   Keys: {list(dataset.keys())}")
            
            # Handle the specific structure: {'dataset': [...], 'metadata': {...}, 'format_info': {...}}
            if 'dataset' in dataset:
                data_list = dataset['dataset']
                print(f"   Dataset contains {len(data_list)} samples")
                
                # Collect all audio samples
                all_audio = []
                for sample in data_list:
                    if isinstance(sample, (list, tuple)) and len(sample) >= 1:
                        all_audio.append(np.asarray(sample[0], dtype=np.float32))
                    elif isinstance(sample, np.ndarray):
                        all_audio.append(sample.astype(np.float32))
                    else:
                        all_audio.append(np.asarray(sample, dtype=np.float32))
                
                # Stack all samples: each is 1024 samples (one CREPE frame)
                audio_frames = np.vstack(all_audio)
                print(f"   Total frames: {audio_frames.shape}")
                
                # Flatten to continuous audio or process as frames directly
                audio_data = audio_frames  # Keep as frames for direct processing
            elif 'audio' in dataset:
                audio_data = np.asarray(dataset['audio'], dtype=np.float32)
            elif 'data' in dataset:
                audio_data = np.asarray(dataset['data'], dtype=np.float32)
            elif 'x' in dataset:
                audio_data = np.asarray(dataset['x'], dtype=np.float32)
            else:
                # Use first value
                first_key = list(dataset.keys())[0]
                audio_data = np.asarray(dataset[first_key], dtype=np.float32)
        elif isinstance(dataset, (list, tuple)):
            audio_data = np.asarray(dataset[0], dtype=np.float32)
        elif isinstance(dataset, np.ndarray):
            audio_data = dataset.astype(np.float32)
        else:
            raise ValueError(f"Unknown dataset type: {type(dataset)}")
        
        # Handle multi-dimensional arrays
        if audio_data.ndim > 1:
            audio_data = audio_data.flatten()
        
        print(f"   Audio shape: {audio_data.shape}")
        print(f"   Duration: {len(audio_data) / 16000:.2f} seconds (assuming 16kHz)")
        
    except FileNotFoundError:
        print(f"   ⚠️ Dataset file not found!")
        print("   Using dummy audio for testing...")
        audio_data = np.random.randn(16000 * 2).astype(np.float32)
    except Exception as e:
        print(f"   ⚠️ Error loading dataset: {e}")
        print("   Using dummy audio for testing...")
        audio_data = np.random.randn(16000 * 2).astype(np.float32)
    
    # Run pitch prediction
    print("\n🎵 Running pitch prediction...")
    
    # Check if audio_data is already frames (2D) or continuous audio (1D)
    if audio_data.ndim == 2 and audio_data.shape[1] == 1024:
        # Already in frames format - process directly
        print(f"   Processing {len(audio_data)} pre-framed samples...")
        frames = normalize_frames(audio_data)
        
        model.eval()
        activations = []
        with torch.no_grad():
            for i in range(0, len(frames), 32):
                batch = frames[i:i + 32]
                batch_tensor = torch.from_numpy(batch).float()
                activation = model(batch_tensor)
                activations.append(activation.cpu().numpy())
        
        activation = np.vstack(activations)
        confidence = activation.max(axis=1)
        cents = activation_to_cents(activation)
        frequency = cents_to_frequency(cents)
        frequency[np.isnan(frequency)] = 0
        time = np.arange(len(confidence)) * 10 / 1000.0  # Assume 10ms step
    else:
        # Continuous audio - use predict_pitch
        time, frequency, confidence, activation = predict_pitch(
            audio_data,
            sr=16000,
            model=model,
            step_size=10  # 10ms hop
        )
    
    print(f"\n📊 Results:")
    print(f"   Activation shape: {activation.shape}")
    print(f"   Time points: {len(time)}")
    if len(time) > 0:
        print(f"   Time range: {time[0]:.3f}s - {time[-1]:.3f}s")
    
    if np.any(frequency > 0):
        valid_freq = frequency[frequency > 0]
        print(f"   Frequency range: {valid_freq.min():.1f} Hz - {valid_freq.max():.1f} Hz")
        print(f"   Mean confidence: {confidence.mean():.3f}")
    else:
        print("   (No valid pitch detected - model has random weights)")
    
    if not weights_loaded:
        print("\n⚠️  Note: Results are from random weights.")
        print("   For accurate pitch, convert official TF weights or download .pth")
    
    print("\n📐 CREPE Architecture Summary (EXACT replica):")
    print("   • Input: 1024 samples (64ms @ 16kHz)")
    print("   • 6 Conv blocks: Conv2D → BatchNorm → ReLU → MaxPool → Dropout")
    print("   • Filters: 1024, 128, 128, 128, 256, 512")
    print("   • Kernels: 512×1, 64×1, 64×1, 64×1, 64×1, 64×1")
    print("   • Output: 360 bins (20 cents each, C1-B7)")
    print("=" * 70)