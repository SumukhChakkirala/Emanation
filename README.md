# Emanation

**Pitch Estimation for RF Signals using Deep Learning**

This project aims to estimate fundamental frequencies (pitch) from RF emanation signals. It uses a Dirac comb + rectangular pulse approach to generate synthetic training data that mimics real-world electromagnetic emanations. Currently using a CREPE-based neural network for research 

## 📋 Table of Contents

- [Overview](#overview)
- [Features](#features)
- [Installation](#installation)
- [Quick Start](#quick-start)
- [Project Structure](#project-structure)
- [Usage](#usage)
- [Results](#results)
- [References](#references)

## 🎯 Overview

This project:

1. **Generates synthetic RF signals** using Dirac comb convolved with rectangular pulses
2. **Trains a CREPE-style CNN** for pitch estimation across 360 frequency bins
3. **Evaluates performance** using Raw Pitch Accuracy (RPA) and Raw Chroma Accuracy (RCA)

### Signal Generation Approach

The signal generation follows this pipeline:
```
Rectangular Pulse × Dirac Comb → Pulse Train → + Complex Noise → IQ Signal
```

- **Dirac Comb**: Impulse train at period T_h = 1/F_h
- **Rectangular Pulse**: Width T = duty_cycle × T_h (default 10%)
- **Convolution**: Creates periodic pulse train with harmonics
- **Complex Noise**: AWGN added as I + jQ components

## ✨ Features

- **CREPE Architecture**: 6-layer CNN with 360 pitch bins (32.7 Hz - 1975 Hz)
- **16 kHz Sampling Rate**: Compatible with audio-domain processing
- **1024-sample Frames**: 64 ms analysis windows
- **Gaussian Label Smoothing**: 25-cent standard deviation as per CREPE paper
- **Configurable SNR**: Train with various noise levels (default: 15-20 dB)

## 🚀 Installation

### Prerequisites

- Python 3.9+ / 3.12.6 (preferrably)
- pip or conda

### Option 1: Using venv (Recommended)

```bash
# Clone the repository
git clone https://github.com/SumukhChakkirala/Emanation.git
cd Emanation

# Create virtual environment
python -m venv .venv

# Activate virtual environment
# On Windows:
.\.venv\Scripts\Activate.ps1
# On macOS/Linux:
source .venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

### Option 2: Using Conda

```bash
# Clone the repository
git clone https://github.com/SumukhChakkirala/Emanation.git
cd Emanation

# Create conda environment
conda create -n emanation python=3.10
conda activate emanation

# Install PyTorch (with CUDA if available)
conda install pytorch torchvision torchaudio pytorch-cuda=11.8 -c pytorch -c nvidia
# Or for CPU only:
# conda install pytorch torchvision torchaudio cpuonly -c pytorch

# Install other dependencies
pip install -r requirements.txt
```

### Verify Installation

```bash
python -c "import torch; print(f'PyTorch: {torch.__version__}'); print(f'CUDA: {torch.cuda.is_available()}')"
```

## ⚡ Quick Start

### Run Full Pipeline

```bash
cd programs/experiments
python run_experiment.py --all
```

This will:
1. Generate synthetic training data (~10,800 or more samples)
2. Train the CREPE model for 30 epochs
3. Evaluate on test set

### Individual Steps

```bash
# Generate data only
python run_experiment.py --generate

# Train only (data must exist)
python run_experiment.py --train

# Evaluate only (model must exist)
python run_experiment.py --evaluate
```

## 📁 Project Structure

```
Emanation/
├── programs/
│   ├── DiracCombPlots.py          # Original signal generation & visualization
│   ├── experiments/
│   │   ├── generate_crepe_data.py # Synthetic dataset generation
│   │   ├── train_crepe.py         # Model training script
│   │   ├── run_experiment.py      # Full pipeline runner
│   │   ├── IQData/                # Generated datasets (.pkl)
│   │   └── models_crepe/          # Saved models & training history
│   └── ...
├── crepe_weights/                  # Pre-trained CREPE weights
├── models_rf/                      # RF model checkpoints
├── Results/                        # Experiment results
├── requirements.txt                # Python dependencies
└── README.md                       # This file
```

## 📖 Usage

### Generating Custom Data

```python
from generate_crepe_data import generate_crepe_dataset_dense

iq_dict = generate_crepe_dataset_dense(
    output_path='./IQData/my_dataset.pkl',
    bins_to_generate=list(range(0, 360, 1)), 
    snr_list=list(range(15, 21)),              # 15-20 dB SNR
    samples_per_bin_snr=10,                    # 10 augmentations
    duty_cycle=0.1,                            # 10% duty cycle
    seed=42
)
```

### Training with Custom Config

```python
from train_crepe import main

# Modify config in train_crepe.py:
config = {
    'data_path': './IQData/iq_dict_crepe_dirac_comb.pkl',
    'batch_size': 32,
    'epochs': 50,
    'lr': 0.0002,
    'dropout': 0.25,
    'gaussian_sigma': 1.25,
    'device': 'cuda',
    'save_dir': './models_crepe/',
}
```

### Loading a Trained Model

```python
import torch
from train_crepe import CREPE

# Load model
model = CREPE(dropout=0.25)
checkpoint = torch.load('./models_crepe/crepe_best.pth', weights_only=False)
model.load_state_dict(checkpoint['model_state_dict'])
model.eval()

# Inference
signal = torch.randn(1, 1, 1024)  # (batch, channels, samples)
logits = model(signal)
predicted_bin = torch.argmax(logits, dim=1)
```

## 📊 Results

### Training Metrics

| Metric | Value |
|--------|-------|
| RPA (50 cents) | ~33% |
| RPA (25 cents) | ~12% |
| RCA | ~34% |
| Mean Error | ~370 cents |


## 🔧 Configuration

### Key Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `CREPE_FS` | 16000 | Sampling rate (Hz) |
| `CREPE_FRAME_LENGTH` | 1024 | Samples per frame |
| `CREPE_N_BINS` | 360 | Number of pitch bins |
| `duty_cycle` | 0.1 | Pulse duty cycle (10%) |
| `snr_range` | (15, 20) | SNR in dB |
| `gaussian_sigma` | 1.25 | Label smoothing (bins) |

## 📚 References

1. **CREPE Paper**: Kim, J. W., et al. "CREPE: A Convolutional Representation for Pitch Estimation." ICASSP 2018.

## 📄 License

This project is for educational and research purposes.

## 👤 Author

Sumukh Chakkirala - [@SumukhChakkirala](https://github.com/SumukhChakkirala)
