"""
CREPE Replication Experiment - Complete Pipeline

This script provides a complete workflow for replicating CREPE:
1. Generate synthetic data with varied pitches
2. Train the model
3. Evaluate with RPA/RCA metrics
4. Visualize results

Usage:
    python run_experiment.py --generate   # Generate data only
    python run_experiment.py --train      # Train only (data must exist)
    python run_experiment.py --all        # Full pipeline
"""

import argparse
import os
import sys


def main():
    parser = argparse.ArgumentParser(description='CREPE Replication Experiment')
    parser.add_argument('--generate', action='store_true', help='Generate training data')
    parser.add_argument('--train', action='store_true', help='Train the model')
    parser.add_argument('--evaluate', action='store_true', help='Evaluate trained model')
    parser.add_argument('--all', action='store_true', help='Run full pipeline')
    
    args = parser.parse_args()
    
    # Default to full pipeline if no args
    if not (args.generate or args.train or args.evaluate or args.all):
        args.all = True
    
    print("=" * 80)
    print("CREPE Replication Experiment")
    print("=" * 80)
    
    # Step 1: Generate Data
    if args.generate or args.all:
        print("\n" + "=" * 80)
        print("STEP 1: Generating Training Data")
        print("=" * 80)
        
        from generate_crepe_data import generate_crepe_dataset_dense, visualize_dataset
        import numpy as np
        
        OUTPUT_PATH = './IQData/iq_dict_crepe_dirac_comb.pkl'
        
        # Use every 2nd bin for faster generation (180 bins)
        bins_to_generate = list(range(0, 360, 2))
        
        iq_dict = generate_crepe_dataset_dense(
            output_path=OUTPUT_PATH,
            bins_to_generate=bins_to_generate,
            snr_list=list(np.arange(20, -22, -2)),  # 20 to -20 dB
            samples_per_bin_snr=5,
            duty_cycle=0.1,
            seed=42
        )
        
        visualize_dataset(iq_dict, n_samples=5)
        print("✓ Data generation complete!")
    
    # Step 2: Train Model
    if args.train or args.all:
        print("\n" + "=" * 80)
        print("STEP 2: Training CREPE Model")
        print("=" * 80)
        
        from train_crepe import main as train_main
        train_main()
        print("✓ Training complete!")
    
    # Step 3: Evaluate
    if args.evaluate or args.all:
        print("\n" + "=" * 80)
        print("STEP 3: Final Evaluation")
        print("=" * 80)
        
        import torch
        import pickle
        import numpy as np
        from train_crepe import CREPE, CREPEDataset, evaluate, CREPE_FRAME_LENGTH
        from torch.utils.data import DataLoader
        import torch.nn as nn
        
        # Load best model
        model_path = './models_crepe/crepe_best.pth'
        if not os.path.exists(model_path):
            print(f"❌ Model not found at {model_path}. Run training first.")
            return
        
        checkpoint = torch.load(model_path, map_location='cpu', weights_only=False)
        config = checkpoint['config']
        
        print(f"Loaded model from epoch {checkpoint['epoch']}")
        print(f"  Training RPA (50 cents): {checkpoint['rpa_50']:.2f}%")
        print(f"  Training RCA: {checkpoint['rca']:.2f}%")
        
        # Load data
        with open(config['data_path'], 'rb') as f:
            iq_dict = pickle.load(f)
        
        # Create test dataset (use lower SNR for harder evaluation)
        test_dataset = CREPEDataset(
            iq_dict=iq_dict,
            snr_range=(-5, 5),  # Harder conditions
            target_length=CREPE_FRAME_LENGTH,
            gaussian_sigma=config['gaussian_sigma']
        )
        test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)
        
        # Create and load model
        model = CREPE(capacity_multiplier=config['capacity'], dropout=0.25)
        model.load_state_dict(checkpoint['model_state_dict'])
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        model = model.to(device)
        
        # Evaluate
        criterion = nn.BCEWithLogitsLoss()
        test_loss, rpa_50, rpa_25, rca, mean_error = evaluate(model, test_loader, criterion, device)
        
        print("\n" + "=" * 80)
        print("📊 Test Results (SNR range: -5 to 5 dB)")
        print("=" * 80)
        print(f"  Test Loss:          {test_loss:.6f}")
        print(f"  RPA (50 cents):     {rpa_50:.2f}%")
        print(f"  RPA (25 cents):     {rpa_25:.2f}%")
        print(f"  RCA:                {rca:.2f}%")
        print(f"  Mean Pitch Error:   {mean_error:.1f} cents")
        
        print("\n✓ Evaluation complete!")
    
    print("\n" + "=" * 80)
    print("🎉 Experiment Complete!")
    print("=" * 80)
    
    print("\nSummary:")
    print("  - Data: ./IQData/iq_dict_crepe_dirac_comb.pkl")
    print("  - Model: ./models_crepe/crepe_best.pth")
    print("  - History: ./models_crepe/training_history.pkl")
    print("  - Visualization: ./dataset_visualization.png")


if __name__ == "__main__":
    main()
