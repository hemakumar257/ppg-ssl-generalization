"""
Phase-2: Advanced Preprocessing Pipeline Execution Script

This script runs the complete preprocessing pipeline on all three datasets:
- PPG-DaLiA (wearable PPG)
- WESAD (wrist BVP as PPG)
- BIDMC (clinical PPG)

Output: Preprocessed data ready for Phase-3 (CNN and CNN-LSTM modeling)
"""

import sys
import os
from pathlib import Path

# Add preprocessing module to path
sys.path.insert(0, str(Path(__file__).parent))

from preprocessing.pipeline import PreprocessingPipeline


def main():
    """
    Run complete preprocessing pipeline.
    """
    print("\n" + "="*80)
    print(" "*20 + "PHASE-2: ADVANCED PREPROCESSING PIPELINE")
    print("="*80)
    
    # Configuration (optimized for MAE reduction)
    config = {
        'target_fs': 64.0,              # Target sampling rate (Hz)
        'window_duration': 10.0,        # Window duration (seconds)
        'overlap': 0.5,                 # 50% overlap for data augmentation
        'quality_threshold': 0.5,       # Minimum quality score (0-1)
        'normalization_method': 'zscore',  # Per-window Z-score normalization
        'train_ratio': 0.7,             # 70% training data
        'val_ratio': 0.15,              # 15% validation data (15% test)
        'output_dir': 'preprocessed_data',
        'random_seed': 42               # For reproducibility
    }
    
    print("\nPipeline Configuration:")
    print("-" * 80)
    for key, value in config.items():
        print(f"  {key:25s}: {value}")
    print("-" * 80)
    
    # Initialize pipeline
    pipeline = PreprocessingPipeline(**config)
    
    # Select datasets to process
    datasets_to_process = ['ppg_dalia', 'wesad', 'bidmc']
    
    print(f"\nDatasets to process: {', '.join([d.upper() for d in datasets_to_process])}")
    print("\nStarting preprocessing...")
    
    # Run pipeline
    try:
        pipeline.run(datasets=datasets_to_process)
        
        print("\n" + "="*80)
        print(" "*25 + "[OK] PREPROCESSING COMPLETE!")
        print("="*80)
        
        print(f"\nPreprocessed data saved to: {config['output_dir']}/")
        print("\nData format:")
        print("  - Shape: (n_samples, 640, 1) for CNN input")
        print("  - Window: 10 seconds @ 64 Hz = 640 samples")
        print("  - Normalization: Per-window Z-score (mean=0, std=1)")
        print("  - Splits: Train/Val/Test = 70%/15%/15%")
        
        print("\nReady for Phase-3: CNN and CNN-LSTM modeling!")
        
    except Exception as e:
        print(f"\n❌ ERROR: Preprocessing failed: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    return 0


if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)
