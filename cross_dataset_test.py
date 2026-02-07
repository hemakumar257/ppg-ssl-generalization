import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import numpy as np
from pathlib import Path
import pandas as pd
from models import CNNBaseline, CNNLSTMHybrid
from train import PPGDataset, evaluate

def cross_dataset_benchmark():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    datasets = ['ppg_dalia', 'wesad', 'bidmc']
    
    # Define models to compare: (Model Type, File Suffix, Display Name)
    model_configs = [
        ('cnn', 'best_model_cnn', 'Supervised CNN'),
        ('cnn', 'best_model_ssl_ft', 'SSL Fine-Tuned (Phase 4.1)')
    ]
    
    results = []
    
    print(f"=== Starting Cross-Dataset Generalization Benchmark on {device} ===")
    
    for train_ds in datasets:
        for model_type, prefix, display_name in model_configs:
            # Construct model path
            model_fname = f"{prefix}_{train_ds}.pth"
            if model_type == 'cnn' and 'ssl' in prefix and train_ds == 'ppg_dalia':
                 # Use the specialized one for DaLiA if available, else standard
                 if Path("best_model_ssl_specialized_ppg_dalia.pth").exists():
                     model_fname = "best_model_ssl_specialized_ppg_dalia.pth"
                     display_name = "SSL Specialized (Phase 4.2)"

            if not Path(model_fname).exists():
                print(f"Skipping {display_name} trained on {train_ds}: {model_fname} not found.")
                continue

            # Load Model
            if model_type == 'cnn':
                model = CNNBaseline().to(device)
            else:
                model = CNNLSTMHybrid().to(device) # Not used in this specific config but kept for extensibility
            
            try:
                model.load_state_dict(torch.load(model_fname, map_location=device))
                model.eval()
            except Exception as e:
                print(f"Error loading {model_fname}: {e}")
                continue

            # Evaluate on ALL datasets (Cross-Domain)
            for test_ds in datasets:
                print(f"Eval: {display_name} (Train: {train_ds}) -> Target: {test_ds}")
                
                try:
                    test_data_path = Path("preprocessed_data") / test_ds / 'test'
                    test_set = PPGDataset(test_data_path)
                    test_loader = DataLoader(test_set, batch_size=64, shuffle=False)
                    
                    criterion = nn.HuberLoss()
                    _, test_mae = evaluate(model, test_loader, criterion, device, use_tqdm=False)
                    
                    results.append({
                        'Method': display_name,
                        'Train Dataset': train_ds,
                        'Test Dataset': test_ds,
                        'MAE': test_mae,
                        'Type': 'In-Domain' if train_ds == test_ds else 'Cross-Domain'
                    })
                except Exception as e:
                    print(f"  Error evaluating on {test_ds}: {e}")

    # Save Results
    df = pd.DataFrame(results)
    if not df.empty:
        df.to_csv("generalization_results_phase6.csv", index=False)
        
        print("\n=== Generalization Matrix (MAE) ===")
        pivot = df.pivot_table(index=['Method', 'Train Dataset'], columns='Test Dataset', values='MAE')
        print(pivot)
        
        # Calculate Average Generalization Gap (Cross-Domain MAE - In-Domain MAE)
        print("\n=== Analysis ===")
        for method in df['Method'].unique():
            subset = df[df['Method'] == method]
            in_domain = subset[subset['Type'] == 'In-Domain']['MAE'].mean()
            cross_domain = subset[subset['Type'] == 'Cross-Domain']['MAE'].mean()
            print(f"{method}: In-Domain Avg={in_domain:.2f}, Cross-Domain Avg={cross_domain:.2f}, Gap={cross_domain-in_domain:.2f}")

    else:
        print("No results generated.")

if __name__ == "__main__":
    cross_dataset_benchmark()
