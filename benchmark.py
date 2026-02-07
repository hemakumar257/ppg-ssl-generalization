import os
import subprocess
import pandas as pd
import json
from pathlib import Path

datasets = ['ppg_dalia', 'wesad', 'bidmc']
models = ['cnn', 'hybrid']
results_list = []

for dataset in datasets:
    for model_type in models:
        print(f"\n{'='*60}")
        print(f"BENCHMARKING: {model_type.upper()} on {dataset.upper()}")
        print(f"{'='*60}")
        
        result_file = f"result_{dataset}_{model_type}.json"
        
        # Run training
        cmd = [
            'python', 'train.py', 
            '--dataset', dataset, 
            '--model', model_type, 
            '--epochs', '15',
            '--no-tqdm',
            '--save-results', result_file
        ]
        
        try:
            print(f"Running: {' '.join(cmd)}")
            subprocess.run(cmd, check=True)
            
            if Path(result_file).exists():
                with open(result_file, 'r') as f:
                    res = json.load(f)
                
                results_list.append({
                    'Dataset': res['dataset'],
                    'Model': res['model'],
                    'Test MAE (BPM)': res['test_mae'],
                    'Training Time (s)': res['training_time']
                })
                print(f"✓ DONE: {dataset} {model_type} -> Test MAE: {res['test_mae']:.2f}")
                
                # Incremental Visualization
                print("  Generating plots...")
                subprocess.run(['python', 'visualize_results.py'], check=False)
                
                # Incremental Generalization (only if at least one model exists)
                print("  Running generalization check...")
                subprocess.run(['python', 'cross_dataset_test.py'], check=False)

            else:
                print(f"✗ FAILED: Result file {result_file} not created.")
                    
        except Exception as e:
            print(f"✗ CRASHED: {dataset} {model_type} -> {e}")

# Save results to CSV
if results_list:
    df = pd.DataFrame(results_list)
    df.to_csv("benchmark_results.csv", index=False)
    print("\nBenchmark Results Summary:")
    print(df)
else:
    print("\nNo results collected.")
