import json
import pandas as pd
from pathlib import Path
import matplotlib.pyplot as plt

import seaborn as sns

def compare_results():
    datasets = ['ppg_dalia', 'wesad', 'bidmc']
    summary = []
    
    for ds in datasets:
        # Load Baseline CNN (Phase 3)
        base_cnn_file = Path(f"result_{ds}_cnn.json")
        if base_cnn_file.exists():
            with open(base_cnn_file, 'r') as f:
                data = json.load(f)
                summary.append({'Dataset': ds.upper(), 'Method': 'Supervised CNN', 'MAE': data['test_mae']})
        
        # Load Baseline Hybrid (Phase 3)
        base_hyb_file = Path(f"result_{ds}_hybrid.json")
        if base_hyb_file.exists():
            with open(base_hyb_file, 'r') as f:
                data = json.load(f)
                summary.append({'Dataset': ds.upper(), 'Method': 'Supervised Hybrid', 'MAE': data['test_mae']})
        
        # Load Domain SSL CNN (Phase 4.1)
        ssl_cnn_file = Path(f"result_ssl_ft_{ds}.json")
        if ssl_cnn_file.exists():
            with open(ssl_cnn_file, 'r') as f:
                data = json.load(f)
                summary.append({'Dataset': ds.upper(), 'Method': 'SSL CNN (Phase 4.1)', 'MAE': data['test_mae']})

        # Load Specialized SSL (Phase 4.2 - DaLiA only)
        if ds == 'ppg_dalia':
            spec_file = Path(f"result_ssl_specialized_{ds}.json")
            if spec_file.exists():
                with open(spec_file, 'r') as f:
                    data = json.load(f)
                    summary.append({'Dataset': ds.upper(), 'Method': 'Specialized SSL CNN', 'MAE': data['test_mae']})

    df = pd.DataFrame(summary)
    print("\n=== Comprehensive Phase 3 & 4 Benchmark ===")
    if not df.empty:
        print(df.pivot(index='Dataset', columns='Method', values='MAE'))
    
    # Visualization
    if not df.empty:
        plt.figure(figsize=(12, 6))
        # Custom color palette
        palette = {
            'Supervised CNN': 'salmon',
            'Supervised Hybrid': 'firebrick',
            'SSL CNN (Phase 4.1)': 'skyblue',
            'Specialized SSL CNN': 'royalblue'
        }
        sns.barplot(data=df, x='Dataset', y='MAE', hue='Method', palette=palette)
        plt.title("Eval: Supervised vs. SSL across Architectures")
        plt.ylabel("MAE (BPM)")
        plt.grid(axis='y', linestyle='--', alpha=0.5)
        plt.tight_layout()
        plt.savefig('final_benchmark_phase4_2.png')
        plt.close()

if __name__ == "__main__":
    compare_results()
