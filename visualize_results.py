import torch
import numpy as np
import matplotlib.pyplot as plt
import json
from pathlib import Path
from models import CNNBaseline, CNNLSTMHybrid
from train import PPGDataset
from torch.utils.data import DataLoader

def plot_loss(result_json_path):
    with open(result_json_path, 'r') as f:
        res = json.load(f)
    
    history = res['history']
    dataset = res['dataset']
    model_type = res['model']
    
    plt.figure(figsize=(10, 5))
    plt.subplot(1, 2, 1)
    plt.plot(history['train_loss'], label='Train Loss')
    plt.plot(history['val_loss'], label='Val Loss')
    plt.title(f'Loss: {model_type} on {dataset}')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)
    
    plt.subplot(1, 2, 2)
    plt.plot(history['train_mae'], label='Train MAE')
    plt.plot(history['val_mae'], label='Val MAE')
    plt.title(f'MAE: {model_type} on {dataset}')
    plt.xlabel('Epoch')
    plt.ylabel('MAE (BPM)')
    plt.legend()
    plt.grid(True)
    
    plt.tight_layout()
    plt.savefig(f'curves_{dataset}_{model_type}.png')
    plt.close()

def plot_predictions(dataset_name, model_path, model_type='cnn'):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    data_dir = Path("preprocessed_data") / dataset_name / 'test'
    
    dataset = PPGDataset(data_dir)
    loader = DataLoader(dataset, batch_size=32)
    
    if model_type == 'cnn':
        model = CNNBaseline().to(device)
    else:
        model = CNNLSTMHybrid().to(device)
        
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()
    
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for signals, labels in loader:
            signals = signals.to(device)
            preds = model(signals)
            all_preds.extend(preds.cpu().numpy().flatten())
            all_labels.extend(labels.numpy())
            
    plt.figure(figsize=(8, 8))
    plt.scatter(all_labels, all_preds, alpha=0.3, color='blue')
    plt.plot([40, 180], [40, 180], 'r--', lw=2)
    plt.xlabel('Ground Truth HR (BPM)')
    plt.ylabel('Predicted HR (BPM)')
    plt.title(f'Predictions: {model_type.upper()} on {dataset_name.upper()}')
    plt.xlim(40, 180)
    plt.ylim(40, 180)
    plt.grid(True, linestyle='--', alpha=1.0)
    plt.savefig(f'scatter_{dataset_name}_{model_type}.png')
    plt.close()

if __name__ == "__main__":
    datasets = ['ppg_dalia', 'wesad', 'bidmc']
    models = ['cnn', 'hybrid']
    
    for ds in datasets:
        for mt in models:
            json_path = f"result_{ds}_{mt}.json"
            model_path = f"best_model_{mt}_{ds}.pth"
            if Path(json_path).exists():
                plot_loss(json_path)
            if Path(model_path).exists():
                plot_predictions(ds, model_path, mt)
