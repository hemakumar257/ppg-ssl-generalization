import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
from pathlib import Path
import json
import matplotlib.pyplot as plt
from tqdm import tqdm
import argparse
import time

from models import CNNBaseline, CNNLSTMHybrid

class PPGDataset(Dataset):
    """
    Standard Dataset for windowed PPG signals and physiological labels.
    """
    def __init__(self, data_path: Path):
        # Robust loading to prevent contention/locking errors
        max_retries = 3
        for attempt in range(max_retries):
            try:
                self.signals = np.load(data_path / 'signals.npy') # (N, T, 1)
                self.labels = np.load(data_path / 'y.npy')       # (N,)
                break
            except Exception as e:
                if attempt == max_retries - 1: raise e
                time.sleep(1)
        
        # Reshape signals to (N, 1, T) for PyTorch Conv1D
        self.signals = self.signals.transpose(0, 2, 1)
        
        # Convert to torch tensors
        self.signals = torch.from_numpy(self.signals).float()
        self.labels = torch.from_numpy(self.labels).float()

    def __len__(self):
        return len(self.signals)

    def __getitem__(self, idx):
        return self.signals[idx], self.labels[idx]

def train_one_epoch(model, loader, criterion, optimizer, device, use_tqdm=True):
    model.train()
    running_loss = 0.0
    mae_sum = 0.0
    
    pbar = tqdm(loader, desc="Training", leave=False) if use_tqdm else loader
    for signals, labels in pbar:
        signals, labels = signals.to(device), labels.to(device).unsqueeze(1)
        
        optimizer.zero_grad()
        outputs = model(signals)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item() * signals.size(0)
        mae_sum += torch.abs(outputs - labels).sum().item()
    
    epoch_loss = running_loss / len(loader.dataset)
    epoch_mae = mae_sum / len(loader.dataset)
    return epoch_loss, epoch_mae

def evaluate(model, loader, criterion, device, use_tqdm=True):
    model.eval()
    running_loss = 0.0
    mae_sum = 0.0
    
    with torch.no_grad():
        pbar = tqdm(loader, desc="Evaluating", leave=False) if use_tqdm else loader
        for signals, labels in pbar:
            signals, labels = signals.to(device), labels.to(device).unsqueeze(1)
            outputs = model(signals)
            loss = criterion(outputs, labels)
            
            running_loss += loss.item() * signals.size(0)
            mae_sum += torch.abs(outputs - labels).sum().item()
            
    val_loss = running_loss / len(loader.dataset)
    val_mae = mae_sum / len(loader.dataset)
    return val_loss, val_mae

def run_training(dataset_name, model_type='cnn', epochs=50, batch_size=64, lr=1e-3, use_tqdm=True):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Training {model_type} on {dataset_name} using {device}")
    
    data_dir = Path("preprocessed_data") / dataset_name
    
    # Load Datasets
    try:
        train_set = PPGDataset(data_dir / 'train')
        val_set = PPGDataset(data_dir / 'val')
        test_set = PPGDataset(data_dir / 'test')
    except Exception as e:
        print(f"Error loading data: {e}")
        return None, None

    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_set, batch_size=batch_size)
    test_loader = DataLoader(test_set, batch_size=batch_size)
    
    # Initialize Model
    if model_type == 'cnn':
        model = CNNBaseline().to(device)
    else:
        model = CNNLSTMHybrid().to(device)
        
    criterion = nn.HuberLoss() 
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5)
    
    history = {'train_loss': [], 'train_mae': [], 'val_loss': [], 'val_mae': []}
    best_val_mae = float('inf')
    
    start_time = time.time()
    for epoch in range(epochs):
        train_loss, train_mae = train_one_epoch(model, train_loader, criterion, optimizer, device, use_tqdm)
        val_loss, val_mae = evaluate(model, val_loader, criterion, device, use_tqdm)
        
        scheduler.step(val_loss)
        
        history['train_loss'].append(train_loss)
        history['train_mae'].append(train_mae)
        history['val_loss'].append(val_loss)
        history['val_mae'].append(val_mae)
        
        print(f"Epoch {epoch+1}/{epochs} | Train Loss: {train_loss:.4f}, MAE: {train_mae:.2f} | Val Loss: {val_loss:.4f}, MAE: {val_mae:.2f}")
        
        if val_mae < best_val_mae:
            best_val_mae = val_mae
            torch.save(model.state_dict(), f"best_model_{model_type}_{dataset_name}.pth")
            print("  ✓ Best model saved")

    total_time = time.time() - start_time
    # Final Evaluation
    best_path = f"best_model_{model_type}_{dataset_name}.pth"
    if Path(best_path).exists():
        model.load_state_dict(torch.load(best_path, map_location=device))
        test_loss, test_mae = evaluate(model, test_loader, criterion, device, use_tqdm)
        print(f"\nFinal Test MAE for {dataset_name}: {test_mae:.2f}")
        
        results = {
            'dataset': dataset_name,
            'model': model_type,
            'test_mae': test_mae,
            'val_mae': best_val_mae,
            'training_time': total_time,
            'history': history
        }
        return results, test_mae
    else:
        print("Error: No best model found to evaluate.")
        return None, None

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, default="ppg_dalia")
    parser.add_argument("--model", type=str, default="cnn")
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--no-tqdm", action="store_true")
    parser.add_argument("--save-results", type=str, default=None)
    args = parser.parse_args()
    
    results, test_mae = run_training(args.dataset, args.model, epochs=args.epochs, use_tqdm=not args.no_tqdm)
    
    if args.save_results and results:
        with open(args.save_results, 'w') as f:
            json.dump(results, f, indent=2)
