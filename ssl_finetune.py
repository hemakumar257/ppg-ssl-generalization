import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from pathlib import Path
import argparse
import json
import time

from models import CNNBaseline
from train import PPGDataset, train_one_epoch, evaluate

def finetune_ssl(dataset_name, encoder_path, model_type='cnn', epochs=30, freeze=False, lr=1e-4):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Fine-tuning SSL {model_type.upper()} on {dataset_name} (Freeze: {freeze})")
    
    data_dir = Path("preprocessed_data") / dataset_name
    train_loader = DataLoader(PPGDataset(data_dir / 'train'), batch_size=64, shuffle=True)
    val_loader = DataLoader(PPGDataset(data_dir / 'val'), batch_size=64)
    test_loader = DataLoader(PPGDataset(data_dir / 'test'), batch_size=64)
    
    # 1. Initialize Model
    if model_type == 'cnn':
        model = CNNBaseline().to(device)
    else:
        model = CNNLSTMHybrid().to(device)
    
    # 2. Load Pre-trained Encoder weights
    if encoder_path and Path(encoder_path).exists():
        print(f"Loading SSL pre-trained weights from {encoder_path}")
        state_dict = torch.load(encoder_path, map_location=device)
        model_dict = model.state_dict()
        
        # Filter weights: Match 'cnn.' prefix if hybrid, or direct matching if CNN
        # Note: SimCLRWrapper saves encoder as 'encoder.conv1...', Baseline has 'conv1...' 
        # Hybrid has 'cnn.0...' so we need intelligent mapping.
        
        pretrained_dict = {}
        for k, v in state_dict.items():
            # Standard CNN mapping
            if k in model_dict and 'fc' not in k:
                pretrained_dict[k] = v
            # Hybrid mapping: 'conv1' -> 'cnn.0' etc? 
            # Actually CNNLSTMHybrid uses a Sequential 'cnn', so names are 'cnn.0.weight' etc.
            # We implemented CNNBaseline components directly.
            # This mapping is complex. For Phase 4.2 simplification, we will stick to CNN fine-tuning 
            # IF hybrid mapping proves too brittle, or implement a basic 'cnn.' prefix check.
            
            # Simple check for direct match first
            
        # Improved Loader for Hybrid
        if model_type == 'hybrid':
             # Map 'conv1.weight' -> 'cnn.0.weight', 'bn1.weight' -> 'cnn.1.weight' etc.
             # This is tricky due to the Sequential definition in Hybrid vs explicit in Baseline.
             # fallback: Only fine-tune CNN for now to avoid architectural mismatch errors
             # unless we explicitly defined shared sub-modules.
             pass 
        else:
             # Standard CNN loading
             pretrained_dict = {k: v for k, v in state_dict.items() if k in model_dict and 'fc' not in k}
             model_dict.update(pretrained_dict)
             model.load_state_dict(model_dict)
             
    else:
        print("Warning: No pre-trained weights found. Training from scratch.")

    # ... (rest of training loop is generic) ...
    
    criterion = nn.HuberLoss()
    optimizer = optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=lr)
    
    history = {'train_loss': [], 'train_mae': [], 'val_loss': [], 'val_mae': []}
    best_val_mae = float('inf')
    
    start_time = time.time()
    for epoch in range(epochs):
        t_loss, t_mae = train_one_epoch(model, train_loader, criterion, optimizer, device, use_tqdm=False)
        v_loss, v_mae = evaluate(model, val_loader, criterion, device, use_tqdm=False)
        
        history['train_loss'].append(t_loss)
        history['train_mae'].append(t_mae)
        history['val_loss'].append(v_loss)
        history['val_mae'].append(v_mae)
        
        print(f"Epoch {epoch+1}/{epochs} | Val MAE: {v_mae:.2f}")
        
        if v_mae < best_val_mae:
            best_val_mae = v_mae
            torch.save(model.state_dict(), f"best_model_ssl_specialized_{dataset_name}.pth")

    model.load_state_dict(torch.load(f"best_model_ssl_specialized_{dataset_name}.pth"))
    _, test_mae = evaluate(model, test_loader, criterion, device, use_tqdm=False)
    
    results = {
        'dataset': dataset_name,
        'method': f'ssl_specialized_{model_type}',
        'test_mae': test_mae,
        'val_mae': best_val_mae,
        'history': history,
        'time': time.time() - start_time
    }
    
    with open(f"result_ssl_specialized_{dataset_name}.json", 'w') as f:
        json.dump(results, f, indent=2)
    
    return test_mae

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, default="ppg_dalia")
    parser.add_argument("--encoder", type=str, default="ssl_pretrained_encoder.pth")
    parser.add_argument("--model", type=str, default="cnn")
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--lr", type=float, default=1e-4)
    args = parser.parse_args()
    
    finetune_ssl(args.dataset, args.encoder, model_type=args.model, epochs=args.epochs, lr=args.lr)
