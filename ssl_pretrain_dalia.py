import torch
import torch.optim as optim
from pathlib import Path
from tqdm import tqdm
import argparse
import os

from models import CNNBaseline
from ppg_ssl.augmentations import PPGTransforms, ContrastiveTransform
from ppg_ssl.dataset import get_ssl_dataloader
from ppg_ssl.models import SimCLRWrapper, nt_xent_loss

def pretrain_dalia_specialized(epochs=50, batch_size=64, lr=1e-3, temperature=0.5):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"=== Specialized SSL Pre-training for PPG-DaLiA on {device} ===")
    print(f"Config: Epochs={epochs}, BS={batch_size}, LR={lr}, Temp={temperature}")
    
    # Exclusive DaLiA Dataset
    loader = get_ssl_dataloader(datasets=['ppg_dalia'], batch_size=batch_size, 
                                transform=ContrastiveTransform(PPGTransforms(domain_type='wearable')))
    if not loader: return
    
    model = SimCLRWrapper(CNNBaseline(), input_dim=256).to(device)
    optimizer = optim.AdamW(model.parameters(), lr=lr)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    
    output_name = "ssl_dalia_specialized.pth"
    
    for epoch in range(epochs):
        model.train(); total_loss = 0
        pbar = tqdm(loader, desc=f"DaLiA-SSL Epoch {epoch+1}")
        for x1, x2 in pbar:
            x1, x2 = x1.to(device), x2.to(device)
            optimizer.zero_grad()
            z1, z2 = model(x1, x2)
            loss = nt_xent_loss(z1, z2, temperature=temperature)
            loss.backward(); optimizer.step(); total_loss += loss.item()
            pbar.set_postfix({'loss': loss.item()})
        print(f"Epoch {epoch+1} Avg Loss: {total_loss/len(loader):.4f}")
        scheduler.step()
        
        # Save checkpoints frequently
        if (epoch+1) % 10 == 0:
            torch.save(model.encoder.state_dict(), output_name)
            
    print(f"✓ Saved specialized weights to {output_name}")

if __name__ == "__main__":
    pretrain_dalia_specialized()
