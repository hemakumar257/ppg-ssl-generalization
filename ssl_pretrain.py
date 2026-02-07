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

def pretrain_ssl(datasets=['ppg_dalia', 'wesad', 'bidmc'], domain_type='wearable', epochs=30, batch_size=128, lr=1e-3, temperature=0.5):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"SSL Pre-training on {device} | Domain: {domain_type} | Datasets: {datasets}")
    
    loader = get_ssl_dataloader(datasets=datasets, batch_size=batch_size, 
                                transform=ContrastiveTransform(PPGTransforms(domain_type=domain_type)))
    if not loader: return
    
    model = SimCLRWrapper(CNNBaseline(), input_dim=256).to(device)
    optimizer = optim.AdamW(model.parameters(), lr=lr)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    
    output_name = f"ssl_{domain_type}_encoder.pth"
    
    for epoch in range(epochs):
        model.train(); total_loss = 0
        pbar = tqdm(loader, desc=f"Epoch {epoch+1}")
        for x1, x2 in pbar:
            x1, x2 = x1.to(device), x2.to(device)
            optimizer.zero_grad()
            z1, z2 = model(x1, x2)
            loss = nt_xent_loss(z1, z2, temperature=temperature)
            loss.backward(); optimizer.step(); total_loss += loss.item()
            pbar.set_postfix({'loss': loss.item()})
        print(f"Epoch {epoch+1} Avg Loss: {total_loss/len(loader):.4f}")
        scheduler.step()
        torch.save(model.encoder.state_dict(), output_name)
    print(f"✓ Saved pre-trained weights to {output_name}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--datasets", nargs='+', default=['ppg_dalia', 'wesad', 'bidmc'])
    parser.add_argument("--domain", type=str, choices=['wearable', 'clinical'], default='wearable')
    parser.add_argument("--epochs", type=int, default=30)
    args = parser.parse_args()
    pretrain_ssl(datasets=args.datasets, domain_type=args.domain, epochs=args.epochs)
