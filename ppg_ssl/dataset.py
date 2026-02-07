import os
import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader
from pathlib import Path

class SSLDataset(Dataset):
    def __init__(self, data_dirs, transform=None):
        self.signals = []
        self.transform = transform
        for data_dir in data_dirs:
            data_path = Path(data_dir)
            if not data_path.exists(): continue
            for split in ['train', 'val']:
                sig_file = data_path / split / 'signals.npy'
                if sig_file.exists():
                    self.signals.append(np.load(sig_file))
        if self.signals:
            self.signals = np.concatenate(self.signals, axis=0)
            if len(self.signals.shape) == 3 and self.signals.shape[2] == 1:
                self.signals = self.signals.transpose(0, 2, 1)
            elif len(self.signals.shape) == 2:
                self.signals = self.signals[:, np.newaxis, :]
        else: self.signals = np.array([])
    def __len__(self): return len(self.signals)
    def __getitem__(self, idx):
        x = self.signals[idx]
        if self.transform: return self.transform(x)
        return torch.from_numpy(x.astype(np.float32))

def get_ssl_dataloader(data_root="preprocessed_data", datasets=['ppg_dalia', 'wesad', 'bidmc'], batch_size=128, transform=None):
    data_dirs = [os.path.join(data_root, d) for d in datasets]
    dataset = SSLDataset(data_dirs, transform=transform)
    return DataLoader(dataset, batch_size=batch_size, shuffle=True, drop_last=True)
