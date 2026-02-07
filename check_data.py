import pickle
import numpy as np
from pathlib import Path

pkl_path = Path('datasets/PPG+Dalia/PPG_FieldStudy/S1/S1.pkl')
if not pkl_path.exists():
    print(f"File not found: {pkl_path}")
else:
    with open(pkl_path, 'rb') as f:
        data = pickle.load(f, encoding='latin1')
    print(f"Keys: {data.keys()}")
    if 'label' in data:
        labels = data['label']
        print(f"Label type: {type(labels)}")
        print(f"Label shape: {np.shape(labels)}")
        print(f"First 5 labels: {labels[:5]}")
        # Check signal length to estimate label SR
        ppg = data['signal']['chest']['PPG'].flatten()
        duration = len(ppg) / 700.0
        label_sr = len(labels) / duration
        print(f"Estimated Label SR: {label_sr:.2f} Hz")
    if 'signal' in data:
        print(f"Signal keys: {data['signal'].keys()}")
