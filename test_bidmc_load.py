from preprocessing.loaders import load_bidmc
from preprocessing.windowing import segment_signal
import numpy as np

try:
    sig, fs, labels, meta = load_bidmc(1)
    print(f"Loaded BIDMC S1. FS: {fs}")
    print(f"Labels keys: {labels.keys() if labels else 'None'}")
    if labels:
        print(f"HR shape: {labels['hr'].shape}")
        
    windows, indices, seg_labels = segment_signal(sig, fs, window_duration=10.0, overlap=0.5, labels=labels)
    print(f"Segmented: {len(windows)} windows")
    print(f"Segmented labels keys: {seg_labels.keys() if seg_labels else 'None'}")
    if seg_labels:
        print(f"Segmented HR shape: {seg_labels['hr'].shape}")
        print(f"First 5 segmented HR: {seg_labels['hr'][:5]}")
        
except Exception as e:
    import traceback
    traceback.print_exc()
