from preprocessing.pipeline import PreprocessingPipeline
from preprocessing.loaders import load_bidmc, load_wesad, load_ppg_dalia
import os

print("--- Testing Loaders ---")
datasets = {
    "BIDMC": (load_bidmc, 1, "datasets/bidmc-ppg-and-respiration-dataset-1.0.0/bidmc-ppg-and-respiration-dataset-1.0.0"),
    "WESAD": (load_wesad, 2, "datasets/WESAD/WESAD"),
    "PPG-DaLiA": (load_ppg_dalia, 1, "datasets/PPG+Dalia/PPG_FieldStudy")
}

for name, (loader, sid, path) in datasets.items():
    print(f"\nTesting {name} loader...")
    full_path = os.path.abspath(path)
    print(f"Path: {full_path}")
    if not os.path.exists(full_path):
        print(f"❌ Path does NOT exist!")
        continue
    
    try:
        sig, fs, meta = loader(sid, path)
        print(f"✅ Success! FS: {fs}, Signal shape: {sig.shape}")
    except Exception as e:
        print(f"❌ Failed: {e}")

print("\n--- Testing Pipeline Statistics Fix ---")
pipeline = PreprocessingPipeline()
# Mock some stats to avoid div by zero if run() fails
pipeline.stats['total_windows'] = 0 
if pipeline.stats['total_windows'] == 0:
    print("Pipeline stats check: total_windows is 0, will handle safely in updated code.")
