from preprocessing.loaders import load_bidmc
import os
import traceback

print("--- Debugging load_bidmc ---")
subject_id = 1
# This is the default path in loaders.py
default_path = "datasets/bidmc-ppg-and-respiration-dataset-1.0.0/bidmc-ppg-and-respiration-dataset-1.0.0"

print(f"Testing subject {subject_id} with path: {default_path}")
print(f"Absolute path: {os.path.abspath(default_path)}")
print(f"Exists: {os.path.exists(default_path)}")

try:
    sig, fs, meta = load_bidmc(subject_id)
    print(f"SUCCESS: sig shape {sig.shape}, fs {fs}")
except Exception as e:
    print(f"FAILED: {e}")
    traceback.print_exc()
