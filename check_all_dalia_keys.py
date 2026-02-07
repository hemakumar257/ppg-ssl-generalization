import pickle
import os

print("--- Checking All PPG-DaLiA Subjects Structure ---")
base_path = r'c:\Users\annam\OneDrive\Desktop\PPG\datasets\PPG+Dalia\PPG_FieldStudy'

for sid in range(1, 16):
    pkl_path = os.path.join(base_path, f'S{sid}', f'S{sid}.pkl')
    if not os.path.exists(pkl_path):
        print(f"S{sid}: File not found")
        continue
    
    try:
        with open(pkl_path, 'rb') as f:
            data = pickle.load(f, encoding='latin1')
            chest_keys = list(data['signal']['chest'].keys())
            wrist_keys = list(data['signal']['wrist'].keys())
            print(f"S{sid} Chest Keys: {chest_keys}")
            print(f"S{sid} Wrist Keys: {wrist_keys}")
            
            if 'PPG' in chest_keys:
                print(f"  [OK] PPG found in CHEST! Shape: {data['signal']['chest']['PPG'].shape}")
            elif 'BVP' in wrist_keys:
                print(f"  [OK] BVP found in WRIST! Shape: {data['signal']['wrist']['BVP'].shape}")
            else:
                print(f"  [ERROR] No PPG (Chest) or BVP (Wrist) found!")
    except Exception as e:
        print(f"S{sid}: Error - {e}")
