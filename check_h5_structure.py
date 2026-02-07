import h5py

def print_structure(name, obj):
    print(name)

h5_path = r'c:\Users\annam\OneDrive\Desktop\PPG\datasets\PPG+Dalia\PPG_FieldStudy\S1\S1_RespiBAN.h5'
with h5py.File(h5_path, 'r') as f:
    f.visititems(print_structure)
