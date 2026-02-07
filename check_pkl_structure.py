import pickle

path = r'c:\Users\annam\OneDrive\Desktop\PPG\datasets\PPG+Dalia\PPG_FieldStudy\S1\S1.pkl'
with open(path, 'rb') as f:
    data = pickle.load(f, encoding='latin1')
    print("Keys:", data.keys())
    if 'signal' in data:
        print("Signal keys:", data['signal'].keys())
        if 'chest' in data['signal']:
            print("Chest keys:", data['signal']['chest'].keys())
    if 'label' in data:
        print("Label shape/type:", type(data['label']))
