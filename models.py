import torch
import torch.nn as nn
import torch.nn.functional as F

class CNNBaseline(nn.Module):
    """
    1D CNN Baseline for PPG signal regression.
    Architecture:
    - 3 Conv blocks (Conv1D -> BatchNorm -> ReLU -> Dropout -> MaxPool)
    - Global Average Pooling
    - Fully Connected head for regression
    """
    def __init__(self, input_channels=1, hidden_dim=64, dropout=0.3):
        super(CNNBaseline, self).__init__()
        
        # Block 1
        self.conv1 = nn.Conv1d(input_channels, hidden_dim, kernel_size=11, stride=1, padding=5)
        self.bn1 = nn.BatchNorm1d(hidden_dim)
        self.pool1 = nn.MaxPool1d(4)
        
        # Block 2
        self.conv2 = nn.Conv1d(hidden_dim, hidden_dim*2, kernel_size=7, stride=1, padding=3)
        self.bn2 = nn.BatchNorm1d(hidden_dim*2)
        self.pool2 = nn.MaxPool1d(4)
        
        # Block 3
        self.conv3 = nn.Conv1d(hidden_dim*2, hidden_dim*4, kernel_size=5, stride=1, padding=2)
        self.bn3 = nn.BatchNorm1d(hidden_dim*4)
        self.pool3 = nn.MaxPool1d(4)
        
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden_dim*4, 1)

    def forward(self, x, return_features=False):
        # x shape: (batch, channels, time) -> (B, 1, 640)
        x = F.relu(self.bn1(self.conv1(x)))
        x = self.pool1(x)
        
        x = F.relu(self.bn2(self.conv2(x)))
        x = self.pool2(x)
        
        x = F.relu(self.bn3(self.conv3(x)))
        x = self.pool3(x)
        
        # Global Average Pooling
        x = torch.mean(x, dim=2)
        
        if return_features:
            return x
            
        x = self.dropout(x)
        return self.fc(x)


class CNNLSTMHybrid(nn.Module):
    """
    CNN-LSTM Hybrid for capturing morphological and temporal dependencies.
    """
    def __init__(self, input_channels=1, hidden_dim=64, lstm_units=128, dropout=0.3):
        super(CNNLSTMHybrid, self).__init__()
        
        # CNN Feature Extractor
        self.cnn = nn.Sequential(
            nn.Conv1d(input_channels, hidden_dim, kernel_size=11, padding=5),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.MaxPool1d(2),
            
            nn.Conv1d(hidden_dim, hidden_dim*2, kernel_size=7, padding=3),
            nn.BatchNorm1d(hidden_dim*2),
            nn.ReLU(),
            nn.MaxPool1d(2)
        )
        
        # LSTM layer
        # Output of CNN will be (B, 128, 160) for 640 input
        self.lstm = nn.LSTM(input_size=hidden_dim*2, 
                            hidden_size=lstm_units, 
                            num_layers=2, 
                            batch_first=True, 
                            bidirectional=True,
                            dropout=dropout)
        
        self.fc = nn.Linear(lstm_units * 2, 1) # *2 for bidirectional
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, return_features=False):
        # CNN: (B, 1, 640) -> (B, 128, 160)
        x = self.cnn(x)
        
        # Reshape for LSTM: (B, hidden, seq) -> (B, seq, hidden)
        x = x.transpose(1, 2)
        
        # LSTM
        x, _ = self.lstm(x)
        
        # Use last time step output
        x = x[:, -1, :]
        
        if return_features:
            return x
            
        x = self.dropout(x)
        return self.fc(x)

if __name__ == "__main__":
    # Test forward pass
    model_cnn = CNNBaseline()
    model_hybrid = CNNLSTMHybrid()
    
    test_input = torch.randn(8, 1, 640)
    
    out_cnn = model_cnn(test_input)
    out_hybrid = model_hybrid(test_input)
    
    print(f"CNN Output shape: {out_cnn.shape}")
    print(f"Hybrid Output shape: {out_hybrid.shape}")
    assert out_cnn.shape == (8, 1)
    assert out_hybrid.shape == (8, 1)
    print("✓ Model forward passes successful!")
