import torch
import torch.nn as nn

class CNN2D_GRU_EEG(nn.Module):
    def __init__(self, num_classes):
        super(CNN2D_GRU_EEG, self).__init__()
        
        # 2D Convolutional layers
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=32, kernel_size=(3, 3), padding=(1, 1))
        self.bn1 = nn.BatchNorm2d(32)  # Batch Normalization after the first conv layer
        self.pool = nn.MaxPool2d(kernel_size=(2, 2)) 
        self.conv2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=(3, 3), padding=(1, 1))
        self.bn2 = nn.BatchNorm2d(64)  # Batch Normalization after the second conv layer
        self.pool = nn.MaxPool2d(kernel_size=(2, 2))
        self.DOut = nn.Dropout(0.25)
        
        # GRU layer
        self.gru = nn.GRU(input_size=64, hidden_size=128, num_layers=2, batch_first=True, bidirectional=True)  # Adjust input_size based on pooling
        
        # Fully connected layer
        self.fc = nn.Linear(128 * 2, num_classes)  # 128 * 2 due to bidirectional GRU

    def forward(self, x):
        # Input shape should be [batch_size, 1, 128, 500]
        x = x.permute(0, 3, 1, 2)
        x = torch.relu(self.bn1(self.conv1(x)))
        x = self.pool(x)
        x = torch.relu(self.bn2(self.conv2(x)))
        x = self.pool(x)
        x = self.DOut(x)

        # Reshape for GRU
        batch_size, channels, height, width = x.size()
        seq_len = height
        feature_size = channels * width

        if self.gru.input_size != feature_size:
            self.gru = nn.GRU(input_size=feature_size, hidden_size=128, num_layers=2, batch_first=True, bidirectional=True)
        
        x = x.permute(0, 2, 1, 3)  # Shape: [batch_size, seq_len, channels, width]
        x = x.contiguous().view(batch_size, seq_len, -1)
        x, _ = self.gru(x)  # Pass through GRU layer

        # Use the last hidden state
        x = x[:, -1, :]  # Get the output of the last time step
        x = self.fc(x)  # Fully connected layer for classification

        return x
