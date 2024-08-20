import torch
import torch.nn as nn

class SmallLSTM(nn.Module):
    def __init__(self, input_shape):
        super(SmallLSTM, self).__init__()
        self.lstm_layers = nn.Sequential(
            nn.LSTM(input_shape[1], 256, batch_first=True, num_layers=4, dropout=0.2)
        )
        self.fc_layers = nn.Sequential(
            nn.Linear(256, 256),
            nn.ELU(),
            nn.Linear(256, 128),
            nn.ELU(),
            nn.Linear(128, 24)
        )

    def forward(self, x):
        x, _ = self.lstm_layers(x)
        x = x[:, -1, :]  # Get the last time step
        x = self.fc_layers(x)
        return x

def build_model(input_shape):
    return SmallLSTM(input_shape)
