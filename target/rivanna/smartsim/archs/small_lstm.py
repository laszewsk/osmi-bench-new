import torch
import torch.nn as nn

class Model(nn.Module):

    input_shape = (8, 48)  # Sequence length, feature size
    output_shape = (24,)  # Total output size
    dtype = torch.float32
    name = "small_lstm"

    @classmethod
    def model_batch(cls, batch):
        return  {
            'inputs': batch,
            'shape': (batch, cls.input_shape[0], cls.input_shape[1]),
            'dtype': cls.dtype
        }
            

    def __init__(self):
        super(Model, self).__init__()
        self.lstm_layers = nn.Sequential(
            nn.LSTM(self.input_shape[1], 256, batch_first=True, num_layers=4, dropout=0.2)
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