import torch
import torch.nn as nn

class Model(nn.Module):

    input_shape = (8, 48)  # Sequence length, feature size
    output_shape = (24,)  # Total output size
    dtype = torch.float32
    name = "small_lstm"

    def __init__(self):
        """
        Initializes the Model class.

        Attributes:
        - input_shape (tuple): The shape of the input data (sequence length, feature size).
        - output_shape (tuple): The shape of the output data.
        - dtype (torch.dtype): The data type of the tensors.
        - name (str): The name of the model.
        - lstm_layers (nn.Sequential): The sequential layers for the LSTM model.
        - fc_layers (nn.Sequential): The sequential layers for the fully connected model.
        """
        super(Model, self).__init__()

        self.input_shape = (8, 48)  # Sequence length, feature size
        self.output_shape = (24,)  # Total output size
        self.dtype = torch.float32
        self.name = "small_lstm"

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

    def model_batch(self, batch):
        """
        Creates a dictionary representing the batch of inputs for the model.

        Parameters:
        - batch: The input batch.

        Returns:
        - A dictionary with the following keys:
            - 'inputs': The input batch.
            - 'shape': The shape of the input batch.
            - 'dtype': The data type of the input batch.
        """
        return  {
            'inputs': batch,
            'shape': (batch, self.input_shape[0], self.input_shape[1]),
            'dtype': self.dtype
        }


    def forward(self, x):
        """
        Forward pass of the small_lstm model.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, sequence_length, input_size).

        Returns:
            torch.Tensor: Output tensor of shape (batch_size, output_size).
        """
        x, _ = self.lstm_layers(x)
        x = x[:, -1, :]  # Get the last time step
        x = self.fc_layers(x)
        return x
    
