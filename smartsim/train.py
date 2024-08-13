"""usage: python train.py {small_lstm|medium_cnn|large_tcnn}"""
import argparse
import importlib
import numpy as np
import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

# Note: AMD accelerators also use the 'cuda' device, even though they use
# the ROCm/HIP stack
TORCH_DEVICES = {
    'cpu': torch.device('cpu'),
    'cuda': torch.device('cuda')
}

parser = argparse.ArgumentParser()
archs = [s.split('.')[0] for s in os.listdir('archs') if s[0:1] != '_']
parser.add_argument('arch', type=str, choices=archs, help='Type of neural network architectures')
parser.add_argument('device', type=str, choices = TORCH_DEVICES.keys(), help='The target device for the trained model')
parser.add_argument('-b', '--batch', type=int, default=1, help='batch size')
args = parser.parse_args()

device = TORCH_DEVICES[args.device]

# Parameters
samples = 100
epochs = 5
batch_size = args.batch

# Compute synthetic data for X and Y
if args.arch == "small_lstm":
    input_shape = (8, 48)  # Sequence length, feature size
    output_shape = (24,)  # Total output size
elif args.arch == "medium_cnn":
    input_shape = (9, 101, 82)  # Channels, Height, Width
    output_shape = (101*82,)  # Flattened output size
elif args.arch == "large_tcnn":
    input_shape = (9, 101, 82)  # Channels, Depth, Height, Width for 3D CNNs, but let's simplify
    output_shape = (101*82,)  # Adjust based on actual model architecture
else:
    raise ValueError("Model not supported. Need to specify input and output shapes")

X = np.random.rand(samples, *input_shape).astype(np.float32)
Y = np.random.rand(samples, *output_shape).astype(np.float32)

# Convert numpy arrays to PyTorch tensors
X_tensor = torch.tensor(X).to(device)
Y_tensor = torch.tensor(Y).to(device)

# Create a DataLoader
dataset = TensorDataset(X_tensor, Y_tensor)
dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

# Define model
model_module = importlib.import_module('archs.' + args.arch)
model = model_module.build_model(input_shape).to(device)
print(model)

# Define loss and optimizer
criterion = nn.MSELoss()  # Changed to MSE for demonstration, adjust as needed
optimizer = torch.optim.Adam(model.parameters())

# Train model
model.train()
for epoch in range(epochs):
    for batch_idx, (data, targets) in enumerate(dataloader):
        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output, targets)
        loss.backward()
        optimizer.step()
    print(f'Epoch {epoch+1}, Loss: {loss.item()}')

# Save model
model.eval() 
example_input = torch.rand(1, *input_shape)
scripted_model = torch.jit.trace(model, example_input)
scripted_model.save(f"{args.arch}_model.jit")
