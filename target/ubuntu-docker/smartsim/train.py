"""usage: python train.py {small_lstm|medium_cnn|large_tcnn}"""
import argparse
import importlib
import numpy as np
import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from cloudmesh.common.StopWatch import StopWatch
from cloudmesh_pytorchinfo import print_gpu_device_properties
from cloudmesh.gpu.gpu import Gpu
from pprint import pprint

StopWatch.start("init")
parser = argparse.ArgumentParser()
archs = [s.split('.')[0] for s in os.listdir('archs') if s[0:1] != '_']
parser.add_argument('arch', type=str, choices=archs, help='Type of neural network architectures')
parser.add_argument('-b', '--batch', type=int, default=1, help='batch size')
args = parser.parse_args()

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

print_gpu_device_properties()
try:
    gpu = Gpu()

    gpu.probe()
    #print ("Vendor:", gpu.smi())
    information = gpu.system()
    pprint(information)
except:
    print ("This is not a cuda device")


StopWatch.stop("init")

StopWatch.start("setup")

X = np.random.rand(samples, *input_shape).astype(np.float32)
Y = np.random.rand(samples, *output_shape).astype(np.float32)

# Convert numpy arrays to PyTorch tensors
X_tensor = torch.tensor(X)
Y_tensor = torch.tensor(Y)

# Create a DataLoader
dataset = TensorDataset(X_tensor, Y_tensor)
dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

# Define model
model_module = importlib.import_module('archs.' + args.arch)
model = model_module.build_model(input_shape)
print(model)

# Define loss and optimizer
criterion = nn.MSELoss()  # Changed to MSE for demonstration, adjust as needed
optimizer = torch.optim.Adam(model.parameters())

StopWatch.stop("setup")

StopWatch.start("train")
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
StopWatch.stop("train")


# Save model
StopWatch.start("save")

device_name = "cuda"  # Default to GPU

# device_name = "cpu"

## model = getattr(models, model_name)(pretrained=True)
model.to(torch.device(device_name))
model.eval()

example_input = torch.rand(1, *input_shape, device=device_name)
scripted_model = torch.jit.trace(model, example_input)
scripted_model.save(f"{args.arch}_model.jit")



StopWatch.stop("save")

StopWatch.benchmark()