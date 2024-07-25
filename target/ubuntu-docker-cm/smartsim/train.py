"""usage: python train.py [--config=config.yaml] -- It assumes that the config.yaml file is in the same directory as this script.
"""
import sys
# import argparse
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
from cloudmesh.common.FlatDict import FlatDict
from cloudmesh.common.console import Console
import argparse


# Parse command line arguments
parser = argparse.ArgumentParser()
parser.add_argument("--config", default="config.yaml", help="Path to config file")
args = parser.parse_args()
config_file = args.config
if not os.path.isfile(config_file):
    print(f"Config file '{config_file}' does not exist")
    sys.exit()


StopWatch.start("init")

config = FlatDict()
config.load(config_file)

print(config)

# check experiment parameters

terminate = False
for key in ["experiment.arch",
            "experiment.repeat", 
            "experiment.samples", 
            "experiment.epochs", 
            "experiment.batch_size",
            "experiment.requests"]:
    if key not in config:
        Console.error(f"{key} not defined in config.yaml")
        terminate = True

if terminate:
    sys.exit()


# Parameters
mode = "train"
repeat = int(config["experiment.repeat"])
samples = int(config["experiment.samples"])
epochs = int(config["experiment.epochs"])
batch_size = int(config["experiment.batch_size"])    
arch = config["experiment.arch"]
requests = int(config["experiment.requests"])

# Compute synthetic data for X and Y
if arch == "small_lstm":
    input_shape = (8, 48)  # Sequence length, feature size
    output_shape = (24,)  # Total output size
elif arch == "medium_cnn":
    input_shape = (9, 101, 82)  # Channels, Height, Width
    output_shape = (101*82,)  # Flattened output size
elif arch == "large_tcnn":
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
model_module = importlib.import_module(f'archs.{arch}')
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
scripted_model.save(f"{arch}_model.jit")



StopWatch.stop("save")

tag = f"mode={mode},repeat={repeat},arch={arch},samples={samples},epochs={epochs},batch_size={batch_size}"

StopWatch.benchmark(tag=tag, 
                    attributes=["timer", "time", "start", "tag", "msg"])


print (tag)