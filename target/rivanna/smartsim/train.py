"""usage: python train.py [--config=config.yaml] -- It assumes that the config.yaml file is in the same directory as this script.
"""
# train.py
import sys
# import argparse
import importlib
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
from cloudmesh.common.util import banner



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

PROFILE = bool(config["system.profile"])
if PROFILE:
    import torch
    import torchvision.models as models
    from torch.profiler import profile, record_function, ProfilerActivity

    total_flops = 0


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
batch_requests = bool(config["experiment.batch_requests"])
replicas = int(config["experiment.replicas"])
num_gpus = int(config["experiment.num_gpus"])
# device_name = "cuda" if num_gpus > 0 else "cpu"
device_name = config["experiment.device"]


#try:
print("A")
model_module = importlib.import_module(f'archs.{arch}')
Model = model_module.Model

print("B")
# model_class = Model.model_batch(Model.input_shape)
print ("C")
#except:
#    Console.error(f"Model {arch} not defined in the archs directory")

input_shape  = Model.input_shape
output_shape = Model.output_shape
dtype = Model.dtype


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

X = torch.rand(samples, *input_shape, dtype=dtype)
Y = torch.rand(samples, *output_shape, dtype=dtype)

# Convert numpy arrays to PyTorch tensors
X_tensor = torch.tensor(X)
Y_tensor = torch.tensor(Y)

# Create a DataLoader
dataset = TensorDataset(X_tensor, Y_tensor)
dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

# Define model
#model_module = importlib.import_module(f'archs.{arch}')
#model = model_module.build_model(input_shape)
model = Model()
print(model)

# Define loss and optimizer
criterion = nn.MSELoss()  # Changed to MSE for demonstration, adjust as needed
optimizer = torch.optim.Adam(model.parameters())

StopWatch.stop("setup")

StopWatch.start("train")

def execution():
    pass

total_flops = 0

# Train model


def execution(data,targets,dataloader):
    optimizer.zero_grad()
    output = model(data)
    loss = criterion(output, targets)
    loss.backward()
    optimizer.step()

if not  PROFILE:
    for epoch in range(epochs):
        model.train()
        for batch_idx, (data, targets) in enumerate(dataloader):
            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, targets)
            loss.backward()
            optimizer.step()
        print(f'Epoch {epoch+1}, Loss: {loss.item()}')

else:

    for epoch in range(epochs):
        model.train()
        for batch_idx, (data, targets) in enumerate(dataloader):
            with profile(activities=[
                    ProfilerActivity.CUDA,
                    ProfilerActivity.CPU,
                    ], 
                    record_shapes=True, 
                    with_flops=True) as prof:

                optimizer.zero_grad()
                output = model(data)
                loss = criterion(output, targets)
                loss.backward()
                optimizer.step()
            events = prof.events()
            flops = sum([int(evt.flops) for evt in events]) 
            print(f"FLOPS: {flops}")
            total_flops += flops

        print(f'Epoch {epoch+1}, Loss: {loss.item()}')



   

    print("Total FLOPS: ", total_flops)
    print("Epochs: ", epochs)   

StopWatch.stop("train")

#if PROFILE:
#    print(prof.key_averages().table(sort_by="cpu_time_total", row_limit=10))

# Save model
StopWatch.start("save")

## model = getattr(models, model_name)(pretrained=True)

model.to(torch.device(device_name))
model.eval()

example_input = torch.rand(1, *input_shape, device=device_name)
scripted_model = torch.jit.trace(model, example_input)

torch.jit.save(scripted_model, f"{arch}_model.jit")


if os.path.isfile(f"{arch}_model.jit"):
    Console.ok("JIT file exists")
else:
    Console.error("JIT file does not exist")



StopWatch.stop("save")


tag = f"device={device_name},mode={mode},repeat={repeat},arch={arch},samples={samples},epochs={epochs},batch_size={batch_size}"

StopWatch.benchmark(tag=tag, 
                    attributes=["timer", "time", "start", "tag", "msg"])


print (tag)

banner("loading model")
m = torch.jit.load(f"{arch}_model.jit")
m.eval()

banner("Model loded")