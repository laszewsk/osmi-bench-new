import argparse
import importlib
import io
import numpy as np
import os
import time
import torch

from smartredis import Client
from tqdm import tqdm

from cloudmesh.common.StopWatch import StopWatch
from cloudmesh_pytorchinfo import print_gpu_device_properties
from cloudmesh.gpu.gpu import Gpu
from pprint import pprint
from cloudmesh.common.FlatDict import FlatDict
from cloudmesh.common.console import Console

StopWatch.start("init")


config = FlatDict()
config.load("config.yaml")

print(config)

# check experiment parameters

terminate = False
for key in ["experiment.arch",
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
samples = int(config["experiment.samples"])
epochs = int(config["experiment.epochs"])
batch = batch_size = int(config["experiment.batch_size"])    
arch = config["experiment.arch"]
num_requests = requests = int(config["experiment.requests"])

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


isd = lambda a, b, c : {'inputs': a, 'shape': b, 'dtype': c}

models = {
          'small_lstm': isd('inputs', (batch, 8, 48), np.float32),
          'medium_cnn': isd('inputs', (batch, 101, 82, 9), np.float32),
          'large_tcnn': isd('inputs', (batch, 3, 101, 82, 9), np.float32),
          'swmodel': isd('dense_input', (batch, 3778), np.float32),
          'lwmodel': isd('dense_input', (batch, 1426), np.float32),
         }
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
data = np.array(np.random.random(models[arch]['shape']), dtype=models[arch]['dtype'])

# Define model
model_module = importlib.import_module('archs.' + arch)
torch_model = model_module.build_model(input_shape)
print(torch_model)

# Load the TorchScript model
model_path = f'{arch}_model.jit'
torch_model = torch.jit.load(model_path)
torch_model.eval()  # Ensure the model is in evaluation mode

# Serialize the loaded TorchScript model into a byte buffer
model_buffer = io.BytesIO()
torch.jit.save(torch_model, model_buffer)
model_buffer.seek(0)  # Reset buffer position to the beginning

# Get the database address and create a SmartRedis client
client = Client(address="localhost:6780", cluster=False)

StopWatch.stop("setup")

StopWatch.start("inference")    

times = list()
# 

# set SR_LOG_LEVEL to switch off redis logging
for _ in tqdm(range(num_requests)):
    tik = time.perf_counter()
    client.put_tensor("input", torch.rand(models[arch]['shape']).numpy())
    # put the PyTorch CNN in the database in GPU memory
    # BUG NEEDS GPU LIST
    client.set_model("cnn", model_buffer.getvalue(), "TORCH", device="GPU:0")
    #client.set_model("cnn", model_buffer.getvalue(), "TORCH", device="CPU")
    # execute the model, supports a variable number of inputs and outputs
    client.run_model("cnn", inputs=["input"], outputs=["output"])
    # get the output
    output = client.get_tensor("output")
    #print(f"Prediction: {output}")
    tok = time.perf_counter()
    times.append(tok - tik)

elapsed = sum(times)
avg_inference_latency = elapsed/num_requests

print(f"elapsed time: {elapsed:.1f}s | average inference latency: {avg_inference_latency:.3f}s | 99th percentile latency: {np.percentile(times, 99):.3f}s | ips: {1/avg_inference_latency:.1f}")
StopWatch.stop("inference")

tag = f"elapsed time={elapsed:.1f}s,average inference latency={avg_inference_latency:.3f}s,99th percentile latency={np.percentile(times, 99):.3f}s,ips={1/avg_inference_latency:.1f}"

StopWatch.benchmark(tag=tag, 
                    attributes=["timer", "time", "start", "tag", "msg"])