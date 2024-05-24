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
import cloudmesh_pytorchinfo

StopWatch.start("init")
parser = argparse.ArgumentParser()
archs = [s.split('.')[0] for s in os.listdir('archs') if s[0:1] != '_']
parser.add_argument('arch', type=str, choices=archs, help='Type of neural network architectures')
parser.add_argument('-b', '--batch', type=int, default=1, help='batch size')
parser.add_argument('-n', default=128, type=int, help='number of requests')
args = parser.parse_args()

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


isd = lambda a, b, c : {'inputs': a, 'shape': b, 'dtype': c}

models = {
          'small_lstm': isd('inputs', (args.batch, 8, 48), np.float32),
          'medium_cnn': isd('inputs', (args.batch, 101, 82, 9), np.float32),
          'large_tcnn': isd('inputs', (args.batch, 3, 101, 82, 9), np.float32),
          'swmodel': isd('dense_input', (args.batch, 3778), np.float32),
          'lwmodel': isd('dense_input', (args.batch, 1426), np.float32),
         }
print_gpu_device_properties()
StopWatch.stop("init")

StopWatch.start("setup")
data = np.array(np.random.random(models[args.arch]['shape']), dtype=models[args.arch]['dtype'])

# Define model
model_module = importlib.import_module('archs.' + args.arch)
torch_model = model_module.build_model(input_shape)
print(torch_model)

# Load the TorchScript model
model_path = f'{args.arch}_model.jit'
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

num_requests = 128

times = list()

for _ in tqdm(range(num_requests)):
    tik = time.perf_counter()
    client.put_tensor("input", torch.rand(models[args.arch]['shape']).numpy())
    # put the PyTorch CNN in the database in GPU memory
    client.set_model("cnn", model_buffer.getvalue(), "TORCH", device="CPU")
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

StopWatch.benchmark()