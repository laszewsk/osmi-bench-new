import argparse
import numpy as np
import os
import time
import torch

from cloudmesh.common.FlatDict import FlatDict
from cloudmesh.common.StopWatch import StopWatch

from smartredis import Client
from tqdm import tqdm

StopWatch.start("inferencer-init")    

config = FlatDict()
config.load("config.yaml")

# Refactor to a dataclass
# Parameters
mode = "inference"
repeat = int(config["experiment.repeat"])
samples = int(config["experiment.samples"])
epochs = int(config["experiment.epochs"])
batch = batch_size = int(config["experiment.batch_size"])    
arch = config["experiment.arch"]
num_requests = requests = int(config["experiment.requests"])
batch_requests = config["experiment.batch_requests"] 
replicas = config["experiment.replicas"]

isd = lambda a, b, c : {'inputs': a, 'shape': b, 'dtype': c}

models = {
          'small_lstm': isd('inputs', (batch, 8, 48), np.float32),
          'medium_cnn': isd('inputs', (batch, 101, 82, 9), np.float32),
          'large_tcnn': isd('inputs', (batch, 3, 101, 82, 9), np.float32),
          'swmodel': isd('dense_input', (batch, 3778), np.float32),
          'lwmodel': isd('dense_input', (batch, 1426), np.float32),
         }

# Create the SmartSim client
client = Client(cluster=False)

times = list()
StopWatch.stop("inferencer-init")    


StopWatch.start("inferencer-loop")    
for _ in tqdm(range(num_requests)):
    tik = time.perf_counter()
    client.put_tensor("input", torch.rand(models[arch]['shape']).numpy())
    # execute the model, supports a variable number of inputs and outputs
    client.run_model("cnn", inputs=["input"], outputs=["output"])
    # get the output
    output = client.get_tensor("output")
    #print(f"Prediction: {output}")
    tok = time.perf_counter()
    times.append(tok - tik)

elapsed = sum(times)
avg_inference_latency = elapsed/num_requests

StopWatch.stop("inferencer-loop")

StopWatch.start("inferencer-finalize")

print(f"elapsed time: {elapsed:.1f}s | average inference latency: {avg_inference_latency:.3f}s | 99th percentile latency: {np.percentile(times, 99):.3f}s | ips: {1/avg_inference_latency:.1f}")

tag_base = f"prg=inferencer.py,mode={mode},repeat={repeat},arch={arch},samples={samples},epochs={epochs},batch_size={batch_size}"

tag = f"{tag_base},elapsed time={elapsed:.1f}s,average inference latency={avg_inference_latency:.3f}s,99th percentile latency={np.percentile(times, 99):.3f}s,ips={1/avg_inference_latency:.1f}"


StopWatch.stop("inferencer-finalize")

StopWatch.benchmark(tag=tag, 
                    attributes=["timer", "time", "start", "tag", "msg"])
