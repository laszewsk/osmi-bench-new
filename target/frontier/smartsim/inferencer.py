import numpy as np
import time
import torch

from cloudmesh.common.FlatDict import FlatDict
from cloudmesh.common.StopWatch import StopWatch
from cloudmesh.common.console import Console    

from smartredis import Client
from tqdm import tqdm
import importlib

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




try:
    model_module = importlib.import_module(f'archs.{arch}')

    model_class = model_module.BuildModel(model_module.input_shape)
except:
    Console.error(f"Model {arch} not defined in the archs directory")

input_shape  = model_class.input_shape
output_shape = model_class.output_shape
dtype = model_class.dtype

model = model_class.model_batch(batch)


# Create the SmartSim client
client = Client(cluster=False)

times = list()
StopWatch.stop("inferencer-init")    


StopWatch.start("inferencer-loop")    
for _ in tqdm(range(num_requests)):
    tik = time.perf_counter()
    client.put_tensor("input", torch.rand(model['shape']).numpy())
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
