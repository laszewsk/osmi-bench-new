import argparse
import numpy as np
import os
import time
import torch

from smartredis import Client
from tqdm import tqdm

REDISAI_DEVICES = {
    'cpu': 'CPU',
    'cuda': 'GPU'
}

parser = argparse.ArgumentParser()
archs = [s.split('.')[0] for s in os.listdir('archs') if s[0:1] != '_']
parser.add_argument('arch', type=str, choices=archs, help='Type of neural network architectures')
parser.add_argument('-b', '--batch', type=int, default=1, help='batch size')
parser.add_argument('-n', default=128, type=int, help='number of requests')
args = parser.parse_args()

isd = lambda a, b, c : {'inputs': a, 'shape': b, 'dtype': c}

models = {
          'small_lstm': isd('inputs', (args.batch, 8, 48), np.float32),
          'medium_cnn': isd('inputs', (args.batch, 101, 82, 9), np.float32),
          'large_tcnn': isd('inputs', (args.batch, 3, 101, 82, 9), np.float32),
          'swmodel': isd('dense_input', (args.batch, 3778), np.float32),
          'lwmodel': isd('dense_input', (args.batch, 1426), np.float32),
         }

# Get the database address and create a SmartRedis client
client = Client(cluster=False)

num_requests = 128

times = list()

for _ in tqdm(range(num_requests)):
    tik = time.perf_counter()
    client.put_tensor("input", torch.rand(models[args.arch]['shape']).numpy())
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
