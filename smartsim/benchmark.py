import argparse
import importlib
import io
import numpy as np
import os
import sys
import torch

from smartsim import Experiment
from smartredis import Client
from tqdm import tqdm
from pathlib import Path

REDISAI_DEVICES = {
    'cpu': 'CPU',
    'cuda': 'GPU'
}
# Get the full path to the inferencer application
INFERENCER = Path(__file__).parent.resolve() / "inferencer.py"
# If batching is requested, wait 10ms to aggregate inference requests
DEFAULT_BATCH_TIMEOUT = 10

parser = argparse.ArgumentParser()
archs = [s.split('.')[0] for s in os.listdir('archs') if s[0:1] != '_']
parser.add_argument('arch', type=str, choices=archs, help='Type of neural network architectures')
parser.add_argument('device', type=str, choices = ["cpu", "cuda"], default="cpu", help='The target device for the trained model')
parser.add_argument('-b', '--batch', type=int, default=1, help='batch size')
parser.add_argument('-n', default=128, type=int, help='number of requests')
parser.add_argument('replicas', default=1, type=int, help='Number of simultaneous inference applications to run')
parser.add_argument('use_batching', type=bool, action='store_true', help='Batch inference requests from multiple clients')
args = parser.parse_args()

redisai_device = REDISAI_DEVICES[args.device]

# TODO: Refactor this to a namedtuple
isd = lambda a, b, c : {'inputs': a, 'shape': b, 'dtype': c}

models = {
          'small_lstm': isd('inputs', (args.batch, 8, 48), np.float32),
          'medium_cnn': isd('inputs', (args.batch, 101, 82, 9), np.float32),
          'large_tcnn': isd('inputs', (args.batch, 3, 101, 82, 9), np.float32),
          'swmodel': isd('dense_input', (args.batch, 3778), np.float32),
          'lwmodel': isd('dense_input', (args.batch, 1426), np.float32),
         }

input_shape = models[args.arch]['shape']
data = np.array(np.random.random(input_shape, dtype=models[args.arch]['dtype']))

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

# Make a SmartSim experiment and start the server
exp = Experiment("Inference-Benchmark", launcher="local")
# TODO: Update this to allow inference server to be scaled across multiple nodes
db = exp.create_database(port=6780, interface="lo")
exp.generate(db)
exp.start(db)

# Get the database address and create a SmartRedis client
client = Client(address=db.get_address()[0], cluster=False)
# put the PyTorch CNN in the database in GPU memory
batch_timeout = DEFAULT_BATCH_TIMEOUT if args.use_batching else 0
batch_size = args.batch*args.replicas if args.use_batching else args.batch
client.set_model(
    "cnn",
    model_buffer.getvalue(),
    "TORCH",
    device=redisai_device,
    min_batch_timeout=batch_timeout,
    batch_size=batch_size
)

# Create the inferencer representative
inferencer_run_settings = exp.create_run_settings(
    exe=sys.executable,
    exe_args=[
        INFERENCER,
        args.arch,
        f"-b {args.batch}",
        f"-n {args.n}",
    ]
)

ensemble = exp.create_ensemble(
    "inferencer",
    inferencer_run_settings,
    replicas=args.replicas
)
# Make sure that keys from each replica do not overlap
for model in ensemble:
    model.register_incoming_entity(model)

exp.start(ensemble, block=True)