import argparse
import importlib
import io
import numpy as np
import sys
import time
import torch

from pathlib import Path

from smartsim import Experiment
from smartredis import Client
from tqdm import tqdm

from cloudmesh.common.StopWatch import StopWatch
from cloudmesh_pytorchinfo import print_gpu_device_properties
from cloudmesh.gpu.gpu import Gpu
from pprint import pprint
from cloudmesh.common.FlatDict import FlatDict
from cloudmesh.common.console import Console

# Get the full path to the inferencer application
INFERENCER = Path(__file__).parent.resolve() / "inferencer.py"
# If batching is requested, wait 10ms to aggregate inference requests
DEFAULT_BATCH_TIMEOUT = 10

StopWatch.start("init")

config = FlatDict()
config.load("config.yaml")

print(config)

# check experiment parameters
required_params = [
    "experiment.arch",
    "experiment.repeat", 
    "experiment.samples", 
    "experiment.epochs", 
    "experiment.batch_size",
    "experiment.requests",
    "experiment.batch_requests",
    "experiment.replicas",
    "experiment.num_gpus",
]

terminate = False
for key in required_params:
    if key not in config:
        Console.error(f"{key} not defined in config.yaml")
        terminate = True

if terminate:
    sys.exit()

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
num_gpus = config["experiment.num_gpus"]

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

# Make a SmartSim experiment and start the server
exp = Experiment("Inference-Benchmark", launcher="local")

# TODO: Update this to allow inference server to be scaled across
# multiple nodes, to launch on 
db = exp.create_database(port=6780, interface="lo")
exp.generate(db)
exp.start(db)

# Get the database address and create a SmartRedis client
client = Client(address=db.get_address()[0], cluster=False)
# put the PyTorch CNN in the database in GPU memory
batch_timeout = DEFAULT_BATCH_TIMEOUT if batch_requests else 0
batch_size = batch*replicas if batch_requests else batch
min_batch_size = batch_size if batch_requests else 0
client.set_model_multigpu(
    "cnn",
    model_buffer.getvalue(),
    "TORCH",
    first_gpu=0,
    num_gpus=num_gpus,
    min_batch_timeout=batch_timeout,
    min_batch_size=min_batch_size,
    batch_size=batch_size
)

# Create the inferencer representative
inferencer_run_settings = exp.create_run_settings(
    exe=sys.executable,
    exe_args=str(INFERENCER),
)

ensemble = exp.create_ensemble(
    "inferencer",
    run_settings=inferencer_run_settings,
    replicas=replicas
)
ensemble.attach_generator_files(to_symlink="config.yaml")

# Make sure that keys from each replica do not overlap
for model in ensemble:
    model.register_incoming_entity(model)

exp.generate(ensemble)

StopWatch.stop("setup")

exp.start(ensemble, block=True)



tag_base = f"prg=benchmark.py,mode={mode},repeat={repeat},arch={arch},samples={samples},epochs={epochs},batch_size={batch_size}"

tag = f"{tag_base},elapsed time={elapsed:.1f}s,average inference latency={avg_inference_latency:.3f}s,99th percentile latency={np.percentile(times, 99):.3f}s,ips={1/avg_inference_latency:.1f}"

StopWatch.benchmark(tag=tag, 
                    attributes=["timer", "time", "start", "tag", "msg"])