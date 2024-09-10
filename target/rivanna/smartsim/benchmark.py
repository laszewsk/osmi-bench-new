import importlib
import io
import numpy as np
import sys
import torch

from pathlib import Path

from smartsim import Experiment
from smartredis import Client

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

StopWatch.start("benchmark-init")

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
mode = "train"
repeat = int(config["experiment.repeat"])
samples = int(config["experiment.samples"])
epochs = int(config["experiment.epochs"])
batch = batch_size = int(config["experiment.batch_size"])    
arch = config["experiment.arch"]
requests = int(config["experiment.requests"])
batch_requests = bool(config["experiment.batch_requests"])
replicas = int(config["experiment.replicas"])
num_gpus = int(config["experiment.num_gpus"])
# device_name = "cuda" if num_gpus > 0 else "cpu"
device_name = config["experiment.device"]





model_module = importlib.import_module(f'archs.{arch}')
Model = model_module.Model()

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
    
StopWatch.stop("benchmark-init")

StopWatch.start("benchmark-setup")
# smartredis requires numpy arrays
#data = np.array(torch.rand(Model.model_batch(batch)['shape']), 
#                dtype=dtype)


data = torch.rand(Model.model_batch(batch)['shape'], dtype=dtype).numpy()


# Define model
torch_model = Model
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

StopWatch.stop("benchmark-setup")


StopWatch.start("benchmark-run-ensemble")

exp.start(ensemble, block=True)

StopWatch.stop("benchmark-run-ensemble")

tag = f"prg=benchmark.py,mode={mode}," \
      f"repeat={repeat}," \
      f"arch={arch}," \
      f"samples={samples}," \
      f"epochs={epochs}," \
      f"batch_size={batch_size}"

StopWatch.benchmark(tag=tag, 
                    attributes=["timer", "time", "start", "tag", "msg"])