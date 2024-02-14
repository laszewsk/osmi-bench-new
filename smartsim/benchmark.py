import io
import numpy as np
import time
import torch
import torch.nn as nn
import torch.nn.functional as F

from tqdm import tqdm
from smartredis import Client


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, 3, 1)
        self.conv2 = nn.Conv2d(32, 64, 3, 1)
        self.dropout1 = nn.Dropout(0.25)
        self.dropout2 = nn.Dropout(0.5)
        self.fc1 = nn.Linear(9216, 128)
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
        x = self.conv1(x)
        x = F.relu(x)
        x = self.conv2(x)
        x = F.relu(x)
        x = F.max_pool2d(x, 2)
        x = self.dropout1(x)
        x = torch.flatten(x, 1)
        x = self.fc1(x)
        x = F.relu(x)
        x = self.dropout2(x)
        x = self.fc2(x)
        output = F.log_softmax(x, dim=1)
        return output

# Initialize an instance of our CNN model
n = Net()
n.eval()

# prepare a sample input to trace on (random noise is fine)
example_forward_input = torch.rand(1, 1, 28, 28)

def create_torch_model(torch_module, example_forward_input):

    # perform the trace of the nn.Module.forward() method
    module = torch.jit.trace(torch_module, example_forward_input)

    # save the traced module to a buffer
    model_buffer = io.BytesIO()
    torch.jit.save(module, model_buffer)
    return model_buffer.getvalue()

traced_cnn = create_torch_model(n, example_forward_input)

#client = Client(address=db.get_address()[0], cluster=False)
client = Client(address="localhost:6780", cluster=False)

num_requests = 128

times = list()

for _ in tqdm(range(num_requests)):
    tik = time.perf_counter()
    client.put_tensor("input", torch.rand(20, 1, 28, 28).numpy())
    # put the PyTorch CNN in the database in GPU memory
    client.set_model("cnn", traced_cnn, "TORCH", device="CPU")
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
