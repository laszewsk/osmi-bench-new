import torch
import torchvision.models as models
from torch.profiler import profile, record_function, ProfilerActivity

total_flops = 0

def profile(f):

    with profile(activities=[
            ProfilerActivity.CUDA,
            ProfilerActivity.CPU,
            ], 
            record_shapes=True, 
            with_flops=True) as prof:
        output = model(data)
    events = prof.events()
    flops = sum([int(evt.flops) for evt in events]) 
    total_flops += flops
