import torch
import torchvision.models as models
from torch.profiler import profile, record_function, ProfilerActivity
import humanize

total_flops = 0

class Profiler:

    def __init__(self):

        self.prof = profile(activities=[
            ProfilerActivity.CUDA,
            ProfilerActivity.CPU,
            ], 
            record_shapes=True, 
            with_flops=True)
        self.flops = 0

    def start(self):
        self.prof.start()

    def stop(self):
        self.prof.stop()

    def benchmark(self, row_limit=100):
        events = self.prof.events()
        self.flops = sum([int(evt.flops) for evt in events]) 
        print(f"FLOPS: {self.flops} ({humanize.intword(self.flops)})")
        print(self.prof.key_averages(group_by_stack_n=5).table(sort_by="self_cuda_time_total", row_limit=row_limit))
        