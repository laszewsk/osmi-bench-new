import torch

# part of cloudmesh

def print_gpu_device_properties():
    try:
        for i in range(torch.cuda.device_count()):
            print (i)
            print("  Name  :", torch.cuda.get_device_properties(i).name)
            print("  Memory:", torch.cuda.get_device_properties(i).total_memory)
            print("  Cores :", torch.cuda.get_device_properties(i).multi_processor_count)
    except:
        print ("This is not a cuda device")