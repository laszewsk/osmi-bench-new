# Setup environment
module use /lustre/orion/gen150/world-shared/smartsim-2023/workshop_scratch/ashao/OLCF_SmartSim2023/modulefiles
module load py-smartsim-0.5.0-gcc-11.2.0

# Test if PyTorch can see the GPU
import torch; torch.cuda.device_count()

# Start RedisAI
python startdb.py

# Test if server is running
lsof -i :6780

# Run benchmark
python benchmark.py
