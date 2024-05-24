# OSMI Setup on various target machines

This documentation is evoling


## Setup environment on Frontier

This is the original documentation from Frontier

```bash
module use /lustre/orion/gen150/world-shared/smartsim-2023/workshop_scratch/ashao/OLCF_SmartSim2023/modulefiles
module load py-smartsim-0.5.0-gcc-11.2.0
```

### Test if PyTorch can see the GPU

```
python -c 'import torch; print(torch.cuda.device_count())'
```

### Start RedisAI

```
python startdb.py
```

### Test if server is running

```
lsof -i :6780
```

### Train model

```
time python train.py small_lstm
```

### Run benchmark

```
time python benchmark.py small_lstm
```

## Setup on Rivanna 

### Get the code

login to rivanna with ssh (see infomall tutorial)

```bash
ssh rivanna
cd /scratch/$USER
git clone git@github.com:laszewsk/osmi-bench-new.git
cd /scratch/$USER/osmi-bench-new
```

if you frequently work on it add into your .bashrc file

export PROJECT_OSMI2=/scratch/$USER/osmi-bench-new
alias osmi2='cd $PROJECT_OSMI2'

this way you can enter the command osmi2 to jump to the right directory

### activate the environment env.sh

To set up the environment on rivanna use 

```bash
source env.sh
```
This includes the setup of the modules installed on Rivanna

```bash
# module load cuda/11.4.2
# module load nccl/2.18.3-CUDA-12.2.2
# module load cudnn/8.9.4.25

# module purge
# module load intel-compilers/2023.1.0  impi/2021.9.0 python/3.9.16
# module load cmake/3.28.1
module purge
module load  gcc/11.4.0  openmpi/4.1.4
module load python/3.11.4
module load cuda/11.8.0
module load cudnn/8.9.7
```

### Install smartsim and other software


After this you can install in your python venv which is called OSMI2 the needed packages. You only need to do tha install onece.

```bash
pip install smartsim
pip install cloudmesh-common
pip install cloudmesh-gpu
wget https://github.com/git-lfs/git-lfs/releases/download/v3.5.1/git-lfs-linux-amd64-v3.5.1.tar.gz

tar xvf  git-lfs-linux-amd64-v3.5.1.tar.gz

export PATH=$PATH:$PWD/git-lfs-3.5.1
```

### Build smartsim

smart clean
smart build --device gpu 
smart validate

### Run the benchmark

```bash
python -c 'import torch; print(torch.cuda.device_count())'
python startdb.py & 
lsof -i :6780
time python train.py small_lstm
time python benchmark.py small_lstm
```

## Using Singularity container

```bash
cd images
make image
```

export VERSION=24.04-py3
export IMAGE=cloudmesh-smartsim-${VERSION}

python startdb.py & 
lsof -i :6780
apptainer exec $IMAGE python startdb.py
apptainer exec $IMAGE python train.py small_lstm


## Ubuntu

For Ubuntu the setup is as follows

```bash
sudo apt install git-lfs
python3.11 -m venv ~/ENV11
source ~/ENV11/bin/activate

pip install onnxruntime-gpu --extra-index-url https://aiinfra.pkgs.visualstudio.com/PublicPackages/_packaging/onnxruntime-cuda-12/pypi/simple/

python3 -m pip install tensorflow[and-cuda]
pip install cloudmesh-common
pip install cloudmesh-gpu
pip install smartsim

smart clean
smart build --device gpu 
smart validate
```

### Run the benchmark

```bash
python -c 'import torch; print(torch.cuda.device_count())'
python startdb.py & 
lsof -i :6780
time python train.py small_lstm
time python benchmark.py small_lstm
```




