# OSMI Setup on docker

This documentation is evoling


## Code 

The code is located in 

https://github.com/laszewsk/osmi-bench-new


Please chec it out and go to the target directory for ubuntu, smartsim, and cloudmesh

```bash
git clone https://github.com/laszewsk/osmi-bench-new
cd osmi-bench-new/target/ubuntu-docker-cm/smartsim
```


## Ubuntu - docker

To installl the cloudmesh experiment executor please install first a python virtual envwith 

```bash
python3.11 -m venv ~/ENV11
source ~/ENV11/bin/activate
pip install cloudmesh-ee
pip install cloudmesh-gpu
pip install cloudmesh-rivanna
```

To create the docker image you can say

```bash
make image
```


For Ubuntu the setup is as follows

Please note that the next steps also include training and write a new model into the experminet directory. This is  useful as it allows to completely independently validate training and infernce in separate experiment directories. In case the same training is to be used with different inferences, it could simply be put into a directory and if its not there training is called, while if it is there it could be read from the exiting file.

To set up the inference with the parameters identified in cloudmesh.in.yaml please run 


```bash
make infer-setup
```

To run them say 

```bash
make infer-run
```

Directories to run them are created in 

```
project-infer
```

In these directories you will find a complete copy of the code as well as the log file results what can then be mined.


## TRaining only run

To setup the training experiments only without inference based on the config.in.yaml file use 

```bash
make train-setup
```

To run them say 

```bash
make train-run
```

Directories to run them are created in 

```
project-train
```



# Appendix


## Ubuntu - Native

First we assume you have a valid python virtual env just in case

For Ubuntu the setup is as follows

```bash
sudo apt install git-lfs
python3.11 -m venv ~/ENV11
source ~/ENV11/bin/activate

pip install onnxruntime-gpu 
# --extra-index-url https://aiinfra.pkgs.visualstudio.com/PublicPackages/_packaging/onnxruntime-cuda-12/pypi/simple/

pip install cloudmesh-common cloudmesh-gpu

python3 -m pip install tensorflow[and-cuda]
python3 -m pip install torch


pip install smartsim
smart clean
smart build --device gpu 
smart validate
```

to run the code with config.yam say 

```
make train
make bench
```


### Run the benchmark by hand

```bash
python -c 'import torch; print(torch.cuda.device_count())'
python startdb.py & 
lsof -i :6780
time python train.py small_lstm
time python benchmark.py small_lstm
```


## Smartsim code for Docker

* <https://github.com/ashao/SmartSim/tree/docker-gpu>

make sure you get the branch docker-gpu

## Samrtsim tutorial

https://github.com/CrayLabs/SmartSim/pkgs/container/smartsim-tutorials

docker pull ghcr.io/craylabs/smartsim-tutorials:latest
docker run -p 8888:8888 ghcr.io/craylabs/smartsim-tutorials:latest
