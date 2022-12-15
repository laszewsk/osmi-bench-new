# OSMI-Bench

The Open Surrogate Model Inference (OSMI) Benchmark is a distributed inference benchmark
for machine-learned surrogate models and is described in the following paper:

Brewer, Wesley, Daniel Martinez, Mathew Boyer, Dylan Jude, Andy Wissink, Ben Parsons, Junqi Yin, and Valentine Anantharaj. "Production Deployment of Machine-Learned Rotorcraft Surrogate Models on HPC." In 2021 IEEE/ACM Workshop on Machine Learning in High Performance Computing Environments (MLHPC), pp. 21-32. IEEE, 2021.

Available from https://ieeexplore.ieee.org/abstract/document/9652868

# Setup environment - on summit login node

    singularity pull docker://haproxy

## Get compute node 

    # Standard production run

    bsub -Is -P ARD143 -nnodes 1 -W 2:00 $SHELL

Note: replace ARD143 with subproject number

## Debug node

    bsub -Is -q debug -P ARD143 -nnodes 1 -W 1:00 $SHELL

# Modify both models.conf and models.py to be consistent with your models

# From launch/batch node

    # launch the TFS servers
    ./1_start_tfs_servers.sh

    # start the load balancer  
    ./2_start_load_balancers.sh

    # run the benchmark sweep
    ./3_run_benchmarks.sh # currently this is using tfs_grpc_client.py
                          # but should be changed to using benchmark.py in the future

    # run an individual benchmark
    python benchmark.py -b 32 -m lwmodel -n 1024

# Submit batch job
bsub batch.lsf 
