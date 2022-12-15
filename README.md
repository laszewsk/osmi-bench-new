# OSMI-Bench

The Open Surrogate Model Inference (OSMI) Benchmark is a distributed inference benchmark
for machine-learned surrogate models and is described in the following paper:

Brewer, Wesley, Daniel Martinez, Mathew Boyer, Dylan Jude, Andy Wissink, Ben Parsons, Junqi Yin, and Valentine Anantharaj. "Production Deployment of Machine-Learned Rotorcraft Surrogate Models on HPC." In 2021 IEEE/ACM Workshop on Machine Learning in High Performance Computing Environments (MLHPC), pp. 21-32. IEEE, 2021.

Available from https://ieeexplore.ieee.org/abstract/document/9652868

1. Setup environment - on Summit login node. Note that this benchmark is currently setup to module load open-ce/1.1.3-py38-0 and cuda/11.0.2. Users on other systems may `pip install -r requirements.txt`. In addition to TensorFlow and gRPC, users also need to install TensorFlow Serving and if wanting to use multiple GPUs may install an HAProxy Singularity container as follows:

        singularity pull docker://haproxy

2. Interactive usage:

        bsub -Is -P ARD143 -nnodes 1 -W 2:00 $SHELL

    *Note: replace ARD143 with subproject number*

    *Modify both models.conf and models.py to be consistent with your models*

4. Launch processes From launch/batch node on Summit:

        # launch the TFS servers
        ./1_start_tfs_servers.sh

        # start the load balancer  
        ./2_start_load_balancers.sh

        # run the benchmark sweep
        ./3_run_benchmarks.sh # currently this is using tfs_grpc_client.py
                              # but should be changed to using benchmark.py in the future

        # run an individual benchmark
        python benchmark.py -b 32 -m lwmodel -n 1024

5. Production run. First update parameters in batch.lsf, then submit to LSF scheduler:

        bsub batch.lsf 
