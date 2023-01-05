# OSMI-Bench

The Open Surrogate Model Inference (OSMI) Benchmark is a distributed inference benchmark
for machine-learned surrogate models and is described in the following paper:

> Brewer, Wesley, Daniel Martinez, Mathew Boyer, Dylan Jude, Andy Wissink, Ben Parsons, Junqi Yin, and Valentine Anantharaj. "Production Deployment of Machine-Learned Rotorcraft Surrogate Models on HPC." In 2021 IEEE/ACM Workshop on Machine Learning in High Performance Computing Environments (MLHPC), pp. 21-32. IEEE, 2021.

Available from https://ieeexplore.ieee.org/abstract/document/9652868. Note that OSMI-Bench differs from SMI-Bench described in the paper only in that the models that are used in OSMI are trained on synthetic data, whereas the models in SMI were trained using data from CFD simulations. 

# Instructions

1. Setup environment - on Summit login node. Note that this benchmark is currently setup to `module load open-ce/1.1.3-py38-0` and `module load cuda/11.0.2`. Users on other systems may `pip install -r requirements.txt`. In addition to TensorFlow and gRPC, users also need to install TensorFlow Serving and if wanting to use multiple GPUs may install an HAProxy Singularity container as follows:

        > singularity pull docker://haproxy

2. Interactive usage:

        > bsub -Is -P ARD143 -nnodes 1 -W 2:00 $SHELL

    *Note: replace ARD143 with subproject number*
    *Modify both models.conf and models.py to be consistent with your models*

3. Preparing model 

    Generate the model in the models directory using:

        > python train.py medium

    Check the model output:

        > saved_model_cli show --all --dir mymodel

    Update name and path in models.conf file. Make sure name of model is defined in models parameter in tfs_grpc_client.py. 

    Launch TensorFlow Serving:

        > tensorflow_model_server --port=8500 --rest_api_port=0 --model_config_file=models.conf >& log & 

    Make sure TF Serving started correctly:

        > lsof -i :8500 

    *Should list a process with status LISTEN if working correctly.*

    Send packets to be inference:

        > python tfs_grpc_client.py -m mymodel -b 32 -n 10 localhost:8500

    Output of timings should be in file results.csv.

4. Launch processes (from launch/batch node)

    If running on more than one GPU, will need to launch up multiple TF Serving processes, each one bound to a specific GPU. This is what the script 1_start_tfs_servers.sh will do. 2_start_load_balancers.sh will launch HAProxy load balancers on each compute node. 3_run_benchmark.sh automates the launch of multiple concurrent client threads for a sweep of batch sizes. 

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
