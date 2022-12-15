# setup - on summit login node

singularity pull docker://haproxy

# get compute node 

# standard production run
bsub -Is -P ARD143 -nnodes 1 -W 2:00 $SHELL

# debug node
bsub -Is -q debug -P ARD143 -nnodes 1 -W 1:00 $SHELL

# modify both models.conf and models.py to be consistent with your models

# from launch/batch node

    # launch the TFS servers
    ./1_start_tfs_servers.sh

    # start the load balancer  
    ./2_start_load_balancers.sh

    # run the benchmark sweep
    ./3_run_benchmarks.sh # currently this is using tfs_grpc_client.py
                          # but should be changed to using benchmark.py in the future

    # run an individual benchmark
    python benchmark.py -b 32 -m lwmodel -n 1024

# submit batch job
bsub batch.lsf   # Summit
bsub < batch.lsf # SCOUT
