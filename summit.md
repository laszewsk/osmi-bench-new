# Summit Instructions

1. Install software

    git clone https://code.ornl.gov/whb/osmi-bench.git

    cd osmi-bench

2. Setup environment

    . benchmark/env.sh

3. Train models

    cd models
    python train.py small_lstm
    python train.py medium_cnn
    python train.py large_tcnn

Note: Edit benchmark/models.conf to modify paths to point to the individual, e.g., /ccs/home/whbrewer/osmi-bench/models/small_lstm

4. Start up tensorflow model server

    jsrun -n1 tensorflow_model_server --port=8500 --rest_api_port=0 --model_config_file=models.conf >& $MEMBERWORK/$PROJ_ID/tfs.log &

Note: make sure PROJ_ID is set project identifier, e.g. `export PROJ_ID=ABC123`

5. Test that server is running

    jsrun -n1 lsof -i :8500

6. Test benchmark

    jsrun -n1 python tfs_grpc_client.py -m small_lstm -b 32 -n 10 localhost:8500
