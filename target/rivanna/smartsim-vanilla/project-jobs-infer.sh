arch=small_lstm samples=100 epochs=5 batch_size=1 requests=128 directive=a100 repeat=1 batch_requests=True replicas=1 num_gpus=1 device=cuda cd /sfs/weka/scratch/thf2bn/osmi-bench-new/target/rivanna/smartsim/project-infer/arch_small_lstm_samples_100_epochs_5_batch_size_1_requests_128_directive_a100_repeat_1_batch_requests_True_replicas_1_num_gpus_1_device_cuda && sbatch bench-infer.sh
