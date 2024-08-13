#SBATCH

module load apptainer
module load cuda/12.4.1

APPTAINER_IMAGE=cloudmesh-smartsim
UID=`id -u`
GID=`id -g`
CODE=/home/green/Desktop/osmi-bench-new/target/rivanna/smartsim

pwd

apptainer run --nv  -it cloudmesh/smartsim bash -c "python3 train.py"

# make train