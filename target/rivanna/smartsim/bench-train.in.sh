#!/usr/bin/env bash

#SBATCH --job-name={identifier}
#SBATCH --output=osmi-{identifier}-%u-%j.out
#SBATCH --error=osmi-{identifier}-%u-%j.err
{slurm.sbatch}
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --mem=64GB
#SBATCH --cpus-per-task=1
#SBATCH --time={ee.time}

WORKDIR=/scratch/$USER/osmi-bench-new/target/rivanna/smartsim

CONTAINER=$WORKDIR/images/cloudmesh-smartsim.sif

PROGRESS() {
    echo "# ###############################################"
    echo "# cloudmesh status=$1 progress=$2 msg=$3 pid=$$"
    echo "# ###############################################"
}

PROGRESS "running" "configure" 1
module load apptainer
module load cuda/12.4.1

PROGRESS "running" "pwd" 2

pwd


echo "============================================================"
echo "PROJECT_ID: {identifier}"
echo "REPEAT: {experiment.repeat}"

PROGRESS "running" "training" 3
PROGRESS "running" "train" 2


# export APPTAINERENV_CUDA_VISIBLE_DEVICES=0 
apptainer run --nv  $CONTAINER bash -c "python3 train.py"

PROGRESS "running" "finished" 2

# make train