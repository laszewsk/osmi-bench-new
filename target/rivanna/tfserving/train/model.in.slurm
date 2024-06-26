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

PROGRESS() {
    echo "# ###############################################"
    echo "# cloudmesh status=$1 progress=$2 msg=$3 pid=$$"
    echo "# ###############################################"
}

PROGRESS "running" "configure" 1

nvidia-smi

PROGRESS "running" "configure" 2

MODELS_DIR=./archs

export USER_SCRATCH=/scratch/$USER


RESULT_DIR=`pwd`

if [ -z "$OSMI_TARGET" ]; then
    echo "OSMI_TARGET is not set"
    exit 1
fi
export CONTAINER_DIR=$OSMI_TRAIN/images

module purge
module load apptainer

MODEL={experiment.model}
CONTAINER=$CONTAINER_DIR/tensorflow_23.10-tf2-py3.sif

echo "============================================================"
echo "PROJECT_ID: {identifier}"
echo "MODELS_DIR: $MODELS_DIR"
echo "MODEL: $MODEL"
echo "REPEAT: {experiment.repeat}"
echo "CONTAINER:" $CONTAINER

PROGRESS "running" "training" 3

# cd $MODELS_DIR

time apptainer exec --nv $CONTAINER python train.py $MODEL

PROGRESS "completed" "done" 100

#tr -cd '\11\12\15\40-\176' < $OUTPUT > tmp-output
#mv tmp-output $OUTPUT
