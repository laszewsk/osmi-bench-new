export USER_SCRATCH=/scratch/$USER
export USER_LOCALSCRATCH=/localscratch/$USER

export CLOUDMESH_CONFIG_DIR=$USER_SCRATCH/.cloudmesh
export APPTAINER_CACHEDIR=$USER_SCRATCH/.apptainer/cache

#export OSMI_PROJECT=`pwd`
export OSMI_PROJECT="$OSMI_PROJECT_BASE"
export OSMI_TARGET=$OSMI_PROJECT/osmi-bench-new/target/rivanna
export OSMI_TRAIN=$OSMI_PROJECT/osmi-bench-new/target/rivanna/tfserving/train

export VENV=OSMI2

export TARGET=rivanna

mkdir -p $APPTAINER_CACHEDIR

echo "============================================================="
echo "OSMI_PROJECT_BASE    " $OSMI_PROJECT_BASE
echo "USER_SCRATCH:        " $USER_SCRATCH
echo "USER_LOCALSCRATCH:   " $USER_LOCALSCRATCH
echo "CLOUDMESH_CONFIG_DIR:" $CLOUDMESH_CONFIG_DIR
echo "APPTAINER_CACHEDIR:  " $APPTAINER_CACHEDIR
echo "OSMI_PROJECT:        " $OSMI_PROJECT
echo "OSMI_TARGET:         " $OSMI_TARGET
echo "OSMI_TRAIN:          " $OSMI_TRAIN
echo "============================================================="

export PATH=$PATH:$PWD/git-lfs-3.5.1

echo -n "Load   apptainer ... "
module load apptainer
echo "done"


echo -n "Load   python/3.11.4, cuda/12.2.2 ... " 
module load gcc/11.4.0  openmpi/4.1.4 python/3.11.4 
# module load cuda/12.2.2
# module load nccl/2.18.3-CUDA-12.2.2
module load cuda/11.8.0
module load cudnn/8.9.7

#module load cuda/11.4.2
#module load cudnn/8.9.4.25



echo "done"
echo -n "Create $USER_SCRATCH/${VENV}/bin/python ... "
python -m venv $USER_SCRATCH/${VENV} # takes about 5.2s
echo "done"
echo -n "Load   Python $USER_SCRATCH/${VENV}/bin/activate ... "
source $USER_SCRATCH/${VENV}/bin/activate
pip install --upgrade pip > /dev/null
echo "done"

echo "============================================================="
echo "Python version: " $(python --version)
echo "Python path:    " $(which python)
echo "Pip version:    " $(pip --version)
echo "============================================================="
