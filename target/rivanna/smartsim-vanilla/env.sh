export USER_SCRATCH=/scratch/$USER

export CLOUDMESH_CONFIG_DIR=$USER_SCRATCH/.cloudmesh
export APPTAINER_CACHEDIR=$USER_SCRATCH/.apptainer/cache

export OSMI_PROJECT=`pwd`
export OSMI_TARGET=$OSMI_PROJECT/osmi-bench-new/target/rivanna/smartsim
export OSMI_TRAIN=$OSMI_TARGET


export VENV=ENV3

export TARGET=rivanna

mkdir -p $APPTAINER_CACHEDIR

echo "============================================================="
echo "USER_SCRATCH:        " $USER_SCRATCH
echo "CLOUDMESH_CONFIG_DIR:" $CLOUDMESH_CONFIG_DIR
echo "APPTAINER_CACHEDIR:  " $APPTAINER_CACHEDIR
echo "OSMI_PROJECT:        " $OSMI_PROJECT
echo "OSMI_TARGET:         " $OSMI_TARGET
echo "OSMI_TRAIN:          " $OSMI_TRAIN
echo "============================================================="

module purge

echo -n "Load   apptainer ... "
module load apptainer
echo "done"

echo -n "Load   cuda/12.4.1" 
module load cuda/12.4.1
echo "done"

echo -n "Load   python/3.11.4, cuda/12.2.2 ... " 
module load gcc/11.4.0  openmpi/4.1.4 python/3.11.4 
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
