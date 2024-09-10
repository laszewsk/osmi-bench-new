#
# source install.sh 
module purge

module load apptainer/1.2.2 pytorch/2.4.0

module load gcc/11.4.0
module load openmpi/4.1.4
module load python/3.11.4

python -m venv /scratch/$USER/ENV3
source /scratch/$USER/ENV3/bin/activate

