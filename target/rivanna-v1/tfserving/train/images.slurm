#!/usr/bin/env bash

#SBATCH --job-name=tf-image
#SBATCH --output=tf-image-%u-%j.out
#SBATCH --error=tf-image%u-%j.err
#SBATCH --gres=gpu:a100:1
#SBATCH --reservation=bi_fox_dgx
#SBATCH --partition=bii-gpu
#SBATCH --account=bii_dsc_community
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --mem=64GB
#SBATCH --cpus-per-task=1
#SBATCH --time=0:30:00

sh images.sh