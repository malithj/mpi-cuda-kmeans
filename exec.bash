#!/bin/bash

#SBATCH --nodes=1
#SBATCH --tasks-per-node=10
#SBATCH --time=0:10:00
#SBATCH --job-name=kmeans
#SBATCH --reservation=eece5640
#SBATCH --partition=gpu
#SBATCH --mem=8Gb
#SBATCH --gres=gpu:k20:1
#SBATCH --output=kmeans-%j.out

#cd /scratch/$USER/eece5640/transpose/

mpirun kmeans

