#!/bin/bash
#SBATCH --job-name=floyd-warshall
#SBATCH --error=err/test-serial.err
#SBATCH --output=out/test-serial.out
#SBATCH --nodes=1
#SBATCH --mem=10000
#SBATCH --gres=gpu:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --cpu-freq=performance
#SBATCH --partition=interactive
##SBATCH --nodelist=node06

## SBATCH --error=err/test-%j.err (commento)
## module load nvhpc/22.11
module load cuda
## module load openmpi

## nvc -O3 -fopenmp -mp=gpu -gpu=cc70 -o floyd-main floyd-main.o lib/floyd-library.o lib/time-library.o

make USE_OMP=n USE_MPI=n PRINT_D=y

## ulimit -s $((10*1024*1024))

# export OMP_NUM_THREADS=8

srun ./floyd-main
