#!/bin/bash
#SBATCH --job-name=floyd-warshall
#SBATCH --error=err/test.err
#SBATCH --output=out/test.out
#SBATCH --nodes=1
#SBATCH --mem=100000
#SBATCH --gres=gpu:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=5
#SBATCH --cpu-freq=performance
#SBATCH --partition=skyvolta

module load cuda/10.2
module load openmpi
module load nvhpc/22.11

## nvc -O3 -fopenmp -mp=gpu -gpu=cc70 -o floyd-main floyd-main.o lib/floyd-library.o lib/time-library.o

make USE_OMP=y USE_MPI=n PRINT_D=n

ulimit -s $((10*1024*1024))

srun ./floyd-main
