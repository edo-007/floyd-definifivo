#!/bin/bash
#SBATCH --job-name=floyd-warshall
#SBATCH --error=err/test.err
#SBATCH --output=out/test.out
#SBATCH --nodes=1
#SBATCH --mem=100000
#SBATCH --gres=gpu:1
#SBATCH --ntasks=2
#SBATCH --cpus-per-task=1
#SBATCH --cpu-freq=performance
#SBATCH --partition=skyvolta

module load cuda/10.2
module load openmpi/4.1.5-cuda10.2 
module load nvhpc/22.11

## nvc -O3 -fopenmp -mp=gpu -gpu=cc70 -o floyd-main floyd-main.o lib/floyd-library.o lib/time-library.o

make USE_OMP=n USE_MPI=y PRINT_D=n

ulimit -s $((10*1024*1024))

mpirun ./floyd-main
