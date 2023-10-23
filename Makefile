FLAGS = -Wall
CC = nvc


# ricorda :- D_COSTANT corrinponde all costante _COSTANT 
# nvc -O3 -fopenmp -mp=gpu -gpu=cc70 -mavx2 -mfma 
ifeq ($(USE_OMP),y)
	FLAGS += -fopenmp -D_USE_OMP=1
endif

ifeq ($(USE_MPI),y)
	FLAGS += -D_USE_MPI=1
	CC = mpicc
endif

ifeq ($(PRINT_D),y)
	FLAGS += -D_PRINT_DISTANCE=1
endif


all: 
	$(CC) $(FLAGS) -g -c lib/floyd-library.c -o lib/floyd-library.o
	$(CC) $(FLAGS) -g -c lib/time-library.c -o lib/time-library.o
	$(CC) $(FLAGS) -g -c floyd-main.c
	$(CC) $(FLAGS) -g -o floyd-main floyd-main.o lib/floyd-library.o lib/time-library.o 


# floyd-main : floyd-main.o lib/floyd-library.o lib/time-library.o 
# 	$(CC) $(FLAGS) -o floyd-main floyd-main.o lib/floyd-library.o lib/time-library.o

# floyd-main.o : floyd-main.c 
# 	$(CC) $(FLAGS) -c floyd-main.c

# floyd-library.o : floyd-library.c floyd-library.h
# 	$(CC) $(FLAGS) -c floyd-library.c

# lib/time-library.o: lib/time-library.c 
# 	gcc -c -o lib/time-library.o lib/time-library.c 


clean:
	rm *.o lib/*.o

# load-module:
# 	module load nvhpc/22.11
# 	module load cuda/10.2
# 	module load openmpi

