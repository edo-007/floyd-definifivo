#include <stdlib.h>
#include <stdio.h>
#include <time.h>
#include <mpi.h>
#include <omp.h>
#include "lib/floyd-library.h"
#include "lib/time-library.h"


#ifdef _USE_OMP
    #include <omp.h>
    #define OMP_PARALLEL_FOR(i,j,distance) _Pragma("omp parallel for private(i,j,distance)")
#else
    #define OMP_PARALLEL_FOR(i,j,distance)
#endif

void FloydAlgorithmm ( int G[N][N] , int C[N][N] , int D[N][N] , int P[N][N], int mpi_size, int mpi_rank) {

    int k,i,j,distance;
    int n_local_row;
/* 
 * =========================================
 *   Algoritmo Floyd                      
 * =========================================
 */
    
    /* Se l' algoritmo è Seriale/OMP, n_local_row è pari al numero di righe di tutta la matrice. */
    n_local_row = N;
       
#ifdef _USE_MPI

    int row_k[N]; 
    int block_id,lk;
    
    n_local_row = N/mpi_size;

    /* Ogni processo. alloca una partizione (Dp) della matrice D*/
    int (*Dp) [ N ];
    if ( posix_memalign ( ( void *)&Dp , 4096 , n_local_row*N*sizeof(int) ) != 0 ) {
        perror("ERROR: allocation of Dp FAILED:" ) ;
        exit(-1);
    } 
    
    /* ROOT sparpaglia D nei buffer Dp di tutti gli altri processi */
    MPI_Scatter( *D, N*n_local_row, MPI_INT, Dp, N*n_local_row, MPI_INT, ROOT, MPI_COMM_WORLD);
    
    // if ( mpi_rank == 0){
    //         printf("N = %d\n", N);
    //         stampa_matrice(D, N, N, 'D');
    //         puts("\n");
    // }
    if ( mpi_rank == 1){
            printf("N = %d\n", N);
            stampa_matrice(Dp, n_local_row, N, (mpi_rank + 65));
            puts("\n");
    }
    

#endif 

    for ( k = 0; k < N ; k++ ) {   
    /*   MAIN-FOR   */

#ifdef _USE_MPI
        block_id = k / n_local_row;
        lk = k % n_local_row;

        /* Prima del broadcasting, ROOT copia la riga k-esima su un array temporaneo.*/
        if ( block_id == ROOT ){
            int t;
            for (t = 0; t < N ; t++)   
                row_k[t] = Dp[lk][t]; 
        }
        /* MPI0 sends the blocks of distance matrix to the other processes */
        MPI_Bcast(row_k, N, MPI_INT, ROOT, MPI_COMM_WORLD);
#endif


OMP_PARALLEL_FOR(i, j, distance)
        for ( i = 0; i < n_local_row; i++ ) {
            for ( j = 0; j < N ; j++ ) {

#ifdef _USE_MPI
                distance = Dp[i][k] + row_k[j];
                if ( Dp[i][j] > distance ){
                    Dp[i][j] = distance;  
                }
#else   /* NOT MPI*/

                distance = D[i][k] + D[k][j];
                if ( D[i][j] > distance ) {
                    D[i][j] = distance; 
#ifdef _SERIAL                                            
                    P[i][j] = P[k][j];
#endif
                } 
#endif
            }
        }
        
    } 


#ifdef _USE_MPI
        MPI_Barrier(MPI_COMM_WORLD);
#endif


#ifdef _USE_MPI
    MPI_Gather(Dp, N*n_local_row, MPI_INT,  D, N*n_local_row, MPI_INT, ROOT , MPI_COMM_WORLD);
    MPI_Barrier(MPI_COMM_WORLD);
#endif


/* End Floyd-Warshall*/ 
}


int main(int argc, char **argv){ /* _______________________________________________ MAIN ______________________________________________*/

    struct timespec start, end;
    double elapsed,mean,sum;
    int mpi_rank, mpi_size;

    mpi_rank = 0.0;
    mpi_size = 0.0;
    
    mean = 0.0;
    sum = 0.0;

    // char *fname = "graph.dot";
    int (*G) [ N ] ; // puntatore ad array di N elementi interi

    int (*C) [ N ] ;    // Costi
    int (*D) [ N ] ;    // Distanze
    int (*P) [ N ] ;    // Predecessori 

/*_____________________________________
 * 
 *          INIZIALIZZAZIONE
 *_____________________________________
*/

#ifdef _USE_OMP
    printf(BOLD_ON "\n__ Using OMP __\n" BOLD_OFF);
#endif

#ifdef _USE_MPI

/*
    memory allocation: only MPI0 allocate memory to store the entire graph, other
    process allocate only minimum storage necessary
*/ 
    
    if ( MPI_Init(&argc, &argv) != MPI_SUCCESS ) {
        fprintf(stderr , "Error in MPI_Init.\n" );
        exit(-1);
    }
    MPI_Comm_size(MPI_COMM_WORLD, &mpi_size);       // Size
    MPI_Comm_rank(MPI_COMM_WORLD, &mpi_rank);       // Rank

    /* root process */
    if (mpi_rank == 0) {

        printf( BOLD_ON "\n\t** Using MPI ** \n\n" BOLD_OFF); 
        printf("- Communicator size = %d\n\n", mpi_size);

        /* Solo il processo root le alloca */    
        posix_memalign_all(&G,&C,&D,&P);
        initGraph (G,C,D,P, SEED);

    }
#else   

    /* NO MPI */                          
    posix_memalign_all(&G,&C,&D,&P);
    initGraph (G,C,D,P, SEED);    

#endif

#if ( !defined _USE_OMP && !defined _USE_MPI ) 
    printf( BOLD_ON "\n:: Serial version ::" BOLD_OFF);
#endif

/*_____________________________________
 * 
 *  ESECUZIONE ALGORITMO FLOYD_WARSHALL
 *_____________________________________
*/

    int rep;
    for ( rep = 0; rep < NREP; rep++){


#ifdef _USE_MPI
        if ( mpi_rank == ROOT){
#endif
            clock_gettime(CLOCK_REALTIME, &start);  /* start */

#ifdef _USE_MPI
        }
#endif


        FloydAlgorithmm(G,C,D,P, mpi_size, mpi_rank);


#ifdef _USE_MPI
        if ( mpi_rank == ROOT) {
#endif
            clock_gettime(CLOCK_REALTIME, &end);     /* End */
            elapsed = simple_sub_timespec(start, end);
            sum += elapsed;
#ifdef _USE_MPI
        }
#endif


    }

#ifdef _USE_MPI
    /* Terminazione di tutti i processi tranne quello root */
    MPI_Finalize();
    if ( mpi_rank != ROOT )
        exit(EXIT_SUCCESS);
#endif

    mean = sum/(float)NREP;
    printf("\nmedia: %lf\n", mean);

#ifdef _PRINT_DISTANCE
    stampa_matrice(D,N,N,'D');
#endif

    // printGraph( G,C, fname );
    // printAPSP(G,C,D,P);

    return 0;
}

