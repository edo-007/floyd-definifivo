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
     
    
/* 
 * =========================================
 *   Algoritmo Floyd                      
 * =========================================
 */
    int inf = 1000;
    int k,i,j, n_local_row;
    int distance;

    // Se il l' algoritmo rimane seriale, n_local_row
    // è pari al numero di righe di tutta la matrice.

    n_local_row = N;

#ifdef _USE_MPI

    int row_k[N]; 
    int block_id,lk;
    
    int root = 0;
    n_local_row = N/mpi_size;

    /* Ogni processo. alloca una partizione (Db) della matrice D*/
    int (*Db) [ N ];
    if ( posix_memalign ( ( void *)&Db , 4096 , n_local_row*N*sizeof(int) ) != 0 ) {
        perror("ERROR: allocation of Db FAILED:" ) ;
        exit(-1);
    } 

    if (mpi_rank == 0)
        stampa_matrice(D, N,N,'I');
        
    /* ROOT sparpaglia D nei buffer Db di tutti gli altri processi */
    MPI_Scatter( D, N*n_local_row, MPI_INT,   Db, N*n_local_row, MPI_INT, ROOT , MPI_COMM_WORLD);
    
#endif 
    
    for ( k = 0; k < N ; k++ ) {                            /*   MAIN-FOR   */

#ifdef _USE_MPI
        block_id = k / n_local_row;
        lk = k % n_local_row;

        /* Prima del broadcasting, ROOT copia la riga k-esima su un array temporaneo.*/
        if ( mpi_rank == block_id ){
            int t;
            for (t=0; t < N ; t++)   
                row_k[t] = Db[lk][t];
    
            MPI_Bcast(row_k, N, MPI_INT, root, MPI_COMM_WORLD);
        }
 
#endif

OMP_PARALLEL_FOR(i,j)
        for ( i = 0; i < N; i++ ) {             /* i < n_local_row*/
            for ( j = 0; j < N ; j++ ) {
#ifdef _USE_MPI
                distance = Db[i][k] + row_k[j];
                if ( Db[i][j] > distance )
                    Db[i][j] = distance;  
#else   /* NOT MPI*/
                distance = D[i][k] + D[k][j];
                if ( D[i][j] > distance ) {
                    D[i][j] = distance;  
#endif


# ifdef _SERIAL                                            
                    P[i][j] = P[k][j];
# endif
                }
            }
        }
    

#ifdef _USE_MPI
        MPI_Barrier(MPI_COMM_WORLD);
#endif

    } 

#ifdef _USE_MPI
    MPI_Gather(Db, N*n_local_row, MPI_INT,  D, N*n_local_row, MPI_INT, ROOT , MPI_COMM_WORLD);
    if (mpi_rank == 0)
    stampa_matrice(D, N,N,'F');
#endif


/* End Floyd-Warshall*/ 
}






int main(int argc, char **argv){

    
    // assumiamo che costi siano tutti positivi
    struct timespec start, end;

    double elapsed,mean,sum;
    mean = 0.0;
    sum = 0.0;
    char *fname = "graph.dot";
    unsigned int seed;
    seed = 7;

    int mpi_rank, mpi_size;
    mpi_rank = 0;
    mpi_size = 0; 

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

    if ( MPI_Init(&argc, &argv) != MPI_SUCCESS ) {
        fprintf(stderr , "Error in MPI_Init.\n" ) ;
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
        initGraph (G,C,D,P, seed);
    }
#else      
    /* NO MPI */       
    printf("N _> %d \n", N);                     
    posix_memalign_all(&G,&C,&D,&P);
    initGraph (G,C,D,P, seed);

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

        clock_gettime(CLOCK_REALTIME, &start);  /* start */
        
        FloydAlgorithmm(G,C,D,P, mpi_size, mpi_rank);
    
        clock_gettime(CLOCK_REALTIME, &end);     /* End */

        elapsed = simple_sub_timespec(start, end);
        sum += elapsed;
    }

#ifdef _USE_MPI
    if ( MPI_Finalize() != MPI_SUCCESS ){
        fprintf(stderr , "Error in MPI_Finalize.\n" ) ;
        exit(-1);
    }

    if ( mpi_rank != 0 ){
        // printf("succes exit [ rank = %d ]\n", mpi_rank);
        exit(EXIT_SUCCESS);
    }
#endif


    mean = sum/(float)NREP;
    printf("\nmedia: %lf\n", mean);


#ifdef _PRINT_DISTANCE
    stampa_matrice(D,N,N,'D');
#endif

    printGraph( G,C, fname );
    printAPSP(G,C,D,P);


    return 0;
}