#include <stdlib.h>
#include <stdio.h>
#include <limits.h>
#include <mpi.h>

/*
 * _Pragma è un operatore preprocessore introdotto nello standard C99.
 * Consente di definire una direttiva #pragma come una stringa costante. 
 * Ciò consente di creare direttive #pragma dinamiche o calcolate durante la fase di preprocessing. 
 * Mentre #pragma è una direttiva preprocessore completa, _Pragma può essere utilizzato per 
 * incorporare direttive #pragma all'interno di costrutti condizionali o macro preprocessore.
 *
 */

// #ifdef _USE_OMP
//     #include <omp.h>
//     #define OMP_PARALLEL_FOR(i,j) _Pragma("omp parallel for private(i,j)")
// #else
//     #define OMP_PARALLEL_FOR(i,j)
// #endif

#include "floyd-library.h"
#include <sys/types.h>
#include <sys/stat.h>
#include <fcntl.h>
#include <string.h>
#include <unistd.h>

void stampa_matrice(int (*mat)[N], int n_row, int n_col, int c){
    int i,j;

    printf("\n%c     ",c);
    for(i = 0; i < n_col; i++) {
        // printf(BOLD_ON "\t\x1b[4m#%d  " BOLD_OFF,i); // i
        printf("\t%d  ",i);
    }
    printf("\n\n");
    for(i = 0; i < n_row; i++) {
        // printf(BOLD_ON "#%d | " BOLD_OFF,i); // h
        printf("#%d | ",i); // h
            
        for(j = 0; j < n_col; j++){
            if (mat[i][j] >= 0 && c == 'P')
                printf("\t  %d  ", mat[i][j]);
            else 
                printf("\t %d  ", mat[i][j]);
        }
    printf("\n");
    }
}


void printPath (int G[N][N] , int P[N][N] , int i , int j , FILE *fp) {

    if ( ( i !=j ) && ( P[i][j] != NIL ) ) 
    {
        printPath (G , P , i , P[i][j] , fp ) ;
        G[P[i][j]][j] = 2;
        fprintf (fp , " %d " , j ) ;
        
    }
}
void printAPSP(int G[N][N], int C[N][N], int D[N][N], int P[N][N])
{
    //int fp;
    //int i, j;
//
    //fp = open("apsp-s.txt", O_WRONLY);
    //for (i = 0; i < N; i++)
    //{
    //    for (j = 0; j < N; j++)
    //    {
    //        if (i != j)
    //        {
    //            fprintf(fp, "%d --> %d: ", i, j);
    //            printPath(G, P, i, j, fp);
    //            fprintf(fp, " \t\t new_cost=%d oldcost:%d\n", D[i][j], C[i][j]);
    //        }
    //    }
    //}
    //fclose(fp);
}


void initGraph ( int (*G)[N], int (*C)[N], int (*D)[N],int (*P)[N] , unsigned int seed ) {
    
    int i , j , nedges = 0 , count ;
    srand(seed);

    // init G, C and D: adiacency , costs and distancesmatrices
    for ( i=0; i < N ; i++ ){
        for ( j=0; j < N ; j++ ){
            G[i][j] = 0;
            C[i][j] = INF;
            D[i][j] = INF;
        }
    }

    for ( i=0; i < N ; i++ ) {
        int n_neighbour = (rand () % N) + 1; // random number between 1 . . N
        j = 0; count = 0;
        
        while ( count < n_neighbour ) {
            
            if ( j == i ) {
                G[i][j] = 0; C[i][j] = 0; D[i][j] = 0;

            } 
            else {
                int alpha = rand ( ) ;  // alpha \in[0,INT_MAX]
                if ( alpha < (ALPHA*INT_MAX) ) { // ALPHA coefficente 0 . . 1

                    // create edge
                    int weight = (rand ( ) % MAX_WEIGHT ) ; G[i][j] = 1; C[i][j] = weight; D[i][j] = weight;
                    count++; nedges++;
                }
            }
            j = (j+1) % N ;
        }
    }
    for ( i=0; i < N ; i++ ) {
        for ( j=0; j < N ; j++ ) {

            if ( ( i != j) && (C[i][j] != INF ) ) {
                P[i][j] = i ;
            } else { 

               for ( i=0; i < N ; i++ ){
                    for ( j=0; j < N ; j++ ){
                        G[i][j] = 0;
                        C[i][j] = INF ;
                        D[i][j] = INF ;
                    }
                }
                for ( i=0; i < N ; i++ ) {

                    int n_neighbour = (rand ( ) % N) + 1; // random number between 1 . . N
                    j = 0; count = 0;

                    while ( count < n_neighbour ) {

                        if ( j == i ) {
                            G[i][j] = 0; C[i][j] = 0; D[i][j] = 0;
                        } else {

                            int alpha = rand ( ) ; // random number between 0 . . INT_MAX
                            if ( alpha < (ALPHA*INT_MAX) ) { // ALPHA 

                                // create edge
                                int weight = (rand ( ) % MAX_WEIGHT ) ; 
                                G[i][j] = 1;  C[i][j] = weight; D[i][j] = weight ;
                                count++; nedges++;
                            }
                        }
                        j = (j+1) % N ;
                    }
                }

                // inizializzazione matrice dei predecessori
                for ( i=0; i < N ; i++ ) {
                    for ( j=0; j < N ; j++ ) {

                        if ( ( i != j) && (C[i][j] != INF ) ) {
                            P[i][j] = i ;
                        }else {
                            P[i][j] = NIL ;
                        }
                    }
                }

            }
        }
    }

}

void printGraph ( int G[N][N] , int C[N][N] , char *fname ) {

    int fd;
    int i , j ;
    char temp[256];
    
    fd = open(fname, O_WRONLY | O_TRUNC | O_CREAT );
    if (fd < 0){
        perror("errore");
    }
    
    char *start = "digraph {\n \t rankdir=LR;\n\n";
    write(fd , start, strlen(start));

    for ( i=0; i<N ; i++ ) {

        for ( j=0; j<N ; j++ ) {
            if ( G[i][j] == 1 ) {
                sprintf(temp , "\t %s%d -> %s%d[label=\"%d\",weight=\"%d\"];\n" , "N" , i , "N" , j , 
                    C[i][j] , C[i][j] ) ;
                write(fd, temp, strlen(temp));
            }   
            if ( G[i][j] == 2 ) {
                sprintf(temp , "\t %s%d -> %s%d[label=\"%d\",weight=\"%d\",color=red,penwidth=3];\n" ,
                "N" , i , "N" , j , C[i][j] , C[i][j] ) ;
                write(fd, temp, strlen(temp));
            }
        }

    }
    
    char *end = "}\n";
    write(fd , end, strlen(end) ) ;
    close(fd) ;
}

void posix_memalign_all( void *G, void *C, void *D, void *P){


    if ( posix_memalign ( ( void *)G , 4096 , N*N*sizeof(int ) ) != 0 ) {
        perror("ERROR: allocation of G FAILED:" ) ;
        exit( -1 );
    }
    if ( posix_memalign ( ( void *)C , 4096 , N*N*sizeof(int ) ) != 0 ) {
        perror("ERROR: allocation of C FAILED:" ) ;
        exit( -1 );
    }
    if ( posix_memalign ( ( void *)D , 4096 , N*N*sizeof(int ) ) != 0 ) {
        perror("ERROR: allocation of D FAILED:" ) ;
        exit( -1 );
    }
    if ( posix_memalign ( ( void *)P , 4096 , N*N*sizeof(int ) ) != 0 ) {
        perror("ERROR: allocation of P FAILED:" ) ;
        exit( -1 );
    }

}



