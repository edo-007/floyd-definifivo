#define MAX_WEIGHT 10  // max value of weight
#define ALPHA 0.25  // thresholdforcreating edges
#define INF (100*MAX_WEIGHT) // infinite value
#define NIL (-1)
#define BOLD_ON  "\e[1m"
#define BOLD_OFF  "\e[m"
#define ROOT 0
#define SEED 7

#ifdef _PRINT_DISTANCE
    // #define N (8)
    #define NREP 1
#else
    #define N (16)
    #define NREP 1
#endif

void stampa_matrice(int (*mat)[N], int n_row, int n_col, int c);
void posix_memalign_all( void *G, void *C, void *D, void *P);
void printAPSP ( int G [ N ] [ N ] , int C [ N ] [ N ] , int D [ N ] [ N ] , int P [ N ] [ N ] );
void initGraph ( int G [ N ] [ N ] , int C [ N ] [ N ] , int D [ N ] [ N ] , int P [ N ] [ N ] , unsigned int seed );
void printGraph ( int G [ N ] [ N ] , int C [ N ] [ N ] , char *fname );
void FloydAlgorithm( int G [N] [N] , int C [N] [N] , int D [N] [N] , int P [N] [N], int mpi_size, int mpi_rank);