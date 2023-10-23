# Intro

### Communicators 
Oggetto che contiene un __gruppo di processi__ ed un set di attributi ad essi associati.
Due processi  possono comunicare solo se fanno parte dello stesso comunicatore.  
_Ogni processo ha un identificatore univoco._  
Ogni direttiva di comunicazione richiede di specificare un communicator tra i parametri.  

> __`MPI_COMM_WORLD`__  
Comunicatore che include tutti i processi;  

> `int MPI_Comm_size( MPI_Comm comm, int *size )`  
Numero di processi associati al comunicatore 

> `int MPI_Comm_rank( MPI_Comm comm, int *rank )`  
MPI permette di creare __gruppi logici di processi__, and __in ogni gruppo__, il singolo processo è identificato dal rank.  

<br>

> `comm` comunicatore,  
`size` numero di processi in comm,  
`rank` identificativo del chiamante nel gruppo comm.  

<br>
<br>


# Funzioni MPI

## MPI_Barrier  

`int MPI_Barrier (MPI_Comm comm )`  

- serve a sincronizzare l' esecuzione.
- Blocca il chiamante fino a che tutti i processi nel communicator chiamano la stessa (barrier), quindi la chiamata ritorna ad ogni processo solo quando tutti i membri del communicator hanno eseguito la chimata.  

* * *

## MPI_Bcast
Comunicazione one-to-all  
__root__ $\to$ __tutti i processi in COMM__  

`int MPI_Bcast( void *buffer, int count, MPI_Datatype datatype, int root, MPI_Comm comm )`  

`buffer - indirizzo iniziale della porzione di memoria`  
`count - numero di elementi`  
`datatype - tipo di dato`  
`root - rank mittente`  
`comm - comunicatore`

MPI_BCAST broadcasts a message from the process with _rank root_ to all processes of the group, __itself included__. It is called by all members of group using the same arguments for comm, root.  
On __return__, the contents of root's communication buffer has been copied to all processes.

* * * 

## MPI Scatter (sparpagliare)

`MPI_Scatter(`  
&nbsp; `void* send_data,`  
&nbsp; `int send_count,`  
&nbsp; `MPI_Datatype send_datatype,`  
&nbsp; `void* recv_data,`  
&nbsp; `int recv_count,`  
&nbsp; `MPI_Datatype recv_datatype,`  
&nbsp; `int root,`  
&nbsp; `MPI_Comm communicator`  
`)`  
    
__send_data__, è un array di dati che risiede nel processo root. __send_count__  e __send_datatype__, specificano quanti elementi e di che tipo saranno inviati ad ogni processo.   

> Se *send_count=1* e *send_datatype=MPI_INT*, allora P0 riceve il primo intero dell'array, P1 riceve il secondo intero e cosi via. 

> Se *send_count=2*, allora P0 prende il primo ed il secondo intero, P1 uno riceve il terzo e il quarto e cosi via.  

*In pratica, send_count è spesso uguale al numero di elementi nell'array diviso il numero di processi.*

__recv_data__ : buffer che contiene __recv_count__ elementi del tipo __recv_datatype__.    
__root__: indica il processo che sta sparpagliando gli elementi dell' the array.  

Alla fine dell' esecuzione gni processo deve spedire il pezzo lavorato al root che ricomporrà il buffer ( __Gather__ ).
Root legge il send-buffer, lo divide e lo manda agli altri. Ci sono quindi 2 buffer per processo, receive e send.


# Alltoall
tutti i processi spediscono il proprio buffer a tutti gli altri. I buffer devono avere dimensioni uguali
# Reduce 
diversi processi lavoro su root una parte del dataset per poi riunire i risultati.

extraer paravir
vtune della intel


# OMP

#pragma omp __parallel__ 
    
    crea un gruppo do thread senza fare altro  

#pragma omp __for__

    divide le iterazioni del for tra tutti i thread disponibili  


## Variabili 

__privata__:
ogni thread ha una copia locale della variabile.

__condivisa__: 
la variabile è condivisa, il che implica che è visibile ed accessibile da ogni thread simultaneamente.
Di default ogni variabile è condivisa eccetto le variabili utilizzate come contatori dei cicli for.




# Slurm file

__--nodes__  
Ogni server corrisponde a un nodo e il loro numero viene specificato tramite la direttiva --nodes  
__--ntasks__   
Numero totale di task per job.
I task corrispondono al numero di processi MPI e il loro numero vengono specificato tramite questa direttiva.  
Ad ogni task viene associato una CPU. Tutti i _ntask_ sono vincolati a essere eseguiti su un sigolo nodo di computazaione.   
__--ntasks-per-node__   
Offre la possibilit`a di controllare il numero di task per singolo nodo  

__--cpus-per-task__   
Imposta quante CPU per task saranno riservate, _in unione con OpenMP rappresenta il numero di core_ per singolo processo OpenMP

## Esempi di script sbatch MPI - OMP

### MPI case 1  

`#!/bin/bash`  
`#SBATCH --ntasks=16`  
`#SBATCH --cpus-per-task=1`  
`#SBATCH --nodes=1`  
`#SBATCH --ntasks-per-node=16`  


Questo esempio richiede 16 task, corrispondenti a 16 processi MPI, a cui viene associata una CPU (_cpus-per-task=1_) per task. Tutti i 16 task sono vincolati a essere eseguiti su un sigolo nodo di computazaione.

* * *

### MPI esempio 2 

`#!/bin/bash`  
`#SBATCH --ntasks=32`  
`#SBATCH --cpus-per-task=1`  
`#SBATCH --nodes=2`  
`#SBATCH --ntasks-per-node=16`  

In questo esempio sono richiesti 32 task suddivisi tra 2 nodi, con 16 task per ogni nodo.

### MPI & OMP esempio 3

`#!/bin/bash`  
`#SBATCH --ntasks=16`  
`#SBATCH --cpus-per-task=4`  
`#SBATCH --nodes=4`  
`#SBATCH --ntasks-per-node=4`  

In questo caso viene eseguita un’applicazione che, per ogni processo MPI, usa core multipli.  
Vengono richiesti 16 task (_--ntasks=16_) e 4 core per task (_--cpus-per-task=4_), quindi 16*4=64 core totali.  
I 16 task vengono ripartiti tra 4 nodi (_--nodes=4_) con 4 task per nodo (_--ntasks-per-node=4_).

# Utilities

> __sacct__

> a per copiare un file dal cluster al proprio host usare il seguente comando  
`$ scp -r username@{login-node}:/home/username /path/to/local/dir/`


# Compilation parameter
## per compilatore _nvc_

__-mp [=option]__  
Enable OpenMP pragmas and directives to explicitly parallelize regions of code for execution by accelerator devices or multi-processor system.  The options allowed are:

              gpu    Compile OpenMP directives for parallel execution on the GPU.

__-mfma__  
Generate (don't generate) __fused multiply-add (FMA)__ instructions for targets that support it.  FMA
instructions are generally faster than separate multiply-add instructions, and can generate higher precision
results since the multiply result is not rounded before the addition.  However, because of this, the result
may be different than the unfused multiply and add instructions.  FMA instructions are enabled with higher
optimization levels.



# Appunti Floyd-Warshall  

**Fonte del parallelismo**   


Durante la _k-esima_ iterazione :  


`for ( i = 0; i < N ; i++ ) {`  
&nbsp;&nbsp;` for ( j = 0; j < N ; j++ ) {`  
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;`if ( D [i][j] > ( D[i][k] + D[k][j]) ) {`  
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;` D [i][j] = D [i][k] + D [k][j]; `  
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;`}`  
&nbsp;&nbsp;` }`  
`}`  

Tutti i valori `D[i][j]` possono essere calcolati in parallelo dato che la _k-esima_ colonna e la _k-esima_ riga non cambiano durante tutta la _k-esima_ iterazione.
IN fatti, dato che i pesi sono **positivi**, per ogni k:  

- for i = k the condition D[k][j] > (D[k][k] + D[k][j]) is false  
- for j = k the condition D[i][k] > (D[i][k] + D[k][k]) is false  

&nbsp;  

Durante la _k-esima_ iterazione ogni elemento della riga k deve essere inviato (in broadcast) ad ogni task della colonna del rispettivo elemento.
