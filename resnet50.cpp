/***************************************************************************************
 *
 * Description: C++/MPI proxy for ResNet-50 distributed training with data parallelism
 * Author: Shigang Li
 * Email: shigangli.cs@gmail.com
 *
 ***************************************************************************************/

#include "mpi.h"
#include <unistd.h>
#include <stdio.h>
#include <string>
#include <time.h>
#include <stdlib.h>

#define RUNS 512
#define WARM_UP 10
#define TOTALSIZE 25559081
#define MSGAGG 1

#ifdef MSGAGG
//message aggregation
#define NUM 6

//pointers of the send/receive buffers
float* grad_ptrs[NUM];
float* sum_grad_ptrs[NUM];

//sizes for the gradients
int msgSize[NUM] = {
3104745,
4461568,
4462592,
4986880,
4468736,
4074560
};

#else
//number of trainable parameters in ResNet-50
#define NUM 161

//pointers of the send/receive buffers
float* grad_ptrs[NUM];
float* sum_grad_ptrs[NUM];

//sizes for the gradients
int msgSize[NUM] = {
1001,
2050048,
2048,
2048,
1048576,
512,
512,
2359296,
512,
512,
1048576,
2048,
2048,
1048576,
512,
512,
2359296,
512,
512,
1048576,
2048,
2048,
1048576,
512,
512,
2359296,
512,
512,
524288,
2048,
2048,
2097152,
1024,
1024,
262144,
256,
256,
589824,
256,
256,
262144,
1024,
1024,
262144,
256,
256,
589824,
256,
256,
262144,
1024,
1024,
262144,
256,
256,
589824,
256,
256,
262144,
1024,
1024,
262144,
256,
256,
589824,
256,
256,
262144,
1024,
1024,
262144,
256,
256,
589824,
256,
256,
262144,
1024,
1024,
262144,
256,
256,
589824,
256,
256,
131072,
1024,
1024,
524288,
512,
512,
65536,
128,
128,
147456,
128,
128,
65536,
512,
512,
65536,
128,
128,
147456,
128,
128,
65536,
512,
512,
65536,
128,
128,
147456,
128,
128,
65536,
512,
512,
65536,
128,
128,
147456,
128,
128,
32768,
512,
512,
131072,
256,
256,
16384,
64,
64,
36864,
64,
64,
16384,
256,
256,
16384,
64,
64,
36864,
64,
64,
16384,
256,
256,
16384,
64,
64,
36864,
64,
64,
4096,
256,
256,
16384,
64,
64,
9408
};
#endif


//allreduce
int run_allreduce(){
    for(int i=0; i<NUM; i++){
        MPI_Allreduce(grad_ptrs[i], sum_grad_ptrs[i], msgSize[i], MPI_FLOAT, MPI_SUM, MPI_COMM_WORLD);
    }
    return 0;
}

//only communicate with two neighbors in ring topology
int run_ring(int rank, int np) {
    int neighborS;
    int neighborP;
    MPI_Request request;
    MPI_Status  status;

    neighborS = (rank+1)%np;
    neighborP = (rank-1+np)%np;
    for(int i=0; i<NUM; i++){
        MPI_Irecv(sum_grad_ptrs[i], msgSize[i], MPI_FLOAT, neighborP, i, MPI_COMM_WORLD, &request);
        MPI_Send(grad_ptrs[i], msgSize[i], MPI_FLOAT, neighborS, i, MPI_COMM_WORLD);
        MPI_Wait(&request, &status);
    }
    return 0;
}


int main(int argc, char *argv[]){
    int rank, world_size;
    double begin, elapse;

    for(int i=0; i<NUM; i++){
        grad_ptrs[i] = (float *)calloc(msgSize[i], sizeof(float));
        sum_grad_ptrs[i] = (float *)calloc(msgSize[i], sizeof(float));
    }

    MPI_Init(&argc,&argv);
    MPI_Comm_size(MPI_COMM_WORLD, &world_size);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    //warmup
    for(int i=0; i<WARM_UP; i++){
        run_allreduce();
    }

    begin = MPI_Wtime();
    for(int i=0; i<RUNS; i++){
        run_allreduce();
    }
    elapse = (MPI_Wtime()-begin)/RUNS;
    printf("Rank = %d, world_size = %d, total_params = %d, ResNet-50 data parallelism (allreduce) runtime for each iteration = %f s\n", rank, world_size, TOTALSIZE, elapse);

    //for(int i=0; i<NUM; i++){
    //    printf("msgsize[%d] = %d\n", i, msgSize[i]);
    //}

    //warmup
    for(int i=0; i<WARM_UP; i++){
        run_ring(rank, world_size);
    }

    begin = MPI_Wtime();
    for(int i=0; i<RUNS; i++){
        run_ring(rank, world_size);
    }
    elapse = (MPI_Wtime()-begin)/RUNS;
    printf("Rank = %d, world_size = %d, total_params = %d, ResNet-50 data parallelism (neighbors in ring) runtime for each iteration = %f s\n", rank, world_size, TOTALSIZE, elapse);


    MPI_Finalize();
}
