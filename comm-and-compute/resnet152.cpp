/*********************************************************************
 *
 * Description: C++/MPI proxy for ResNet-152 distributed training 
 *              with data parallelism
 * Author: Shigang Li
 * Email: shigangli.cs@gmail.com
 *
 *********************************************************************/

#include <mpi.h>
#include <unistd.h>
#include <stdio.h>
#include <string>
#include <time.h>
#include <stdlib.h>
#include <assert.h>

#define RUNS 512
#define WARM_UP 10

// allreduce sizes for gradients with message aggregation
#define NUM_B 10
int allreduce_sizes[NUM_B] = {6511592, 6567936, 5905920, 6113280, 6176256, 6112768, 6176256, 6112768, 5321216, 5194816};

// batchsize = 128
// A100 GPU
// runtime in us (10E-6) for each iteration 
int fwd_rt_whole_model = 119000;
int bwd_rt_per_B = 23800;

int run_data_parallel(float** grad_ptrs, float** sum_grad_ptrs, MPI_Comm* conv_allreduce_comms){
    
    //forward
    usleep(fwd_rt_whole_model); //compute

    //backward
    MPI_Request grad_allreduce_reqs[NUM_B];
    //must initialize with MPI_REQUEST_NULL
    for(int i=0; i<NUM_B; i++)
        grad_allreduce_reqs[i] = MPI_REQUEST_NULL;

    int index, flag;
    for(int i=0; i<NUM_B; i++){
	if(i > 1)
            MPI_Testany(NUM_B, grad_allreduce_reqs, &index, &flag, MPI_STATUSES_IGNORE); //advancing MPI in the background

        usleep(bwd_rt_per_B); //compute

        MPI_Iallreduce(grad_ptrs[i], sum_grad_ptrs[i], allreduce_sizes[i], MPI_FLOAT, MPI_SUM, conv_allreduce_comms[i], &grad_allreduce_reqs[i]);	
    }

    MPI_Waitall(NUM_B, grad_allreduce_reqs, MPI_STATUSES_IGNORE); 
    return 0;
}

int main(int argc, char *argv[]){
    int rank, world_size;
    
    MPI_Init(&argc,&argv);
    MPI_Comm_size(MPI_COMM_WORLD, &world_size);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm conv_allreduce_comms[NUM_B];
    for(int i=0; i<NUM_B; i++)
        MPI_Comm_dup(MPI_COMM_WORLD, &conv_allreduce_comms[i]); //duplicated for nb colls

    float* grad_ptrs[NUM_B];
    float* sum_grad_ptrs[NUM_B];
    for(int i=0; i<NUM_B; i++){
        grad_ptrs[i] = (float *)calloc(allreduce_sizes[i], sizeof(float));
        sum_grad_ptrs[i] = (float *)calloc(allreduce_sizes[i], sizeof(float));
    }

    MPI_Barrier(MPI_COMM_WORLD);

    //warmup
    for(int wmp = 0; wmp < WARM_UP; wmp++){
        run_data_parallel(grad_ptrs, sum_grad_ptrs, conv_allreduce_comms);
    }

    double begin, elapse;
    begin = MPI_Wtime();
    for(int iter = 0; iter < RUNS; iter++){
        run_data_parallel(grad_ptrs, sum_grad_ptrs, conv_allreduce_comms);
    }
    elapse = (MPI_Wtime()-begin)/RUNS;

    int total_params = 0;
    for(int i=0; i<NUM_B; i++){
	total_params += allreduce_sizes[i];	
    }

    if(rank == 0){
        printf("Rank = %d, world_size = %d, data_shards = %d, total_params = %d, global_batch_size = %d. \n", rank, world_size, world_size, total_params, 128*world_size);
        printf("ResNet-152 with pure data parallelism runtime for each iteration = %f s. \n", elapse);
    }

    MPI_Finalize();
}
