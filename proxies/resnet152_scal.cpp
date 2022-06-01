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

#define WARM_UP 8
#define RUNS 128

// allreduce sizes for gradients with message aggregation
#define NUM_B 10
int allreduce_sizes[NUM_B] = {6511592, 6567936, 5905920, 6113280, 6176256, 6112768, 6176256, 6112768, 5321216, 5194816};

// Global batch_size <= 32 K
// A100 GPU
// runtime in us (10E-6) for each iteration
// corresponding to local batch size = {128, 64, 32, 16}
int local_batch_size_arr[4] = {128, 64, 32, 16};
int fwd_rt_whole_model_arr[4] = {119000, 63000, 36000, 27667};
int bwd_rt_per_B_arr[4] = {23800, 12600, 7200, 5533};

int run_data_parallel(float** grad_ptrs, float** sum_grad_ptrs, int fwd_rt_whole_model, int bwd_rt_per_B){
    
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

        MPI_Iallreduce(grad_ptrs[i], sum_grad_ptrs[i], allreduce_sizes[i], MPI_FLOAT, MPI_SUM, MPI_COMM_WORLD, &grad_allreduce_reqs[i]);	
    }

    MPI_Waitall(NUM_B, grad_allreduce_reqs, MPI_STATUSES_IGNORE); 
    return 0;
}

int main(int argc, char *argv[]){
    int rank, world_size;
    
    MPI_Init(&argc,&argv);
    MPI_Comm_size(MPI_COMM_WORLD, &world_size);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    float* grad_ptrs[NUM_B];
    float* sum_grad_ptrs[NUM_B];
    for(int i=0; i<NUM_B; i++){
        grad_ptrs[i] = (float *)calloc(allreduce_sizes[i], sizeof(float));
        sum_grad_ptrs[i] = (float *)calloc(allreduce_sizes[i], sizeof(float));
    }

    int fwd_rt_whole_model, bwd_rt_per_B, local_batch_size;

    MPI_Barrier(MPI_COMM_WORLD);
    switch(world_size){
        case 256:
	    fwd_rt_whole_model = fwd_rt_whole_model_arr[0]; 
            bwd_rt_per_B = bwd_rt_per_B_arr[0];
	    local_batch_size = local_batch_size_arr[0];
            break;
        case 512:
	    fwd_rt_whole_model = fwd_rt_whole_model_arr[1]; 
            bwd_rt_per_B = bwd_rt_per_B_arr[1];
	    local_batch_size = local_batch_size_arr[1];
            break;
        case 1024:
	    fwd_rt_whole_model = fwd_rt_whole_model_arr[2]; 
            bwd_rt_per_B = bwd_rt_per_B_arr[2];
	    local_batch_size = local_batch_size_arr[2];
            break;
        case 2048:
	    fwd_rt_whole_model = fwd_rt_whole_model_arr[3]; 
            bwd_rt_per_B = bwd_rt_per_B_arr[3];
	    local_batch_size = local_batch_size_arr[3];
            break;
        default:
            printf("Unsupported MPI_Comm_World size!\n");
	    return 0;
    }


    //warmup
    for(int wmp = 0; wmp < WARM_UP; wmp++){
        run_data_parallel(grad_ptrs, sum_grad_ptrs, fwd_rt_whole_model, bwd_rt_per_B);
    }

    double begin, elapse;
    begin = MPI_Wtime();
    for(int iter = 0; iter < RUNS; iter++){
        run_data_parallel(grad_ptrs, sum_grad_ptrs, fwd_rt_whole_model, bwd_rt_per_B);
    }
    elapse = (MPI_Wtime()-begin)/RUNS;

    int total_params = 0;
    for(int i=0; i<NUM_B; i++){
	total_params += allreduce_sizes[i];	
    }

    if(rank == 0){
        printf("Rank = %d, world_size = %d, data_shards = %d, total_params = %d, global_batch_size = %d. \n", rank, world_size, world_size, total_params, local_batch_size*world_size);
        printf("ResNet-152 with pure data parallelism runtime for each iteration = %f s. \n", elapse);
    }

    MPI_Finalize();
}
