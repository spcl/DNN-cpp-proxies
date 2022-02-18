/*********************************************************************
 *
 * Description: C++/MPI proxy for CosmoFlow model distributed training 
 *              with a hybrid of model and data parallelism
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

#define RUNS 256
#define WARM_UP 10

#define NUM_L 8
// we set model shards = 4
// batchsize = 8
// A100 GPU
// runtime in us (10E-6) for each model shard 
int fwd_rt_per_layer[NUM_L] = {6567, 13135, 6567, 3283, 1641, 5, 3, 1};
int bwd_rt_per_layer[NUM_L] = {2, 6, 10, 3283, 6567, 13135, 26270, 13135};

#define NUM_Conv_L 5
// 2x2 2D spatial decomposation for 3D tensors
// Note that each worker has two neighbors in 2D decomposation

// conv layer halo exchange message sizes in forward
int conv_fwd_halo_sizes[NUM_Conv_L-1] = {2097152, 1048576, 524288, 262144};

// conv layer halo exchange message sizes in backward
int conv_bwd_halo_sizes[NUM_Conv_L-1] = {131072, 262144, 524288, 1048576};

#define NUM_Dense_L 3
// dense layer allgather msg sizes in forward
int dense_fwd_allgather_sizes[NUM_Dense_L] = {65536, 256, 128};

// dense layer reduce_scatter msg sizes in backward
//int dense_bwd_reduce_scatter_sizes[NUM_Dense_L] = {512, 1024, 262144};
int dense_bwd_reduce_scatter_sizes[NUM_Dense_L] = {128, 256, 65536};

// allreduce sizes for gradients with message aggregation
// aggregate all dense layers:  Dense2-0  Conv4   Conv3   Conv2   Conv1  Conv0
int allreduce_sizes[NUM_L-2] = {1050737, 3539456, 884992, 221312, 55360, 3488};

int run_model_data_parallel(float** fwd_halo_send_buff0_ptrs,
                            float** fwd_halo_send_buff1_ptrs,
                            float** fwd_halo_recv_buff0_ptrs,
                            float** fwd_halo_recv_buff1_ptrs,
                            float** bwd_halo_send_buff0_ptrs,
                            float** bwd_halo_send_buff1_ptrs,
                            float** bwd_halo_recv_buff0_ptrs,
                            float** bwd_halo_recv_buff1_ptrs,
                            float** dense_fwd_allgather_sbuff_ptrs,
                            float** dense_fwd_allgather_rbuff_ptrs,
                            float** dense_bwd_rs_sbuff_ptrs,
                            float** dense_bwd_rs_rbuff_ptrs,
                            float** grad_ptrs,
                            float** sum_grad_ptrs,
		            MPI_Comm model_parallel_comm,
		            MPI_Comm dense_allreduce_comm,
		            MPI_Comm* conv_allreduce_comms){

    
    //forward
    int mp_group_rank;
    MPI_Comm_rank(model_parallel_comm, &mp_group_rank);
    for(int i=0; i<NUM_L; i++){
        if(i>=1 && i<NUM_Conv_L){ //halo exchange for conv layers
	    int msg_idx = i-1;
            MPI_Request requests[4];
            MPI_Isend(fwd_halo_send_buff0_ptrs[msg_idx], conv_fwd_halo_sizes[msg_idx], MPI_FLOAT, mp_group_rank^1, i, model_parallel_comm, &requests[0]);
            MPI_Isend(fwd_halo_send_buff1_ptrs[msg_idx], conv_fwd_halo_sizes[msg_idx], MPI_FLOAT, mp_group_rank^2, i, model_parallel_comm, &requests[1]);
            MPI_Irecv(fwd_halo_recv_buff0_ptrs[msg_idx], conv_fwd_halo_sizes[msg_idx], MPI_FLOAT, mp_group_rank^1, i, model_parallel_comm, &requests[2]);
            MPI_Irecv(fwd_halo_recv_buff1_ptrs[msg_idx], conv_fwd_halo_sizes[msg_idx], MPI_FLOAT, mp_group_rank^2, i, model_parallel_comm, &requests[3]);
	    MPI_Waitall(4, requests, MPI_STATUSES_IGNORE);
	}
	else if(i>=NUM_Conv_L){ //all gather for dense layers
	    int msg_idx = i-NUM_Conv_L;
	    MPI_Allgather(dense_fwd_allgather_sbuff_ptrs[msg_idx], dense_fwd_allgather_sizes[msg_idx], MPI_FLOAT, dense_fwd_allgather_rbuff_ptrs[msg_idx], dense_fwd_allgather_sizes[msg_idx], MPI_FLOAT, model_parallel_comm);
	}

        usleep(fwd_rt_per_layer[i]); //compute
    }
    
    //backward
    MPI_Request grad_allreduce_reqs[NUM_Conv_L+1];
    for(int i=0; i<NUM_Conv_L+1; i++)
        grad_allreduce_reqs[i] = MPI_REQUEST_NULL;

    int index, flag;
    for(int i=0; i<NUM_L; i++){
	if(i > NUM_Dense_L)
            MPI_Testany(NUM_Conv_L+1, grad_allreduce_reqs, &index, &flag, MPI_STATUSES_IGNORE); //advancing MPI in the background 

        usleep(bwd_rt_per_layer[i]); //compute

	if(i < NUM_Dense_L){ //dense layers
            MPI_Reduce_scatter_block(dense_bwd_rs_sbuff_ptrs[i], dense_bwd_rs_rbuff_ptrs[i], dense_bwd_reduce_scatter_sizes[i], MPI_FLOAT, MPI_SUM, model_parallel_comm);
	}
	else if(i < NUM_L-1){ //conv layers
	    int msg_idx = i-NUM_Dense_L;
            MPI_Request requests[4];
            MPI_Isend(bwd_halo_send_buff0_ptrs[msg_idx], conv_bwd_halo_sizes[msg_idx], MPI_FLOAT, mp_group_rank^1, i, model_parallel_comm, &requests[0]);
            MPI_Isend(bwd_halo_send_buff1_ptrs[msg_idx], conv_bwd_halo_sizes[msg_idx], MPI_FLOAT, mp_group_rank^2, i, model_parallel_comm, &requests[1]);
            MPI_Irecv(bwd_halo_recv_buff0_ptrs[msg_idx], conv_bwd_halo_sizes[msg_idx], MPI_FLOAT, mp_group_rank^1, i, model_parallel_comm, &requests[2]);
            MPI_Irecv(bwd_halo_recv_buff1_ptrs[msg_idx], conv_bwd_halo_sizes[msg_idx], MPI_FLOAT, mp_group_rank^2, i, model_parallel_comm, &requests[3]);
	    MPI_Waitall(4, requests, MPI_STATUSES_IGNORE);
	}

	if(i == NUM_Dense_L-1){
            MPI_Iallreduce(grad_ptrs[0], sum_grad_ptrs[0], allreduce_sizes[0], MPI_FLOAT, MPI_SUM, dense_allreduce_comm, &grad_allreduce_reqs[0]);	
	}
	else if(i > NUM_Dense_L-1){
            MPI_Iallreduce(grad_ptrs[i-NUM_Dense_L+1], sum_grad_ptrs[i-NUM_Dense_L+1], allreduce_sizes[i-NUM_Dense_L+1], MPI_FLOAT, MPI_SUM, conv_allreduce_comms[i-NUM_Dense_L], &grad_allreduce_reqs[i-NUM_Dense_L+1]);	
	}
    }

    MPI_Waitall(NUM_Conv_L+1, grad_allreduce_reqs, MPI_STATUSES_IGNORE); 
    return 0;
}

int main(int argc, char *argv[]){
    int rank, world_size;
    
    int model_shards = 4; // do not change this
    
    MPI_Init(&argc,&argv);
    MPI_Comm_size(MPI_COMM_WORLD, &world_size);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm conv_allreduce_comms[NUM_Conv_L];
    for(int i=0; i<NUM_Conv_L; i++)
        MPI_Comm_dup(MPI_COMM_WORLD, &conv_allreduce_comms[i]); //duplicated for nb colls

    int dense_allreduce_group_rank, mp_group_rank;
    int dense_allreduce_group_size, mp_group_size;

    //the number of processes should be a multiple of model_shards = 4
    assert(world_size % model_shards == 0);
    int dense_allreduce_group_color = rank % model_shards;

    MPI_Comm dense_allreduce_comm;
    MPI_Comm_split(MPI_COMM_WORLD, dense_allreduce_group_color, rank, &dense_allreduce_comm);

    MPI_Comm_rank(dense_allreduce_comm, &dense_allreduce_group_rank);
    MPI_Comm_size(dense_allreduce_comm, &dense_allreduce_group_size);

    MPI_Comm model_parallel_comm;
    MPI_Comm_split(MPI_COMM_WORLD, dense_allreduce_group_rank, rank, &model_parallel_comm);
    MPI_Comm_rank(model_parallel_comm, &mp_group_rank);
    MPI_Comm_size(model_parallel_comm, &mp_group_size);

    assert(dense_allreduce_group_color == mp_group_rank);
    assert(model_shards == mp_group_size);

    float* fwd_halo_send_buff0_ptrs[NUM_Conv_L-1];
    float* fwd_halo_send_buff1_ptrs[NUM_Conv_L-1];
    float* fwd_halo_recv_buff0_ptrs[NUM_Conv_L-1];
    float* fwd_halo_recv_buff1_ptrs[NUM_Conv_L-1];

    float* bwd_halo_send_buff0_ptrs[NUM_Conv_L-1];
    float* bwd_halo_send_buff1_ptrs[NUM_Conv_L-1];
    float* bwd_halo_recv_buff0_ptrs[NUM_Conv_L-1];
    float* bwd_halo_recv_buff1_ptrs[NUM_Conv_L-1];
    for(int i=0; i<NUM_Conv_L-1; i++){
        fwd_halo_send_buff0_ptrs[i] = (float *)calloc(conv_fwd_halo_sizes[i], sizeof(float));
        fwd_halo_send_buff1_ptrs[i] = (float *)calloc(conv_fwd_halo_sizes[i], sizeof(float));
        fwd_halo_recv_buff0_ptrs[i] = (float *)calloc(conv_fwd_halo_sizes[i], sizeof(float));
        fwd_halo_recv_buff1_ptrs[i] = (float *)calloc(conv_fwd_halo_sizes[i], sizeof(float));

        bwd_halo_send_buff0_ptrs[i] = (float *)calloc(conv_bwd_halo_sizes[i], sizeof(float));
        bwd_halo_send_buff1_ptrs[i] = (float *)calloc(conv_bwd_halo_sizes[i], sizeof(float));
        bwd_halo_recv_buff0_ptrs[i] = (float *)calloc(conv_bwd_halo_sizes[i], sizeof(float));
        bwd_halo_recv_buff1_ptrs[i] = (float *)calloc(conv_bwd_halo_sizes[i], sizeof(float));
    }

    float* dense_fwd_allgather_sbuff_ptrs[NUM_Dense_L];
    float* dense_fwd_allgather_rbuff_ptrs[NUM_Dense_L];
    float* dense_bwd_rs_sbuff_ptrs[NUM_Dense_L];
    float* dense_bwd_rs_rbuff_ptrs[NUM_Dense_L];
    for(int i=0; i<NUM_Dense_L; i++){
        dense_fwd_allgather_sbuff_ptrs[i] = (float *)calloc(dense_fwd_allgather_sizes[i], sizeof(float));
	dense_fwd_allgather_rbuff_ptrs[i] = (float *)calloc(dense_fwd_allgather_sizes[i]*model_shards, sizeof(float));
	dense_bwd_rs_sbuff_ptrs[i] = (float *)calloc(dense_bwd_reduce_scatter_sizes[i]*model_shards, sizeof(float));
	dense_bwd_rs_rbuff_ptrs[i] = (float *)calloc(dense_bwd_reduce_scatter_sizes[i], sizeof(float));
    }

    float* grad_ptrs[NUM_L-2];
    float* sum_grad_ptrs[NUM_L-2];
    for(int i=0; i<NUM_L-2; i++){
        grad_ptrs[i] = (float *)calloc(allreduce_sizes[i], sizeof(float));
        sum_grad_ptrs[i] = (float *)calloc(allreduce_sizes[i], sizeof(float));
    }

    MPI_Barrier(MPI_COMM_WORLD);

    //warmup
    for(int wmp = 0; wmp < WARM_UP; wmp++){
        run_model_data_parallel(fwd_halo_send_buff0_ptrs,
                                fwd_halo_send_buff1_ptrs,
                                fwd_halo_recv_buff0_ptrs,
                                fwd_halo_recv_buff1_ptrs,
                                bwd_halo_send_buff0_ptrs,
                                bwd_halo_send_buff1_ptrs,
                                bwd_halo_recv_buff0_ptrs,
                                bwd_halo_recv_buff1_ptrs,
                                dense_fwd_allgather_sbuff_ptrs,
                                dense_fwd_allgather_rbuff_ptrs,
                                dense_bwd_rs_sbuff_ptrs,
                                dense_bwd_rs_rbuff_ptrs,
                                grad_ptrs,
                                sum_grad_ptrs,
                                model_parallel_comm,
                                dense_allreduce_comm,
                                conv_allreduce_comms);
    }

    double begin, elapse;
    begin = MPI_Wtime();
    for(int iter = 0; iter < RUNS; iter++){
        run_model_data_parallel(fwd_halo_send_buff0_ptrs,
                                fwd_halo_send_buff1_ptrs,
                                fwd_halo_recv_buff0_ptrs,
                                fwd_halo_recv_buff1_ptrs,
                                bwd_halo_send_buff0_ptrs,
                                bwd_halo_send_buff1_ptrs,
                                bwd_halo_recv_buff0_ptrs,
                                bwd_halo_recv_buff1_ptrs,
                                dense_fwd_allgather_sbuff_ptrs,
                                dense_fwd_allgather_rbuff_ptrs,
                                dense_bwd_rs_sbuff_ptrs,
                                dense_bwd_rs_rbuff_ptrs,
                                grad_ptrs,
                                sum_grad_ptrs,
                                model_parallel_comm,
                                dense_allreduce_comm,
                                conv_allreduce_comms);
    }
    elapse = (MPI_Wtime()-begin)/RUNS;

    int total_params;
    for(int i=0; i<NUM_L-2; i++){
	if(i == 0)
	    total_params = allreduce_sizes[i] * model_shards;
        else
	    total_params += allreduce_sizes[i];	
    }

    if(rank == 0)
        printf("Rank = %d, world_size = %d, model_shards = %d, data_shards = %d, total_params = %d, global_batch_size = %d, CosmoFlow model-data parallelism runtime for each iteration = %f s\n", rank, world_size, mp_group_size, dense_allreduce_group_size, total_params, 8*dense_allreduce_group_size, elapse);

    MPI_Finalize();
}
