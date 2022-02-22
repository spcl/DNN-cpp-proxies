/*********************************************************************
 *
 * Description: C++/MPI proxy for GPT3 (175 B) distributed training 
 *              with a hybrid of data, pipeline parallism and MoEs
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

#define RUNS 128
#define WARM_UP 8

#define NUM_L 96
#define ACC_STEP_SCALE 2

// msg sizes for GPT-3 (M_dim=12288) with micro-batch size=1 and seq_len=2048
#define PIPE_P2P_SIZE       25165824
#define MP_ALLREDUCE_SIZE   25165824
#define MOE_ALL2ALL_SIZE    25165824
#define MHA_SIZE   603979776 // num params of mha in a layer
#define MLP_SIZE   1207959552 // num params of mlp in a layer

// runtime in us (10E-6)
#define FWD_MHA 22367
#define BWD_MHA 44734
#define FWD_MLP 41293
#define BWD_MLP 82586

int run_data_moe_pipe(int grad_acc_step, int stage_id, int num_stage, int num_moe,
    		      float *grad_ptr,
                      float *sum_grad_ptr,
                      float *fwd_send_buff,
                      float *fwd_recv_buff,
                      float *bwd_send_buff,
                      float *bwd_recv_buff,
                      float **moe_fwd_alltoall_send_ptrs,
                      float **moe_fwd_alltoall_recv_ptrs,
                      float **moe_bwd_alltoall_send_ptrs,
                      float **moe_bwd_alltoall_recv_ptrs,
                      MPI_Comm dp_moe_comm,
                      MPI_Comm pp_p2p_comm){

    MPI_Request fwd_reqs[2];
    MPI_Request bwd_reqs[2];
    for(int i=0; i<2; i++){
        fwd_reqs[i] = MPI_REQUEST_NULL;
        bwd_reqs[i] = MPI_REQUEST_NULL;
    }

    //forward
    for(int i=0; i<grad_acc_step; i++){
        if(stage_id == 0){
            MPI_Wait(&fwd_reqs[0], MPI_STATUS_IGNORE);
            usleep(FWD_MHA); //compute
            usleep(FWD_MLP/num_moe); 
            MPI_Isend(fwd_send_buff, PIPE_P2P_SIZE, MPI_FLOAT, stage_id+1, i, pp_p2p_comm, &fwd_reqs[0]);
        }
        else if(stage_id == num_stage-1){
            MPI_Irecv(fwd_recv_buff, PIPE_P2P_SIZE, MPI_FLOAT, stage_id-1, i, pp_p2p_comm, &fwd_reqs[1]);
            MPI_Wait(&fwd_reqs[1], MPI_STATUS_IGNORE);
            usleep(FWD_MHA); //compute
            usleep(FWD_MLP/num_moe); 
        }
        else{
            MPI_Irecv(fwd_recv_buff, PIPE_P2P_SIZE, MPI_FLOAT, stage_id-1, i, pp_p2p_comm, &fwd_reqs[1]);
            MPI_Wait(&fwd_reqs[1], MPI_STATUS_IGNORE);
            MPI_Wait(&fwd_reqs[0], MPI_STATUS_IGNORE);
            usleep(FWD_MHA); //compute
            usleep(FWD_MLP/num_moe); 
            MPI_Isend(fwd_send_buff, PIPE_P2P_SIZE, MPI_FLOAT, stage_id+1, i, pp_p2p_comm, &fwd_reqs[0]);
        }
        for(int j=0; j<2; j++){
            MPI_Alltoall(moe_fwd_alltoall_send_ptrs[j], MOE_ALL2ALL_SIZE/num_moe, MPI_FLOAT, moe_fwd_alltoall_recv_ptrs[j], MOE_ALL2ALL_SIZE/num_moe, MPI_FLOAT, dp_moe_comm);
        }
    }

    //backward
    for(int i=0; i<grad_acc_step; i++){
        if(stage_id == 0){
            MPI_Irecv(bwd_recv_buff, PIPE_P2P_SIZE, MPI_FLOAT, stage_id+1, i, pp_p2p_comm, &bwd_reqs[1]);
            MPI_Wait(&bwd_reqs[1], MPI_STATUS_IGNORE);
            usleep(BWD_MHA); //compute
            usleep(BWD_MLP/num_moe); 
        }
        else if(stage_id == num_stage-1){
            MPI_Wait(&bwd_reqs[0], MPI_STATUS_IGNORE);
            usleep(BWD_MHA); //compute
            usleep(BWD_MLP/num_moe); 
            MPI_Isend(bwd_send_buff, PIPE_P2P_SIZE, MPI_FLOAT, stage_id-1, i, pp_p2p_comm, &bwd_reqs[0]);
        }
        else{
            MPI_Irecv(bwd_recv_buff, PIPE_P2P_SIZE, MPI_FLOAT, stage_id+1, i, pp_p2p_comm, &bwd_reqs[1]);
            MPI_Wait(&bwd_reqs[1], MPI_STATUS_IGNORE);
            MPI_Wait(&bwd_reqs[0], MPI_STATUS_IGNORE);
            usleep(BWD_MHA); //compute
            usleep(BWD_MLP/num_moe); 
            MPI_Isend(bwd_send_buff, PIPE_P2P_SIZE, MPI_FLOAT, stage_id-1, i, pp_p2p_comm, &bwd_reqs[0]);
        }
        for(int j=0; j<2; j++){
            MPI_Alltoall(moe_bwd_alltoall_send_ptrs[j], MOE_ALL2ALL_SIZE/num_moe, MPI_FLOAT, moe_bwd_alltoall_recv_ptrs[j], MOE_ALL2ALL_SIZE/num_moe, MPI_FLOAT, dp_moe_comm);
        }
    }

    MPI_Allreduce(grad_ptr, sum_grad_ptr, MHA_SIZE, MPI_FLOAT, MPI_SUM, dp_moe_comm);

    return 0;
}


int main(int argc, char *argv[]){
    int rank, world_size;
    double begin, elapse;
    
    //number of pipeline stages
    int num_stage = NUM_L;
    int num_layer = NUM_L;
    int acc_step_scale = ACC_STEP_SCALE;
    //number of micro-batches in an iteration
    int grad_acc_step = num_stage * acc_step_scale;

    if(argc == 2){
        num_stage = atoi(argv[1]);
        num_layer = atoi(argv[1]);
    }
    if(argc == 3){
        num_stage = atoi(argv[1]);
        num_layer = atoi(argv[1]);
        acc_step_scale = atoi(argv[2]);
        grad_acc_step = num_stage * acc_step_scale;
    }

    MPI_Init(&argc,&argv);
    MPI_Comm_size(MPI_COMM_WORLD, &world_size);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm dp_moe_comm;
    MPI_Comm pp_p2p_comm;

    //the number of processes should be a multiple of num_stage
    assert(world_size % num_stage == 0);
    int num_moe = world_size / num_stage;

    int dp_moe_group_rank, pp_p2p_group_rank;
    int dp_moe_group_size, pp_p2p_group_size;

    int dp_moe_group_color = rank % num_stage;
    MPI_Comm_split(MPI_COMM_WORLD, dp_moe_group_color, rank, &dp_moe_comm);
    MPI_Comm_rank(dp_moe_comm, &dp_moe_group_rank);
    MPI_Comm_size(dp_moe_comm, &dp_moe_group_size);

    MPI_Comm_split(MPI_COMM_WORLD, dp_moe_group_rank, rank, &pp_p2p_comm);
    MPI_Comm_rank(pp_p2p_comm, &pp_p2p_group_rank);
    MPI_Comm_size(pp_p2p_comm, &pp_p2p_group_size);

    assert(pp_p2p_group_size == num_stage);
    assert(dp_moe_group_size == num_moe);

    int stage_id = pp_p2p_group_rank;

    float* grad_ptr = (float *)calloc(MHA_SIZE, sizeof(float));
    float* sum_grad_ptr = (float *)calloc(MHA_SIZE, sizeof(float));

    float* fwd_send_buff = (float *)calloc(PIPE_P2P_SIZE, sizeof(float));
    float* fwd_recv_buff = (float *)calloc(PIPE_P2P_SIZE, sizeof(float));
    float* bwd_send_buff = (float *)calloc(PIPE_P2P_SIZE, sizeof(float));
    float* bwd_recv_buff = (float *)calloc(PIPE_P2P_SIZE, sizeof(float));
     
    float* moe_fwd_alltoall_send_ptrs[2];
    float* moe_fwd_alltoall_recv_ptrs[2];
    float* moe_bwd_alltoall_send_ptrs[2];
    float* moe_bwd_alltoall_recv_ptrs[2];
    for(int i=0; i<2; i++){
        moe_fwd_alltoall_send_ptrs[i] = (float *)calloc(MOE_ALL2ALL_SIZE, sizeof(float));
        moe_fwd_alltoall_recv_ptrs[i] = (float *)calloc(MOE_ALL2ALL_SIZE, sizeof(float));
        moe_bwd_alltoall_send_ptrs[i] = (float *)calloc(MOE_ALL2ALL_SIZE, sizeof(float));
        moe_bwd_alltoall_recv_ptrs[i] = (float *)calloc(MOE_ALL2ALL_SIZE, sizeof(float));
    }

    MPI_Barrier(MPI_COMM_WORLD);

    //warmup
    for(int wmp = 0; wmp < WARM_UP; wmp++){
        run_data_moe_pipe(grad_acc_step, stage_id, num_stage, num_moe,
        		  grad_ptr,
                          sum_grad_ptr,
                          fwd_send_buff,
                          fwd_recv_buff,
                          bwd_send_buff,
                          bwd_recv_buff,
			  moe_fwd_alltoall_send_ptrs,
			  moe_fwd_alltoall_recv_ptrs,
			  moe_bwd_alltoall_send_ptrs,
			  moe_bwd_alltoall_recv_ptrs,
                          dp_moe_comm,
                          pp_p2p_comm);
    }

    begin = MPI_Wtime();
    for(int iter = 0; iter < RUNS; iter++){
        run_data_moe_pipe(grad_acc_step, stage_id, num_stage, num_moe,
        		  grad_ptr,
                          sum_grad_ptr,
                          fwd_send_buff,
                          fwd_recv_buff,
                          bwd_send_buff,
                          bwd_recv_buff,
			  moe_fwd_alltoall_send_ptrs,
			  moe_fwd_alltoall_recv_ptrs,
			  moe_bwd_alltoall_send_ptrs,
			  moe_bwd_alltoall_recv_ptrs,
                          dp_moe_comm,
                          pp_p2p_comm);
    }
    elapse = (MPI_Wtime()-begin)/RUNS;

    if(rank == 0)
        printf("MoEs: Rank = %d, world_size = %d, layers = %d, stages = %d, num_moe = %d, acc_step = %d, total_params = %d B, global batch = %d, GPT-3 DP-MoE-PP runtime for each iteration = %f s\n", rank, world_size, num_layer, num_stage, num_moe, grad_acc_step, 1811939328/1024*num_layer/1024/1024, world_size*acc_step_scale, elapse);

    MPI_Finalize();
}
