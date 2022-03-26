#include <mpi.h>
#include <unistd.h>
#include <stdio.h>
#include <string>
#include <time.h>
#include <stdlib.h>
#include <assert.h>

#define WARM_UP 0
#define RUNS 1

#define NUM_L 96
#define ACC_STEP_SCALE 2
#define MODEL_SHARDS 4

// msg sizes for GPT-3 (M_dim=12288) with micro-batch size=1 and seq_len=2048
// we set model shards = 4
#define PIPE_P2P_SIZE       25165824
#define MP_ALLREDUCE_SIZE   25165824
#define MOE_ALL2ALL_SIZE    25165824
//#define DP_ALLREDUCE_SIZE   452984832+154389504
#define DP_ALLREDUCE_SIZE   452984832 // num params of one shard of a layer

// runtime in us (10E-6) for each model shard of each layer
#define FWD_RT 15915
#define BWD_RT 31830
#define BWD_RT_GPIPE 47745

int run_one_step_pipe_model(int grad_acc_step, int stage_id, int num_stage,
    		            float *grad_ptr,
                            float *sum_grad_ptr,
                            float *fwd_send_buff,
                            float *fwd_recv_buff,
                            float *bwd_send_buff,
                            float *bwd_recv_buff,
                            float **mp_fwd_inter_ptrs,
                            float **sum_mp_fwd_inter_ptrs,
                            float **mp_bwd_grad_ptrs,
                            float **sum_mp_bwd_grad_ptrs,
                            MPI_Comm dp_allreduce_comm,
                            MPI_Comm mp_allreduce_comm,
                            MPI_Comm pp_p2p_comm){

    MPI_Request reqs[2];

    if(stage_id % 2 == 0){
        MPI_Irecv(bwd_recv_buff, PIPE_P2P_SIZE, MPI_FLOAT, stage_id+1, 1, pp_p2p_comm, &reqs[0]);// receive input for next mb
        usleep(FWD_RT); //compute fwd
        for(int j=0; j<2; j++){
            MPI_Allreduce(mp_fwd_inter_ptrs[j], sum_mp_fwd_inter_ptrs[j], MP_ALLREDUCE_SIZE, MPI_FLOAT, MPI_SUM, mp_allreduce_comm);
        }
        MPI_Isend(fwd_send_buff, PIPE_P2P_SIZE, MPI_FLOAT, stage_id+1, 2, pp_p2p_comm, &reqs[1]);// send output of current mb
        MPI_Waitall(2, reqs, MPI_STATUS_IGNORE);
    }else{ 
        MPI_Irecv(fwd_recv_buff, PIPE_P2P_SIZE, MPI_FLOAT, stage_id-1, 2, pp_p2p_comm, &reqs[1]);// receive input for next mb
        usleep(BWD_RT); //compute bwd
        for(int j=0; j<2; j++){
            MPI_Allreduce(mp_bwd_grad_ptrs[j], sum_mp_bwd_grad_ptrs[j], MP_ALLREDUCE_SIZE, MPI_FLOAT, MPI_SUM, mp_allreduce_comm);
        }
        MPI_Isend(bwd_send_buff, PIPE_P2P_SIZE, MPI_FLOAT, stage_id-1, 1, pp_p2p_comm, &reqs[0]);// send output of current mb
        MPI_Waitall(2, reqs, MPI_STATUS_IGNORE);
    }

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
    MPI_Comm dp_allreduce_comm;
    MPI_Comm mp_pp_comm;
    MPI_Comm mp_allreduce_comm;
    MPI_Comm pp_p2p_comm;

    //the number of processes should be a multiple of num_stage*MODEL_SHARDS = 384
    assert(world_size % (num_stage*MODEL_SHARDS) == 0);

    int dp_allreduce_group_rank, mp_pp_group_rank, mp_allreduce_group_rank, pp_p2p_group_rank;
    int dp_allreduce_group_size, mp_pp_group_size, mp_allreduce_group_size, pp_p2p_group_size;

    int dp_allreduce_group_color = rank % (num_stage*MODEL_SHARDS);
    MPI_Comm_split(MPI_COMM_WORLD, dp_allreduce_group_color, rank, &dp_allreduce_comm);
    MPI_Comm_rank(dp_allreduce_comm, &dp_allreduce_group_rank);
    MPI_Comm_size(dp_allreduce_comm, &dp_allreduce_group_size);

    MPI_Comm_split(MPI_COMM_WORLD, dp_allreduce_group_rank, rank, &mp_pp_comm);
    MPI_Comm_rank(mp_pp_comm, &mp_pp_group_rank);
    MPI_Comm_size(mp_pp_comm, &mp_pp_group_size);

    int mp_allreduce_group_color = mp_pp_group_rank % num_stage;
    MPI_Comm_split(mp_pp_comm, mp_allreduce_group_color, mp_pp_group_rank, &mp_allreduce_comm);

    MPI_Comm_rank(mp_allreduce_comm, &mp_allreduce_group_rank);
    MPI_Comm_size(mp_allreduce_comm, &mp_allreduce_group_size);

    MPI_Comm_split(mp_pp_comm, mp_allreduce_group_rank, mp_pp_group_rank, &pp_p2p_comm);
    MPI_Comm_rank(pp_p2p_comm, &pp_p2p_group_rank);
    MPI_Comm_size(pp_p2p_comm, &pp_p2p_group_size);

    assert(pp_p2p_group_size == num_stage);
    assert(mp_allreduce_group_size == MODEL_SHARDS);
    assert(dp_allreduce_group_size == world_size/(num_stage*MODEL_SHARDS));

    int stage_id = pp_p2p_group_rank;

    float* grad_ptr = (float *)calloc(DP_ALLREDUCE_SIZE, sizeof(float));
    float* sum_grad_ptr = (float *)calloc(DP_ALLREDUCE_SIZE, sizeof(float));

    float* fwd_send_buff = (float *)calloc(PIPE_P2P_SIZE, sizeof(float));
    float* fwd_recv_buff = (float *)calloc(PIPE_P2P_SIZE, sizeof(float));
    float* bwd_send_buff = (float *)calloc(PIPE_P2P_SIZE, sizeof(float));
    float* bwd_recv_buff = (float *)calloc(PIPE_P2P_SIZE, sizeof(float));

    float* mp_fwd_inter_ptrs[2];
    float* sum_mp_fwd_inter_ptrs[2];
    float* mp_bwd_grad_ptrs[2];
    float* sum_mp_bwd_grad_ptrs[2];
    for(int i=0; i<2; i++){
        mp_fwd_inter_ptrs[i] = (float *)calloc(MP_ALLREDUCE_SIZE, sizeof(float));
        sum_mp_fwd_inter_ptrs[i] = (float *)calloc(MP_ALLREDUCE_SIZE, sizeof(float));
        mp_bwd_grad_ptrs[i] = (float *)calloc(MP_ALLREDUCE_SIZE, sizeof(float));
        sum_mp_bwd_grad_ptrs[i] = (float *)calloc(MP_ALLREDUCE_SIZE, sizeof(float));
    }
     
    float* moe_fwd_alltoall_ptrs[2];
    float* moe_bwd_alltoall_ptrs[2];
    for(int i=0; i<2; i++){
        moe_fwd_alltoall_ptrs[i] = (float *)calloc(MOE_ALL2ALL_SIZE, sizeof(float));
        moe_bwd_alltoall_ptrs[i] = (float *)calloc(MOE_ALL2ALL_SIZE, sizeof(float));
    }

    MPI_Barrier(MPI_COMM_WORLD);

    //warmup
    for(int wmp = 0; wmp < WARM_UP; wmp++){
        run_one_step_pipe_model(grad_acc_step, stage_id, num_stage,
        		        grad_ptr,
                                sum_grad_ptr,
                                fwd_send_buff,
                                fwd_recv_buff,
                                bwd_send_buff,
                                bwd_recv_buff,
                                mp_fwd_inter_ptrs,
                                sum_mp_fwd_inter_ptrs,
                                mp_bwd_grad_ptrs,
                                sum_mp_bwd_grad_ptrs,
                                dp_allreduce_comm,
                                mp_allreduce_comm,
                                pp_p2p_comm);
    }

    begin = MPI_Wtime();
    for(int iter = 0; iter < RUNS; iter++){
        run_one_step_pipe_model(grad_acc_step, stage_id, num_stage,
        		        grad_ptr,
                                sum_grad_ptr,
                                fwd_send_buff,
                                fwd_recv_buff,
                                bwd_send_buff,
                                bwd_recv_buff,
                                mp_fwd_inter_ptrs,
                                sum_mp_fwd_inter_ptrs,
                                mp_bwd_grad_ptrs,
                                sum_mp_bwd_grad_ptrs,
                                dp_allreduce_comm,
                                mp_allreduce_comm,
                                pp_p2p_comm);
    }
    elapse = (MPI_Wtime()-begin)/RUNS;

    if(rank == 0)
        printf("1F1B: Rank = %d, world_size = %d, layers = %d, stages = %d, acc_step = %d, total_params = %d B, global batch = %d, GPT-3 runtime for one pipeline step = %f s\n", rank, world_size, num_layer, num_stage, grad_acc_step, 1811939328/1024*num_layer/1024/1024, world_size*acc_step_scale/MODEL_SHARDS, elapse);

    MPI_Finalize();
}
