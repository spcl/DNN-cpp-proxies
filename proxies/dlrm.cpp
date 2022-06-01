/*********************************************************************
 *
 * Description: DLRM
 *  python -m torch.distributed.launch --nproc_per_node=4 dlrm_s_pytorch.py 
 *  --arch-embedding-size="80000-80000-80000-80000" --arch-sparse-feature-size=128 
 *  --arch-mlp-bot="128-128-128-128" --arch-mlp-top="512-512-512-256-1" 
 *  --max-ind-range=40000000 --data-generation=random --loss-function=bce 
 *  --round-targets=True --learning-rate=1.0 --mini-batch-size=2048 --print-freq=2 
 *  --print-time --test-freq=16 --test-mini-batch-size=1024 
 *  --memory-map --use-gpu --num-batches=32 --dist-backend=nccl  
 *
 *********************************************************************/

#include <mpi.h>
#include <unistd.h>
#include <stdio.h>
#include <string>
#include <time.h>
#include <stdlib.h>
#include <assert.h>

#define RUNS 1
#define WARM_UP 0


#define BOT_MLP_SIZE   49536
#define TOP_MLP_SIZE   728065 
#define EMB_ALL2ALL_SIZE   262144  //2048*128

// runtime in us (10E-6)
#define FWD_BOT_MLP 341
#define FWD_TOP_MLP 455
#define FWD_INTER 209
#define FWD_EMB 95

int run_dlrm(int num_proc,
    	     float *top_grad_ptr,
             float *sum_top_grad_ptr,
    	     float *bot_grad_ptr,
             float *sum_bot_grad_ptr,
             float *fwd_alltoall_send_ptrs,
             float *fwd_alltoall_recv_ptrs,
             float *bwd_alltoall_send_ptrs,
             float *bwd_alltoall_recv_ptrs){

    MPI_Request grad_allreduce_reqs[2];
    usleep(FWD_EMB); //fwd
    //alltoall
    MPI_Alltoall(fwd_alltoall_send_ptrs, EMB_ALL2ALL_SIZE/num_proc, MPI_FLOAT, fwd_alltoall_recv_ptrs, EMB_ALL2ALL_SIZE/num_proc, MPI_FLOAT, MPI_COMM_WORLD);

    usleep(FWD_BOT_MLP); //fwd
    usleep(FWD_INTER); //fwd

    usleep(FWD_TOP_MLP); //fwd

    usleep(FWD_TOP_MLP*2); //bwd
    //allreduce
    //MPI_Allreduce(top_grad_ptr, sum_top_grad_ptr, TOP_MLP_SIZE, MPI_FLOAT, MPI_SUM, MPI_COMM_WORLD);
    MPI_Iallreduce(top_grad_ptr, sum_top_grad_ptr, TOP_MLP_SIZE, MPI_FLOAT, MPI_SUM, MPI_COMM_WORLD, &grad_allreduce_reqs[0]);

    usleep(FWD_INTER); //bwd
    usleep(FWD_BOT_MLP*2); //bwd
    //allreduce
    //MPI_Allreduce(bot_grad_ptr, sum_bot_grad_ptr, BOT_MLP_SIZE, MPI_FLOAT, MPI_SUM, MPI_COMM_WORLD);
    MPI_Iallreduce(bot_grad_ptr, sum_bot_grad_ptr, BOT_MLP_SIZE, MPI_FLOAT, MPI_SUM, MPI_COMM_WORLD, &grad_allreduce_reqs[1]);

    //alltoall
    MPI_Alltoall(bwd_alltoall_send_ptrs, EMB_ALL2ALL_SIZE/num_proc, MPI_FLOAT, bwd_alltoall_recv_ptrs, EMB_ALL2ALL_SIZE/num_proc, MPI_FLOAT, MPI_COMM_WORLD);
    usleep(FWD_EMB*2); //bwd

    MPI_Waitall(2, grad_allreduce_reqs, MPI_STATUSES_IGNORE); 

    return 0;
}


int main(int argc, char *argv[]){
    int rank, world_size;
    double begin, elapse;

    MPI_Init(&argc,&argv);
    MPI_Comm_size(MPI_COMM_WORLD, &world_size);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    float* top_grad_ptr = (float *)calloc(TOP_MLP_SIZE, sizeof(float));
    float* sum_top_grad_ptr = (float *)calloc(TOP_MLP_SIZE, sizeof(float));
    float* bot_grad_ptr = (float *)calloc(BOT_MLP_SIZE, sizeof(float));
    float* sum_bot_grad_ptr = (float *)calloc(BOT_MLP_SIZE , sizeof(float));
     
    float* fwd_alltoall_send_ptrs = (float *)calloc(EMB_ALL2ALL_SIZE, sizeof(float));
    float* fwd_alltoall_recv_ptrs = (float *)calloc(EMB_ALL2ALL_SIZE, sizeof(float));
    float* bwd_alltoall_send_ptrs = (float *)calloc(EMB_ALL2ALL_SIZE, sizeof(float));
    float* bwd_alltoall_recv_ptrs = (float *)calloc(EMB_ALL2ALL_SIZE, sizeof(float));

    MPI_Barrier(MPI_COMM_WORLD);

    //warmup
    for(int wmp = 0; wmp < WARM_UP; wmp++){
        run_dlrm(world_size,
            	 top_grad_ptr,
                 sum_top_grad_ptr,
            	 bot_grad_ptr,
                 sum_bot_grad_ptr,
                 fwd_alltoall_send_ptrs,
                 fwd_alltoall_recv_ptrs,
                 bwd_alltoall_send_ptrs,
                 bwd_alltoall_recv_ptrs);
    }

    begin = MPI_Wtime();
    for(int iter = 0; iter < RUNS; iter++){
        run_dlrm(world_size,
            	 top_grad_ptr,
                 sum_top_grad_ptr,
            	 bot_grad_ptr,
                 sum_bot_grad_ptr,
                 fwd_alltoall_send_ptrs,
                 fwd_alltoall_recv_ptrs,
                 bwd_alltoall_send_ptrs,
                 bwd_alltoall_recv_ptrs);
    }
    elapse = (MPI_Wtime()-begin)/RUNS;

    if(rank == 0)
        printf("MoEs: Rank = %d, world_size = %d, global batch = %d, DLRM runtime per iteration = %f s\n", rank, world_size, 2048, elapse);

    MPI_Finalize();
}
