/*********************************************************************
 *
 * Description: C++/MPI proxy for BERT-large distributed training 
 *              with a hybrid pipeline and data parallelism
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

//p2p msg size for Bert with micro-batch size=8 and seq_length=128
#define P2PSIZE 1049600

#define BEGINNUM 21
#define INTERNUM 16
#define ENDNUM 26
//sizes for the gradients per layer of bert-large, 1024-hidden, 16-heads
int first_layer_grad_sizes[BEGINNUM] = {31254528, 524288, 2048, 1048576, 1048576, 1048576, 1048576, 4194304, 4194304, 1024, 1024, 1024, 1024, 1024, 1024, 1024, 1024, 4096, 1024, 1024, 1024};
int intermediate_layer_grad_sizes[INTERNUM] = {1048576, 1048576, 1048576, 1048576, 4194304, 4194304, 1024, 1024, 1024, 1024, 1024, 1024, 4096, 1024, 1024, 1024};
int end_layer_grad_sizes[ENDNUM] = {1048576, 1048576, 1048576, 1048576, 4194304, 4194304, 1048576, 1048576, 31254528, 2048, 1024, 1024, 1024, 1024, 1024, 1024, 4096, 1024, 1024, 1024, 1024, 30522, 1024, 1024, 1024, 2};


#define BEGINSIZE 44379136
#define INTERSIZE 12596224
#define ENDSIZE 45984572

////message aggregation
//#define BEGINNUM 1
//#define INTERNUM 1
//#define ENDNUM 1
//int first_layer_grad_sizes[BEGINNUM] = {BEGINSIZE};
//int intermediate_layer_grad_sizes[INTERNUM] = {INTERSIZE};
//int end_layer_grad_sizes[ENDNUM] = {ENDSIZE};


int main(int argc, char *argv[]){
    int rank, world_size;
    double begin, elapse;

    //number of basic Transformer layers
    int num_layer = 24;
    //number of pipeline stages
    int num_stage = 4;
    //number of micro-batches in an iteration
    int grad_acc_step = 8;

    if(argc == 3){
        num_layer = atoi(argv[1]);
        num_stage = atoi(argv[2]);
    }
    if(argc == 4){
        num_layer = atoi(argv[1]);
        num_stage = atoi(argv[2]);
        grad_acc_step = atoi(argv[3]);
    }
    
    MPI_Init(&argc,&argv);
    MPI_Comm_size(MPI_COMM_WORLD, &world_size);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm allreduce_comm;
    MPI_Comm p2p_comm;
    int allreduce_group_rank, p2p_group_rank;
    int allreduce_group_size, p2p_group_size;

    //the number of processes should be a multiple of num_stage
    assert((world_size>num_stage) && (world_size%num_stage == 0));
    int allreduce_group_color = rank % num_stage;

    MPI_Comm_split(MPI_COMM_WORLD, allreduce_group_color, rank, &allreduce_comm);
    MPI_Comm_rank(allreduce_comm, &allreduce_group_rank);
    MPI_Comm_size(allreduce_comm, &allreduce_group_size);

    MPI_Comm_split(MPI_COMM_WORLD, allreduce_group_rank, rank, &p2p_comm);
    MPI_Comm_rank(p2p_comm, &p2p_group_rank);
    MPI_Comm_size(p2p_comm, &p2p_group_size);

    int stage_id = allreduce_group_color;
    assert(allreduce_group_color == p2p_group_rank);


    int num_layer_per_stage = (int)(num_layer/num_stage);
    //printf("num_layer = %d, num_stage = %d, num_layer_per_stage = %d \n", num_layer, num_stage, num_layer_per_stage);
    //printf("Global rank = %d, allreduce group size/rank = %d, %d, p2p group size/rank = %d, %d\n", rank, allreduce_group_size, allreduce_group_rank, p2p_group_size, p2p_group_rank);

    int begin_stage_grad_num = BEGINNUM + (num_layer_per_stage - 1)*INTERNUM;
    int end_stage_grad_num = ENDNUM + (num_layer_per_stage - 1)*INTERNUM;
    int inter_stage_grad_num = num_layer_per_stage * INTERNUM;

    //pointers of the send/receive buffers
    float* begin_stage_grad_ptrs[begin_stage_grad_num];
    float* sum_begin_stage_grad_ptrs[begin_stage_grad_num];
    
    float* intermediate_stage_grad_ptrs[inter_stage_grad_num];
    float* sum_intermediate_stage_grad_ptrs[inter_stage_grad_num];
    
    float* end_stage_grad_ptrs[end_stage_grad_num];
    float* sum_end_stage_grad_ptrs[end_stage_grad_num];

    int num_grad_per_stage = -1;
    if(stage_id == 0){
        num_grad_per_stage = BEGINNUM + (num_layer_per_stage-1) * INTERNUM;
    }
    else if(stage_id == num_stage-1){
        num_grad_per_stage = ENDNUM + (num_layer_per_stage-1) * INTERNUM;
    }
    else{
        num_grad_per_stage = num_layer_per_stage * INTERNUM;
    }

    int stage_grad_sizes[num_grad_per_stage] = {0}; 

    if(stage_id == 0){
        for(int i=0; i<BEGINNUM; i++){
            begin_stage_grad_ptrs[i] = (float *)calloc(first_layer_grad_sizes[i], sizeof(float));
            sum_begin_stage_grad_ptrs[i] = (float *)calloc(first_layer_grad_sizes[i], sizeof(float));
            stage_grad_sizes[i] = first_layer_grad_sizes[i];
        }
        for(int j=0; j<num_layer_per_stage-1; j++){
            for(int k=0; k<INTERNUM; k++){
                begin_stage_grad_ptrs[INTERNUM*j+k+BEGINNUM] = (float *)calloc(intermediate_layer_grad_sizes[k], sizeof(float));
                sum_begin_stage_grad_ptrs[INTERNUM*j+k+BEGINNUM] = (float *)calloc(intermediate_layer_grad_sizes[k], sizeof(float));
                stage_grad_sizes[INTERNUM*j+k+BEGINNUM] = intermediate_layer_grad_sizes[k];                
            }
        }
    }
    else if(stage_id == num_stage-1){
        for(int j=0; j<num_layer_per_stage-1; j++){
            for(int k=0; k<INTERNUM; k++){
                end_stage_grad_ptrs[INTERNUM*j+k] = (float *)calloc(intermediate_layer_grad_sizes[k], sizeof(float));
                sum_end_stage_grad_ptrs[INTERNUM*j+k] = (float *)calloc(intermediate_layer_grad_sizes[k], sizeof(float));
                stage_grad_sizes[INTERNUM*j+k] = intermediate_layer_grad_sizes[k];                
            }
        }
        for(int i=0; i<ENDNUM; i++){
            end_stage_grad_ptrs[INTERNUM*(num_layer_per_stage-1)+i] = (float *)calloc(end_layer_grad_sizes[i], sizeof(float));
            sum_end_stage_grad_ptrs[INTERNUM*(num_layer_per_stage-1)+i] = (float *)calloc(end_layer_grad_sizes[i], sizeof(float));
            stage_grad_sizes[INTERNUM*(num_layer_per_stage-1)+i] = end_layer_grad_sizes[i]; 
        }
    }
    else{
        for(int j=0; j<num_layer_per_stage; j++){
            for(int k=0; k<INTERNUM; k++){
                intermediate_stage_grad_ptrs[INTERNUM*j+k] = (float *)calloc(intermediate_layer_grad_sizes[k], sizeof(float));
                sum_intermediate_stage_grad_ptrs[INTERNUM*j+k] = (float *)calloc(intermediate_layer_grad_sizes[k], sizeof(float));
                stage_grad_sizes[INTERNUM*j+k] = intermediate_layer_grad_sizes[k];                
            }
        }
    }
    MPI_Barrier(MPI_COMM_WORLD);

    //warmup
    for(int wmp = 0; wmp < WARM_UP; wmp++){
        float *send_buffer = (float *)calloc(P2PSIZE, sizeof(float));
        float *recv_buffer = (float *)calloc(P2PSIZE, sizeof(float));
    
        //p2p forward
        for(int i=0; i<grad_acc_step; i++){
            if(stage_id == 0){
                MPI_Request request;
                MPI_Isend(send_buffer, P2PSIZE, MPI_FLOAT, stage_id+1, i, p2p_comm, &request);
                MPI_Wait(&request, MPI_STATUS_IGNORE);
            }
            else if(stage_id == num_stage-1){
                MPI_Request request;
                MPI_Irecv(recv_buffer, P2PSIZE, MPI_FLOAT, stage_id-1, i, p2p_comm, &request);
                MPI_Wait(&request, MPI_STATUS_IGNORE);
            }
            else{
                MPI_Request requests[2];
                MPI_Isend(send_buffer, P2PSIZE, MPI_FLOAT, stage_id+1, i, p2p_comm, &requests[0]);
                MPI_Irecv(recv_buffer, P2PSIZE, MPI_FLOAT, stage_id-1, i, p2p_comm, &requests[1]);
                MPI_Waitall(2, requests, MPI_STATUSES_IGNORE);
            }
        }
    
        //p2p backward
        for(int i=0; i<grad_acc_step; i++){
            if(stage_id == 0){
                MPI_Request request;
                MPI_Irecv(recv_buffer, P2PSIZE, MPI_FLOAT, stage_id+1, i, p2p_comm, &request);
                MPI_Wait(&request, MPI_STATUS_IGNORE);
            }
            else if(stage_id == num_stage-1){
                MPI_Request request;
                MPI_Isend(send_buffer, P2PSIZE, MPI_FLOAT, stage_id-1, i, p2p_comm, &request);
                MPI_Wait(&request, MPI_STATUS_IGNORE);
            }
            else{
                MPI_Request requests[2];
                MPI_Isend(send_buffer, P2PSIZE, MPI_FLOAT, stage_id-1, i, p2p_comm, &requests[0]);
                MPI_Irecv(recv_buffer, P2PSIZE, MPI_FLOAT, stage_id+1, i, p2p_comm, &requests[1]);
                MPI_Waitall(2, requests, MPI_STATUSES_IGNORE);
            }
        }
    
        //allreduce on gradients
        if(stage_id == 0){
            for(int i=0; i<num_grad_per_stage; i++){
                MPI_Allreduce(begin_stage_grad_ptrs[i], sum_begin_stage_grad_ptrs[i], stage_grad_sizes[i], MPI_FLOAT, MPI_SUM, allreduce_comm);
            }
        }
        else if(stage_id == num_stage-1){
            for(int i=0; i<num_grad_per_stage; i++){
                MPI_Allreduce(end_stage_grad_ptrs[i], sum_end_stage_grad_ptrs[i], stage_grad_sizes[i], MPI_FLOAT, MPI_SUM, allreduce_comm);
            }
        }
        else{
            for(int i=0; i<num_grad_per_stage; i++){
                MPI_Allreduce(intermediate_stage_grad_ptrs[i], sum_intermediate_stage_grad_ptrs[i], stage_grad_sizes[i], MPI_FLOAT, MPI_SUM, allreduce_comm);
            }
        }
    }

    begin = MPI_Wtime();
    for(int iter = 0; iter < RUNS; iter++){
        float *send_buffer = (float *)calloc(P2PSIZE, sizeof(float));
        float *recv_buffer = (float *)calloc(P2PSIZE, sizeof(float));
    
        //p2p forward
        for(int i=0; i<grad_acc_step; i++){
            if(stage_id == 0){
                MPI_Request request;
                MPI_Isend(send_buffer, P2PSIZE, MPI_FLOAT, stage_id+1, i, p2p_comm, &request);
                MPI_Wait(&request, MPI_STATUS_IGNORE);
            }
            else if(stage_id == num_stage-1){
                MPI_Request request;
                MPI_Irecv(recv_buffer, P2PSIZE, MPI_FLOAT, stage_id-1, i, p2p_comm, &request);
                MPI_Wait(&request, MPI_STATUS_IGNORE);
            }
            else{
                MPI_Request requests[2];
                MPI_Isend(send_buffer, P2PSIZE, MPI_FLOAT, stage_id+1, i, p2p_comm, &requests[0]);
                MPI_Irecv(recv_buffer, P2PSIZE, MPI_FLOAT, stage_id-1, i, p2p_comm, &requests[1]);
                MPI_Waitall(2, requests, MPI_STATUSES_IGNORE);
            }
        }
    
        //p2p backward
        for(int i=0; i<grad_acc_step; i++){
            if(stage_id == 0){
                MPI_Request request;
                MPI_Irecv(recv_buffer, P2PSIZE, MPI_FLOAT, stage_id+1, i, p2p_comm, &request);
                MPI_Wait(&request, MPI_STATUS_IGNORE);
            }
            else if(stage_id == num_stage-1){
                MPI_Request request;
                MPI_Isend(send_buffer, P2PSIZE, MPI_FLOAT, stage_id-1, i, p2p_comm, &request);
                MPI_Wait(&request, MPI_STATUS_IGNORE);
            }
            else{
                MPI_Request requests[2];
                MPI_Isend(send_buffer, P2PSIZE, MPI_FLOAT, stage_id-1, i, p2p_comm, &requests[0]);
                MPI_Irecv(recv_buffer, P2PSIZE, MPI_FLOAT, stage_id+1, i, p2p_comm, &requests[1]);
                MPI_Waitall(2, requests, MPI_STATUSES_IGNORE);
            }
        }
    
        //allreduce on gradients
        if(stage_id == 0){
            for(int i=0; i<num_grad_per_stage; i++){
                MPI_Allreduce(begin_stage_grad_ptrs[i], sum_begin_stage_grad_ptrs[i], stage_grad_sizes[i], MPI_FLOAT, MPI_SUM, allreduce_comm);
            }
        }
        else if(stage_id == num_stage-1){
            for(int i=0; i<num_grad_per_stage; i++){
                MPI_Allreduce(end_stage_grad_ptrs[i], sum_end_stage_grad_ptrs[i], stage_grad_sizes[i], MPI_FLOAT, MPI_SUM, allreduce_comm);
            }
        }
        else{
            for(int i=0; i<num_grad_per_stage; i++){
                MPI_Allreduce(intermediate_stage_grad_ptrs[i], sum_intermediate_stage_grad_ptrs[i], stage_grad_sizes[i], MPI_FLOAT, MPI_SUM, allreduce_comm);
            }
        }
    }
    elapse = (MPI_Wtime()-begin)/RUNS;

    printf("Rank = %d, world_size = %d, layers = %d, stages = %d, total_params = %d, Bert-large pipeline and data parallelism runtime for each iteration = %f s\n", rank, world_size, num_layer, num_stage, BEGINSIZE+ENDSIZE+INTERSIZE*(num_layer-2), elapse);

    MPI_Finalize();
}
