# DNN-cpp-proxies
C++/MPI proxies for distributed training of deep neural networks, including `ResNet-50`, `ResNet-152`, `BERT-large`, `DLRM`, `GPT-2`, `GPT-3`, etc. These proxies cover `data parallelism`, `operator parallelism`, `pipeline parallelism`, and `hybrid parallelism`.

## Demo
Compile:

`mpicxx gpt2_large.cpp -o gpt2`

Run:

`mpirun -n 32 ./gpt2`

Setup the number of Transformer layers and the number of pipeline stages:

`mpirun -n 32 ./gpt2 64 8`
