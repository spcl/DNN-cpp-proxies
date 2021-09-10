# deep-nn-proxy
C++/MPI proxies for deep neural networks.

## Demo
Compile:
`mpicxx gpt2_large.cpp -o gpt2`

Run:
`mpirun -n 32 ./gpt2`

Setup the number of Transformer layers and the number of pipeline stages from command line:
`mpirun -n 32 ./gpt2 64 8`
