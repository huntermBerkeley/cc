#include <chrono>
#include <cstddef>
#include <cstdio>
#include <cstdlib>
#include <list>
#include <numeric>
#include <set>
#include <vector>
#include  <stdlib.h>
#include <iostream>
#include <cuda_runtime_api.h>
#include <cuda.h>
#include <random>

#include "test_kernels.hpp"

//Init cuda here

// __device__ void VecTest(float * A, float* B, size_t n){
//
//   size_t tid = threadIdx.x;
//   if (tid < n){
//     B[N]  = A[N];
//   }
//
//   __syncthreads();
//   return;
// }


using namespace std;



//This is a test of integrating cuda into upcxx funtions
int main(int argc, char** argv) {

    //first define sparse matrix
    int nnz = 100000000;
    int * valA;
    int * rowIndA;
    int * colIndA;

    cudaMalloc((void ** )&valA, nnz*sizeof(int));
    cudaMalloc((void ** )&rowIndA, nnz*sizeof(int));
    cudaMalloc((void ** )&colIndA, nnz*sizeof(int));


    int * vecB;

    int * output;


    cudaMalloc((void ** )&vecB, nnz*sizeof(int));
    cudaMalloc((void ** )&output, nnz*sizeof(int));


    auto start = std::chrono::high_resolution_clock::now();

    fill_wrapper(nnz, valA,  rowIndA, colIndA);

    fill_vector(nnz, vecB);

    //rn matrix in csc format
    spmv(nnz, valA,rowIndA,colIndA, vecB, output);

    auto end = std::chrono::high_resolution_clock::now();

    auto duration = std::chrono::duration_cast<std::chrono::microseconds>( end - start ).count();

    std::cout << "Filled matrix of " << nnz << " items in " << duration << " microseconds"  << endl;

    return 0;
}
