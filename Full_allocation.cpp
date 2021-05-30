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
    int nnz = 2;

    char *valLoc = new char[2];
    int *rowLoc = new int[2];
    int *colLoc = new int[2];

    valLoc[0] = 'A';
    rowLoc[0] = 1;
    colLoc[0] = 0;

    valLoc[1]  = 'C';
    rowLoc[1] = 0;
    colLoc[1] = 1;


    char * valA;
    int * rowIndA;
    int * colIndA;

    cudaMalloc((void ** )&valA, nnz*sizeof(char));
    cudaMalloc((void ** )&rowIndA, nnz*sizeof(int));
    cudaMalloc((void ** )&colIndA, nnz*sizeof(int));


    //memcopys
    cudaMemcpy(valA, valLoc, nnz*sizeof(char), cudaMemcpyHostToDevice);
    cudaMemcpy(rowIndA, rowLoc, nnz*sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(colIndA, colLoc, nnz*sizeof(int), cudaMemcpyHostToDevice);


    char * vecLoc = new char[2];
    vecLoc[0] = 'T';
    vecLoc[1] = 'G';

    char * outLoc = new char[2];
    outLoc[0] = 0x20;
    outLoc[1] =  0x20;

    char * vecB;

    char * output;


    cudaMalloc((void ** )&vecB, nnz*sizeof(char));
    cudaMalloc((void ** )&output, nnz*sizeof(char));

    //memcopy to vec B and output;
    cudaMemcpy(vecB, vecLoc, nnz*sizeof(char), cudaMemcpyHostToDevice);
    cudaMemcpy(output, outLoc, nnz*sizeof(char), cudaMemcpyHostToDevice);

    auto start = std::chrono::high_resolution_clock::now();

    //fill_wrapper(nnz, valA,  rowIndA, colIndA);

    //fill_vector(nnz, vecB);

    //rn matrix in csc format
    semiring_spmv(nnz, valA,rowIndA,colIndA, vecB, output);

    cudaMemcpy(outLoc, output,  nnz*sizeof(char), cudaMemcpyDeviceToHost);

    auto end = std::chrono::high_resolution_clock::now();

    auto duration = std::chrono::duration_cast<std::chrono::microseconds>( end - start ).count();

    std::cout << "Filled matrix of " << nnz << " items in " << duration << " microseconds"  << endl;

    std::cout  << "Output: " << outLoc[0] << " " << outLoc[1] << "." << endl;

    return 0;
}
