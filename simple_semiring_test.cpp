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
#include <string>

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

#ifndef MAX_VEC
#define MAX_VEC 100
#endif

using namespace std;

void printVec(int vecLen, char * vec, int*lengths){

  std::cout << "Vector: "  << endl;
  for (int i=0; i < vecLen; i++){

    std::cout << "[ ";
    for (int j = 0; j < lengths[i]; j++){
      cout << vec[i*MAX_VEC+j];
    }
    std:: cout << " ]" << endl;
  }

}

//Allocate a set amount of memory determined by the user
// and use this to create a vector with the correct output
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
    int * lenA;
    int * rowIndA;
    int * colIndA;


    //don't need to allocate for  end of string, use as a character array is safe if there is a
    //secondary buffer for lengths
    cudaMalloc((void ** )&valA, nnz*sizeof(char));
    cudaMalloc((void ** )&rowIndA, nnz*sizeof(int));
    cudaMalloc((void ** )&colIndA, nnz*sizeof(int));


    //memcopys
    cudaMemcpy(valA, valLoc, nnz*sizeof(char), cudaMemcpyHostToDevice);

    cudaMemcpy(rowIndA, rowLoc, nnz*sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(colIndA, colLoc, nnz*sizeof(int), cudaMemcpyHostToDevice);


    char * vecLoc = new char[2*MAX_VEC];
    vecLoc[0] = 'T';
    vecLoc[1*MAX_VEC] = 'G';
    int * lenVec = new int[2];
    lenVec[0] = 1;
    lenVec[1] = 1;

    char * outLoc = new char[2*MAX_VEC];
    outLoc[0] = 0x20;
    outLoc[1*MAX_VEC] =  0x20;
    int * lenOut = new int[2];
    lenOut[0] = 1;
    lenOut[1] = 1;

    char * vecB;

    char * output;

    int *vecL;
    int * outL;

    //define locks
    int * hostLocks = new int[2];
    int * locks;
    //init just for safety
    hostLocks[0]= 1;
    hostLocks[1] = 1;

    cudaMalloc((void **)&locks, nnz*sizeof(int));
    cudaMemcpy(vecB, vecLoc, nnz*MAX_VEC*sizeof(char), cudaMemcpyHostToDevice);

    cudaMalloc((void ** )&vecB, nnz*MAX_VEC*sizeof(char));
    cudaMalloc((void ** )&output, nnz*MAX_VEC*sizeof(char));

    cudaMalloc((void **)&vecL, nnz*sizeof(int));
    cudaMalloc((void **)&outL, nnz*sizeof(int));

    //memcopy to vec B and output;
    cudaMemcpy(vecB, vecLoc, nnz*MAX_VEC*sizeof(char), cudaMemcpyHostToDevice);
    cudaMemcpy(output, outLoc, nnz*MAX_VEC*sizeof(char), cudaMemcpyHostToDevice);

    //copy over sizing pointers
    cudaMemcpy(vecL, lenVec, nnz*sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(outL, lenOut, nnz*sizeof(int), cudaMemcpyHostToDevice);

    auto start = std::chrono::high_resolution_clock::now();

    //fill_wrapper(nnz, valA,  rowIndA, colIndA);

    //fill_vector(nnz, vecB);

    //print before
    cout << "Before" << endl;
    printVec(2, vecLoc, lenVec);

    //rn matrix in csc format

    //fun flips
    for (int i = 0; i < 50; i++){



      semiring_spmv(nnz, valA,rowIndA,colIndA, vecB, vecL, output, outL);

      //swap these bois
      char * temp = output;
      int * tempLens = outL;

      output = vecB;
      outL = vecL;

      vecB = temp;
      vecL = tempLens;


    }
    //one more for good measure
    semiring_spmv(nnz, valA,rowIndA,colIndA, vecB, vecL, output, outL);

    //copy back
    cudaMemcpy(outLoc, output,  nnz*MAX_VEC*sizeof(char), cudaMemcpyDeviceToHost);
    cudaMemcpy(lenOut, outL, nnz*sizeof(int), cudaMemcpyDeviceToHost);

    auto end = std::chrono::high_resolution_clock::now();

    auto duration = std::chrono::duration_cast<std::chrono::microseconds>( end - start ).count();

    std::cout << "Filled matrix of " << nnz << " items in " << duration << " microseconds"  << endl;

    //and print
    printVec(2, outLoc, lenOut);

    return 0;
}
