#include <chrono>
#include <cstddef>
#include <cstdio>
#include <cstdlib>
#include <list>
#include <numeric>
#include <set>
#include <vector>
#include <stdlib.h>
#include <iostream>
#include <cuda_runtime_api.h>
#include <cuda.h>
#include <random>
#include <cstring>
#include <inttypes.h>
#include <algorithm>

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


//Allocate a set amount of memory determined by the user
// and use this to create a vector with the correct output
int main(int argc, char** argv) {

  return cudaMain(argc, argv);

}
