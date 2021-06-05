
#include <cstdio>
#include <cstdlib>
#include <cuda.h>
#include <cusparse_v2.h>
#include <stdio.h>
#include <string>
#include <inttypes.h>
#include <iostream>
#include <vector>
#include <assert.h>
#include <fstream>
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

typedef unsigned long long int uint64_cu;

#ifndef MAX_VEC
#define MAX_VEC 4000
#endif

//credit to stackoverflow
//https://stackoverflow.com/questions/5447570/cuda-atomic-operations-on-unsigned-chars

using namespace std;

__device__ static inline char atomicCAS(char* address, char expected, char desired) {
    size_t long_address_modulo = (size_t) address & 3;
    auto* base_address = (unsigned int*) ((char*) address - long_address_modulo);
    unsigned int selectors[] = {0x3214, 0x3240, 0x3410, 0x4210};

    unsigned int sel = selectors[long_address_modulo];
    unsigned int long_old, long_assumed, long_val, replacement;
    char old;

    long_val = (unsigned int) desired;
    long_old = *base_address;
    do {
        long_assumed = long_old;
        replacement = __byte_perm(long_old, long_val, sel);
        long_old = atomicCAS(base_address, long_assumed, replacement);
        old = (char) ((long_old >> (long_address_modulo * 8)) & 0x000000ff);
    } while (expected == old && long_assumed != long_old);

    return old;
}

void printrowkern(uint64_t row, char * vec, uint64_t*lengths){

    std::cout << "len: " << lengths[row] << endl;
    std::cout << "[ ";
    for (int j = 0; j < lengths[row]; j++){
      cout << vec[row*MAX_VEC+j];
    }
    std:: cout << " ]" << endl;

}

void printLenskern(std::vector<uint64_t> rows, uint64_t*lengths){

  std::cout << "[ ";
    for (int j = 0; j < rows.size(); j++){
      std::cout << rows.at(j) << ": " << lengths[rows.at(j)] << ", ";
    }
    std:: cout << " ]" << endl;

}

void printCudaVec(uint64_t nnz, uint64_t* cudaVec){

  uint64_t * copy;

  copy = new uint64_t[nnz];

  cudaMemcpy(copy, cudaVec,  nnz*sizeof(uint64_t), cudaMemcpyDeviceToHost);

  for (uint64_t i =0; i < nnz; i++){
    cout << i << ": [ " << copy[i] << " ]  ";

    if (i % 5 == 4){
      cout << endl;
    }
  }
  cout << endl;

  delete copy;
}

void printCudaStars(uint64_t nnz, bool* cudaVec){

  bool * copy;

  copy = new bool[nnz];

  cudaMemcpy(copy, cudaVec,  nnz*sizeof(bool), cudaMemcpyDeviceToHost);

  for (uint64_t i =0; i < nnz; i++){
    cout << i << ": [ " << copy[i] << " ]" << endl;
  }

  delete copy;
}

//convert sparse char mat to boolean ints
__global__ void mat_char_to_int(uint64_t nnz, uint64_t * Arows, uint64_t * Acols, char* Avals, uint64_t *Bcols, uint64_t * Brows, uint64_t * Bvals){

  uint64_t tid = threadIdx.x +  blockIdx.x*blockDim.x;

  if (tid >= nnz) return;

  Brows[tid] = Arows[tid];
  Bcols[tid] = Acols[tid];
  Bvals[tid] = 1;
}




__global__ void naive_cond_hook(uint64_t nnz, uint64_t * Arows, uint64_t * Acols, char * Avals, uint64_t * parent, bool * star){

  uint64_t tid = threadIdx.x +  blockIdx.x*blockDim.x;

  if (tid >= nnz) return;

  uint64_t u = Arows[tid];
  uint64_t v = Acols[tid];

  uint64_t parent_u = parent[u];
  uint64_t parent_v = parent[v];


  //retreive f earlier
  uint64_t gparent_u = parent[parent[u]];
  uint64_t old;

  //star hook procedure
  if (star[u] && parent[u] > parent[v]){
    old = (uint64_t) atomicCAS( (uint64_cu *) parent+parent_u, (uint64_cu) gparent_u, (uint64_cu) parent_v);
    //if this is the case we must have succeeded
    if (old == gparent_u){
      return;
    }
    parent_v = parent[v];
    parent_u = parent[u];
    gparent_u = parent[parent_u];
  }


}

__global__ void parent_cond_hook(uint64_t nnz, uint64_t * parent, uint64_t* gparent, bool * star, char* contigs, uint64_t * contig_lens, char* contigs_holder, uint64_t * contig_lens_holder){

  uint64_t tid = threadIdx.x +  blockIdx.x*blockDim.x;

  if (tid >= nnz) return;

  //for parent cond hook, if I am not a star, set my parent to my grandparent

  //if star[u]
  // parent[u] = parent[parent[u]]
  uint64_t gparent_u = gparent[tid];
  uint64_t parent_u = parent[tid];

  if (star[tid]){

    //absorb from your parent
    //first copy over your material

    //compress parents
    uint64_t my_contig_len = contig_lens[tid];
    uint64_t my_parent_len = contig_lens[parent_u];
    char * my_contig = contigs + MAX_VEC*tid;
    char * my_parent = contigs+MAX_VEC*parent_u;
    char * my_output = contigs_holder + MAX_VEC*tid;

    //copy from me
    for (int i = 0; i < my_contig_len; i++){
      my_output[i] = my_contig[i];
    }

    //copy from my parent
    for (int i =0; i < my_parent_len; i++){
      my_output[i+my_contig_len] = my_parent[i];
    }

    //copy to new len
    contig_lens_holder[tid] = my_contig_len+my_parent_len;

    //and absorb
    parent[tid] = gparent_u;



  } else {

    uint64_t my_contig_len = contig_lens[tid];
    char * my_contig = contigs + MAX_VEC*tid;
    char * my_output = contigs_holder + MAX_VEC*tid;

    //copy from me
    for (int i = 0; i < my_contig_len; i++){
      my_output[i] = my_contig[i];
    }

    contig_lens_holder[tid] = my_contig_len;

  }
  // if (parent[tid] =  parent[parent[tid]]){
  //   star[tid] = false;
  // }


}


__global__ void naive_uncond_hook(uint64_t nnz, uint64_t * Arows, uint64_t * Acols, char * Avals, uint64_t * parent, bool * star){

  uint64_t tid = threadIdx.x +  blockIdx.x*blockDim.x;

  if (tid >= nnz) return;

  uint64_t u = Arows[tid];
  uint64_t v = Acols[tid];

  uint64_t parent_u = parent[u];
  uint64_t parent_v = parent[v];

  //retreive f earlier
  uint64_t gparent_u = parent[parent[u]];
  uint64_t old;

  //star hook procedure
  if (star[u] && parent[u] != parent[v]){
    old = (uint64_t) atomicCAS( (uint64_cu *) parent+parent_u, (uint64_cu) gparent_u, (uint64_cu) parent_v);
    //if this is the case we must have succeeded
    if (old == gparent_u){
      return;
    }
    parent_v = parent[v];
    parent_u = parent[u];
    gparent_u = parent[parent_u];
  }


}

__global__ void shortcutting(uint64_t nnz, uint64_t * parents, uint64_t * gparents, bool * stars){

  //assume gparents already defined
  uint64_t tid = threadIdx.x +  blockIdx.x * blockDim.x;

  //double check this is numcols
  if (tid >= nnz) return;

  uint64_t v = tid;

  //star hook procedure
  if (!stars[v]){

    parents[v] = gparents[v];

  }


}

__global__ void setGrandparents(uint64_t nnz, uint64_t * parents, uint64_t * grandparents){

  //assume gparents already defined
  uint64_t tid = threadIdx.x +  blockIdx.x*blockDim.x;

  //double check this is numcols
  if (tid >= nnz) return;

  grandparents[tid] = parents[parents[tid]];

  return;

}

__global__ void reset_star(uint64_t nnz, bool * stars){

  uint64_t tid = threadIdx.x +  blockIdx.x*blockDim.x;

  //double check this is numcols
  if (tid >= nnz) return;

  stars[tid] = true;


}


//initialize every thread to be it's own parent
__global__ void init_parent(uint64_t nnz,  uint64_t* parent){

  uint64_t tid = threadIdx.x +  blockIdx.x*blockDim.x;

  if (tid < nnz){
    parent[tid] = tid;
  }

}

//initialize lengths to be 0
__global__ void init_contig_lens(uint64_t nnz,  uint64_t* contig_lens){

  uint64_t tid = threadIdx.x +  blockIdx.x*blockDim.x;

  if (tid < nnz){
    contig_lens[tid] = 0;
  }

}

//assert lengths are null
__global__ void assert_contig_lens(uint64_t nnz,  uint64_t* contig_lens){

  uint64_t tid = threadIdx.x +  blockIdx.x*blockDim.x;

  if (tid < nnz){
    if (contig_lens[tid] != 0){
      printf("contiig length %llu not 0\n", tid);
    }
  }

}

//initialize lengths to be 0
__global__ void init_contigs(uint64_t nnz, uint64_t num_vert, uint64_t* Arows, uint64_t* Acols, char* Avals, char* contigs, uint64_t* contig_lens){

  uint64_t tid = threadIdx.x +  blockIdx.x*blockDim.x;

  if (tid < nnz){

    //grab row
    char my_val = Avals[tid];
    contigs[Arows[tid]*MAX_VEC] = my_val;
    contig_lens[Arows[tid]] = 1;
  }

}



__device__ char semiring_multiply(char a, char b){

  printf("Multiplying %c, %c\n", a,b);
  if (a == 'z' ||  b == 'z')
    return 'z';

  if (a == 0x20) return b;

  return a;
}

__device__ char semiring_add(char a, char b){

  printf("adding %c, %c\n", a,b);
  if (a == 0x20){
    return b;
  }
  if (b == 0x20){
    return a;
  }
  //both nonzero, bad path
  //this will corrupt any future adds to this index as well
  return 'z';
}

__global__ void copy_kernel(double * to_copy, double* items, size_t n) {
  int tid = threadIdx.x +  blockIdx.x*blockDim.x;



  if (tid < n) {
    to_copy[tid] = items[tid];
  }
}

__global__ void uint_copy_kernel(uint64_t* to_fill, uint64_t* to_copy, size_t n) {
  uint64_t tid = threadIdx.x +  blockIdx.x*blockDim.x;



  if (tid < n) {
    to_fill[tid] = to_copy[tid];
  }
}

__global__ void copy_kernel_char(char * to_copy, char* items, size_t n) {
  int tid = threadIdx.x +  blockIdx.x*blockDim.x;



  if (tid < n) {
    printf("Tid %d reporting\n", tid);
    to_copy[tid] = items[tid];
  }
}

//__global__ void kmer_copy_kernel(uint64_t contig_num, char * contigs, )

//After the conditional hooking step, we should push any updated reads into the contigs
//because this happens first, the len of the contigs must be 1
__global__ void update_leads(uint64_t nnz, char * contigs, uint64_t * contig_lens, uint64_t num_updates, char * updates, uint64_t * update_lens, uint64_t * parent){

  uint64_t tid = threadIdx.x +  blockIdx.x*blockDim.x;

  if (tid >= num_updates) return;


  uint64_t contig_index = parent[tid];

  assert(contig_lens[contig_index] == 1);

  contig_lens[contig_index] += update_lens[tid];

  //move the first intem back to the last index in preparation for the copy kernel
  contigs[MAX_VEC*contig_index+contig_lens[contig_index]-1] = contigs[MAX_VEC*contig_index];


  //copy kernel moved from cc
  for (int i = 0; i < contig_lens[contig_index]; i++){

    contigs[MAX_VEC*contig_index+i] = updates[MAX_VEC*tid+i];

  }

  //finished>
  return;

}

//kernel to fill the matrix with random values - every row gets one item, and a col that is 10+ my row number
//this sparse matrix loops 10x
__global__  void fill_matrix(int nnz, int* vals, int*rows, int*cols){

  int tid = threadIdx.x +  blockIdx.x*blockDim.x;

  if (tid < nnz){
    vals[tid]  = 1;
    cols[tid] = tid;
    rows[tid] = ( tid+10 ) % nnz;
  }

}


//assume dataset is cleaned to have no columns with more than 1 value
//when this is the case you don't need memory checks
__global__ void multiply_semiring(uint64_t matNonZeros, char* matVals, uint64_t*matRows, uint64_t*matCols, char* vecB, uint64_t* vecLens, char* output, uint64_t * outLens){


  uint64_t tid = threadIdx.x + blockIdx.x * blockDim.x;

  if (tid >= matNonZeros) return;


  //look into a and get my col index
  uint64_t my_col  =  matCols[tid];
  uint64_t my_row = matRows[tid];
  char my_val = matVals[tid];

  char* my_vec = vecB + my_col*MAX_VEC;
  uint64_t my_vec_len = vecLens[my_col];
  uint64_t new_len = my_vec_len+1;


  if (my_row == 75){
    printf("Tid %" PRIu64 " reporting\n", tid);
    printf("Row %" PRIu64 "\n", my_row);
    printf("Col %" PRIu64 "\n", my_col);
    printf("Val %c\n", my_val);
    printf("New len %" PRIu64 "\n",new_len);
  }

  //printf("Value %d,%d: %s reporting with new len %d\n", my_row,my_col,my_val, new_len);

  //terminate call, no need to retreive


  //who cares, we are erasing -- probably clean this up
  char* my_output = output + my_row*MAX_VEC;
  //uint64_t my_old_len = outLens[my_row];

  //if no item, terminate
  //will these ever be called?
  //no
  //e for exit lmao


  //grab lock here - keep swapping in one while
  // uint64_t old = 0;
  // do {
  //   old = atomicExch(locks+my_row, 1);
  // } while (old != 0);

  outLens[my_row] = new_len;

  // if (my_row == 2832){
  //     printf("Outlen %" PRIu64 "\n",outLens[my_row]);
  // }

  //copy values, ask about faster method
  for (uint64_t i = 0; i < my_vec_len; i++){
    my_output[i] = my_vec[i];
  }
  //and replace old length
  my_output[my_vec_len] = my_val;

  //replace lock
  //old = atomicExch(locks+my_row, 0);


}

__global__ void multiply_kernel(int matNonZeros, int* matVals, int* matRows, int* matCols, int* vecB, int * output){


  int tid = threadIdx.x + blockIdx.x * blockDim.x;

  if (tid >= matNonZeros) return;

  //look into a and get my col index
  int my_col  =  matCols[tid];
  int my_row = matRows[tid];
  int my_val = matVals[tid];
  int my_vec = vecB[my_col];
  int old = output[my_row];
  int assumed;

  //prep my value and cas
  int my_output =  my_val * my_vec;

  do {
    assumed = old;
    old = atomicCAS(output + my_row, assumed, my_output+assumed);

  } while (assumed != old);

}

__global__ void vec_kernel(int nnz, int* vec){
  int tid = threadIdx.x +  blockIdx.x * blockDim.x;

  if (tid >= nnz) return;


  vec[tid] = 1;
}



__global__ void clear_kernel(int nnz, char*vec){
  int tid = threadIdx.x +  blockIdx.x * blockDim.x;

  if (tid >= nnz) return;


  vec[tid] =0x20;

}

__device__ void reduce_recursive(uint64_t* x, size_t n) {
  size_t tid = threadIdx.x + blockIdx.x * blockDim.x;

  size_t k = n/2;



  // TODO: Write reduction implementation here.
  if (tid < k && tid+k < n){
    //sum
    //printf("Val: %d\n", x[tid]);
    x[tid] = max(x[tid],x[tid+k]);
  }

  //sync threads
  __syncthreads();

  if (k == 1){
    return;
  }
  reduce_recursive(x, k);


}

__global__ void reduce(uint64_t* x, size_t n) {
  reduce_recursive(x, n);
}

__device__ void reduce_parents_recursive(uint64_t* x, size_t n) {
  size_t tid = threadIdx.x + blockIdx.x * blockDim.x;

  size_t k = n/2;



  // TODO: Write reduction implementation here.
  if (tid < k && tid+k < n){
    //sum
    //printf("Val: %d\n", x[tid]);
    x[tid] = x[tid] + x[tid+k];
  }

  //sync threads
  __syncthreads();

  if (k == 1){
    return;
  }
  reduce_recursive(x, k);


}

__global__ void check_stars(uint64_t nnz,  bool * stars, int* converged){

    size_t tid = threadIdx.x + blockIdx.x * blockDim.x;

    if (tid >= nnz) return;

    if (stars[tid]){
      //printf("We should not converge: %llu\n", tid);

      //swap 1 with 0 if it hasn't happened
      //come back and time this
      converged[0] = 0;
      //cas works, let's test regular convergence
      //atomicCAS(converged,1,0);

    }

    return;

}

//replace old parent with comparison
__device__ void overwrite_old_parents(uint64_t* new_parents, uint64_t* old_parents, size_t n){

  size_t tid = threadIdx.x + blockIdx.x * blockDim.x;

  if (tid < n){

    uint64_t old_p = old_parents[tid];
    uint64_t new_p = new_parents[tid];

    if (new_p==old_p){
      //set to zero - if sum is zero all are equal
      old_parents[tid] = 0;
    } else {
      old_parents[tid] = 1;
    }
  }

}

__global__ void reduce_parents(uint64_t* new_parents, uint64_t* old_parents, size_t n) {

  overwrite_old_parents(new_parents, old_parents,  n);

  //parents are set, `reduce`
  reduce_parents_recursive(old_parents, n);
}

void  fill_wrapper(int nnz, int*vals, int*rows, int*cols){

  int blocknums  = (nnz - 1)/ 1024 + 1;

  fill_matrix<<<blocknums, 1024>>>(nnz, vals,rows,cols);

}

void copy_wrapper(double * to_copy, double* items, size_t n){

  copy_kernel<<<1,n>>>(to_copy, items, n);

}

//check if all items in vec are false: if true, converged
bool starConverged(uint64_t nnz, bool*stars){

  int * converged;

  cudaMallocManaged((void **)&converged,1*sizeof(int));

  //set to true initially
  converged[0] = 1;

  uint64_t blocknums = (nnz -1)/1024 + 1;

  check_stars<<<blocknums, 1024>>>(nnz, stars, converged);
  cudaDeviceSynchronize();

  bool result = true;

  result = (converged[0] == 1);

  std::cout << "Result: " << result << "." << std::endl;
  cudaFree(converged);

  return result;

}

void  fill_vector(int nnz, int*vector){

  int blocknums  = (nnz - 1)/ 1024 + 1;

  vec_kernel<<<blocknums, 1024>>>(nnz, vector);

}

void spmv(int nnz, int* matVals, int* matRows, int* matCols, int* vecB, int * output){

  int blocknums  = (nnz - 1)/ 1024 + 1;
  multiply_kernel<<<blocknums, 1024>>>(nnz, matVals, matRows, matCols, vecB, output);


}

void semiring_spmv(uint64_t nnz, char* matVals, uint64_t* matRows, uint64_t* matCols, char* vecB, uint64_t* vecLens, char * output, uint64_t* outLens){
  uint64_t blocknums = (nnz -1)/1024 + 1;

  //dummy kernel to copy over results to assert that cuda is working
  //copy_kernel_char<<<blocknums, 1024>>>(output, matVals, nnz);
  multiply_semiring<<<blocknums, 1024>>>(nnz, matVals, matRows, matCols, vecB, vecLens, output, outLens);

  cudaDeviceSynchronize();
}

//given an array of lengths, return the largest
//use cuda reduction, stay on machine
uint64_t arr_max(uint64_t* lens, uint64_t n){

  int blocknums = (n -1)/1024 + 1;

  uint64_t * tempLens;
  cudaMalloc((void ** )&tempLens, n*sizeof(uint64_t));

  //copy over to temp vector
  uint_copy_kernel<<<blocknums, 1024>>>(tempLens, lens, n);

  //printf("Entering reduce\n");
  //and reduce
  reduce<<<blocknums, 1024>>>(tempLens, n);

  //copy to uint64_t before deletion.
  uint64_t max;
  cudaMemcpy(&max, tempLens, sizeof(uint64_t), cudaMemcpyDeviceToHost);

  cudaFree(tempLens);

  return max;

}

//build grandparents - needs to happen as independent kernel call
uint64_t * build_grandparents(uint64_t nnz, uint64_t * parents){

  uint64_t * grandparents;

  cudaMalloc((void **)&grandparents,nnz*sizeof(uint64_t));

  uint64_t blocknums = (nnz -1)/1024 + 1;

  setGrandparents<<<blocknums,1024>>>(nnz, parents, grandparents);

  return grandparents;


}

__global__ void star_gp_compare(uint64_t nnz, uint64_t*parents, uint64_t* grandparents, bool* stars){

  int tid = threadIdx.x +  blockIdx.x * blockDim.x;

  if (tid >= nnz) return;

  uint64_t gp = grandparents[tid];
  uint64_t parent = parents[tid];

  if (gp != parent){
    stars[tid] = false;
    stars[gp] = false;
  }


}

__global__ void parent_star_gp_compare(uint64_t nnz, uint64_t*parents, uint64_t* grandparents, bool* stars){

  int tid = threadIdx.x +  blockIdx.x * blockDim.x;

  if (tid >= nnz) return;

  uint64_t gp = grandparents[tid];
  uint64_t parent = parents[tid];

  if (gp == parent){
    stars[tid] = false;
  }


}

__global__ void star_parent(uint64_t nnz, uint64_t*parents, bool* stars){

  int tid = threadIdx.x +  blockIdx.x * blockDim.x;

  if (tid >= nnz) return;

  uint64_t parent = parents[tid];

  stars[tid] = stars[parent];


}

//update stars based on AS starcheck
void star_check(uint64_t nnz, uint64_t * parents, bool *stars){

  uint64_t blocknums = (nnz -1)/1024 + 1;

  //first, build grandparents and reset star
  reset_star<<<blocknums, 1024>>>(nnz, stars);
  uint64_t * grandparents = build_grandparents(nnz, parents);

  //next step
  //if gp[v] != p[v]
  //star[v] and star[gp[v]] = false;
  star_gp_compare<<<blocknums, 1024>>>(nnz, parents, grandparents, stars);

  //inherit parent's condition
  cudaFree(grandparents);
  star_parent<<<blocknums, 1024>>>(nnz, parents, stars);



}

//update stars based on AS starcheck
//simpler version
void parent_star_check(uint64_t nnz, uint64_t * parents, bool *stars){

  uint64_t blocknums = (nnz -1)/1024 + 1;

  //first, build grandparents and reset star
  reset_star<<<blocknums, 1024>>>(nnz, stars);
  uint64_t * grandparents = build_grandparents(nnz, parents);

  //next step
  //if gp[v] != p[v]
  //star[v] and star[gp[v]] = false;
  parent_star_gp_compare<<<blocknums, 1024>>>(nnz, parents, grandparents, stars);

  //inherit parent's condition
  cudaFree(grandparents);



}

__global__ void count_contigs(uint64_t nnz, uint64_t * parents){

  uint64_t tid = threadIdx.x +  blockIdx.x * blockDim.x;

  if (tid >= nnz) return;

  if (tid != 2832) return;

  //iterate through parents
  uint64_t count = 0;

  for (int i = 0;i <nnz; i++){
    if (parents[i] == tid) count++;
  }

  if (count != 0){
    printf("Component %llu: %lli\n", tid, count);
  }

}



uint64_t * old_cc(uint64_t nnz, uint64_t* Arows, uint64_t* Acols, char* Avals){


  uint64_t blocknums = (nnz -1)/1024 + 1;

  uint64_t * parents;

  cudaMalloc((void **)&parents,nnz*sizeof(uint64_t));

  uint64_t * old_parents;

  cudaMalloc((void **)&old_parents,nnz*sizeof(uint64_t));

  init_parent<<<blocknums, 1024>>>(nnz, parents);

  //copy over to check
  uint_copy_kernel<<<blocknums, 1024>>>(old_parents, parents, nnz);

  //init stars
  bool * stars;

  cudaMalloc((void ** )&stars, nnz*sizeof(bool));

  reset_star<<<blocknums, 1024>>>(nnz, stars);

  uint64_t count;

  uint64_t * grandparents;

  uint64_t iters = 0;

  //start with conditional hook
  //this encodes the connections between zz

  do  {

    printf("Before\n");
    printCudaVec(nnz, parents);
    printf("old\n");
    printCudaVec(nnz, old_parents);

    //main code
    naive_cond_hook<<<blocknums, 1024>>>(nnz, Arows, Acols, Avals, parents, stars);


    star_check(nnz, parents, stars);

    naive_uncond_hook<<<blocknums, 1024>>>(nnz, Arows, Acols, Avals, parents, stars);

    grandparents = build_grandparents(nnz, parents);

    shortcutting<<<blocknums, 1024>>>(nnz,parents,grandparents, stars);

    printf("after\n");
    printCudaVec(nnz, parents);
    printf("old\n");
    printCudaVec(nnz, old_parents);

    reduce_parents<<<blocknums, 1024>>>(parents, old_parents, nnz);

    //copy to uint64_t before deletion.
    cudaMemcpy(&count, old_parents, sizeof(uint64_t), cudaMemcpyDeviceToHost);

    //and reset parents
    uint_copy_kernel<<<blocknums, 1024>>>(old_parents, parents, nnz);
    cudaFree(grandparents);

    //printf("Counts\n");
    //count_contigs<<<blocknums, 1024>>>(nnz, parents);
    printf("Done with iteration %llu: %llu \n", iters, count);
    iters++;

  } while (count != 0);

  printf("Converged\n");

  //parents are converged
  cudaFree(old_parents);
  return parents;

}


void cc(uint64_t nnz, uint64_t num_vert, uint64_t* Arows, uint64_t* Acols, char* Avals, std::vector<uint64_t> outputRows, uint64_t maxOut, char*kmerVals, uint64_t*kmerLens, uint64_t * kmerParents){


  uint64_t blocknums = (nnz -1)/1024 + 1;
  uint64_t block_vert = (num_vert -1)/1024 + 1;


  //for each vertex init with parent
  char * contigs;
  uint64_t * contig_lens;


  cudaMallocManaged((void **)&contigs,num_vert*MAX_VEC*sizeof(char));

  cudaMallocManaged((void **)&contig_lens,num_vert*sizeof(uint64_t));

  //expose some extra memory so we don't get weird overwrite bugs
  char * contigs_holder;
  uint64_t * contig_lens_holder;

  cudaMallocManaged((void **)&contigs_holder,num_vert*MAX_VEC*sizeof(char));

  cudaMallocManaged((void **)&contig_lens_holder,num_vert*sizeof(uint64_t));



  //init both
  printf("If failure, Below this.\n");
  init_contig_lens<<<block_vert, 1024>>>(num_vert, contig_lens);
  assert_contig_lens<<<block_vert, 1024>>>(num_vert, contig_lens);
  cudaDeviceSynchronize();
  printf("Things worked out\n");
  init_contigs<<<blocknums, 1024>>>(nnz,num_vert, Arows, Acols, Avals, contigs, contig_lens);



  uint64_t * parents;

  cudaMalloc((void **)&parents,num_vert*sizeof(uint64_t));

  uint64_t * old_parents;

  cudaMalloc((void **)&old_parents,num_vert*sizeof(uint64_t));

  init_parent<<<block_vert, 1024>>>(num_vert, parents);

  //copy over to check
  uint_copy_kernel<<<block_vert, 1024>>>(old_parents, parents, num_vert);

  //init stars
  bool * stars;

  cudaMalloc((void ** )&stars, num_vert*sizeof(bool));

  reset_star<<<block_vert, 1024>>>(num_vert, stars);


  uint64_t * grandparents;

  uint64_t iters = 0;


  // printf("stars\n");
  // printCudaStars(num_vert, stars);

  //start with conditional hook
  //this encodes the connections between vertices
  naive_uncond_hook<<<blocknums, 1024>>>(nnz, Arows, Acols, Avals, parents, stars);


  //after unconditional hook, we need to add back in the starts so that the final items aren't one kmer short
  update_leads<<<blocknums, 1024>>>(nnz, contigs, contig_lens, maxOut, kmerVals, kmerLens, kmerParents);

  //printf("Before\n");
  //printCudaVec(num_vert, parents);

  //parent_star_check(num_vert, parents, stars);
  bool converged= false;


  do  {

    // printf("Before\n");
    // printCudaVec(num_vert, parents);
    // printf("stars\n");
    // printCudaStars(num_vert, stars);

    //main code
    grandparents = build_grandparents(num_vert, parents);
    parent_cond_hook<<<block_vert, 1024>>>(num_vert, parents, grandparents, stars, contigs, contig_lens, contigs_holder, contig_lens_holder);
    //naive_uncond_hook<<<blocknums, 1024>>>(nnz, Arows, Acols, Avals, parents, stars);

    //update contigs
    char * temp = contigs;
    uint64_t * temp_lens = contig_lens;
    contigs = contigs_holder;
    contig_lens = contig_lens_holder;

    contigs_holder = temp;
    contig_lens_holder = temp_lens;

    parent_star_check(num_vert, parents, stars);

    //naive_uncond_hook<<<blocknums, 1024>>>(nnz, Arows, Acols, Avals, parents, stars);


    printLenskern(outputRows, contig_lens);
    //shortcutting<<<blocknums, 1024>>>(nnz,parents,grandparents, stars);

    //printf("after\n");
    //printCudaVec(num_vert, parents);
    // printf("stars\n");
    // printCudaStars(num_vert, stars);

    //this is buggy
    //appears to not be consistent
    //reduce_parents<<<block_vert, 1024>>>(parents, old_parents, num_vert);

    //copy to uint64_t before deletion.
    //cudaMemcpy(&count, old_parents, sizeof(uint64_t), cudaMemcpyDeviceToHost);
    converged = starConverged(num_vert, stars);

    //and reset parents
    uint_copy_kernel<<<block_vert, 1024>>>(old_parents, parents, num_vert);
    cudaFree(grandparents);

    //printf("Counts\n");
    //count_contigs<<<blocknums, 1024>>>(nnz, parents);
    //printf("Done with iteration %llu: %llu %llu \n", iters, count, num_vert>>(iters-3));
    printf("Done with iter %llu\n", iters);
    iters++;

  } while (!converged);

  printf("Converged\n");
  //if we've really converged we need to syncronize so that the lens are guaranteed
  cudaDeviceSynchronize();


  //when done, print outrows
  for (int i =0; i < outputRows.size(); i++){
    printrowkern(outputRows.at(i), contigs, contig_lens);
  }

  //time to write to output
  std::ofstream fout;
  fout.open("output.dat");

  for (int i = 0; i < outputRows.size(); i++){

    uint64_t row = outputRows.at(i);
    cout << "len: " << contig_lens[row] << endl;
    for (uint64_t j = 0; j < contig_lens[row]; j++){
      fout << contigs[i*MAX_VEC+j];
    }
    fout << endl;

  }
  fout.close();

  //parents are converged
  cudaFree(old_parents);


}
