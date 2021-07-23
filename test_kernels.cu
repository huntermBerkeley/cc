
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
#include <chrono>
//Init cuda here
#include <thrust/device_vector.h>
#include <thrust/host_vector.h>
#include <thrust/transform.h>

//includes needed for 
#include <map>
#include "read_kmers.hpp"
#include "kmer_t.hpp"
#include "cudaCounter.hpp"
#include "noLock_cudaHashMap.hpp"

#ifndef MAX_VEC
#define MAX_VEC 8000
#endif

using namespace std;

//struct for transform
struct kmer_to_pkmer : public thrust::unary_function<kmer_pair,pkmer_t>
{
  __host__ 
  pkmer_t operator()(kmer_pair x) { return x.next_kmer(); }
};

struct kmer_to_start : public thrust::unary_function<kmer_pair,pkmer_t>
{
  __host__ 
  pkmer_t operator()(kmer_pair x) { return x.last_kmer(); }
};



#ifndef CUDA_CHECK
#define CUDA_CHECK(ans)                                                                  \
        gpuAssert((ans), __FILE__, __LINE__);
inline void gpuAssert(cudaError_t code, const char* file, int line, bool abort = true)
{
	if (code != cudaSuccess)
	{
		printf("GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
		if (abort)
			exit(code);
}
}
#endif

typedef unsigned long long int uint64_cu;


//copied host code for handling kmers


//building off of 'perf' adj matrix
//this constructs the matrix from first available
//TODO: parallelize: this should be embarassingly parallel with just a tad of locking
__host__ std::vector<std::pair<kmer_pair, uint64_t>> build_adj_mat(std::vector<kmer_pair> kmers, uint64_t * nnz, char ** vals, uint64_t ** rows, uint64_t ** cols){




  //main kmer counter - at the end of all of this it should equal kmers.size;
  uint64_t counter = 0;

  //the map we'll need: kmer->uint64_t
  std::map<std::string, uint64_t> kmer_to_num;

  uint64_t _nnz = 0;

  //start by adding all kmers to map from pkmer_t - kmer_pair
  //no this is not very efficient O(log n) vs O(1), but this is for debugging only
  //The original issue I suspect stems from something funky with a hash of a hash,
  //so I would like to avoid that as much as possible.
  for (uint64_t i = 0; i < kmers.size(); i++){
    //printf("current: %s\n", kmers[i].kmer.get().c_str());
    //can this be extended into the main loop?
    if (kmers[i].forwardExt() != 'F'){
      //valid edge!
      _nnz +=1;
    }
  }

  printf("Forward pass done with nnz %llu\n", _nnz);

  std::vector<std::pair<kmer_pair, uint64_t>> starts;

  //pass back - we want this for the next step
  *nnz = _nnz;

  //now allocate mat
  char * _vals = new char[_nnz];
  uint64_t * _rows = new uint64_t[_nnz];
  uint64_t * _cols = new uint64_t[_nnz];
  uint64_t slot = 0;

  //iterate through all kmers to find starts
  for (uint64_t i = 0; i < kmers.size(); i++){

    kmer_pair next_kmer = kmers[i];

    //continue on next instruction
    //all instructions valid
    if (next_kmer.forwardExt() == 'F') continue;



    //start file write!


    //start of the main loop
    uint64_t my_val;
    uint64_t next_val;

    //set value of main kmer if DNE
    if (kmer_to_num.count(next_kmer.kmer.get()) == 0){
        kmer_to_num[next_kmer.kmer.get()] = counter;
        counter++;
    }

    //use value stored in map
    my_val = kmer_to_num[next_kmer.kmer.get()];

    //set value of next kmer if DNE
    if (kmer_to_num.count(next_kmer.next_kmer().get()) == 0){
        kmer_to_num[next_kmer.next_kmer().get()] = counter;
        counter++;
    }
    //spooky bug part not needed, not spooky

    next_val = kmer_to_num[next_kmer.next_kmer().get()];

    //mats are allocated, insert into next available slot
    assert (slot < _nnz);

    _vals[slot] = next_kmer.forwardExt();
    _rows[slot] = my_val;
    _cols[slot] = next_val;

    //check if start
    if (next_kmer.backwardExt() == 'F'){
      //save to starts
      //this info is a tad redundant, maybe reduce to just string
      starts.push_back(std::make_pair(next_kmer, slot));
    }

    slot++;

    //update kmer to the next one in the chain
    //next_kmer = kmer_to_kmer[next_kmer.next_kmer().get()];





  }

  printf("Regular construct done, counter is %llu\n", counter);
  fflush(stdout);


  //close sample solution

  //set output
  *vals = _vals;
  *rows = _rows;
  *cols = _cols;

  return starts;

}

//take the starts from a perf construction and insert them into cuda mats
__host__ void prep_starts(std::vector<std::pair<kmer_pair, uint64_t>> starts, uint64_t * rows, uint64_t * startsNnz, char ** startVals, uint64_t** startLens, uint64_t ** startRows){


  uint64_t _startsNnz = starts.size();

  char * _startVals;
  uint64_t * _startLens;
  uint64_t * _startRows;

  //cudaMallocManaged for debugging, replace with cudaMemcpy for speed later
  cudaMallocManaged((void **)&_startVals, _startsNnz*MAX_VEC*sizeof(char));

  cudaMallocManaged((void **)&_startLens, _startsNnz*sizeof(uint64_t));

  cudaMallocManaged((void **)&_startRows, _startsNnz*sizeof(uint64_t));


  //iterate through outKmers

  for(int i=0; i < starts.size(); i++){

    //can never be too safe
    assert (i < _startsNnz);

    uint64_t slot = std::get<1>(starts.at(i));
    kmer_pair contig_start = std::get<0>(starts.at(i));

    pkmer_t kmer = contig_start.kmer;
    for(int j=0; j < kmer.get().size(); j++){

      //index into the results
      _startVals[i*MAX_VEC+j] = kmer.get()[j];



    }
    _startLens[i] = kmer.get().size();

    //set parent via cond hook
    _startRows[i] = rows[slot];


  }


  //done iterating, output to vecs
  *startsNnz = _startsNnz;
  *startVals = _startVals;
  *startLens = _startLens;
  *startRows = _startRows;


}

__host__ void copy_to_cuda(uint64_t nnz, char * originalVals, uint64_t * originalRows, uint64_t* originalCols, char** newVals, uint64_t ** newRows, uint64_t ** newCols){

  //malloc space for cuda Arrays
  char * _newVals;
  uint64_t * _newRows;
  uint64_t * _newCols;

  cudaMalloc((void ** )&_newVals, nnz*sizeof(char));
  cudaMalloc((void ** )&_newRows, nnz*sizeof(uint64_t));
  cudaMalloc((void ** )&_newCols, nnz*sizeof(uint64_t));



  //memcopys
  cudaMemcpy(_newVals, originalVals, nnz*sizeof(char), cudaMemcpyHostToDevice);

  cudaMemcpy(_newRows, originalRows, nnz*sizeof(uint64_t), cudaMemcpyHostToDevice);
  cudaMemcpy(_newCols, originalCols, nnz*sizeof(uint64_t), cudaMemcpyHostToDevice);

  *newVals = _newVals;
  *newRows = _newRows;
  *newCols = _newCols;
  //throw in a syncronize just in case, this one is probably clear to remove but you never know
  cudaDeviceSynchronize();
}

//generate a list of the vector ids to point to for output
__host__ std::vector<uint64_t> gen_outRows(std::vector<std::pair<kmer_pair, uint64_t>> starts, uint64_t * rows){

  std::vector<uint64_t> outRows;

  for (int i = 0; i < starts.size(); i++){

    uint64_t slot = std::get<1>(starts.at(i));

    outRows.push_back(rows[slot]);
  }

  return outRows;


}


//have everyone attempt to insert into the hashmap
__global__ void insert_all_onethread(uint64_t nnz, kmer_pair* kmers, cudaHashMap * map){

  uint64_t tid = threadIdx.x +  blockIdx.x*blockDim.x;

  if (tid != 0) return;


  for (uint64_t i =0; i < nnz; i++){

    kmer_pair my_kmer = kmers[i];

    map->insert(my_kmer.kmer, i);

  }
  


}

__global__ void insert_all(uint64_t nnz, kmer_pair* kmers, cudaHashMap * map){

  uint64_t tid = threadIdx.x +  blockIdx.x*blockDim.x;

  if (tid >= nnz) return;

  kmer_pair my_kmer = kmers[tid];

  map->insert(my_kmer.kmer, tid);



}

__global__ void assertInserts(uint64_t nnz, kmer_pair* kmers, cudaHashMap * map){


  uint64_t tid = threadIdx.x +  blockIdx.x*blockDim.x;

  if (tid >= nnz) return;

  kmer_pair my_kmer = kmers[tid];

  
  uint64_t val = map->get(my_kmer.kmer);

  if (val == map->size+1){

    printf("Kmer %llu failed to retreive, had val: %llu\n", tid, val);

    uint64_t val2 = map->get(my_kmer.kmer);

    printf("Kmer %llu val 2: %llu\n", tid, val2);


  }


}

__global__ void assertInserts_onethread(uint64_t nnz, kmer_pair* kmers, cudaHashMap * map){


  uint64_t tid = threadIdx.x +  blockIdx.x*blockDim.x;

  if (tid !=0) return;

  for (uint64_t i = 0; i < nnz; i++){


  kmer_pair my_kmer = kmers[i];

  
  uint64_t val = map->get(my_kmer.kmer);

  if (val == map->size+1){

    printf("Kmer %llu failed to retreive, had val: %llu\n", i, val);

    uint64_t val2 = map->get(my_kmer.kmer);

    printf("Kmer %llu val 2: %llu", i, val2);


  }

}


}

//end of copied code for handlers

__device__ uint16_t get_lock_nowait(uint32_t * locks, int index) {
  //set lock to 1 to claim
  //returns 0 if success
  uint32_t zero = 0;
  uint32_t one = 1;
  return atomicCAS(&locks[index], zero, one);
}

__device__ void get_lock(uint32_t * locks, int index) { 
  
  uint16_t result = 1;

  do {
    result = get_lock_nowait(locks, index);
  } while (result !=0);

}

__device__ void free_lock(uint32_t * locks, int index) {

  //set lock to 0 to release
  uint32_t zero = 0;
  uint32_t one = 1;
  //TODO: might need a __threadfence();
  atomicCAS(&locks[index], one, zero);

}

__global__ void counterComp(cudaCounter * counter, uint64_t nnz, uint64_t * counter_holder, uint32_t * locks){

  uint64_t tid = threadIdx.x +  blockIdx.x*blockDim.x;

  if (tid >= nnz) return;

  uint64_t my_val = counter->get(tid);

  if (my_val > nnz) return;


  assert(my_val < nnz+1000);

  while (true){

    uint16_t result = get_lock_nowait(locks, my_val);

    if (result ==0){

      if (counter_holder[my_val] == 0){

        counter_holder[my_val] = 1;

      } else {

        printf("Thread %llu with val %llu received a counter to a filled index\n", tid, my_val);

      }

    free_lock(locks, my_val);

    return;

    }

  }

  

}

__global__ void counterCheck(cudaCounter * counter){

  uint64_t tid = threadIdx.x +  blockIdx.x*blockDim.x;

  uint64_t my_val = counter->get(tid);

  //printf("%llu got val %llu\n", tid, my_val);

}

__global__ void counterAtomicCheck(uint64_t * counter){


  //uint64_t tid = threadIdx.x +  blockIdx.x*blockDim.x;


  atomicAdd((uint64_cu *) counter, 1);

}

//test cases for counter

//how bad is a regular atomic?
__host__ void testAtomic(){

  //uint64_t nnz = 100000;

  uint64_t * counter;

  cudaMalloc((void **)&counter, sizeof(uint64_t));

  counterAtomicCheck<<<10000000,100>>>(counter);
}

__host__ void testCounter(){

  uint64_t nnz = 100000;

  cudaCounter * counter;

  initCounter(&counter);

  uint64_t * counter_holder;
  uint32_t * locks;

  CUDA_CHECK(cudaMalloc((void**)&counter_holder, (nnz+1000)*sizeof(uint64_t)));
  CUDA_CHECK(cudaMalloc((void**)&locks, (nnz+1000)*sizeof(uint32_t)));

  CUDA_CHECK(cudaMemset(counter_holder, 0, (nnz+1000)*sizeof(uint64_t)));
  CUDA_CHECK(cudaMemset(locks, 0, (nnz+1000)*sizeof(uint32_t)));

  counterComp<<<nnz, 10>>>(counter, nnz, counter_holder, locks);
  cudaDeviceSynchronize();

  CUDA_CHECK(cudaFree(counter_holder));
  CUDA_CHECK(cudaFree(locks));
  freeCounter(counter);

}

//ask for some numbers from the counter
__host__ void testCounterNoCheck(){


  cudaCounter * counter;
  initCounter(&counter);

  counterCheck<<<10000000,100>>>(counter);

  freeCounter(counter);

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


//initialize every thread to be it's own parent
__global__ void init_parent(uint64_t nnz,  uint64_t* parent){

  uint64_t tid = threadIdx.x +  blockIdx.x*blockDim.x;

  if (tid < nnz){
    parent[tid] = tid;
  }

}

//the counter variable is significantly less than it should be - why?
__global__ void set_parents_from_hashmap(uint64_t nnz, kmer_pair* kmers, pkmer_t * pkmers, uint64_t * parents, char * extensions, cudaHashMap * map, uint64_t * count){

  uint64_t tid = threadIdx.x +  blockIdx.x*blockDim.x;

  if (tid >= nnz) return;

  //query myself and my parents
  uint64_t my_slot = map->get(kmers[tid].kmer);

  //this would be a bug
  assert(my_slot != map->size+1);

  //this must fail at least once - otherwise how do you know the kmer ended?
  uint64_t my_parent = map->get(pkmers[tid]);
  //uint64_t my_parent = map->get(kmers[tid].next_kmer());

  if(my_parent != map->size+1){



    parents[my_slot] = my_parent;

    extensions[my_slot] = kmers[tid].forwardExt();

    if(extensions[my_slot] == 'F'){

      printf("Tid failed: kmer %llu with forwardExt %c claims to have parent at %llu\n", my_slot, kmers[tid].forwardExt(), my_parent);
      map->get(pkmers[tid]);

    }

   

    return;

  } else {

    assert(kmers[tid].forwardExt() == 'F');
    atomicAdd((long long unsigned int *) count, (long long unsigned int ) 1);

  }


}

__global__ void print_count(uint64_t * count){


  uint64_t tid = threadIdx.x +  blockIdx.x*blockDim.x;

  if (tid == 0) printf("Num starts: %llu\n", *count);

}

__host__ uint64_t prep_parents(uint64_t nnz, kmer_pair* kmers, pkmer_t * pkmers, uint64_t * parents, char * extensions, cudaHashMap * map){


uint64_t blocksize = 1024;
uint64_t num_blocks = (nnz-1)/blocksize+1;

uint64_t * count;
//this is slow but pretty convenient
cudaMallocManaged((void **)&count, sizeof(uint64_t));

count[0] = 0;

//initialize every thread to be it's own parent
init_parent<<<num_blocks, blocksize>>>(nnz, parents);


set_parents_from_hashmap<<<num_blocks, blocksize>>>(nnz, kmers, pkmers, parents, extensions, map, count);

cudaDeviceSynchronize();

print_count<<<1,1>>>(count);

cudaDeviceSynchronize();

uint64_t to_return = *count;

cudaFree(count);

return to_return;

}


__global__ void find_starts_kernel(uint64_t nnz, pkmer_t * pkmers, uint64_t * starts, uint64_t * counter, cudaHashMap* hashMap){

  uint64_t tid = threadIdx.x +  blockIdx.x*blockDim.x;

  if (tid >= nnz) return;

  if (hashMap->get(pkmers[tid]) == hashMap->size+1){

    uint64_t my_index = atomicAdd((unsigned long long int  *) counter, (long long unsigned int) 1);

    starts[my_index] = tid;

  }


}

__host__ void find_starts_cuda(uint64_t nnz, pkmer_t * pkmers, uint64_t startNnz, uint64_t ** startIds, cudaHashMap * map){


  uint64_t * counter;
  uint64_t * starts;

  cudaMallocManaged((void **)&counter, sizeof(uint64_t));
  cudaMallocManaged((void **)&starts, sizeof(uint64_t)*startNnz);


  counter[0] = 0;
  uint64_t blocksize = 1024;
  uint64_t num_blocks = (nnz-1)/blocksize+1;

  find_starts_kernel<<<num_blocks, blocksize>>>(nnz, pkmers, starts, counter, map);


  print_count<<<1,1>>>(counter);

  cudaFree(counter);

  *startIds = starts;

}

//discern parents from pkmers


// __global__ void naive_cond_hook(uint64_t nnz, uint64_t * Arows, uint64_t * Acols, char * Avals, uint64_t * parent, bool * star){

//   uint64_t tid = threadIdx.x +  blockIdx.x*blockDim.x;

//   if (tid >= nnz) return;

//   uint64_t u = Arows[tid];
//   uint64_t v = Acols[tid];

//   uint64_t parent_u = parent[u];
//   uint64_t parent_v = parent[v];


//   //retreive f earlier
//   uint64_t gparent_u = parent[parent[u]];
//   uint64_t old;

//   //star hook procedure
//   if (star[u] && parent[u] > parent[v]){
//     old = (uint64_t) atomicCAS( (uint64_cu *) parent+parent_u, (uint64_cu) gparent_u, (uint64_cu) parent_v);
//     //if this is the case we must have succeeded
//     if (old == gparent_u){
//       return;
//     }
//     parent_v = parent[v];
//     parent_u = parent[u];
//     gparent_u = parent[parent_u];
//   }


// }

__global__ void parent_cond_hook_no_branch(uint64_t nnz, uint64_t * parent, uint64_t * parent_holder, uint64_t * gparent, bool * star, char* contigs, uint64_t * contig_lens, char* contigs_holder, uint64_t * contig_lens_holder){

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
    parent_holder[tid] = gparent_u;


    //having a branch here is really bad and should  not happen
  }


}

//following
__global__ void map_contigs(uint64_t maxOut, char * startVals, uint64_t * startLens, uint64_t * startParents, uint64_t * contig_index, uint64_t * parents, uint64_t * contig_map, uint64_t * contig_map_lens){

  uint64_t tid = threadIdx.x +  blockIdx.x*blockDim.x;

  if (tid >= maxOut) return;

  //now find correct length and copy over into buffer
  uint64_t my_parent = startParents[tid];
  uint64_t my_contig = parents[my_parent];

  contig_map[tid] = my_contig;
  contig_map_lens[tid] = contig_index[my_parent] + startLens[tid];


}


//output the mappings
__global__ void print_mappings(uint64_t maxOut, uint64_t * contig_map, uint64_t * contig_map_lens){

  uint64_t tid = threadIdx.x +  blockIdx.x*blockDim.x;

  if (tid != 0) return;

  for (uint64_t i =0; i < maxOut; i++){

    printf("Contig %llu -> %llu, len %llu\n", i, contig_map[i], contig_map_lens[i]);
  }

}

__host__ void printBuftype(std::string bufname, void* buffer){

  cudaPointerAttributes bufStats;
  CUDA_CHECK(cudaPointerGetAttributes(&bufStats, buffer));

  cout << "buffer " << bufname << " of type ";

  if (bufStats.type == cudaMemoryTypeUnregistered) cout << "unregistered";
  else if (bufStats.type == cudaMemoryTypeHost) cout << "Host";
  else if (bufStats.type == cudaMemoryTypeDevice) cout << "Device";
  else if (bufStats.type == cudaMemoryTypeManaged) cout << "Managed";
  else cout << "failure on type somehow";

  cout << endl;

}

//this is bad - google cuda heap allocation vs cudaMalloc to learn more
//basically this creates memory that is unusable by host
__global__ void mallocContigs(uint64_t maxOut, char ** final_contigs, uint64_t * contig_map_lens){

  uint64_t tid = threadIdx.x +  blockIdx.x*blockDim.x;

  if (tid >= maxOut) return;

  char * temp_contig;

  cudaMalloc((void **)&temp_contig,contig_map_lens[tid]*sizeof(char));
  //allocate device_side memory


  final_contigs[tid] = temp_contig;


}


//this should work better - create host version and malloc over
__host__ void mallocHostContigs(uint64_t maxOut, char ** final_contigs, uint64_t * contig_map_lens){

  char ** host_final_contigs;
  cudaMallocHost((void **)&host_final_contigs,maxOut*sizeof(char * ));

  uint64_t * _lens; // = (uint64_t *) malloc(maxOut*sizeof(uint64_t));
  CUDA_CHECK(cudaMallocHost((void **)&_lens,maxOut*sizeof(uint64_t)));

  CUDA_CHECK(cudaMemcpy(_lens, contig_map_lens, maxOut*sizeof(uint64_t), cudaMemcpyDeviceToHost));


  for (uint64_t tid =0; tid < maxOut; tid++){


    char * temp_contig;

    CUDA_CHECK(cudaMalloc((void **)&temp_contig,_lens[tid]*sizeof(char)));
    CUDA_CHECK(cudaMemset(temp_contig, 'F', _lens[tid]*sizeof(char)));
    //allocate device_side memory


    host_final_contigs[tid] = temp_contig;

  }

  CUDA_CHECK(cudaMemcpy(final_contigs, host_final_contigs, maxOut*sizeof(char *), cudaMemcpyHostToDevice));


  CUDA_CHECK(cudaFreeHost(host_final_contigs));
}


__host__ void freeHostContigs (uint64_t maxOut, char ** final_contigs){

  char ** host_final_contigs;
  cudaMallocHost((void **)&host_final_contigs,maxOut*sizeof(char * ));



  CUDA_CHECK(cudaMemcpy(host_final_contigs, final_contigs, maxOut*sizeof(char*), cudaMemcpyDeviceToHost));

  for (uint64_t i =0; i < maxOut; i++){
    CUDA_CHECK(cudaFree(host_final_contigs[i]));
  }

  cudaDeviceSynchronize();

  CUDA_CHECK(cudaFreeHost(host_final_contigs));

}

__global__ void freeContigs(uint64_t maxOut, char ** final_contigs){

  uint64_t tid = threadIdx.x +  blockIdx.x*blockDim.x;

  if (tid >= maxOut) return;

  cudaFree(final_contigs[tid]);


}

__global__ void fill_contigs_starts(uint64_t maxOut, char ** final_contigs, char * startVals, uint64_t* startLens){

  uint64_t tid = threadIdx.x +  blockIdx.x*blockDim.x;

  if (tid >= maxOut) return;

  for (uint64_t i=0; i < startLens[tid]; i++){

    final_contigs[tid][i] = startVals[tid*MAX_VEC+i];


  }

}


//fill all contigs
//each vertex finds its contig
// then its insert position into that contig
// and then writes in parallel
//idea! use 2d blocking
//x is vertex num -- if this is not fast enough do vertex tidy = block.x to have all threads access same value
//y is maxOut num - try all array indices in parallel, kill all threads that don't continue
__global__ void fill_contigs(uint64_t num_verts, char ** final_contigs, uint64_t maxOut, uint64_t * contig_map, uint64_t * contig_map_lens, char * contigs, uint64_t * contig_index, uint64_t * contig_lens, uint64_t * parents){

  uint64_t tid = threadIdx.x +  blockIdx.x*blockDim.x;

  uint64_t tidy = threadIdx.y + blockIdx.y*blockDim.y;

  //threads are launching, just not inputting correctly


  if (tid >= num_verts) return;
  if (tidy >= maxOut) return;


  uint64_t my_contig = parents[tid];

  uint64_t my_index_contig = contig_map[tidy];

  //printf("Thread %llu, %llu launching, comp %llu, %llu with len %llu\n", tid, tidy, my_contig, my_index_contig, contig_lens[tid]);


  if (my_contig != my_index_contig) return;

  if (contig_lens[tid] == 0){
    return;
  }



  //printf("Vertex %llu lines up with contig %llu\n", tid, tidy);

  //tidy is now the contig to access
  uint64_t my_start = contig_map_lens[tidy] - contig_index[tid];

  //correct index is 593
  final_contigs[tidy][my_start] = contigs[tid];

}

__global__ void move_contigs_to_host(uint64_t maxOut, char** final_contigs, char** host_final_contigs, uint64_t * contig_lens){

  uint64_t tid = threadIdx.x +  blockIdx.x*blockDim.x;

  if (tid >= maxOut) return;

  //find my id, and memcpy async
  cudaMemcpyAsync(host_final_contigs[tid], final_contigs[tid], contig_lens[tid]*sizeof(char), cudaMemcpyDeviceToHost);


}

__global__ void check_host_contig(char * contig, uint64_t len){

  uint64_t tid = threadIdx.x +  blockIdx.x*blockDim.x;

  if (tid != 0) return;

  printf("Dev contig from host pointer:\n");
  for (uint64_t i = 0; i < len; i++){
    printf("%c", contig[i]);
  }
  printf("\n");

}

//copy over contigs to host, then save to a file
__host__ void save_contigs(std::string filename, uint64_t maxOut, char ** final_contigs, uint64_t * final_lens){

  //the first step is to copy over the lengths, so we know how large of a buffer to allocate

  uint64_t * _lens; // = (uint64_t *) malloc(maxOut*sizeof(uint64_t));
  CUDA_CHECK(cudaMallocHost((void **)&_lens,maxOut*sizeof(uint64_t)));

  char ** host_final_contigs; // = (char **) malloc(maxOut*sizeof(char *));
  CUDA_CHECK(cudaMallocHost((void **)&host_final_contigs,maxOut*sizeof(char * )));

  CUDA_CHECK(cudaMemcpy(_lens, final_lens, maxOut*sizeof(uint64_t), cudaMemcpyDeviceToHost));

  CUDA_CHECK(cudaMemcpy(host_final_contigs, final_contigs, maxOut*sizeof(char * ), cudaMemcpyDeviceToHost));
  cudaDeviceSynchronize();

  std::ofstream fout;
  fout.open(filename);

  // printBuftype("lens", _lens);
  // printBuftype("host_final_contigs", host_final_contigs);

  for (uint64_t i =0 ; i < maxOut; i++){

    char * buffer; //= (char * ) malloc(_lens[i]*sizeof(char));
    CUDA_CHECK(cudaMallocHost((void **)&buffer,_lens[i]*sizeof(char)));

    //buffer = host_final_contigs[i];

    // printBuftype("temp_buf", buffer);


    char * cudaBuf = host_final_contigs[i];

    // printBuftype("host_final_contigs[i]", host_final_contigs[i]);
    // printBuftype("cudaBuf", cudaBuf);

    //printf("Copy size vs copy: %llu vs %llu \n", _lens[i], _lens[i]*sizeof(char));

    CUDA_CHECK(cudaMemcpy(buffer, cudaBuf, _lens[i]*sizeof(char), cudaMemcpyDefault));


    //cudaDeviceSynchronize();
    //host final contig is still a cuda pointer
    //so have a thread read the device memory
    //check_host_contig<<<1,1>>>(host_final_contigs[i], _lens[i]);

    //cudaDeviceSynchronize();
    fflush(stdout);


    //file write
    //cout << "Contig:" << i << " with len " << _lens[i] <<  endl;
    for (uint64_t j=0; j < _lens[i]; j++){
      //cout << buffer[i];
      //printf("%c", buffer[j]);
      fout << buffer[j];
    }
    //printf("\n");
    fout << endl;

    fflush(stdout);
    //done with save
    CUDA_CHECK(cudaFreeHost(buffer));
  }

  CUDA_CHECK(cudaFreeHost(_lens));
  CUDA_CHECK(cudaFreeHost(host_final_contigs));

}

__global__ void check_contig(uint64_t contig_id, char ** final_contigs, uint64_t * lens){

  uint64_t tid = threadIdx.x +  blockIdx.x*blockDim.x;

  if (tid != 0) return;

  //printf("Looking at contig %llu with len %llu\n", contig_id, lens[contig_id]);

  for (uint64_t i=0; i <lens[contig_id]; i++){
    printf("%c", final_contigs[contig_id][i]);
  }
  printf("\n");
}

//fill the start of every contig based input starts
__global__ void parent_cond_hook(uint64_t nnz, uint64_t * parent, uint64_t * parent_holder, uint64_t * gparent, bool * star, char* contigs, uint64_t * contig_lens, char* contigs_holder, uint64_t * contig_lens_holder){

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
    parent_holder[tid] = gparent_u;


    //having a branch here is really bad and should  not happen
  } else {


    //TODO: Move this section to the star check
    //atm we are repeating work

    uint64_t my_contig_len = contig_lens[tid];
    char * my_contig = contigs + MAX_VEC*tid;
    char * my_output = contigs_holder + MAX_VEC*tid;

    //copy from me
    for (int i = 0; i < my_contig_len; i++){
      my_output[i] = my_contig[i];
    }

    contig_lens_holder[tid] = my_contig_len;

    parent_holder[tid] =  parent_u;

  }
  // if (parent[tid] =  parent[parent[tid]]){
  //   star[tid] = false;
  // }


}

__global__ void len_cond_hook(uint64_t nnz, uint64_t * parent, uint64_t * parent_holder, bool * star, uint64_t * contig_index, uint64_t * contig_index_holder){

  uint64_t tid = threadIdx.x +  blockIdx.x*blockDim.x;

  if (tid >= nnz) return;

  //for parent cond hook, if I am not a star, set my parent to my grandparent

  //if star[u]
  // parent[u] = parent[parent[u]]

  uint64_t parent_u = parent[tid];
  uint64_t gparent_u = parent[parent_u];
  uint64_t parent_len = 0;
  uint64_t my_contig_index = contig_index[tid];

  if (star[tid]){

    //absorb from your parent
    //first copy over your material

    //compress parents

    parent_len = contig_index[parent_u];

  }


    //copy to new len
    contig_index_holder[tid] = my_contig_index+parent_len;

    //and absorb
    parent_holder[tid] = gparent_u;

}

//perform initial setup
//given an adj matrix,
__global__ void simple_adj_hook(uint64_t nnz, uint64_t * Arows, uint64_t * Acols, char* Avals, uint64_t * parent, bool * stars, char* contigs, uint64_t * contig_lens){

  uint64_t tid = threadIdx.x +  blockIdx.x*blockDim.x;

  if (tid >= nnz) return;

  uint64_t my_index = Arows[tid];

  parent[my_index] = Acols[tid];

  //and init contigs
  char my_val = Avals[tid];
  contigs[my_index*MAX_VEC] = my_val;
  contig_lens[my_index] = 1;

}

__global__ void len_adj_hook(uint64_t nnz, uint64_t * Arows, uint64_t * Acols, char* Avals, uint64_t * parent, bool * stars, char* contigs, uint64_t * contig_lens){

  uint64_t tid = threadIdx.x +  blockIdx.x*blockDim.x;

  if (tid >= nnz) return;

  uint64_t my_index = Arows[tid];

  parent[my_index] = Acols[tid];

  //and init contigs
  char my_val = Avals[tid];
  contigs[my_index] = my_val;
  contig_lens[my_index] = 1;

}

//the sum of contig sections should be constant
__global__ void count_bases(uint64_t nnz, char * contigs){

  uint64_t tid = threadIdx.x +  blockIdx.x*blockDim.x;

  if (tid != 0) return;

  uint64_t counter = 0;
  for (uint64_t i =0; i < nnz; i++){
    if (contigs[i] == 'A' || contigs[i] == 'C' || contigs[i] == 'T' || contigs[i] == 'G'){
      counter +=1;
    }
  }
  printf("Conting counter: %llu\n", counter);

}

__global__ void sum_lens(uint64_t nnz, uint64_t * lens){

  uint64_t tid = threadIdx.x +  blockIdx.x*blockDim.x;

  if (tid != 0) return;

  uint64_t counter = 0;
  for (uint64_t i =0; i < nnz; i++){
    counter += lens[i];
  }
  printf("lens counter: %llu\n", counter);

}

//unconditional hook - this is frankly bizarre
//and im not sure how it's 'worked' so far
// __global__ void naive_uncond_hook(uint64_t nnz, uint64_t * Arows, uint64_t * Acols, char * Avals, uint64_t * parent, bool * star){

//   uint64_t tid = threadIdx.x +  blockIdx.x*blockDim.x;

//   if (tid >= nnz) return;

//   uint64_t u = Arows[tid];
//   uint64_t v = Acols[tid];

//   uint64_t parent_u = parent[u];
//   uint64_t parent_v = parent[v];

//   //retreive f earlier
//   uint64_t gparent_u = parent[parent[u]];
//   uint64_t old;

//   //star hook procedure
//   if (star[u] && parent[u] != parent[v]){
//     old = (uint64_t) atomicCAS( (uint64_cu *) parent+parent_u, (uint64_cu) gparent_u, (uint64_cu) parent_v);
//     //if this is the case we must have succeeded
//     if (old == gparent_u){
//       return;
//     }
//     parent_v = parent[v];
//     parent_u = parent[u];
//     gparent_u = parent[parent_u];
//   }


// }

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



// __device__ char semiring_multiply(char a, char b){
//
//   printf("Multiplying %c, %c\n", a,b);
//   if (a == 'z' ||  b == 'z')
//     return 'z';
//
//   if (a == 0x20) return b;
//
//   return a;
// }

// __device__ char semiring_add(char a, char b){
//
//   printf("adding %c, %c\n", a,b);
//   if (a == 0x20){
//     return b;
//   }
//   if (b == 0x20){
//     return a;
//   }
//   //both nonzero, bad path
//   //this will corrupt any future adds to this index as well
//   return 'z';
// }

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
  //0th index to

  //was a -1 on the left
  //lets split this into parts
  contigs[MAX_VEC*contig_index+contig_lens[contig_index]-1] = contigs[MAX_VEC*contig_index];


  //copy kernel moved from cc
  for (int i = 0; i < contig_lens[contig_index]-1; i++){

    contigs[MAX_VEC*contig_index+i] = updates[MAX_VEC*tid+i];

  }

  //finished>
  return;

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



// void  fill_wrapper(int nnz, int*vals, int*rows, int*cols){
//
//   int blocknums  = (nnz - 1)/ 1024 + 1;
//
//   fill_matrix<<<blocknums, 1024>>>(nnz, vals,rows,cols);
//
// }

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

  std::cout << "converged: " << result << "." << std::endl;
  cudaFree(converged);

  return result;

}

void  fill_vector(int nnz, int*vector){

  int blocknums  = (nnz - 1)/ 1024 + 1;

  vec_kernel<<<blocknums, 1024>>>(nnz, vector);

}





//build grandparents - needs to happen as independent kernel call
uint64_t * build_grandparents(uint64_t nnz, uint64_t * parents){

  uint64_t * grandparents;

  cudaMalloc((void **)&grandparents,nnz*sizeof(uint64_t));

  uint64_t blocknums = (nnz -1)/1024 + 1;

  setGrandparents<<<blocknums,1024>>>(nnz, parents, grandparents);

  return grandparents;


}



__global__ void parent_star_gp_compare(uint64_t nnz, uint64_t*parents, bool* stars){

  int tid = threadIdx.x +  blockIdx.x * blockDim.x;

  if (tid >= nnz) return;

  uint64_t parent = parents[tid];
  uint64_t gp = parents[parent];

  if (gp == parent){
    stars[tid] = false;
  }


}

__global__ void parent_star_gp_compare_one_update(uint64_t nnz, uint64_t*parents, bool* stars, char*contigs, uint64_t* contig_lens, char* contigs_holder, uint64_t * contig_lens_holder){

  int tid = threadIdx.x +  blockIdx.x * blockDim.x;

  if (tid >= nnz) return;
  //only look at stars that have not reset
  if (!stars[tid]) return;

  uint64_t parent = parents[tid];
  uint64_t gp = parents[parent];

  if (gp == parent){
    stars[tid] = false;

    //and update
    uint64_t my_contig_len = contig_lens[tid];
    char * my_contig = contigs + MAX_VEC*tid;
    char * my_output = contigs_holder + MAX_VEC*tid;

    //copy from me
    for (int i = 0; i < my_contig_len; i++){
      my_output[i] = my_contig[i];
    }

    contig_lens_holder[tid] = my_contig_len;

    //doesn't need to be set - gets propogated in a later copy
    //parent_holder[tid] =  parent_u;

  }


}







__global__ void star_parent(uint64_t nnz, uint64_t*parents, bool* stars){

  int tid = threadIdx.x +  blockIdx.x * blockDim.x;

  if (tid >= nnz) return;

  uint64_t parent = parents[tid];

  stars[tid] = stars[parent];


}



//update stars based on AS starcheck
//simpler version
//this is soo slow
void parent_star_check(uint64_t nnz, uint64_t * parents, bool *stars){

  uint64_t blocknums = (nnz -1)/1024 + 1;

  //first, build grandparents and reset star
  reset_star<<<blocknums, 1024>>>(nnz, stars);
  //uint64_t * grandparents = build_grandparents(nnz, parents);
  cudaDeviceSynchronize();
  //printf("Reset Star\n");
  //fflush(stdout);

  //next step
  //if gp[v] != p[v]
  //star[v] and star[gp[v]] = false;
  parent_star_gp_compare<<<blocknums, 1024>>>(nnz, parents, stars);

  cudaDeviceSynchronize();
  //printf("Set stars\n");
  //fflush(stdout);

  //inherit parent's condition
  //cudaFree(grandparents);



}

void parent_star_check_noreset(uint64_t nnz, uint64_t * parents, bool *stars, char*contig, uint64_t* contig_lens, char* contig_holder, uint64_t * contig_lens_holder){

  uint64_t blocknums = (nnz -1)/1024 + 1;

  //first, build grandparents and reset star


  //next step
  //if gp[v] != p[v]
  //star[v] and star[gp[v]] = false;
  parent_star_gp_compare_one_update<<<blocknums, 1024>>>(nnz, parents, stars, contig, contig_lens, contig_holder, contig_lens_holder);

  //inherit parent's condition
  //cudaFree(grandparents);



}


void cc(uint64_t nnz, uint64_t num_vert, uint64_t* Arows, uint64_t* Acols, char* Avals, std::vector<uint64_t> outputRows, uint64_t maxOut, char*kmerVals, uint64_t*kmerLens, uint64_t * kmerParents){


  uint64_t blocknums = (nnz -1)/1024 + 1;
  uint64_t block_vert = (num_vert -1)/1024 + 1;


  //for each vertex init with parent
  char * contigs;
  uint64_t * contig_lens;

  auto start = std::chrono::high_resolution_clock::now();


  cudaMallocManaged((void **)&contigs,num_vert*MAX_VEC*sizeof(char));


  cudaMallocManaged((void **)&contig_lens,num_vert*sizeof(uint64_t));


  //expose some extra memory so we don't get weird overwrite bugs
  char * contigs_holder;
  uint64_t * contig_lens_holder;

  cudaMallocManaged((void **)&contigs_holder,num_vert*MAX_VEC*sizeof(char));

  cudaMallocManaged((void **)&contig_lens_holder,num_vert*sizeof(uint64_t));



  //init both
  //printf("If failure, Below this.\n");
  init_contig_lens<<<block_vert, 1024>>>(num_vert, contig_lens);
  assert_contig_lens<<<block_vert, 1024>>>(num_vert, contig_lens);

  init_contig_lens<<<block_vert, 1024>>>(num_vert, contig_lens_holder);
  assert_contig_lens<<<block_vert, 1024>>>(num_vert, contig_lens_holder);
  cudaDeviceSynchronize();

  auto contig_setup = std::chrono::high_resolution_clock::now();

  std::chrono::duration<double> diff = contig_setup-start;

  std::cout << "Time required for contig setup: " << diff.count() << " s\n";
  //printf("Things worked out\n");
  //init_contigs<<<blocknums, 1024>>>(nnz,num_vert, Arows, Acols, Avals, contigs, contig_lens);



  uint64_t * parents;

  cudaMalloc((void **)&parents,num_vert*sizeof(uint64_t));

  uint64_t * parents_holder;

  cudaMalloc((void **)&parents_holder,num_vert*sizeof(uint64_t));

  init_parent<<<block_vert, 1024>>>(num_vert, parents);
  //init_parent<<<block_vert, 1024>>>(num_vert, parents_holder);

  //copy over to check
  uint_copy_kernel<<<block_vert, 1024>>>(parents_holder, parents, num_vert);

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
  //this isn't right :ADF:ASD
  //naive_uncond_hook<<<blocknums, 1024>>>(nnz, Arows, Acols, Avals, parents, stars);
  simple_adj_hook<<<blocknums, 1024>>>(nnz, Arows, Acols, Avals, parents, stars, contigs, contig_lens);



  //print statements - use with device syncronize
  cudaDeviceSynchronize();
  // for (int i =0; i < outputRows.size(); i++){
  //   printrowkern(outputRows.at(i), contigs, contig_lens);
  // }

  //after unconditional hook, we need to add back in the starts so that the final items aren't one kmer short
  update_leads<<<blocknums, 1024>>>(nnz, contigs, contig_lens, maxOut, kmerVals, kmerLens, kmerParents);


  cudaDeviceSynchronize();
  // for (int i =0; i < outputRows.size(); i++){
  //   printrowkern(outputRows.at(i), contigs, contig_lens);
  // }

  auto full_setup = std::chrono::high_resolution_clock::now();

  diff = full_setup-contig_setup;

  std::cout << "Time required for final setup: " << diff.count() << " s\n";


  //printf("Before\n");
  //printCudaVec(num_vert, parents);

  //parent_star_check(num_vert, parents, stars);
  bool converged= false;


  do  {

    auto iter_start = std::chrono::high_resolution_clock::now();

    // printf("Before\n");
    // printCudaVec(num_vert, parents);
    // printf("stars\n");
    // printCudaStars(num_vert, stars);

    //main code
    grandparents = build_grandparents(num_vert, parents);
    parent_cond_hook<<<block_vert, 1024>>>(num_vert, parents, parents_holder, grandparents, stars, contigs, contig_lens, contigs_holder, contig_lens_holder);
    //naive_uncond_hook<<<blocknums, 1024>>>(nnz, Arows, Acols, Avals, parents, stars);

    //update contigs
    char * temp = contigs;
    uint64_t * temp_lens = contig_lens;
    contigs = contigs_holder;
    contig_lens = contig_lens_holder;

    contigs_holder = temp;
    contig_lens_holder = temp_lens;

    cudaDeviceSynchronize();

    auto cond_hook = std::chrono::high_resolution_clock::now();

    diff = cond_hook-iter_start;

    std::cout << "Time required for cond hook: " << diff.count() << " s\n";


    //trade parents
    //doing it this way lets us reuse the memory efficiently
    uint64_t * temp_parents = parents;
    parents =  parents_holder;
    parents_holder  = temp_parents;

    printf("Entering star check\n");
    fflush(stdout);


    parent_star_check(num_vert, parents, stars);

    //parent_star_check_noreset(num_vert, parents, stars, contigs, contig_lens, contigs_holder, contig_lens_holder);

    cudaDeviceSynchronize();

    auto star_hook = std::chrono::high_resolution_clock::now();

    diff = star_hook - cond_hook;

    std::cout << "Time required for star check: " << diff.count() << " s\n";

    //naive_uncond_hook<<<blocknums, 1024>>>(nnz, Arows, Acols, Avals, parents, stars);


    //printLenskern(outputRows, contig_lens);
    //shortcutting<<<blocknums, 1024>>>(nnz,parents,grandparents, stars);

    //printf("after\n");
    //printCudaVec(num_vert, parents);
    // printf("stars\n");
    // printCudaStars(num_vert, stars);

    //this is buggy
    //appears to not be consistent
    //reduce_parents<<<block_vert, 1024>>>(parents, parents_holder, num_vert);

    //copy to uint64_t before deletion.
    //cudaMemcpy(&count, parents_holder, sizeof(uint64_t), cudaMemcpyDeviceToHost);
    converged = starConverged(num_vert, stars);

    //and reset parents
    uint_copy_kernel<<<block_vert, 1024>>>(parents_holder, parents, num_vert);
    cudaFree(grandparents);

    cudaDeviceSynchronize();

    //printf("Counts\n");
    //count_contigs<<<blocknums, 1024>>>(nnz, parents);
    //printf("Done with iteration %llu: %llu %llu \n", iters, count, num_vert>>(iters-3));
    printf("Done with iter %llu\n", iters);

    auto iter_end = std::chrono::high_resolution_clock::now();

    diff = iter_end-iter_start;

    std::cout << "Time required for whole iter: " << diff.count() << " s" << endl;

    iters++;

  } while (!converged);

  printf("Converged\n");
  //if we've really converged we need to syncronize so that the lens are guaranteed
  cudaDeviceSynchronize();


  //when done, print outrows
  // for (int i =0; i < outputRows.size(); i++){
  //   printrowkern(outputRows.at(i), contigs, contig_lens);
  // }

  //time to write to output
  std::ofstream fout;
  fout.open("cc_output.dat");

  for (int i = 0; i < outputRows.size(); i++){

    uint64_t row = outputRows.at(i);
    cout << "len: " << contig_lens[row];

    if (contig_lens[row] >= MAX_VEC){
      cout << " TOO LARGE";
    }
    cout <<  endl;
    for (uint64_t j = 0; j < contig_lens[row]; j++){
      fout << contigs[row*MAX_VEC+j];
    }
    fout << endl;

  }
  fout.close();

  //last call to assert correctness
  cudaDeviceSynchronize();

  //parents are converged
  //free up memory
  cudaFree(parents);
  cudaFree(parents_holder);
  cudaFree(contigs);
  cudaFree(contig_lens);
  cudaFree(stars);
  cudaFree(contigs_holder);
  cudaFree(contig_lens_holder);


}

//iteratively solve the cc problem by jumping through the parents array
__global__ void cuda_solver_kernel(uint64_t nnz, uint64_t num_vert, char* contigs, uint64_t* contig_lens, uint64_t * parents, uint64_t startNnz, char * startVals, uint64_t * startLens, uint64_t * startRows){

  uint64_t tid = threadIdx.x +  blockIdx.x * blockDim.x;

  if (tid >= startNnz) return;

  uint64_t row = startRows[tid];

  //else
  printf("Tid %llu working on kmer that starts at %llu\n", tid, row);

  //first step, copy over my parent - if we have a typo it may show up here
  assert(contig_lens[startRows[tid]] == 1);

  //now steal character
  char my_first_extension = contigs[row*MAX_VEC];

  //and fill in
  for (uint64_t i = 0; i < startLens[tid]; i++){

    contigs[row*MAX_VEC +i] = startVals[tid*MAX_VEC +i];
  }
  contigs[row*MAX_VEC + startLens[tid]] = my_first_extension;
  contig_lens[row] += startLens[tid];

  //start looks good, lets test the rest!
  uint64_t my_parent = parents[row];
  while (my_parent != parents[my_parent]){

    //with 0 mutations, everyone ahead of me should have exactly 1 base
    assert(contig_lens[my_parent] == 1);
    //copy over data
    for (uint64_t i = 0; i < contig_lens[my_parent]; i++){

      contigs[row*MAX_VEC+contig_lens[row]+i] = contigs[my_parent*MAX_VEC + i];

    }
    //update length
    contig_lens[row] += contig_lens[my_parent];

    //and update parent
    parents[row] = parents[parents[row]];
    my_parent = parents[row];

  }

}


//separate the approach a little from cc
//simplest version does a one off trace - builds all contigs in parallel locally
//this now works!
void iterative_cuda_solver(uint64_t nnz, uint64_t num_vert, uint64_t* Arows, uint64_t* Acols, char* Avals, std::vector<uint64_t> outputRows, uint64_t outNnz, char*kmerVals, uint64_t*kmerLens, uint64_t * kmerParents){

  //block sizes for mat and vert based ops
  uint64_t blocknums = (nnz -1)/1024 + 1;
  uint64_t block_vert = (num_vert -1)/1024 + 1;

  //define some memory to work with
  char * contigs;
  uint64_t * contig_lens;

  auto start = std::chrono::high_resolution_clock::now();

  cudaMallocManaged((void **)&contigs,num_vert*MAX_VEC*sizeof(char));

  cudaMallocManaged((void **)&contig_lens,num_vert*sizeof(uint64_t));




  //setup! want to do parent_cond hook, and then complete iteratively
  init_contig_lens<<<block_vert, 1024>>>(num_vert, contig_lens);
  assert_contig_lens<<<block_vert, 1024>>>(num_vert, contig_lens);

  cudaDeviceSynchronize();


  auto contig_setup = std::chrono::high_resolution_clock::now();

  std::chrono::duration<double> diff = contig_setup-start;

  std::cout << "Time required for contig setup: " << diff.count() << " s\n";

  //init_contigs<<<blocknums, 1024>>>(nnz,num_vert, Arows, Acols, Avals, contigs, contig_lens);

  uint64_t * parents;

  cudaMalloc((void **)&parents,num_vert*sizeof(uint64_t));

  init_parent<<<block_vert, 1024>>>(num_vert, parents);

  bool * stars;

  cudaMalloc((void ** )&stars, num_vert*sizeof(bool));

  reset_star<<<block_vert, 1024>>>(num_vert, stars);


  simple_adj_hook<<<blocknums, 1024>>>(nnz, Arows, Acols, Avals, parents, stars, contigs, contig_lens);

  cudaDeviceSynchronize();

  //now that the device is ready, launch iterative solver
  cuda_solver_kernel<<<block_vert, 1024>>>(nnz, num_vert, contigs, contig_lens, parents, outNnz, kmerVals, kmerLens, kmerParents);

  cudaDeviceSynchronize();
  fflush(stdout);

  std::ofstream fout;
  fout.open("iterative_output.dat");

  for (int i = 0; i < outputRows.size(); i++){

    uint64_t row = outputRows.at(i);
    cout << "len: " << contig_lens[row];

    if (contig_lens[row] >= MAX_VEC){
      cout << " TOO LARGE";
    }
    cout <<  endl;
    for (uint64_t j = 0; j < contig_lens[row]; j++){
      fout << contigs[row*MAX_VEC+j];
    }
    fout << endl;

  }
  fout.close();


}



void cc_len(uint64_t nnz, uint64_t num_vert, uint64_t* Arows, uint64_t* Acols, char* Avals, std::vector<uint64_t> outputRows, uint64_t maxOut, char*kmerVals, uint64_t*kmerLens, uint64_t * kmerParents){

  uint64_t blocknums = (nnz -1)/1024 + 1;
  uint64_t block_vert = (num_vert -1)/1024 + 1;


  //move from adj matrix to forward extension per
  char * contigs;
  uint64_t * contig_index;

  auto start = std::chrono::high_resolution_clock::now();

  uint64_t * contig_index_holder;

  uint64_t * contig_lens;

  cudaMallocManaged((void **)&contig_index_holder,num_vert*sizeof(uint64_t));

  cudaMallocManaged((void **)&contigs,num_vert*sizeof(char));
  cudaMallocManaged((void **)&contig_index,num_vert*sizeof(uint64_t));

  cudaMallocManaged((void **)&contig_lens,num_vert*sizeof(uint64_t));

  uint64_t * parents;

  cudaMalloc((void **)&parents,num_vert*sizeof(uint64_t));

  uint64_t * parents_holder;

  cudaMalloc((void **)&parents_holder,num_vert*sizeof(uint64_t));

  init_parent<<<block_vert, 1024>>>(num_vert, parents);

  uint_copy_kernel<<<block_vert, 1024>>>(parents_holder, parents, num_vert);

  bool * stars;

  cudaMalloc((void ** )&stars, num_vert*sizeof(bool));


  reset_star<<<block_vert, 1024>>>(num_vert, stars);

  init_contig_lens<<<block_vert, 1024>>>(num_vert, contig_index);
  assert_contig_lens<<<block_vert, 1024>>>(num_vert, contig_index);

  init_contig_lens<<<block_vert, 1024>>>(num_vert, contig_index_holder);
  assert_contig_lens<<<block_vert, 1024>>>(num_vert, contig_index_holder);

  // init_contig_lens<<<block_vert, 1024>>>(num_vert, contig_lens);
  // assert_contig_lens<<<block_vert, 1024>>>(num_vert, contig_lens);

  cudaDeviceSynchronize();

  len_adj_hook<<<blocknums, 1024>>>(nnz, Arows, Acols, Avals, parents, stars, contigs, contig_index);

  //copy over from index at step 1 - these are the correct output lens of the items
  uint_copy_kernel<<<block_vert, 1024>>>(contig_lens, contig_index, num_vert);

  cudaDeviceSynchronize();

  //prints
  count_bases<<<1,1>>>(num_vert, contigs);
  sum_lens<<<1,1>>>(num_vert, contig_index);

  cudaDeviceSynchronize();
  printf("and lens\n");
  sum_lens<<<1,1>>>(num_vert, contig_lens);
  cudaDeviceSynchronize();

  auto full_setup = std::chrono::high_resolution_clock::now();

  std::chrono::duration<double> diff = full_setup-start;

  std::cout << "Time required for internal setup: " << diff.count() << " s" << endl;

  fflush(stdout);

  //no errors yet! time to start iterating
  //there is no lead update this time :D

  bool converged = false;
  uint64_t iters = 0;

  do {

    auto iter_start = std::chrono::high_resolution_clock::now();


    //first up is the cond hook
    len_cond_hook<<<block_vert, 1024>>>(num_vert, parents, parents_holder, stars, contig_index, contig_index_holder);

    //then swap pointers

    uint64_t * temp_indices = contig_index;
    contig_index = contig_index_holder;
    contig_index_holder = temp_indices;

    uint64_t * temp_parents = parents;
    parents =  parents_holder;
    parents_holder  = temp_parents;

    //and check stars
    parent_star_check(num_vert, parents, stars);

    converged = starConverged(num_vert, stars);

    cudaDeviceSynchronize();

    auto iter_end = std::chrono::high_resolution_clock::now();

    diff = iter_end-iter_start;

    std::cout << "Time required for  iter " << iters << ": " << diff.count() << " s" << endl;

    iters++;

  } while (!converged);

  auto fill_start = std::chrono::high_resolution_clock::now();

  //now that we've converged, we need to map from contig_id to actual values
  uint64_t * contig_map;
  uint64_t * contig_map_lens;

  cudaMalloc((void **)&contig_map,maxOut*sizeof(uint64_t));
  cudaMalloc((void **)&contig_map_lens,maxOut*sizeof(uint64_t));

  printf("Max output: %llu\n", maxOut);

  uint64_t maxOutBlock = (maxOut -1)/1024 + 1;

  map_contigs<<<maxOutBlock, 1024>>>(maxOut, kmerVals, kmerLens, kmerParents, contig_index, parents, contig_map, contig_map_lens);
  //print_mappings<<<1,1>>>(maxOut, contig_map, contig_map_lens);

  cudaDeviceSynchronize();
  fflush(stdout);


  //now that we have the mappings, allocate memory
  char ** final_contigs;
  cudaMalloc((void **)&final_contigs,maxOut*sizeof(char * ));

  //char ** host_final_contigs;

  //cudaMalloc((void **)&host_final_contigs, maxOut*sizeof(char *));

  mallocHostContigs(maxOut, final_contigs, contig_map_lens);

  //mallocHostContigs(maxOut, host_final_contigs, contig_map_lens);

  //now fill contigs
  fill_contigs_starts<<<maxOutBlock, 1024>>>(maxOut, final_contigs, kmerVals, kmerLens);

  //set up dim3

  //x dim - num vert
  uint64_t x_size = 24;
  uint64_t y_size = 24;
  uint64_t fill_x_block = (num_vert -1)/x_size + 1;
  uint64_t fill_y_block = (maxOut - 1)/y_size + 1;

  dim3 blockShape = dim3(x_size, y_size);
  dim3 gridShape = dim3(fill_x_block, fill_y_block);

  printf("Grid shape: (%llu, %llu)\n", gridShape.x, gridShape.y);
  printf("block shape: (%llu, %llu)\n", blockShape.x, blockShape.y);
  fflush(stdout);

  fill_contigs<<<gridShape, blockShape>>>(num_vert, final_contigs, maxOut, contig_map, contig_map_lens, contigs, contig_index, contig_lens, parents);

  cudaDeviceSynchronize();
  fflush(stdout);

  // check_contig<<<1,1>>>(0, final_contigs, contig_map_lens);
  // cudaDeviceSynchronize();
  // fflush(stdout);

  auto fill_end = std::chrono::high_resolution_clock::now();

  diff = fill_end-fill_start;

  std::cout << "Time required to fill contig buffers: " << diff.count() << " s" << endl;

  //copy over to host
  // move_contigs_to_host<<<maxOutBlock, 1024>>>(maxOut, final_contigs,host_final_contigs, contig_map_lens);
  // cudaDeviceSynchronize();

  save_contigs("cc_len.dat", maxOut, final_contigs, contig_map_lens);
  cudaDeviceSynchronize();

  auto write_end = std::chrono::high_resolution_clock::now();

  diff = write_end - fill_end;

  std::cout << "Wrote to file in : " << diff.count() << " s" << endl;


  //and at the end free them

  printf("before host Contigs\n");

  freeHostContigs(maxOut, final_contigs);
  cudaDeviceSynchronize();
  printf("Error not in cc_len");
  fflush(stdout);

}









__host__ int cudaMain(int argc, char** argv){

  //start timing
  auto start = std::chrono::high_resolution_clock::now();

  // testAtomic();
  // cudaDeviceSynchronize();
  // fflush(stdout);


  auto afterCounter = std::chrono::high_resolution_clock::now();

  std::chrono::duration<double> diff = afterCounter-start;


  std::cout << "Tested Counter in " << diff.count() << " s\n";



  //prep matrix info
  std::string kmer_fname = std::string(argv[1]);
  int ks = kmer_size(kmer_fname);

  if (ks != KMER_LEN) {
      throw std::runtime_error("Error: " + kmer_fname + " contains " + std::to_string(ks) +
                               "-mers, while this binary is compiled for " +
                               std::to_string(KMER_LEN) +
                               "-mers.  Modify packing.hpp and recompile.");
  }

    size_t n_kmers = line_count(kmer_fname);


    std::map<string, size_t> map;

    //additional map to assert that the kmer pairs aren't overlapping
    std::map<uint64_t, kmer_pair> err_map;

    //load kmers
    std::vector<kmer_pair> kmers = read_kmers(kmer_fname);

    //host pkmer setup can run asynchronously
    thrust::host_vector<kmer_pair> host_kmers(kmers);


    thrust::host_vector<pkmer_t> host_pkmers(host_kmers.size());

    kmer_pair * host_kmers_ptr = thrust::raw_pointer_cast( host_kmers.data() );

    thrust::transform(host_kmers.begin(), host_kmers.end(), host_pkmers.begin(), kmer_to_pkmer());

    thrust::device_vector<pkmer_t> next_pkmers(host_pkmers);

    pkmer_t * host_pkmers_ptr = thrust::raw_pointer_cast( host_pkmers.data() );

    pkmer_t * dev_pkmers = thrust::raw_pointer_cast( next_pkmers.data() );

    printf("host_pkmers_ptr %p  host_kmers %p\n", host_pkmers_ptr, host_kmers_ptr);

    thrust::device_vector<kmer_pair> dev_kmers_vector(kmers);

    kmer_pair* dev_kmers = thrust::raw_pointer_cast( dev_kmers_vector.data() );


    auto beforeHash = std::chrono::high_resolution_clock::now();

    cudaHashMap * hashMap;

    hashMap = initMap(dev_kmers_vector.size());

    printHashMap(hashMap);

    uint64_t insert_block = (dev_kmers_vector.size() -1)/1024 + 1;

    //original size:  dev_kmers_vector.size()
    insert_all<<<insert_block, 1024>>>(dev_kmers_vector.size(), dev_kmers, hashMap);
    cudaDeviceSynchronize();

    printf("Insert completed\n");
    fflush(stdout);


    printHashMap(hashMap);


    //and test correctness
    assertInserts<<<insert_block, 1024>>>(dev_kmers_vector.size(), dev_kmers, hashMap);

    
    auto afterHash = std::chrono::high_resolution_clock::now();

    diff = afterHash-beforeHash;
    std::cout << "Hash table inserts completed in " << diff.count() << " s\n";


    uint64_t * parents;
    cudaMalloc((void **)& parents, dev_kmers_vector.size()*sizeof(uint64_t));

    char * extensions;
    cudaMalloc((void **)&extensions, dev_kmers_vector.size()*sizeof(char));

    uint64_t num_starts = prep_parents(dev_kmers_vector.size(), dev_kmers, dev_pkmers, parents, extensions, hashMap);

    printf("Num starts: %llu\n", num_starts);

    
    //get kmer leads
    //this is more unavoidable overhead

    thrust::host_vector<pkmer_t> host_leads(host_kmers.size());
    thrust::transform(host_kmers.begin(), host_kmers.end(), host_leads.begin(), kmer_to_start());
    thrust::device_vector<pkmer_t> leads(host_leads);

 
    pkmer_t* lead_pointer = thrust::raw_pointer_cast( leads.data() );




    //now construct starts
    uint64_t startNnz = num_starts;
    char * startVals;
    uint64_t * startLens;
    uint64_t * startRows;
    uint64_t * startIds;

    //locate the starts via cuda, then move to memory based on 
    //find_starts_cuda(dev_kmers_vector.size(), lead_pointer, startNnz, &startIds, hashMap);


    freeCudaHashMap(hashMap);
    return 0;
    
    //everything after this is good but doesn't matter 

    char* perfvals;
    uint64_t * perfrows;
    uint64_t * perfcols;
    uint64_t perf_nnz;

    std::vector<std::pair<kmer_pair, uint64_t>> perf_starts = build_adj_mat(kmers, &perf_nnz, &perfvals, &perfrows, &perfcols);

    //print out some samples

    // for (int i =0; i < 10; i++){
    //   printf("%llu -> %llu: %c\n", perfrows[i], perfcols[i], perfvals[i]);
    // }

    //now run test
    //this is successful for all runtimes
    //build_kmers_from_adj(perf_starts, perf_nnz, perfvals, perfrows, perfcols);


    

    //fill starts mats for cuda
    prep_starts(perf_starts, perfrows, &startNnz, &startVals, &startLens, &startRows);

    //check output for verify
    //on test case looks good
    // printf("Visual sanity check on starts\n");
    // int min_size = 10;
    // if (perf_starts.size() < min_size){
    //   min_size = perf_starts.size();
    // }
    // for (int i=0; i < min_size; i++){
    //   cout << i << ": " << std::get<0>(perf_starts.at(i)).kmer_str() << endl;
    //
    //   cout << i << ": ";
    //   for (int j = 0; j < 10; j++){
    //     cout << startVals[i*MAX_VEC+j];
    //   }
    //   cout << endl;
    // }

    std::vector<uint64_t> outRows2 = gen_outRows(perf_starts, perfrows);

    //what info needs to be updated for the next pass?
    //sync just in case
    cudaDeviceSynchronize();


    char* perfvalsCuda;
    uint64_t * perfrowsCuda;
    uint64_t * perfcolsCuda;

    copy_to_cuda(perf_nnz, perfvals, perfrows, perfcols, &perfvalsCuda, &perfrowsCuda, &perfcolsCuda);

    auto midpoint = std::chrono::high_resolution_clock::now();

    diff = midpoint-start;

    std::cout << "Time required for setup: " << diff.count() << " s\n";

    //connected components call
    //iterative_cuda_solver(perf_nnz, n_kmers, perfrowsCuda, perfcolsCuda, perfvalsCuda, outRows2, startNnz, startVals, startLens, startRows);
    cc_len(perf_nnz, n_kmers, perfrowsCuda, perfcolsCuda, perfvalsCuda, outRows2, startNnz, startVals, startLens, startRows);

    cudaDeviceSynchronize();

    auto end = std::chrono::high_resolution_clock::now();

    diff = end-midpoint;

    std::cout << "Time required for cc: " << diff.count() << " s\n";


    return 0;
}
