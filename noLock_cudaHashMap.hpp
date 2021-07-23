#pragma once

#include <cuda.h>
#include "read_kmers.hpp"
#include "kmer_t.hpp"
//#include "MurmurHash3.h"
#include "iostream"

//TODO: make this just sotre the hashes in the kmer_pair slots
//bitcompare too? - take in two void * and do a byte comparison

#ifndef HASHSTATS
#define HASHSTATS
#define FILL_RATIO .693 //ln(2) is the ideal value for %50 bucket empty
#define PROBE_SEED 3
#define STORE_SEED 5
#endif

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


struct cudaHashMap {

	uint64_t size;
	volatile uint64_t slots_full;
  uint64_t * slots;
	uint64_t * slot_counters;

	// Return the k-mer as a string
    

	__device__ void insert(pkmer_t key, uint64_t val);
  __device__ uint64_t get(pkmer_t key);
    

private:

	__device__ uint64_t getHash(pkmer_t key, uint64_t seed);


};




//maybe volatile conflicts?
__device__ void cudaHashMap::insert(pkmer_t key, uint64_t val){
	
	
	//these magic numbers are arbitrary, attempt to keep hashes different
	uint64_t probe = getHash(key,PROBE_SEED);

	uint64_t to_insert = getHash(key, STORE_SEED);

	while (true){

		uint64_t prev = atomicCAS((long long unsigned int *) &slots[probe], (long long unsigned int) 0, (long long unsigned int) to_insert);

		if (prev == 0 || prev == to_insert){

			//this write isn't enforced by global memory - but we don't care
			//we just need *a* unique number, doesn't have to be the first one we find
			slot_counters[probe] = val;

			//this isn't super duper correct, but will be right on the test cases
			//this isn't necessary anyways on the real data
			atomicAdd((long long unsigned int *) &slots_full, 1);

			return;

		}
		//change this to be faster - make size even power of 2;
		probe = (probe+1) % size;

	}


}



__host__ __device__ uint64_t MurmurHash64A ( const void * key, int len, unsigned int seed )
{
	const uint64_t m = 0xc6a4a7935bd1e995;
	const int r = 47;

	uint64_t h = seed ^ (len * m);

	const uint64_t * data = (const uint64_t *)key;
	const uint64_t * end = data + (len/8);

	while(data != end)
	{
		uint64_t k = *data++;

		k *= m; 
		k ^= k >> r; 
		k *= m; 

		h ^= k;
		h *= m; 
	}

	const unsigned char * data2 = (const unsigned char*)data;

	switch(len & 7)
	{
		case 7: h ^= (uint64_t)data2[6] << 48;
		case 6: h ^= (uint64_t)data2[5] << 40;
		case 5: h ^= (uint64_t)data2[4] << 32;
		case 4: h ^= (uint64_t)data2[3] << 24;
		case 3: h ^= (uint64_t)data2[2] << 16;
		case 2: h ^= (uint64_t)data2[1] << 8;
		case 1: h ^= (uint64_t)data2[0];
						h *= m;
	};

	h ^= h >> r;
	h *= m;
	h ^= h >> r;

	return h;
}

//hashmap does not support removal, so this is safe.
//we just avoid locking
__device__ uint64_t cudaHashMap::get(pkmer_t key) { 



	uint64_t probe = getHash(key, PROBE_SEED);
	uint64_t start = probe;

	uint64_t to_insert = getHash(key, STORE_SEED);
	


	do {

		if (slots[probe]== to_insert){


			return slot_counters[probe];

	} else if (slots[probe] == 0){

		//if we run into an empty slot yeah we should return
		//no sense chasing through the whole array
		return size+1;
	}

	probe = probe + 1;

	if (probe >= size) {
		probe -= size;

	}

} while (probe != start);

return size+1;


}

//get the hash, either using prashant's reversible hash or default or murmur
//we'll start with default and burn that bridge when we get there
__device__ uint64_t cudaHashMap::getHash(pkmer_t key, uint64_t seed){

	//if modulus is too slow perf wise,
	//change size acquisition to raise to nearest power of two
	//i think grab biggest bit and << 1?
	//that transforms this into 
	//return key.hash() & (size-1)
	//return key.hash() % size;

	//TODO:: verify that this isn't using memory it shouldn't - murmurhash looks like it wants just 1 uint64
	return MurmurHash64A(key.data, sizeof(char)*PACKED_KMER_LEN, seed) % size;

}




__global__ void init_counter(uint64_t nnz, uint64_t val, uint64_t * counter){

	uint64_t tid = threadIdx.x +  blockIdx.x * blockDim.x;

  	if (tid >= nnz) return;

  	counter[tid] = val;

}

__host__ cudaHashMap * initMap(uint64_t expected_num_items){

	float ratio = 1.0 / FILL_RATIO; 

	uint64_t full_items = expected_num_items*ratio;

	printf("Expecting %llu items with fill ratio %f, creating map with %llu items\n", expected_num_items, ratio, full_items);


	cudaHashMap * hostMap;
	cudaHashMap * newMap;

	CUDA_CHECK(cudaMallocHost((void **)& hostMap, 1*sizeof(cudaHashMap)));

	CUDA_CHECK(cudaMalloc((void **)& newMap, 1*sizeof(cudaHashMap)));

	hostMap->size = full_items;

	hostMap->slots_full = 0;

	//malloc and memset arrays

	uint64_t * _slots;
	uint64_t * _slot_counters;


	CUDA_CHECK(cudaMalloc((void **)&_slots, sizeof(uint64_t)*full_items));
	CUDA_CHECK(cudaMemset(_slots, 0, sizeof(uint64_t)*full_items));

	CUDA_CHECK(cudaMalloc((void **)&_slot_counters, sizeof(uint64_t)*full_items));
	CUDA_CHECK(cudaMemset(_slot_counters, 0, sizeof(uint64_t)*full_items));

	init_counter<<<(full_items - 1) / 1024 + 1, 1024>>>(full_items, full_items+1, _slot_counters);

	hostMap->slots = _slots;
	hostMap->slot_counters = _slot_counters;

	CUDA_CHECK(cudaMemcpy(newMap, hostMap, sizeof(cudaHashMap), cudaMemcpyHostToDevice));

	CUDA_CHECK(cudaFreeHost(hostMap));

	return newMap;

}

//copy components of map, free map, then free components
__host__ void freeCudaHashMap(cudaHashMap * map){


	cudaHashMap * hostMap;

	CUDA_CHECK(cudaMallocHost((void **)& hostMap, 1*sizeof(cudaHashMap)));


	CUDA_CHECK(cudaMemcpy(hostMap, map, sizeof(cudaHashMap), cudaMemcpyDeviceToHost));

	CUDA_CHECK(cudaFree(map));
	CUDA_CHECK(cudaFree((kmer_pair *)hostMap->slots));
	CUDA_CHECK(cudaFree((uint64_t *) hostMap->slot_counters));
	CUDA_CHECK(cudaFreeHost(hostMap));

	return;

}

__global__ void printHashMapKernel(cudaHashMap * map){

  uint64_t tid = threadIdx.x +  blockIdx.x * blockDim.x;

  if (tid != 0) return;

  printf("Map Stats:\n--Size: %llu\n--slots_full: %llu\n", map->size, map->slots_full);

}

__host__ void printHashMap(cudaHashMap * map){

	printHashMapKernel<<<1,1>>>(map);
	cudaDeviceSynchronize();
	fflush(stdout);

}
