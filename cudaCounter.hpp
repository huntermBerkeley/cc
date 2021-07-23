#pragma once

#include <cuda.h>

//this struct is a way to atomically request values with a logarithmic overhead - at the expense of some slots being empty

#ifndef COUNTERSTATS
#define COUNTERSTATS
#define NUM_COUNTERS 1000
#define MAX_COUNT 2
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

struct cudaCounter {

	uint64_t main_counter;
	uint64_t counters[NUM_COUNTERS];
	uint32_t * locks;

	// Return the k-mer as a string
    __device__ void get_lock(int index);
    __device__ void free_lock(int index);

    __device__ uint64_t get(uint64_t tid);

    __device__ void init(uint64_t tid);

    

private:

	__device__ void update_counter(int index);

	__device__ uint16_t get_lock_nowait(int index);


};




__device__ void cudaCounter::init(uint64_t tid){


	if (tid >= NUM_COUNTERS) return;

	counters[tid] = 0;
	locks[tid] = 0;

	update_counter(tid);

	assert(locks[tid] ==0);


	//printf("Counter %llu starts at %llu, lock %lu\n", tid, counters[tid]);

}

__device__ uint16_t cudaCounter::get_lock_nowait(int index) {
	//set lock to 1 to claim
	//returns 0 if success
	uint32_t zero = 0;
	uint32_t one = 1;
	return atomicCAS(locks+index, zero, one);
}


//don't use this
//threadteam can get stuck in some cases
__device__ void cudaCounter::get_lock(int index) { 


	//printf("grabbing lock %d, lock address: %p, mixed_address: %p, look inside: %lu \n", index, &locks, locks+index, locks[index]);

	
	uint16_t result = 1;

	do {
		result = get_lock_nowait(index);
	} while (result !=0);

}

__device__ void cudaCounter::free_lock(int index) {

	//set lock to 0 to release
	uint32_t zero = 0;
	uint32_t one = 1;
	//TODO: might need a __threadfence();
	atomicCAS(locks + index, one, zero);

}

__device__ uint64_t cudaCounter::get(uint64_t tid) { 

	int index = tid % NUM_COUNTERS;

	//get_lock(index);

	while (true){

	
		uint16_t result = get_lock_nowait(index);

		//successfully locked
		if (result == 0){

			//printf("%llu Acquired lock %d", tid, index);

			uint64_t val = atomicAdd((uint64_cu *) counters+index, (uint64_cu) 1);

			if (counters[index] % MAX_COUNT == 0){
				update_counter(index);
			}


			free_lock(index);


			return val;

		} //else {
		// 	printf("%llu failed to Acquire lock %d\n", tid, index);
		// }

	}

	

	

}

__device__ void cudaCounter::update_counter(int index) { 

	//can safely assume lock is already grabbed
	//printf("Counter %d must be updated\n", index);
	counters[index] = atomicAdd((uint64_cu *) &main_counter, (uint64_cu) MAX_COUNT);

 }

 __global__ void initMainKernel(cudaCounter * counter){


 	uint64_t tid = threadIdx.x +  blockIdx.x * blockDim.x;


 	if (tid!=0) return;

 	counter->main_counter = 0;
 }

 __global__ void initCounterKernel(cudaCounter * counter){


 	uint64_t tid = threadIdx.x +  blockIdx.x * blockDim.x;

 	counter->init(tid);
 }



__host__ void freeCounter(cudaCounter * counter){


	cudaDeviceSynchronize();

	printf("Errors shouldn't occur before here\n");
	fflush(stdout);

	cudaCounter * hostCounter = new cudaCounter;

	CUDA_CHECK(cudaMemcpy(hostCounter, counter, sizeof(cudaCounter), cudaMemcpyDeviceToHost));

	CUDA_CHECK(cudaFree(hostCounter->locks));

	CUDA_CHECK(cudaFree(counter));

	free(hostCounter);
}


__host__ void initCounter(cudaCounter ** counter_to_fill){

	//init device memory

	cudaCounter * newCounter = new cudaCounter;

	cudaCounter* cudaNewCounter;

	cudaMalloc((void**)& cudaNewCounter, 1*sizeof(cudaCounter));

	uint32_t * locks;
	cudaMalloc((void **)&locks, NUM_COUNTERS*sizeof(uint32_t));
	cudaMemset(locks, 0, NUM_COUNTERS*sizeof(uint32_t));

	newCounter->locks = locks;


	cudaMemcpy(cudaNewCounter, newCounter, sizeof(cudaCounter), cudaMemcpyHostToDevice);

	//counter->main_counter = 0;

	initMainKernel<<<1,1>>>(cudaNewCounter);

	initCounterKernel<<<NUM_COUNTERS, 1>>>(cudaNewCounter);

	*counter_to_fill = cudaNewCounter; 




}