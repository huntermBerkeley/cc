#pragma once
#include <cuda.h>


__host__ __device__ uint64_t MurmurHash64A ( const void * key, int len, unsigned int seed );

uint64_t hash_64(uint64_t key, uint64_t mask);

uint64_t hash_64i(uint64_t key, uint64_t mask);