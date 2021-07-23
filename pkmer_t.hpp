#pragma once

#include "packing.hpp"
#include <cstring>

using namespace std;

struct pkmer_t {
    unsigned char data[PACKED_KMER_LEN];

    // Get the k-kmer string, hash the k-mer.
    std::string get() const noexcept;

    __host__ __device__ uint64_t hash() const noexcept;

    // Various C++ lifetime stuff.
    pkmer_t(const std::string& kmer);

    pkmer_t() = default;
    pkmer_t(const pkmer_t& pkmer) = default;
    pkmer_t(const char* kmer);

    pkmer_t& operator=(const pkmer_t& pkmer) = default;

    bool operator==(const pkmer_t& pkmer) const noexcept;
    bool operator!=(const pkmer_t& pkmer) const noexcept;

    void init(const unsigned char data[PACKED_KMER_LEN]);
};

std::string pkmer_t::get() const noexcept {
    char kmer[KMER_LEN];
    unpackKmer(data, kmer);
    return std::string(kmer, KMER_LEN);
}



__host__ __device__ uint64_t pkmer_t::hash() const noexcept {
    unsigned long hashval = 5381;
    for (int i = 0; i < PACKED_KMER_LEN; i++) {
        hashval = data[i] + (hashval << 5) + hashval;
    }
    return hashval;
}

pkmer_t::pkmer_t(const std::string& kmer) { packKmer(kmer.data(), data); }

//new definition that takes in a char * representing string data
__host__ __device__ pkmer_t::pkmer_t(const char* kmer) { packKmer(kmer, data); }

bool pkmer_t::operator==(const pkmer_t& pkmer) const noexcept {
    return memcmp(pkmer.data, data, PACKED_KMER_LEN) == 0;
}

bool pkmer_t::operator!=(const pkmer_t& pkmer) const noexcept { return !(*this == pkmer); }

void pkmer_t::init(const unsigned char data[PACKED_KMER_LEN]) {
    for (int i = 0; i < PACKED_KMER_LEN; i++) {
        this->data[i] = data[i];
    }
}
