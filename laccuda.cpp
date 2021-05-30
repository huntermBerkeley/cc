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
#include <cstring>
#include <inttypes.h>
#include <algorithm>

#include "test_kernels.hpp"

//add support for reading kmer files
#include "read_kmers.hpp"
#include "kmer_t.hpp"

//map for allocating numbers
#include <unordered_map>

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
#define MAX_VEC 4000
#endif

using namespace std;

void printVec(int vecLen, char * vec, uint64_t*lengths){

  std::cout << "Vector: "  << endl;
  for (int i=0; i < vecLen; i++){

    std::cout << i << ": [ ";
    for (int j = 0; j < lengths[i]; j++){
      cout << vec[i*MAX_VEC+j];
    }
    std:: cout << " ] " << lengths[i] << endl;
  }

}

void printLens(int vecLen, uint64_t* lengths){

  cout << "Lengths: " << endl;

  uint64_t i = 0;
  while (i < vecLen){

    if (i+5 < vecLen){

      for (int j = i; j <  i+5; j++){

        cout << j << ": " << lengths[j] << "    ";

      }

    } else {

        for (int j = i; j <  vecLen; j++){

          cout << j << ": " << lengths[j] << "    ";

        }
    }

    cout << endl;

    i+=5;
  }


}




void printrow(uint64_t row, char * vec, uint64_t*lengths){

    std::cout << "len: " << lengths[row] << endl;
    std::cout << "[ ";
    for (int j = 0; j < lengths[row]; j++){
      cout << vec[row*MAX_VEC+j];
    }
    std:: cout << " ]" << endl;

}

void printNull(int vecLen, char* vec, uint64_t * lengths){

  printf("Rows that are null:\n");
  for (uint64_t i =0; i < vecLen; i++){
    if (lengths[i] == 0){
      cout << i << " ";
      printrow(i, vec, lengths);
    }
  }

}


void printMat(uint64_t nnz, char* characters, uint64_t *rows, uint64_t* cols){

  for (uint64_t i = 0; i < nnz; i++){
    cout << rows[i] << ", " << cols[i] << ": " << characters[i] << endl;
  }
}

//Allocate a set amount of memory determined by the user
// and use this to create a vector with the correct output
int main(int argc, char** argv) {


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


    std::unordered_map<uint64_t, size_t> map;

    //load kmers
    std::vector<kmer_pair> kmers = read_kmers(kmer_fname);


    //iterate and count
    size_t counter = 0;
    for (size_t i = 0; i < kmers.size(); i++){
      pkmer_t kmer = kmers.at(i).kmer;

      if (map.count(kmer.hash()) != 0){
        cout << "Repeat hit, these should be cleaned" << endl;
        continue;
      } else {
        map[kmer.hash()] = counter;
        counter++;
      }
    }

    //counter gets reused and i don't want to break Things
    //so reset to be a new max val, we will want to increment this later
    size_t max_counter = counter;


    //NEW add ons
    //start kmers need to be appended as an item in the matrix:
    //they are a nonexistent kmer that connects to themselves
    //firstly, we need to allocate space
    //ever kmer has


    //potentially sort
    //would this improve performance?
    //would definitely improve memmory accesses on one dim, but we're already in row format rn :D

    uint64_t nnz = 0;
    for (int i = 0; i < kmers.size(); i++){

      if (kmers.at(i).forwardExt() != 'F'){
        nnz+=1;
      }


    }


    //first define sparse matrix
    std::vector<uint64_t> outRows;

    //grab start extensions, we'll repair these  at the end
    //this is opposed to initial modifications because there isn't
    //really a  point in  copying these over log(n) times, especially
    // because we know right where they go.
    std::vector<kmer_pair> outKmers;

    uint64_t num_vert = counter;

    //generic setup
    // uint64_t nnz = 5;
    // uint64_t num_vert = 6;
    // cout << "Working on " << nnz << " nonzeros." << endl;
    //
    // //local copy of matrix
    char *matLocal = new char[nnz];
    uint64_t *matRowLocal = new uint64_t[nnz];
    uint64_t *matColLocal = new uint64_t[nnz];
    //
    // matRowLocal[0] = 1;
    // matColLocal[0] = 0;
    // matLocal[0] = 'C';
    //
    // matRowLocal[1] = 2;
    // matColLocal[1] = 1;
    // matLocal[1] = 'T';
    //
    // matRowLocal[2] = 3;
    // matColLocal[2] = 2;
    // matLocal[2] = 'G';
    //
    // matRowLocal[3] = 4;
    // matColLocal[3] = 3;
    // matLocal[3] = 'C';
    //
    // matRowLocal[4] = 5;
    // matColLocal[4] =5;
    // matLocal[4] = 'A';

    // 0 0 0 0 0 0
    // 1 0 0 0 0 0
    // 0 1 0 0 0 0
    // 0 0 1 0 0 0
    // 0 0 0 1 0 0
    // 0 0 0 0 1 0





    counter = 0;
    for (int i = 0; i < kmers.size(); i++){



      pkmer_t kmer = kmers.at(i).kmer;
      uint64_t row = map.at(kmer.hash());

      pkmer_t forward = kmers.at(i).next_kmer();
      uint64_t fhash = forward.hash();

      //fhash is the next kmer
      if (map.count(fhash) == 0){



        cout << "end at " <<  map.at(kmer.hash()) << endl;

        continue;

      } else {

        //positive match, add to matrix
        //csr format
        matLocal[counter] = kmers.at(i).forwardExt();
        //was originall row/col
        matRowLocal[counter] = row;
        matColLocal[counter] = map.at(fhash);

        if (kmers.at(i).backwardExt() == 'F'){
          cout << "Start at " << i << endl;
          cout << matRowLocal[counter] << " points to " << matColLocal[counter] << endl;
          outRows.push_back(matRowLocal[counter]);
          outKmers.push_back(kmers.at(i));

        }

        //cut off last char for len count
        //does this work?
        counter++;


        }


    }

    cout << counter << " filled" << endl;
    printf("num_vert %llu\n",num_vert);
    printf("size of mat %llu\n",nnz);
    fflush(stdout);


    char * matValsCuda;
    uint64_t * matRowsCuda;
    uint64_t * matColsCuda;


    //don't need to allocate for  end of string, use as a character array is safe if there is a
    //secondary buffer for lengths
    cudaMalloc((void ** )&matValsCuda, nnz*sizeof(char));
    cudaMalloc((void ** )&matRowsCuda, nnz*sizeof(uint64_t));
    cudaMalloc((void ** )&matColsCuda, nnz*sizeof(uint64_t));


    //memcopys
    cudaMemcpy(matValsCuda, matLocal, nnz*sizeof(char), cudaMemcpyHostToDevice);

    cudaMemcpy(matRowsCuda, matRowLocal, nnz*sizeof(uint64_t), cudaMemcpyHostToDevice);
    cudaMemcpy(matColsCuda, matColLocal, nnz*sizeof(uint64_t), cudaMemcpyHostToDevice);



    //printLens(nnz, lenVecLocal);




    //printVec(nnz, vecLocal, lenVecLocal);

    //print outrows


    auto start = std::chrono::high_resolution_clock::now();

    //fill_wrapper(nnz, valA,  rowIndA, colIndA);
    //std::cout << "Output rows are : ";

    // for(int i=0; i < outRows.size(); i++)
    // std::cout << outRows.at(i) << ' ';
    // cout << endl;

    //fill_vector(nnz, vecB);

    //print before
    // cout << "Before" << endl;
    // printVec(2, vecLoc, lenVec);

    //rn matrix in csc format


    //NEW: Prep outKmers as a trio of vals to be passed in
    uint64_t maxOut = outKmers.size();

    char * kmersVals;
    uint64_t * kmersLens;

    cudaMallocManaged((void **)&kmersVals,maxOut*MAX_VEC*sizeof(char));

    cudaMallocManaged((void **)&kmersLens,maxOut*sizeof(uint64_t));

    //iterate through maxKmers

    for(int i=0; i < maxKmers.size(); i++){

      pkmer_t kmer = maxKmers.at(i);
      for(int j=0; j < kmer.get().size() )
      maxKmers.at(i).kmer.get(i);

    }




    //fun flips
    cc(nnz, num_vert, matRowsCuda, matColsCuda, matValsCuda, outRows);


    //passing  along kmers is a shitshow
    //i guess build a managed list of chars/lens
    //and pass  it along


    //copy back
    //cudaMemcpy(outLocal, components,  nnz*sizeof(uint64_t), cudaMemcpyDeviceToHost);


    auto end = std::chrono::high_resolution_clock::now();

    auto duration = std::chrono::duration_cast<std::chrono::microseconds>( end - start ).count();
    //printVec(nnz, outLocal, lenOutLocal);

    //printLens(nnz, lenOutLocal);

    std::cout << "This is on the laccuda side:" << std::endl;

    //this segaults, repair by passing pkmer_t info to kernels and pass in strings
    // for (int i =0; i < outRows.size(); i++){
    //   printrow(outRows.at(i), contigs, contig_lens);
    // }


    std::cout << "Filled matrix of " << nnz << " items in " << duration << " microseconds"  << endl;


    //printMat(nnz, matLocal, matRowLocal, matColLocal);
    //and print
    //printVec(2, outLoc, lenOut);


    cout << "Saved to " << "output.dat" << endl;


    return 0;
}
