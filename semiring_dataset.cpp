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


    //potentially sort
    //would this improve performance?
    //would definitely improve memmory accesses on one dim, but we're really in row format rn :D



    //first define sparse matrix
    std::vector<uint64_t> outRows;

    uint64_t nnz = counter;
    cout << "Working on " << nnz << " nonzeros." << endl;

    //local copy of matrix
    char *matLocal = new char[nnz];
    uint64_t *matRowLocal = new uint64_t[nnz];
    uint64_t *matColLocal = new uint64_t[nnz];

    //local copy of vector
    char * vecLocal = new char[nnz*MAX_VEC];
    uint64_t * lenVecLocal = new uint64_t[nnz];

    //local copy of output
    //set to 0x20 - empty space
    //set lens to 0 - no char to display
    char * outLocal = new char[nnz*MAX_VEC];
    uint64_t * lenOutLocal = new uint64_t[nnz];



    counter = 0;
    for (int i = 0; i < kmers.size(); i++){

      if (kmers.at(i).backwardExt() == 'F'){
        cout << "Start at " << i << endl;
      }

      pkmer_t kmer = kmers.at(i).kmer;
      uint64_t row = map.at(kmer.hash());

      pkmer_t forward = kmers.at(i).next_kmer();
      uint64_t fhash = forward.hash();

      //fhash is the next kmer
      if (map.count(fhash) == 0){

        //add to list of output vectors
        outRows.push_back(row);
        //make sure to copy these boys too
        strcpy(vecLocal+row*MAX_VEC, kmer.get().c_str());
        lenVecLocal[row] = kmer.get().length();

      } else {

        //positive match, add to matrix
        //csr format
        matLocal[counter] = kmers.at(i).forwardExt();
        //was originall row/col
        matColLocal[counter] = row;
        matRowLocal[counter] = map.at(fhash);


        //cut off last char for len count
        //does this work?
        strcpy(vecLocal+row*MAX_VEC, kmer.get().c_str());
        lenVecLocal[row] = kmer.get().length();
        counter++;

        }


    }

    printf("This should not be 0\n");

    for(int i=0; i < outRows.size(); i++)
    printrow(outRows[i], vecLocal, lenVecLocal);





    //prep output vector
    for (int i =0; i < nnz; i++){
      outLocal[i*MAX_VEC] = 0x20;
      lenOutLocal[i] = 0;
    }

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


    char * vecCuda;

    char * outputCuda;

    uint64_t * lenVecCuda;
    uint64_t * lenOutCuda;

    //allocate space and copy over
    cudaMalloc((void ** )&vecCuda, nnz*MAX_VEC*sizeof(char));
    cudaMalloc((void ** )&outputCuda, nnz*MAX_VEC*sizeof(char));

    cudaMemcpy(vecCuda, vecLocal, nnz*MAX_VEC*sizeof(char), cudaMemcpyHostToDevice);
    cudaMemcpy(outputCuda, outLocal, nnz*MAX_VEC*sizeof(char), cudaMemcpyHostToDevice);

    cudaMalloc((void **)&lenVecCuda, nnz*sizeof(uint64_t));
    cudaMalloc((void **)&lenOutCuda, nnz*sizeof(uint64_t));


    //copy over sizing pointers
    cudaMemcpy(lenVecCuda, lenVecLocal, nnz*sizeof(uint64_t), cudaMemcpyHostToDevice);
    cudaMemcpy(lenOutCuda, lenOutLocal, nnz*sizeof(uint64_t), cudaMemcpyHostToDevice);

    //print
    //printVec(nnz, vecLocal, lenVecLocal);

    //print outrows


    auto start = std::chrono::high_resolution_clock::now();

    //fill_wrapper(nnz, valA,  rowIndA, colIndA);
    std::cout << "Output rows are : ";

    for(int i=0; i < outRows.size(); i++)
    std::cout << outRows.at(i) << ' ';
    cout << endl;

    //fill_vector(nnz, vecB);

    //print before
    // cout << "Before" << endl;
    // printVec(2, vecLoc, lenVec);

    //rn matrix in csc format

    //fun flips
    bool isChanging = true;
    uint64_t oldLen = 0;
    uint64_t newLen = 0;
    uint64_t iters = 0;

    //while (isChanging){
    for (int i =0; i< 4000; i++){

      iters++;

      semiring_spmv(nnz, matValsCuda,matRowsCuda,matColsCuda, vecCuda, lenVecCuda, outputCuda, lenOutCuda);


      newLen = arr_max(lenOutCuda, nnz);
      //cout << "Iteration " << iters << ", prevLen of " << oldLen << ", newLen " << newLen << "." << endl;

      if (newLen <= oldLen){
        isChanging = false;
      }
      oldLen = newLen;

      //swap these bois
      char * temp = outputCuda;
      uint64_t * tempLens = lenOutCuda;

      outputCuda = vecCuda;
      lenOutCuda = lenVecCuda;

      vecCuda = temp;
      lenVecCuda = tempLens;



    }

    //one more for good measure
    //semiring_spmv(nnz, matValsCuda,matRowsCuda,matColsCuda, vecCuda, lenVecCuda, outputCuda, lenOutCuda);
    char * temp = outputCuda;
    uint64_t * tempLens = lenOutCuda;

    outputCuda = vecCuda;
    lenOutCuda = lenVecCuda;

    vecCuda = temp;
    lenVecCuda = tempLens;



    //copy back
    cudaMemcpy(outLocal, outputCuda,  nnz*MAX_VEC*sizeof(char), cudaMemcpyDeviceToHost);
    cudaMemcpy(lenOutLocal, lenOutCuda, nnz*sizeof(uint64_t), cudaMemcpyDeviceToHost);


    printNull(nnz, outLocal, lenOutLocal);

    auto end = std::chrono::high_resolution_clock::now();

    auto duration = std::chrono::duration_cast<std::chrono::microseconds>( end - start ).count();
    //printVec(nnz, outLocal, lenOutLocal);

    //printLens(nnz, lenOutLocal);

    std::cout << "Filled matrix of " << nnz << " items in " << duration << " microseconds"  << endl;


    //printMat(nnz, matLocal, matRowLocal, matColLocal);
    //and print
    //printVec(2, outLoc, lenOut);
    // cout << outRows.size() << "End Contigs:" << endl;
    std::ofstream fout("output.dat");
    for (int i =0; i < outRows.size(); i++){

      uint64_t row = outRows.at(i);
      for (int j = 0; j < lenOutLocal[row]; j++){
        fout << outLocal[row*MAX_VEC+j];
      }
      fout << endl;

      cout << "Done" << endl;
    }
    fout.close();

    cout << "Saved to " << "output.dat" << endl;

    printrow(75, outLocal, lenOutLocal);

    return 0;
}
