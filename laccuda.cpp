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
#include <map>

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
#define MAX_VEC 8000
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

//serial code to construct a 'perfect ordering'
//input - list of kmers
//output - a filled adjacency matrix representing the debruijn graph

//for each kmer:
//if in dict
std::vector<std::pair<kmer_pair, uint64_t>> build_perf_adj_mat(std::vector<kmer_pair> kmers, uint64_t * nnz, char ** vals, uint64_t ** rows, uint64_t ** cols){

  std::ofstream fout;
  fout.open("serial_solution.dat");



  //main kmer counter - at the end of all of this it should equal kmers.size;
  uint64_t counter = 0;

  //the two maps we'll need: kmer->kmer_pair, and kmer->uint64_t
  std::map<std::string, kmer_pair> kmer_to_kmer;
  std::map<std::string, uint64_t> kmer_to_num;

  uint64_t _nnz = 0;

  //start by adding all kmers to map from pkmer_t - kmer_pair
  //no this is not very efficient O(log n) vs O(1), but this is for debugging only
  //The original issue I suspect stems from something funky with a hash of a hash,
  //so I would like to avoid that as much as possible.
  for (uint64_t i = 0; i < kmers.size(); i++){
    //printf("current: %s\n", kmers[i].kmer.get().c_str());
    kmer_to_kmer[kmers[i].kmer.get()] = kmers[i];

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
    if (next_kmer.backwardExt() != 'F') continue;


    //Start of building a contig

    //this section is a repeat because i need the first section
    // uint64_t to build the starts vector
    if (kmer_to_num.count(next_kmer.kmer.get()) == 0){
        kmer_to_num[next_kmer.kmer.get()] = counter;
        counter++;
    }

    //use value stored in map
    starts.push_back(std::make_pair(next_kmer, slot));

    //start file write!
    fout << next_kmer.kmer.get();


    //start of the main loop
    while (next_kmer.forwardExt() != 'F'){

      fout << next_kmer.forwardExt();

      //printf("Working on pair: %s, %s\n", next_kmer.kmer.get().c_str(), next_kmer.next_kmer().get().c_str());

      //get uints for this kmer -> neighbor, insert, and then
      uint64_t my_val;
      uint64_t next_val;

      if (kmer_to_num.count(next_kmer.kmer.get()) == 0){
          kmer_to_num[next_kmer.kmer.get()] = counter;
          counter++;
      }

      //use value stored in map
      my_val = kmer_to_num[next_kmer.kmer.get()];

      if (kmer_to_num.count(next_kmer.next_kmer().get()) == 0){
          kmer_to_num[next_kmer.next_kmer().get()] = counter;
          counter++;
      } else {
        //This implies that a kmer went ahead of me, which I don't think can happen on this dataset
        printf("Spooky bug, this shouldn't happen %s, %llu\n", next_kmer.next_kmer().get().c_str(), kmer_to_num[next_kmer.next_kmer().get()]);
      }

      next_val = kmer_to_num[next_kmer.next_kmer().get()];

      //mats are allocated, insert into next available slot
      assert (slot < _nnz);

      _vals[slot] = next_kmer.forwardExt();
      _rows[slot] = my_val;
      _cols[slot] = next_val;
      slot++;

      //update kmer to the next one in the chain
      next_kmer = kmer_to_kmer[next_kmer.next_kmer().get()];

    }
    fout << endl;




  }

  printf("Perf construct done, counter is %llu\n", counter);
  fflush(stdout);


  //close sample solution
  fout.close();

  //set output
  *vals = _vals;
  *rows = _rows;
  *cols = _cols;

  return starts;

}


//building off of 'perf' adj matrix
//this constructs the matrix from first available
//TODO: parallelize: this should be embarassingly parallel with just a tad of locking
std::vector<std::pair<kmer_pair, uint64_t>> build_adj_mat(std::vector<kmer_pair> kmers, uint64_t * nnz, char ** vals, uint64_t ** rows, uint64_t ** cols){




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


//loop through the starts, take kmer start index and trace through adj matrix
//this function is to assert that the correct information is stored in the adj matrix
//and that regular construction (i.e., random uint64_t values) is viable
void build_kmers_from_adj(std::vector<std::pair<kmer_pair, uint64_t>> starts, uint64_t nnz, char * vals, uint64_t * rows, uint64_t * cols){


  //build map of next connections
  std::map<uint64_t, uint64_t> col_to_index;

  for (uint64_t i=0; i < nnz; i++){
    col_to_index[rows[i]] = i;
  }

  std::ofstream fout;
  fout.open("mat_solution.dat");

  for (uint64_t i =0; i < starts.size(); i++){

    uint64_t slot = std::get<1>(starts.at(i));
    kmer_pair contig_start = std::get<0>(starts.at(i));

    fout << contig_start.kmer.get();

    uint64_t count = 0;
    while (true) {

      //safety check
      //This currently fails?!?
      //issue with the main section
      assert(slot < nnz);

      fout << vals[slot];

      uint64_t col = cols[slot];

      //check count here - if dne done

      if (col_to_index.count(col) == 0){
        break;
      }
      slot = col_to_index[col];
      assert (rows[slot] == col);
      count += 1;

    }

    //goofy carriage return print
    cout << "kmer " << i+1 << " / " << starts.size() << " finished " << '\r' << flush;
    //printf("kmer %d/%d finished with length %llu\n", i+1, starts.size(), count);
    fout << endl;
  }
  cout << endl;
  fout.close();
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


//take the starts from a perf construction and insert them into cuda mats
void prep_starts(std::vector<std::pair<kmer_pair, uint64_t>> starts, uint64_t * rows, uint64_t * startsNnz, char ** startVals, uint64_t** startLens, uint64_t ** startRows){


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

void copy_to_cuda(uint64_t nnz, char * originalVals, uint64_t * originalRows, uint64_t* originalCols, char** newVals, uint64_t ** newRows, uint64_t ** newCols){

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
std::vector<uint64_t> gen_outRows(std::vector<std::pair<kmer_pair, uint64_t>> starts, uint64_t * rows){

  std::vector<uint64_t> outRows;

  for (int i = 0; i < starts.size(); i++){

    uint64_t slot = std::get<1>(starts.at(i));

    outRows.push_back(rows[slot]);
  }

  return outRows;


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


    std::map<string, size_t> map;

    //additional map to assert that the kmer pairs aren't overlapping
    std::map<uint64_t, kmer_pair> err_map;

    //load kmers
    std::vector<kmer_pair> kmers = read_kmers(kmer_fname);

    char* perfvals;
    uint64_t * perfrows;
    uint64_t * perfcols;
    uint64_t perf_nnz;

    std::vector<std::pair<kmer_pair, uint64_t>> perf_starts = build_adj_mat(kmers, &perf_nnz, &perfvals, &perfrows, &perfcols);

    //print out some samples

    for (int i =0; i < 10; i++){
      printf("%llu -> %llu: %c\n", perfrows[i], perfcols[i], perfvals[i]);
    }

    //now run test
    //this is successful for perf runtime
    build_kmers_from_adj(perf_starts, perf_nnz, perfvals, perfrows, perfcols);


    //now construct starts
    uint64_t startNnz;
    char * startVals;
    uint64_t * startLens;
    uint64_t * startRows;

    //fill starts mats for cuda
    prep_starts(perf_starts, perfrows, &startNnz, &startVals, &startLens, &startRows);

    //check output for verify
    //on test case looks good
    printf("Visual sanity check on starts\n");
    int min_size = 10;
    if (perf_starts.size() < min_size){
      min_size = perf_starts.size();
    }
    for (int i=0; i < min_size; i++){
      cout << i << ": " << std::get<0>(perf_starts.at(i)).kmer_str() << endl;

      cout << i << ": ";
      for (int j = 0; j < 10; j++){
        cout << startVals[i*MAX_VEC+j];
      }
      cout << endl;
    }

    std::vector<uint64_t> outRows2 = gen_outRows(perf_starts, perfrows);

    //what info needs to be updated for the next pass?
    //sync just in case
    cudaDeviceSynchronize();


    char* perfvalsCuda;
    uint64_t * perfrowsCuda;
    uint64_t * perfcolsCuda;

    copy_to_cuda(perf_nnz, perfvals, perfrows, perfcols, &perfvalsCuda, &perfrowsCuda, &perfcolsCuda);

    //connected components call
    iterative_cuda_solver(perf_nnz, n_kmers, perfrowsCuda, perfcolsCuda, perfvalsCuda, outRows2, startNnz, startVals, startLens, startRows);
    return 0;


    //iterate and count
    size_t counter = 0;
    for (size_t i = 0; i < kmers.size(); i++){
      pkmer_t kmer = kmers.at(i).kmer;


      //IDEA: Could these be counted more than once???

      //hash of a hash is funky?


      if (map.count(kmer.get()) != 0){
        cout << "Repeat hit, these should be cleaned" << endl;
        cout << "Original: " << err_map[kmer.hash()].kmer_str() << " "<< err_map[kmer.hash()].forwardExt() << endl;
        cout << "New: " << kmers.at(i).kmer_str() << " "<< kmers.at(i).forwardExt() << endl;
        continue;
      } else {
        map[kmer.get()] = counter;
        err_map[kmer.hash()] = kmers.at(i);
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

      //valid edge
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
      uint64_t row = map.at(kmer.get());

      pkmer_t forward = kmers.at(i).next_kmer();
      //uint64_t fhash = forward.hash();

      //fhash is the next kmer
      if (map.count(forward.get()) == 0){



        cout << "end at " <<  map.at(kmer.get()) << endl;

        continue;

      } else {

        //positive match, add to matrix
        //csr format
        matLocal[counter] = kmers.at(i).forwardExt();
        //was originall row/col
        matRowLocal[counter] = row;
        matColLocal[counter] = map.at(forward.get());

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

    char * kmerVals;
    uint64_t * kmerLens;
    uint64_t * kmerParents;

    cudaMallocManaged((void **)&kmerVals,maxOut*MAX_VEC*sizeof(char));

    cudaMallocManaged((void **)&kmerLens,maxOut*sizeof(uint64_t));

    cudaMallocManaged((void **)&kmerParents,maxOut*sizeof(uint64_t));


    //iterate through outKmers

    for(int i=0; i < outKmers.size(); i++){

      pkmer_t kmer = outKmers.at(i).kmer;
      for(int j=0; j < kmer.get().size(); j++){

        //index into the results
        kmerVals[i*MAX_VEC+j] = kmer.get()[j];



      }
      kmerLens[i] = kmer.get().size();

      //set parent via cond hook
      kmerParents[i] = map.at(kmer.get());


    }




    //fun flips
    cc(nnz, num_vert, matRowsCuda, matColsCuda, matValsCuda, outRows, maxOut, kmerVals, kmerLens, kmerParents);


    //passing  along kmers is a shitshow
    //i guess build a managed list of chars/lens
    //and pass  it along


    //copy back
    //cudaMemcpy(outLocal, components,  nnz*sizeof(uint64_t), cudaMemcpyDeviceToHost);


    auto end = std::chrono::high_resolution_clock::now();

    auto duration = std::chrono::duration_cast<std::chrono::microseconds>( end - start ).count();
    //printVec(nnz, outLocal, lenOutLocal);

    //printLens(nnz, lenOutLocal);



    std::cout << "Filled matrix of " << nnz << " items in " << duration << " microseconds"  << endl;


    //printMat(nnz, matLocal, matRowLocal, matColLocal);
    //and print
    //printVec(2, outLoc, lenOut);


    cout << "Saved to " << "output.dat" << endl;


    return 0;
}
