
void copy_wrapper(double * to_copy, double* items, size_t n);

//void fill_wrapper(int nnz, int* vals, int*  rows, int* cols);

void fill_vector(int nnz, int* vector);


void cc(uint64_t nnz, uint64_t num_vert, uint64_t* Arows, uint64_t* Acols, char*Avals, std::vector<uint64_t> outputRows, uint64_t maxOut, char*kmerVals, uint64_t*kmerLens, uint64_t * kmerParents);

//testing solver
void iterative_cuda_solver(uint64_t nnz, uint64_t num_vert, uint64_t* Arows, uint64_t* Acols, char* Avals, std::vector<uint64_t> outputRows, uint64_t maxOut, char*kmerVals, uint64_t*kmerLens, uint64_t * kmerParents);

//new solver - builds absolute position from end, then reverses and inserts into buffer
void cc_len(uint64_t nnz, uint64_t num_vert, uint64_t* Arows, uint64_t* Acols, char* Avals, std::vector<uint64_t> outputRows, uint64_t maxOut, char*kmerVals, uint64_t*kmerLens, uint64_t * kmerParents);


int cudaMain(int argc, char** argv);