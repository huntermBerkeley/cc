
void copy_wrapper(double * to_copy, double* items, size_t n);

//void fill_wrapper(int nnz, int* vals, int*  rows, int* cols);

void fill_vector(int nnz, int* vector);


void cc(uint64_t nnz, uint64_t num_vert, uint64_t* Arows, uint64_t* Acols, char*Avals, std::vector<uint64_t> outputRows, uint64_t maxOut, char*kmerVals, uint64_t*kmerLens, uint64_t * kmerParents);
