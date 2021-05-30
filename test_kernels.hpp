
void copy_wrapper(double * to_copy, double* items, size_t n);

void fill_wrapper(int nnz, int* vals, int*  rows, int* cols);

void fill_vector(int nnz, int* vector);

void spmv(uint64_t nnz, int* matVals, int* matRows, int* matCols, int* vecB, int * output);

void semiring_spmv(uint64_t nnz, char* matVals, uint64_t* matRows, uint64_t* matCols, char* vecB, uint64_t* vecLens, char * output, uint64_t* outLens);

uint64_t arr_max(uint64_t* lens, uint64_t n);


void cc(uint64_t nnz, uint64_t num_vert, uint64_t* Arows, uint64_t* Acols, char*Avals, std::vector<uint64_t> outputRows);
