ifeq (, $(shell which CC))
	CXX = g++
else
	CXX = CC
endif


all: laccuda


cuda_run: cuda_run.cpp test_kernels.hpp test_kernels.o
		nvcc cuda_run.cpp test_kernels.o -o cuda_run -std=c++17  -O3 -lstdc++ -lcuda

test_kernels.o:
		nvcc -c test_kernels.cu -O3 -o test_kernels.o

simple_semiring_test:
	nvcc simple_semiring_test.cpp test_kernels.o -o simple_semiring -std=c++17  -O3 -lstdc++ -lcuda

semiring_dataset: semiring_dataset.cpp test_kernels.hpp test_kernels.o kmer_t.hpp read_kmers.hpp pkmer_t.hpp packing.hpp
	nvcc semiring_dataset.cpp test_kernels.o -o semiring_dataset -std=c++17  -O3 -lstdc++ -lcuda

laccuda: laccuda.cpp test_kernels.hpp test_kernels.o kmer_t.hpp read_kmers.hpp pkmer_t.hpp packing.hpp
	nvcc laccuda.cpp test_kernels.o -o laccuda -std=c++17  -O3 -lstdc++ -lcuda


clean:
	rm -rf test_kernels.o
	rm -rf cuda_run
	rm -rf simple_semiring
	rm -rf semiring_dataset
	rm -rf laccuda
