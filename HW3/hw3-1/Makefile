NVFLAGS  := -std=c++11 -O3 -Xptxas="-v" -arch=sm_61 
LDFLAGS  := -lm
XCFLAGS := -fopenmp
CC = gcc
CXX = g++
CXXFLAGS = -O3 -pthread
CFLAGS = -O3 -lm -pthread

EXES  := hw3-1 hw3-2 hw3-3

all: $(EXES)

clean:
	rm -f $(EXES)

seq: seq.cc
	$(CXX) $(CXXFLAGS) -o $@ $?

hw3-1: hw3-1.cc
	$(CXX) $(CXXFLAGS) -o $@ $?

hw3-2: hw3-2.cu
	nvcc $(NVFLAGS) -o $@ $?

hw3-3: hw3-3.cu
	nvcc $(NVFLAGS) $(LDFLAGS) -Xcompiler="$(XCFLAGS)" -o $@ $?



