NVCC = /usr/local/cuda/bin/nvcc
CXX = g++
CUDA_FLAGS = -I/usr/local/cuda/include -L/usr/local/cuda/lib64
CUDA_LIBS = -lcuda -lcudart

.PHONY: all clean test

all: libhec_bridge.so

libhec_bridge.so: hec_bridge.cpp atlas_kernels.ptx
	$(CXX) -shared -fPIC -O3 $(CUDA_FLAGS) hec_bridge.cpp -o $@ $(CUDA_LIBS)

test: test_bridge
	./test_bridge

test_bridge: test_bridge.cpp libhec_bridge.so
	$(CXX) -O3 $(CUDA_FLAGS) test_bridge.cpp -L. -lhec_bridge -o $@ $(CUDA_LIBS)

clean:
	rm -f *.so *.o test_bridge

status:
	nvidia-smi --query-gpu=index,name,memory.free --format=csv
