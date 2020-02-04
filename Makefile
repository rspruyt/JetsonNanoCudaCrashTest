all: 
	/usr/local/cuda-10.0/bin/nvcc TestCuda.cu -gencode arch=compute_53,code=sm_53 -o TestCuda $(NVCCFLAGS) -I -I/usr/local/cuda-10.0/targets/aarch64-linux/include -g -O0 -lcuda
	
