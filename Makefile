all: 
	/usr/local/cuda-10.0/bin/nvcc TestCuda.cu -gencode arch=compute_53,code=sm_53 -lcuda -o TestCuda $(NVCCFLAGS)
	
