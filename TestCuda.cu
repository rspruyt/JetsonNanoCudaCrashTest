#include <cuda.h>
#include <cuda_runtime.h>
#include <unistd.h>
#include <future>
#include <mutex>
#include <stdio.h>

__global__ void testKernel() {
	printf("Thread Kernel running\n");
}

void testCuda() {
	testKernel<<<1,1>>>();
	cudaError_t err = cudaDeviceSynchronize();
	if (err != cudaSuccess) {
		printf("SYNC FAILED\n\n\n");
	}
}

std::mutex m;

// This works fine with a mutex, but crashes with a sigbus error when not using a mutex
//#define USE_MUTEX

struct MyThread {
	void run() {
		int threadLoop = 0;
		while(1) {
#ifdef USE_MUTEX
			m.lock();
#endif
			printf("Thread Run (loop %d)\n", threadLoop++);
			// run kernel
			testCuda();
#ifdef USE_MUTEX
			m.unlock();
#endif
			usleep(0);
		}
	}
};

int main(int argc, char** argv) {
	MyThread thread;
	auto threadFuture = std::async(std::launch::async, &MyThread::run, thread);
	int loop = 0;
	while(1){
#ifdef USE_MUTEX
		m.lock();
#endif
		int* temp = nullptr;
		printf("*** Main Allocating (loop = %d)\n", loop++);
		cudaError_t err = cudaMallocManaged(&temp, sizeof(int));
		if (err != cudaSuccess) {
			printf("Failed to cudaMallocManaged()\n");
			return -1;
		}
		*temp = 0;	// <-- SIGBUS occurs here if don't use a mutex
		printf("*** Main Finished Allocating value: %d\n", *temp);
#ifdef USE_MUTEX
		m.unlock();
#endif
		usleep(0);
	}
}