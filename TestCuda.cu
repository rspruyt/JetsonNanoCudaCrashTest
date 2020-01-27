#include <cuda.h>
#include <cuda_runtime.h>
#include <unistd.h>
#include <future>
#include <mutex>
#include <stdio.h>

// This works fine with a mutex, but crashes with a sigbus error when not using a mutex
// #define USE_MUTEX

#ifdef USE_MUTEX
std::mutex m;
#endif

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
		int* tempHost = nullptr;
		int* tempDevice = nullptr;

		// 1.) Allocate mapped memory on the host
		printf("*** Main 1.) Allocating (loop = %d)\n", loop++);		
		cudaError_t err = cudaHostAlloc(&tempHost, sizeof(int), cudaHostAllocMapped);
		if (err != cudaSuccess) {
			printf("Failed to cudaHostAlloc()\n");
		}

		// get the device pointer (that is really mapped to the same memory as the host pointer)
		err = cudaHostGetDevicePointer(&tempDevice, tempHost, 0);
		if (err != cudaSuccess) {
			printf("Failed to cudaHostGetDevicePointer()\n");
		}

		// 2.) Set the host pointer to some value and read it
		*tempHost = 50;
		printf("*** Main 2.) Allocated mapped host value: %d\n", *tempHost);

		// 3.) If we copy the Device pointer back to a host value, we should get the same '50' value, 
		//	   since this is mapped memory
		int tempVal = 0;
		err = cudaMemcpy(&tempVal, tempDevice, sizeof(int), cudaMemcpyDeviceToHost);
		if (err != cudaSuccess) {
			printf("Failed to cudaMemcpy () #1\n");
		}
		printf("*** Main 3.) Checking device value: %d\n", tempVal);

		// 4.) Copy new value of '89' from CPU to memory mapped device ptr
		tempVal = 89;
		err = cudaMemcpy(tempDevice, &tempVal, sizeof(int), cudaMemcpyHostToDevice);
		if (err != cudaSuccess) {
			printf("Failed to cudaMemcpy() #2\n");
		}
		
		// reset tempVal
		tempVal = 0;
		
		// copy the device value back to tempVal
		err = cudaMemcpy(&tempVal, tempDevice, sizeof(int), cudaMemcpyDeviceToHost);
		if (err != cudaSuccess) {
			printf("Failed to cudaMemcpy() #3\n");
		}
		printf("*** Main 4.) Copy Back to Device: %d\n", tempVal);

		// 5.) Access tempHost to see if it is 89 as well, since tempDevice and tempHost are the same memory
		printf("*** Main 5.) Host Value: %d\n", *tempHost);

#ifdef USE_MUTEX
		m.unlock();
#endif
		usleep(0);
	}
}
