#include <cuda.h>
#include <cuda_runtime.h>
#include <unistd.h>
#include <stdio.h>
#include <string>

// This works fine with a mutex, but crashes with a sigbus error when not using a mutex
// #define USE_MUTEX

void printFreeMem(const std::string& prefix, size_t& free) {
	size_t total = 0;
	cudaError_t err = cudaMemGetInfo(&free, &total);
	if (err == cudaSuccess) {
		printf("%s: %llu bytes = %d%% free\n", prefix.c_str(), (long long int)free, (int)(((float)free / total)*100));
	} else {
		printf("!!!! Failed to cudaMemGetInfo(), err: %d\n", err);
	}
}

int main(int argc, char** argv) {
	for (int i = 0; i < 20; ++i) {
		size_t freeBefore = 0;
		printFreeMem("Test " + std::to_string(i) + " - before", freeBefore);
		// 10,000 x 10,000 byts = ~ 100 MB
		CUDA_ARRAY_DESCRIPTOR arrDesc;
		memset(&arrDesc, 0, sizeof(arrDesc));

		arrDesc.Format = CU_AD_FORMAT_UNSIGNED_INT8;
		arrDesc.Width = 10000;
		arrDesc.Height = 10000;
		arrDesc.NumChannels = 1;
		CUarray cuArr;
		CUresult result = cuArrayCreate(&cuArr, &arrDesc);
		if (result != CUDA_SUCCESS) {
			printf("!!!! cuArrayCreate() failed\n");
			return -1;
		}
		size_t freeAfter = 0;
		printFreeMem("Test " + std::to_string(i) + " - after", freeAfter);
		printf("diff: = %d\n\n", (int)(freeBefore - freeAfter));
	}

	return 0;
}