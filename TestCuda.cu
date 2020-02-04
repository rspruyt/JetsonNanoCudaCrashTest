#include <cuda.h>
#include <cuda_runtime.h>
#include <unistd.h>
#include <future>
#include <mutex>
#include <stdio.h>

// For testing 8-bit and 16-bit surfaces

typedef uint16_t T;
//typedef uint8_t T;

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

int main(int argc, char** argv) {
	CUresult result;
	CUDA_ARRAY_DESCRIPTOR arrDesc;
	CUDA_RESOURCE_DESC resDesc;
	CUarray array = 0;
	CUsurfObject surf = 0;

	// initializes cuda
	cudaFree(NULL);
	printf("\nUsing T size: %d bytes\n\n", (int)sizeof(T));

	memset(&arrDesc, 0, sizeof(arrDesc));
	memset(&resDesc, 0, sizeof(resDesc));

	// initialize data
	int width = 5;
	int height = 5;

	T* data = (T*)calloc(width * height, sizeof(T));
	T* down = (T*)calloc(width * height, sizeof(T));
	
	for (int i = 0; i < width * height; ++i) {
		data[i] = i * 10;
		printf("data[%d] = %d\n", i, data[i]);
	}

	// create cuda array
	arrDesc.Format = 8 * sizeof(T) <= 8 ? CU_AD_FORMAT_UNSIGNED_INT8 : CU_AD_FORMAT_UNSIGNED_INT16;
	arrDesc.Width = width;
	arrDesc.Height = height;
	arrDesc.NumChannels = 1;
	result = cuArrayCreate(&array, &arrDesc);
	if (result != CUDA_SUCCESS) {
		printf("Failed to create CUDA Array\n");
	}
	resDesc.resType = CU_RESOURCE_TYPE_ARRAY;
	resDesc.res.array.hArray = array;
	result = cuSurfObjectCreate(&surf, &resDesc);
	if (result != CUDA_SUCCESS) {
		printf("Failed to cuSurfObjectCreate\n");
	}

	// Copy from Host to Surface
	CUDA_MEMCPY2D copyParam;
	memset(&copyParam, 0, sizeof(copyParam));
	int rowBytes = width * sizeof(T);
	copyParam.dstMemoryType = CU_MEMORYTYPE_ARRAY;
	copyParam.dstArray = array;
	copyParam.srcMemoryType = CU_MEMORYTYPE_HOST;
	copyParam.srcHost = data;
	copyParam.srcPitch = rowBytes;
	copyParam.WidthInBytes = rowBytes;
	copyParam.Height = height;
	
	printf("\nUploading Data to Surface\n");
	result = cuMemcpy2D(&copyParam);
	if (result != CUDA_SUCCESS) {
		printf("Failed to copy to surface\n");
	}

	// Copy from surface back to Host	
	CUDA_MEMCPY2D copy2Param;
	memset(&copy2Param, 0, sizeof(copy2Param));
	copy2Param.srcMemoryType = CU_MEMORYTYPE_ARRAY;
	copy2Param.srcArray = array;
	copy2Param.dstMemoryType = CU_MEMORYTYPE_HOST;
	copy2Param.dstHost = down;
	copy2Param.dstPitch = rowBytes;
	copy2Param.WidthInBytes = rowBytes;
	copy2Param.Height = height;
	printf("Download Data from Surface\n");
	result = cuMemcpy2D(&copy2Param);
	if (result != CUDA_SUCCESS) {
		printf("Failed to copy from surface\n");
	}

	printf("\n");
	for (int i = 0; i < width * height; ++i) {
		printf("down[%d] = %d\n", i, down[i]);
	}
}
