#include <cuda.h>
#include <cuda_runtime.h>
#include <stdio.h>


#define USE_16_BIT	// This works for 8 bit but not 16 (comment this out for 8-bit)


#ifdef USE_16_BIT
	typedef uint16_t T;	// 16-bit
#else
	typedef uint8_t T;	// 8-bit
#endif

#define WIDTH 3
#define HEIGHT 3

__global__ void testKernel(CUsurfObject surf) {
	for (int i=0; i < HEIGHT; ++i) { // y
		for (int j=0; j < WIDTH; ++j) { // x
			T val = surf2Dread<T>(surf, j, i, cudaBoundaryModeClamp);
			printf("x=%d y=%d, surf=%llu val=%d\n", j, i, surf, val);
		}
	}
}

int main(int argc, char** argv) {
	CUarray array = 0;
	CUsurfObject surf = 0;
	CUDA_ARRAY_DESCRIPTOR arrDesc;
	CUDA_RESOURCE_DESC resDesc;
	
	// clear the descriptors
	memset(&arrDesc, 0, sizeof(arrDesc));
	memset(&resDesc, 0, sizeof(resDesc));
	
	// init CUDA
	cudaFree(NULL);
	
	// create an 8 or 16 bit array
	arrDesc.Format = sizeof(T) * 8 == 8 ? CU_AD_FORMAT_UNSIGNED_INT8 : CU_AD_FORMAT_UNSIGNED_INT16;
	arrDesc.Width = WIDTH;
	arrDesc.Height = HEIGHT;
	arrDesc.NumChannels = 1;
	CUresult result = cuArrayCreate(&array, &arrDesc);
	if (result != CUDA_SUCCESS) {
		printf("Failed to create CUDA Array\n");
		return -1;
	}

	// create a surface from the array
	resDesc.resType = CU_RESOURCE_TYPE_ARRAY;
	resDesc.res.array.hArray = array;
	result = cuSurfObjectCreate(&surf, &resDesc);
	if (result != CUDA_SUCCESS) {
		printf("Failed to create Surface\n");
		return -1;
	}
	printf("\nCreated surface %llu\n\n", surf);

	// create some host data to copy to the surface
	T* data = (T*)calloc(WIDTH * HEIGHT, sizeof(T));
	for (int i = 0; i < WIDTH * HEIGHT; ++i) {
		data[i] = i;
		printf("data[%d] = %d\n", i, data[i]);
	}

	// copy data from Host to Surface
	CUDA_MEMCPY2D copyParam;
	memset(&copyParam, 0, sizeof(copyParam));
	int rowBytes = WIDTH * sizeof(T);
	copyParam.dstMemoryType = CU_MEMORYTYPE_ARRAY;
	copyParam.dstArray = array;
	copyParam.srcMemoryType = CU_MEMORYTYPE_HOST;
	copyParam.srcHost = data;
	copyParam.srcPitch = rowBytes;
	copyParam.WidthInBytes = rowBytes;
	copyParam.Height = HEIGHT;
	
	printf("\nUploading Data to Surface\n");
	result = cuMemcpy2D(&copyParam);
	if (result != CUDA_SUCCESS) {
		printf("Failed to copy to surface\n");
		return -1;
	}

	// run the kernel
	testKernel<<<1,1>>>(surf);
	cudaError_t err = cudaDeviceSynchronize();
	if (err == cudaSuccess) {
		printf("\nSuccess!\n");
	} else {
		printf("Kernel failed: %d\n", err);
		return -1;
	}	
}
