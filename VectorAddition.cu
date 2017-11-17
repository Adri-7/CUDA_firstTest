
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <stdio.h>

#include "wb.h"

__global__ void vecAdd(float *in1, float *in2, float *out, int len) {
	int i = threadIdx.x + blockDim.x * blockIdx.x;

	out[i] = in1[i] + in2[i];
}

int main(int argc, char **argv) {
	//vars
	wbArg_t args;
	int inputLength;
	float *hostInput1;
	float *hostInput2;
	float *hostOutput;
	float *deviceInput1;
	float *deviceInput2;
	float *deviceOutput;
	cudaError_t cudaStatus;

	//Input args
	args = wbArg_read(argc, argv);
	wbTime_start(Generic, "Importing data and creating memory on host");
	hostInput1 =
		(float *)wbImport(wbArg_getInputFile(args, 0), &inputLength);
	hostInput2 =
		(float *)wbImport(wbArg_getInputFile(args, 1), &inputLength);
	hostOutput = (float *)malloc(inputLength * sizeof(float));
	wbTime_stop(Generic, "Importing data and creating memory on host");
	wbLog(TRACE, "The input length is ", inputLength);

	//GPU memory allocation
	wbTime_start(GPU, "Allocating GPU memory.");

	cudaStatus = cudaMalloc(&deviceInput1, inputLength * sizeof(float));
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMalloc failed!");
		goto Error;
	}

	cudaStatus = cudaMalloc(&deviceInput2, inputLength * sizeof(float));
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMalloc failed!");
		goto Error;
	}

	cudaStatus = cudaMalloc(&deviceOutput, inputLength * sizeof(float));
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMalloc failed!");
		goto Error;
	}

	wbTime_stop(GPU, "Allocating GPU memory.");

	//Memory copy to GPU
	wbTime_start(GPU, "Copying input memory to the GPU.");

	cudaStatus = cudaMemcpy(deviceInput1, hostInput1, inputLength * sizeof(float), cudaMemcpyHostToDevice);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMemcpy failed!");
		goto Error;
	}

	cudaStatus = cudaMemcpy(deviceInput2, hostInput2, inputLength * sizeof(float), cudaMemcpyHostToDevice);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMemcpy failed!");
		goto Error;
	}

	wbTime_stop(GPU, "Copying input memory to the GPU.");

	// Launch computations
	wbTime_start(Compute, "Performing CUDA computation");

	vecAdd<<<1, 256>>>(deviceInput1, deviceInput2, deviceOutput, inputLength);

	cudaDeviceSynchronize();
	wbTime_stop(Compute, "Performing CUDA computation");
	
	//Copying result from device to host
	wbTime_start(Copy, "Copying output memory to the CPU");

	cudaStatus = cudaMemcpy(hostOutput, deviceOutput, inputLength * sizeof(float), cudaMemcpyDeviceToHost);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMemcpy failed!");
		goto Error;
	}

	wbTime_stop(Copy, "Copying output memory to the CPU");

Error:
	wbTime_start(GPU, "Freeing GPU Memory");

	cudaFree(deviceInput1);
	cudaFree(deviceInput2);
	cudaFree(deviceOutput);

	wbTime_stop(GPU, "Freeing GPU Memory");
	wbSolution(args, hostOutput, inputLength);

	free(hostInput1);
	free(hostInput2);
	free(hostOutput);

	return cudaStatus;
}