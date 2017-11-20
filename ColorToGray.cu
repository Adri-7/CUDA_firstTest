#include "cuda_runtime.h"
#include "wb.h"

#define wbCheck(stmt)\
do {\
	cudaError_t err = stmt;\
	if (err != cudaSuccess) {\
		wbLog(ERROR, "Failed to run stmt ", #stmt);\
		wbLog(ERROR, "Got CUDA error ... ", cudaGetErrorString(err));\
		return -1;\
	}\
} while (0)\

__global__ void colorToGray(float *input, float *output, int width, int height) {
	int x = threadIdx.x + blockIdx.x * blockDim.x;
  int y = threadIdx.y + blockIdx.y * blockDim.y;

	if ( x < width && y < height){
		int i = x + y * width;

		int r = input[3*i];
		int g = input[3*i+1];
		int b = input[3*i+2];

		output[i] = (0.21*r+0.71*g+0.07*b);
	}
}

int main(int argc, char *argv[]) {
	wbArg_t args;
	int imageChannels;
	int imageWidth;
	int imageHeight;
	char *inputImageFile;
	wbImage_t inputImage;
	wbImage_t outputImage;
	float *hostInputImageData;
	float *hostOutputImageData;
	float *deviceInputImageData;
	float *deviceOutputImageData;
	args = wbArg_read(argc, argv); /* parse the input arguments */
	inputImageFile = wbArg_getInputFile(args, 0);
	inputImage = wbImport(inputImageFile);
	imageWidth = wbImage_getWidth(inputImage);
	imageHeight = wbImage_getHeight(inputImage);
	// For this lab the value is always 3
	imageChannels = wbImage_getChannels(inputImage);
	// Since the image is monochromatic, it only contains one channel
	outputImage = wbImage_new(imageWidth, imageHeight, 1);
	hostInputImageData = wbImage_getData(inputImage);
	hostOutputImageData = wbImage_getData(outputImage);
	wbTime_start(GPU, "Doing GPU Computation (memory + compute)");
	wbTime_start(GPU, "Doing GPU memory allocation");
	cudaMalloc((void **)&deviceInputImageData,
		imageWidth * imageHeight * imageChannels * sizeof(float));
	cudaMalloc((void **)&deviceOutputImageData,
		imageWidth * imageHeight * sizeof(float));
	wbTime_stop(GPU, "Doing GPU memory allocation");
	wbTime_start(Copy, "Copying data to the GPU");
	cudaMemcpy(deviceInputImageData, hostInputImageData,
		imageWidth * imageHeight * imageChannels * sizeof(float),
		cudaMemcpyHostToDevice);
	wbTime_stop(Copy, "Copying data to the GPU");
	wbTime_start(Compute, "Doing the computation on the GPU");

	dim3 blockDim(16,16,1);
  dim3 gridDim(imageWidth/blockDim.x+1, imageHeight/blockDim.y+1,1);

	colorToGray<<<gridDim, blockDim>>>(deviceInputImageData, deviceOutputImageData, imageWidth, imageHeight);

	wbTime_stop(Compute, "Doing the computation on the GPU");

	wbTime_start(Copy, "Copying data from the GPU");
	cudaMemcpy(hostOutputImageData, deviceOutputImageData,
		imageWidth * imageHeight * sizeof(float),
		cudaMemcpyDeviceToHost);
	wbTime_stop(Copy, "Copying data from the GPU");
	wbTime_stop(GPU, "Doing GPU Computation (memory + compute)");


	int i, j;
	FILE *fp = fopen("first.ppm", "wb"); /* b - binary mode */
	(void) fprintf(fp, "P6\n%d %d\n255\n", imageWidth, imageHeight);
	for (j = 0; j < imageHeight; ++j)
	{
			for (i = 0; i < imageWidth; ++i)
			{
					static unsigned char color[3];
					color[0] = hostOutputImageData[i+j*imageWidth]* 255;  /* red */
					color[1] = hostOutputImageData[i+j*imageWidth]* 255;  /* green */
					color[2] = hostOutputImageData[i+j*imageWidth]* 255;  /* blue */

					(void) fwrite(color, 1, 3, fp);
			}
	}
	(void) fclose(fp);


	cudaFree(deviceInputImageData);
	cudaFree(deviceOutputImageData);
	wbImage_delete(outputImage);
	wbImage_delete(inputImage);
	return 0;
}
