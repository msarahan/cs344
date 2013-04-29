// Homework 2
// Image Blurring
//
// In this homework we are blurring an image. To do this, imagine that we have
// a square array of weight values. For each pixel in the image, imagine that we
// overlay this square array of weights on top of the image such that the center
// of the weight array is aligned with the current pixel. To compute a blurred
// pixel value, we multiply each pair of numbers that line up. In other words, we
// multiply each weight with the pixel underneath it. Finally, we add up all of the
// multiplied numbers and assign that value to our output for the current pixel.
// We repeat this process for all the pixels in the image.

// To help get you started, we have included some useful notes here.

//****************************************************************************

// For a color image that has multiple channels, we suggest separating
// the different color channels so that each color is stored contiguously
// instead of being interleaved. This will simplify your code.

// That is instead of RGBARGBARGBARGBA... we suggest transforming to three
// arrays (as in the previous homework we ignore the alpha channel again):
//  1) RRRRRRRR...
//  2) GGGGGGGG...
//  3) BBBBBBBB...
//
// The original layout is known an Array of Structures (AoS) whereas the
// format we are converting to is known as a Structure of Arrays (SoA).

// As a warm-up, we will ask you to write the kernel that performs this
// separation. You should then write the "meat" of the assignment,
// which is the kernel that performs the actual blur. We provide code that
// re-combines your blurred results for each color channel.

//****************************************************************************

// You must fill in the gaussian_blur kernel to perform the blurring of the
// inputChannel, using the array of weights, and put the result in the outputChannel.

// Here is an example of computing a blur, using a weighted average, for a single
// pixel in a small image.
//
// Array of weights:
//
//  0.0  0.2  0.0
//  0.2  0.2  0.2
//  0.0  0.2  0.0
//
// Image (note that we align the array of weights to the center of the box):
//
//    1  2  5  2  0  3
//       -------
//    3 |2  5  1| 6  0       0.0*2 + 0.2*5 + 0.0*1 +
//      |       |
//    4 |3  6  2| 1  4   ->  0.2*3 + 0.2*6 + 0.2*2 +   ->  3.2
//      |       |
//    0 |4  0  3| 4  2       0.0*4 + 0.2*0 + 0.0*3
//       -------
//    9  6  5  0  3  9
//
//         (1)                         (2)                 (3)
//
// A good starting place is to map each thread to a pixel as you have before.
// Then every thread can perform steps 2 and 3 in the diagram above
// completely independently of one another.

// Note that the array of weights is square, so its height is the same as its width.
// We refer to the array of weights as a filter, and we refer to its width with the
// variable filterWidth.

//****************************************************************************

// Your homework submission will be evaluated based on correctness and speed.
// We test each pixel against a reference solution. If any pixel differs by
// more than some small threshold value, the system will tell you that your
// solution is incorrect, and it will let you try again.

// Once you have gotten that working correctly, then you can think about using
// shared memory and having the threads cooperate to achieve better performance.

//****************************************************************************

// Also note that we've supplied a helpful debugging function called checkCudaErrors.
// You should wrap your allocation and copying statements like we've done in the
// code we're supplying you. Here is an example of the unsafe way to allocate
// memory on the GPU:
//
// cudaMalloc(&d_red, sizeof(unsigned char) * numRows * numCols);
//
// Here is an example of the safe way to do the same thing:
//
// checkCudaErrors(cudaMalloc(&d_red, sizeof(unsigned char) * numRows * numCols));
//
// Writing code the safe way requires slightly more typing, but is very helpful for
// catching mistakes. If you write code the unsafe way and you make a mistake, then
// any subsequent kernels won't compute anything, and it will be hard to figure out
// why. Writing code the safe way will inform you as soon as you make a mistake.

// Finally, remember to free the memory you allocate at the end of the function.

//****************************************************************************

#include "utils.h"

__global__
void gaussian_blur(const uchar4* const inputImageRGBA,
                   uchar4* const outputImageRGBA,
                   int numRows, int numCols,
                   const float* const filter, const int filterWidth)
{ 
	// NOTE: Be sure to compute any intermediate results in floating point
	// before storing the final result as unsigned char.

	// NOTE: Be careful not to try to access memory that is outside the bounds of
	// the image. You'll want code that performs the following check before accessing
	// GPU memory:

	int absolute_image_position_x = blockIdx.x*blockDim.x + threadIdx.x;
	int absolute_image_position_y = blockIdx.y*blockDim.y + threadIdx.y;
	int absolute_ptr_offset = absolute_image_position_y * numCols + absolute_image_position_x;
    
	int filter_size = filterWidth*filterWidth;
	extern __shared__ float sh_filter[];
	// copy filter into shared memory for this block
	int local_offset = blockDim.x*threadIdx.y+threadIdx.x;
	if (local_offset < filter_size)
	{
		sh_filter[local_offset] = filter[local_offset];
		//sh_filter[threadIdx.x] = filter[threadIdx.x];
	}
      
	__syncthreads();

	if ( absolute_image_position_x >= numCols ||
		absolute_image_position_y >= numRows )
	{
		return;
	}

	int image_r, image_c;
	float filter_value;
	float4 buffer ={ 0.0f, 0.0f, 0.0f, 1.0f};
    
	//#pragma unroll
	for (int filter_r = -filterWidth/2; filter_r <= filterWidth/2; ++filter_r) 
	{
		//#pragma unroll
        for (int filter_c = -filterWidth/2; filter_c <= filterWidth/2; ++filter_c) 
		{
			//Find the global image position for this filter position
			//clamp to boundary of the image
			image_r = min(max(absolute_image_position_y + filter_r, 0), (numRows - 1));
			image_c = min(max(absolute_image_position_x + filter_c, 0), (numCols - 1));

			// use shared memory filter
			filter_value = sh_filter[(filter_r + filterWidth/2) * filterWidth + filter_c + filterWidth/2];
			// use global memory filter
			//filter_value = filter[(filter_r + filterWidth/2) * filterWidth + filter_c + filterWidth/2];

			buffer.x += inputImageRGBA[image_r * numCols + image_c].x * filter_value;
			buffer.y += inputImageRGBA[image_r * numCols + image_c].y * filter_value;
			buffer.z += inputImageRGBA[image_r * numCols + image_c].z * filter_value;
        }
  }
  outputImageRGBA[absolute_ptr_offset].x = (unsigned char) buffer.x;
  outputImageRGBA[absolute_ptr_offset].y = (unsigned char) buffer.y;
  outputImageRGBA[absolute_ptr_offset].z = (unsigned char) buffer.z;
  outputImageRGBA[absolute_ptr_offset].w = (unsigned char) buffer.w;
}

float         *d_filter;

void allocateMemoryAndCopyToGPU(const size_t numRowsImage, const size_t numColsImage,
                                const float* const h_filter, const size_t filterWidth)
{

  //TODO:
  //Allocate memory for the filter on the GPU
  //Use the pointer d_filter that we have already declared for you
  //You need to allocate memory for the filter with cudaMalloc
  //be sure to use checkCudaErrors like the above examples to
  //be able to tell if anything goes wrong
  //IMPORTANT: Notice that we pass a pointer to a pointer to cudaMalloc
  checkCudaErrors(cudaMalloc(&d_filter, sizeof(float)*filterWidth*filterWidth));

  //TODO:
  //Copy the filter on the host (h_filter) to the memory you just allocated
  //on the GPU.  cudaMemcpy(dst, src, numBytes, cudaMemcpyHostToDevice);
  //Remember to use checkCudaErrors!
  checkCudaErrors(cudaMemcpy(d_filter, h_filter, sizeof(float)*filterWidth*filterWidth, cudaMemcpyHostToDevice));

}

void your_gaussian_blur(const uchar4 * const h_inputImageRGBA, uchar4 * const d_inputImageRGBA,
                        uchar4* const d_outputImageRGBA, const size_t numRows, const size_t numCols,
                        const int filterWidth)
{
	int BLOCK_SIZE = 16;
	//TODO: Set reasonable block size (i.e., number of threads per block)
	const dim3 blockSize = dim3(BLOCK_SIZE,BLOCK_SIZE,1);

	//TODO:
	//Compute correct grid size (i.e., number of blocks per kernel launch)
	//from the image size and and block size.
	const dim3 gridSize = dim3(numCols/blockSize.x+1, numRows/blockSize.y+1, 1);

	//TODO: Call your convolution kernel here 3 times, once for each color channel.
	gaussian_blur<<<gridSize, blockSize, sizeof(float)*filterWidth*filterWidth>>>(d_inputImageRGBA,
                                             d_outputImageRGBA,
                                             numRows,
                                             numCols,
											 d_filter, 
											 filterWidth);

	// Again, call cudaDeviceSynchronize(), then call checkCudaErrors() immediately after
	// launching your kernel to make sure that you didn't make any mistakes.
	cudaDeviceSynchronize(); checkCudaErrors(cudaGetLastError());
}


//Free all the memory that we allocated
//TODO: make sure you free any arrays that you allocated
void cleanup() {
	checkCudaErrors(cudaFree(d_filter));
}
