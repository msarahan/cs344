/* Udacity Homework 3
   HDR Tone-mapping

  Background HDR
  ==============

  A High Definition Range (HDR) image contains a wider variation of intensity
  and color than is allowed by the RGB format with 1 byte per channel that we
  have used in the previous assignment.  

  To store this extra information we use single precision floating point for
  each channel.  This allows for an extremely wide range of intensity values.

  In the image for this assignment, the inside of church with light coming in
  through stained glass windows, the raw input floating point values for the
  channels range from 0 to 275.  But the mean is .41 and 98% of the values are
  less than 3!  This means that certain areas (the windows) are extremely bright
  compared to everywhere else.  If we linearly map this [0-275] range into the
  [0-255] range that we have been using then most values will be mapped to zero!
  The only thing we will be able to see are the very brightest areas - the
  windows - everything else will appear pitch black.

  The problem is that although we have cameras capable of recording the wide
  range of intensity that exists in the real world our monitors are not capable
  of displaying them.  Our eyes are also quite capable of observing a much wider
  range of intensities than our image formats / monitors are capable of
  displaying.

  Tone-mapping is a process that transforms the intensities in the image so that
  the brightest values aren't nearly so far away from the mean.  That way when
  we transform the values into [0-255] we can actually see the entire image.
  There are many ways to perform this process and it is as much an art as a
  science - there is no single "right" answer.  In this homework we will
  implement one possible technique.

  Background Chrominance-Luminance
  ================================

  The RGB space that we have been using to represent images can be thought of as
  one possible set of axes spanning a three dimensional space of color.  We
  sometimes choose other axes to represent this space because they make certain
  operations more convenient.

  Another possible way of representing a color image is to separate the color
  information (chromaticity) from the brightness information.  There are
  multiple different methods for doing this - a common one during the analog
  television days was known as Chrominance-Luminance or YUV.

  We choose to represent the image in this way so that we can remap only the
  intensity channel and then recombine the new intensity values with the color
  information to form the final image.

  Old TV signals used to be transmitted in this way so that black & white
  televisions could display the luminance channel while color televisions would
  display all three of the channels.
  

  Tone-mapping
  ============

  In this assignment we are going to transform the luminance channel (actually
  the log of the luminance, but this is unimportant for the parts of the
  algorithm that you will be implementing) by compressing its range to [0, 1].
  To do this we need the cumulative distribution of the luminance values.

  Example
  -------

  input : [2 4 3 3 1 7 4 5 7 0 9 4 3 2]
  min / max / range: 0 / 9 / 9

  histo with 3 bins: [4 7 3]

  cdf : [4 11 14]


  Your task is to calculate this cumulative distribution by following these
  steps.

*/

#include "utils.h"
#include "stdio.h"

void __global__ generateHistogram(const float* d_logLuminance, unsigned int *d_histogram, const float minLogLum, 
                                  const float lumRange, const int numBins, const int numRows,
                                 const int numCols)
{
    extern __shared__ unsigned int s_histogram[];
    
    // Zero the shared memory histogram
    
    int width_position = blockDim.x*blockIdx.x + threadIdx.x;
    int height_position = blockDim.y*blockIdx.y + threadIdx.y;
    int lum_idx = height_position*numCols + width_position;
    int bin;
    // This might not be copying all of the bins?
    int local_location = threadIdx.y*blockDim.x+threadIdx.x;
    if (local_location < numBins)
    {
            s_histogram[local_location] = d_histogram[local_location];
    }
    __syncthreads();
    if (width_position < numCols && height_position < numRows)
    {
        bin = min((numBins - 1), (unsigned int)((d_logLuminance[lum_idx] - minLogLum) / lumRange * numBins));
        //printf("%d\n",bin);
        atomicAdd(&(d_histogram[bin]), 1);
        //atomicInc(&(s_histogram[bin]), sizeof(unsigned int));
        //printf("bin:%d, value:%d\n",bin, d_histogram[bin]);
    }
    //__syncthreads();
    // copy the shared histogram somehow back into the global device histogram
    //if (local_location < numBins)
    //{
//        atomicAdd(&(d_histogram[local_location]), s_histogram[local_location]);
  //  }
}

void __global__ prefixScan(unsigned int *g_idata, unsigned int *g_odata, const int numBins)
{
   extern __shared__ float temp[];  // allocated on invocation  
   int thid = threadIdx.x;  
   int offset = 1;
   int n = numBins;
   
   //temp[2*thid] = d_cdf[2*thid]; // load input into shared memory  
   //temp[2*thid+1] = d_cdf[2*thid+1]; 
   if (2*thid+1 < numBins)
   {
	temp[2*thid] = g_idata[2*thid]; // load input into shared memory  
	temp[2*thid+1] = g_idata[2*thid+1]; 
   }
    
   for (int d = n>>1; d > 0; d >>= 1)                    // build sum in place up the tree  
   {   
   __syncthreads();  
       if (thid < d)  
       {
           int ai = offset*(2*thid+1)-1;  
           int bi = offset*(2*thid+2)-1;
           temp[bi] += temp[ai];  
       }  
       offset *= 2;
   }
   if (thid == 0) { temp[n - 1] = 0; } // clear the last element  
   for (int d = 1; d < n; d *= 2) // traverse down tree & build scan  
   {  
       offset >>= 1;  
       __syncthreads();  
       if (thid < d)                       
      {
           int ai = offset*(2*thid+1)-1;  
           int bi = offset*(2*thid+2)-1; 
           float t = temp[ai];  
           temp[ai] = temp[bi];  
           temp[bi] += t;   
      }  
   }  
   __syncthreads();
    
   if (2*thid+1 < numBins)
   {
     g_odata[2*thid] = temp[2*thid]; // write results to device memory  
     g_odata[2*thid+1] = temp[2*thid+1];
   }

   /*
   int storage_offset=1;
   
   extern __shared__ unsigned int s_cdf[];
    
   if (2*threadIdx.x+1 < numBins)
   {
       s_cdf[2*threadIdx.x] = d_cdf[2*threadIdx.x];
       s_cdf[2*threadIdx.x+1] = d_cdf[2*threadIdx.x+1];
   }
    
   // The reduce phase
   for (unsigned int s = numBins>>2; s > 0; s>>=1)
   {
       if (threadIdx.x < s)
       {
           s_cdf[storage_offset*(2*threadIdx.x+1)-1] += s_cdf[storage_offset*(2*threadIdx.x+2)-1];
       }
       storage_offset *= 2;
       __syncthreads();
   }
   
   if (threadIdx.x == 0)
   {
       s_cdf[numBins-1] = 0;
   }
   __syncthreads();
   // The down-sweep
   
   //float t=0;
   int e0, e1;
   for (unsigned int s = 1; s < numBins; s<<=2)
   {
       storage_offset >>=1;
       if (threadIdx.x < s)
       {
           e0 = storage_offset*(2*threadIdx.x+1)-1;
           e1 = storage_offset*(2*threadIdx.x+2)-1;
           float t = s_cdf[e0];
           s_cdf[e0] = s_cdf[e1];
           s_cdf[e1]+=t;
       }
       __syncthreads();
   }
    if (2*threadIdx.x+1 < numBins)
   {
        d_cdf[2*threadIdx.x] = s_cdf[2*threadIdx.x]; 
        d_cdf[2*threadIdx.x+1] = s_cdf[2*threadIdx.x+1]; 
    }
    */
}

void your_histogram_and_prefixsum(const float* const d_logLuminance,
                                  unsigned int* const d_cdf,
                                  float &min_logLum,
                                  float &max_logLum,
                                  const size_t numRows,
                                  const size_t numCols,
                                  const size_t numBins)
{
  //TODO
    //1) find the minimum and maximum value in the input logLuminance channel
    //   store in min_logLum and max_logLum
    
    //2) subtract them to find the range
    
    //3) generate a histogram of all the values in the logLuminance channel using
    //   the formula: bin = (lum[i] - lumMin) / lumRange * numBins
    
    //4) Perform an exclusive scan (prefix sum) on the histogram to get
    //   the cumulative distribution of luminance values (this should go in the
    //   incoming d_cdf pointer which already has been allocated for you)
    float *h_logLuminance;
	h_logLuminance=(float*)malloc(sizeof(float)*numRows*numCols);
    checkCudaErrors(cudaMemcpy(h_logLuminance, d_logLuminance, numRows*numCols*sizeof(float), cudaMemcpyDeviceToHost));
    min_logLum = h_logLuminance[0];
    max_logLum = h_logLuminance[0];
      
    dim3 blockSize = dim3(16, 16, 1);
    dim3 gridSize = dim3(numCols/16+1, numRows/16+1, 1);
    
  //Step 1
  //first we find the minimum and maximum across the entire image
  for (size_t i = 1; i < numCols * numRows; ++i) {
    min_logLum = min(h_logLuminance[i], min_logLum);
    max_logLum = max(h_logLuminance[i], max_logLum);
  }

  //Step 2
  float logLumRange = max_logLum - min_logLum;

  unsigned int *d_histogram;
  checkCudaErrors(cudaMalloc(&d_histogram, sizeof(unsigned int) * numBins));
  // exercise the nuclear option on the cdf
  //checkCudaErrors(cudaMemset(d_cdf, 0, sizeof(unsigned int) * numBins));
  checkCudaErrors(cudaMemset(d_histogram, 0, sizeof(unsigned int) * numBins));
    
  //Step 3
  //next we use the now known range to compute
  //a histogram of numBins bins
  
      //Step 3
  //next we use the now known range to compute
  //a histogram of numBins bins
  generateHistogram<<<gridSize, blockSize, sizeof(unsigned int) * numBins>>>(d_logLuminance, 
                           d_histogram, min_logLum, logLumRange, numBins, numRows, numCols);

  //unsigned int h_cdf[numBins];
  //unsigned int histo[numBins];
  //checkCudaErrors(cudaMemcpy(d_cdf, histo, numBins*sizeof(unsigned int), cudaMemcpyHostToDevice));
  //checkCudaErrors(cudaMemcpy(h_cdf, d_cdf, numBins*sizeof(unsigned int), cudaMemcpyDeviceToHost));
  //memcpy(histo, h_cdf, numBins*sizeof(unsigned int));
    
  //Step 4
  //finally we perform and exclusive scan (prefix sum)
  //on the histogram to get the cumulative distribution
  //h_cdf[0] = 0;
  //for (size_t i = 1; i < numBins; ++i) {
//    h_cdf[i] = h_cdf[i - 1] + histo[i - 1];
  //}
    
  prefixScan<<<1,numBins,2*numBins*sizeof(unsigned int)>>>(d_histogram, d_cdf, numBins);
  //checkCudaErrors(cudaMemcpy(d_cdf, h_cdf, numBins*sizeof(unsigned int), cudaMemcpyHostToDevice));
  //cudaDeviceSynchronize();
    checkCudaErrors(cudaFree(d_histogram));
}