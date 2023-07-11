#include <cmath>
#include <iostream>
#include "gpu-new-forward.h"
#define TILE_WIDTH 16
__global__ void conv_forward_kernel(float* __restrict__ output, const float* __restrict__ input, const float* __restrict__ mask, const int Batch, const int Map_out, const int Channel, const int Height, const int Width, const int K)
{
    /*
    Modify this function to implement the forward pass described in Chapter 16.
    We have added an additional dimension to the tensors to support an entire mini-batch
    The goal here is to be correct AND fast.

    Function paramter definitions:
    output - output
    input - input
    mask - convolution kernel
    Batch - batch_size (number of images in x)
    Map_out - number of output feature maps
    Channel - number of input feature maps
    Height - input height dimension
    Width - input width dimension
    K - kernel height and width (K x K)
    */

    const int Height_out = Height - K + 1;
    const int Width_out = Width - K + 1;
    const int Width_grid = (Width_out + TILE_WIDTH - 1) / TILE_WIDTH;


    // We have some nice #defs for you below to simplify indexing. Feel free to use them, or create your own.
    // An example use of these macros:
    // float a = in_4d(0,0,0,0)
    // out_4d(0,0,0,0) = a

    #define out_4d(i3, i2, i1, i0) output[(i3) * (Map_out * Height_out * Width_out) + (i2) * (Height_out * Width_out) + (i1) * (Width_out) + i0]
    #define in_4d(i3, i2, i1, i0) input[(i3) * (Channel * Height * Width) + (i2) * (Height * Width) + (i1) * (Width) + i0]
    #define mask_4d(i3, i2, i1, i0) mask[(i3) * (Channel * K * K) + (i2) * (K * K) + (i1) * (K) + i0]

    // Insert your GPU convolution kernel code here
    int bx = blockIdx.x;
    int bz = blockIdx.z;
    int w_tilenum = (Width_out + TILE_WIDTH - 1) / TILE_WIDTH;
    int btx = (blockIdx.y % w_tilenum) * TILE_WIDTH + threadIdx.x;
    int bty = (blockIdx.y / w_tilenum) * TILE_WIDTH + threadIdx.y;
    float sum = 0.0;

    if((bty < Height_out) && (btx < Width_out)){
        for(int i = 0; i < Channel; i++){
            sum += in_4d(bz,i,bty+0,btx+0) * mask_4d(bx,i,0,0);
            sum += in_4d(bz,i,bty+0,btx+1) * mask_4d(bx,i,0,1);
            sum += in_4d(bz,i,bty+0,btx+2) * mask_4d(bx,i,0,2);
            sum += in_4d(bz,i,bty+0,btx+3) * mask_4d(bx,i,0,3);
            sum += in_4d(bz,i,bty+0,btx+4) * mask_4d(bx,i,0,4);
            sum += in_4d(bz,i,bty+0,btx+5) * mask_4d(bx,i,0,5);
            sum += in_4d(bz,i,bty+0,btx+6) * mask_4d(bx,i,0,6);

            sum += in_4d(bz,i,bty+1,btx+0) * mask_4d(bx,i,1,0);
            sum += in_4d(bz,i,bty+1,btx+1) * mask_4d(bx,i,1,1);
            sum += in_4d(bz,i,bty+1,btx+2) * mask_4d(bx,i,1,2);
            sum += in_4d(bz,i,bty+1,btx+3) * mask_4d(bx,i,1,3);
            sum += in_4d(bz,i,bty+1,btx+4) * mask_4d(bx,i,1,4);
            sum += in_4d(bz,i,bty+1,btx+5) * mask_4d(bx,i,1,5);
            sum += in_4d(bz,i,bty+1,btx+6) * mask_4d(bx,i,1,6);

            sum += in_4d(bz,i,bty+2,btx+0) * mask_4d(bx,i,2,0);
            sum += in_4d(bz,i,bty+2,btx+1) * mask_4d(bx,i,2,1);
            sum += in_4d(bz,i,bty+2,btx+2) * mask_4d(bx,i,2,2);
            sum += in_4d(bz,i,bty+2,btx+3) * mask_4d(bx,i,2,3);
            sum += in_4d(bz,i,bty+2,btx+4) * mask_4d(bx,i,2,4);
            sum += in_4d(bz,i,bty+2,btx+5) * mask_4d(bx,i,2,5);
            sum += in_4d(bz,i,bty+2,btx+6) * mask_4d(bx,i,2,6);  

            sum += in_4d(bz,i,bty+3,btx+0) * mask_4d(bx,i,3,0);
            sum += in_4d(bz,i,bty+3,btx+1) * mask_4d(bx,i,3,1);
            sum += in_4d(bz,i,bty+3,btx+2) * mask_4d(bx,i,3,2);
            sum += in_4d(bz,i,bty+3,btx+3) * mask_4d(bx,i,3,3);
            sum += in_4d(bz,i,bty+3,btx+4) * mask_4d(bx,i,3,4);
            sum += in_4d(bz,i,bty+3,btx+5) * mask_4d(bx,i,3,5);
            sum += in_4d(bz,i,bty+3,btx+6) * mask_4d(bx,i,3,6);

            sum += in_4d(bz,i,bty+4,btx+0) * mask_4d(bx,i,4,0);
            sum += in_4d(bz,i,bty+4,btx+1) * mask_4d(bx,i,4,1);
            sum += in_4d(bz,i,bty+4,btx+2) * mask_4d(bx,i,4,2);
            sum += in_4d(bz,i,bty+4,btx+3) * mask_4d(bx,i,4,3);
            sum += in_4d(bz,i,bty+4,btx+4) * mask_4d(bx,i,4,4);
            sum += in_4d(bz,i,bty+4,btx+5) * mask_4d(bx,i,4,5);
            sum += in_4d(bz,i,bty+4,btx+6) * mask_4d(bx,i,4,6);

            sum += in_4d(bz,i,bty+5,btx+0) * mask_4d(bx,i,5,0);
            sum += in_4d(bz,i,bty+5,btx+1) * mask_4d(bx,i,5,1);
            sum += in_4d(bz,i,bty+5,btx+2) * mask_4d(bx,i,5,2);
            sum += in_4d(bz,i,bty+5,btx+3) * mask_4d(bx,i,5,3);
            sum += in_4d(bz,i,bty+5,btx+4) * mask_4d(bx,i,5,4);
            sum += in_4d(bz,i,bty+5,btx+5) * mask_4d(bx,i,5,5);
            sum += in_4d(bz,i,bty+5,btx+6) * mask_4d(bx,i,5,6);

            sum += in_4d(bz,i,bty+6,btx+0) * mask_4d(bx,i,6,0);
            sum += in_4d(bz,i,bty+6,btx+1) * mask_4d(bx,i,6,1);
            sum += in_4d(bz,i,bty+6,btx+2) * mask_4d(bx,i,6,2);
            sum += in_4d(bz,i,bty+6,btx+3) * mask_4d(bx,i,6,3);
            sum += in_4d(bz,i,bty+6,btx+4) * mask_4d(bx,i,6,4);
            sum += in_4d(bz,i,bty+6,btx+5) * mask_4d(bx,i,6,5);
            sum += in_4d(bz,i,bty+6,btx+6) * mask_4d(bx,i,6,6);
        }
        out_4d(bz, bx, bty, btx) = sum;
    }

    #undef out_4d
    #undef in_4d
    #undef mask_4d
}

	
__host__ void GPUInterface::conv_forward_gpu_prolog(const float *host_output, const float *host_input, const float *host_mask, float **device_output_ptr, float **device_input_ptr, float **device_mask_ptr, const int Batch, const int Map_out, const int Channel, const int Height, const int Width, const int K)
{
    // Allocate memory and copy over the relevant data structures to the GPU

    // We pass double pointers for you to initialize the relevant device pointers,
    //  which are passed to the other two functions.

    // Useful snippet for error checking
    // cudaError_t error = cudaGetLastError();
    // if(error != cudaSuccess)
    // {
    //     std::cout<<"CUDA error: "<<cudaGetErrorString(error)<<std::endl;
    //     exit(-1);
    // }

    const int Height_out = Height - K + 1;
    const int Width_out = Width - K + 1;

    cudaMalloc((void**)device_output_ptr, Batch * Map_out * Height_out * Width_out * sizeof(float));
    cudaMalloc((void**)device_input_ptr, Batch * Channel * Height * Width * sizeof(float));
    cudaMalloc((void**)device_mask_ptr, Map_out * Channel * K * K * sizeof(float));

    cudaMemcpy(*device_input_ptr, host_input, Batch * Channel * Height * Width * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(*device_mask_ptr, host_mask, Map_out * Channel * K * K * sizeof(float), cudaMemcpyHostToDevice);
}


__host__ void GPUInterface::conv_forward_gpu(float *device_output, const float *device_input, const float *device_mask, const int Batch, const int Map_out, const int Channel, const int Height, const int Width, const int K)
{
    // Set the kernel dimensions and call the kernel
    const int Height_out = Height - K + 1;
    const int Width_out = Width - K + 1;
    int w_tilenum = (Width_out + TILE_WIDTH - 1) / TILE_WIDTH;
    int h_tilenum = (Height_out + TILE_WIDTH - 1) / TILE_WIDTH;
    int grid_out = w_tilenum * h_tilenum;

    dim3 blockDim(TILE_WIDTH, TILE_WIDTH, 1);
    dim3 gridDim(Map_out, grid_out, Batch);
    conv_forward_kernel<<<gridDim, blockDim>>>(device_output, device_input, device_mask, Batch, Map_out, Channel, Height, Width, K);
    cudaDeviceSynchronize();
}


__host__ void GPUInterface::conv_forward_gpu_epilog(float *host_output, float *device_output, float *device_input, float *device_mask, const int Batch, const int Map_out, const int Channel, const int Height, const int Width, const int K)
{
    // Copy the output back to host
    const int Height_out = Height - K + 1;
    const int Width_out = Width - K + 1;
    
    cudaMemcpy(host_output, device_output, Batch * Map_out * Height_out * Width_out * sizeof(float), cudaMemcpyDeviceToHost);
    // Free device memory
    cudaFree(host_output);
    cudaFree(device_input);
    cudaFree(device_mask);
}


__host__ void GPUInterface::get_device_properties()
{
    int deviceCount;
    cudaGetDeviceCount(&deviceCount);

    for(int dev = 0; dev < deviceCount; dev++)
    {
        cudaDeviceProp deviceProp;
        cudaGetDeviceProperties(&deviceProp, dev);

        std::cout<<"Device "<<dev<<" name: "<<deviceProp.name<<std::endl;
        std::cout<<"Computational capabilities: "<<deviceProp.major<<"."<<deviceProp.minor<<std::endl;
        std::cout<<"Max Global memory size: "<<deviceProp.totalGlobalMem<<std::endl;
        std::cout<<"Max Constant memory size: "<<deviceProp.totalConstMem<<std::endl;
        std::cout<<"Max Shared memory size per block: "<<deviceProp.sharedMemPerBlock<<std::endl;
        std::cout<<"Max threads per block: "<<deviceProp.maxThreadsPerBlock<<std::endl;
        std::cout<<"Max block dimensions: "<<deviceProp.maxThreadsDim[0]<<" x, "<<deviceProp.maxThreadsDim[1]<<" y, "<<deviceProp.maxThreadsDim[2]<<" z"<<std::endl;
        std::cout<<"Max grid dimensions: "<<deviceProp.maxGridSize[0]<<" x, "<<deviceProp.maxGridSize[1]<<" y, "<<deviceProp.maxGridSize[2]<<" z"<<std::endl;
        std::cout<<"Warp Size: "<<deviceProp.warpSize<<std::endl;
    }
}