// Histogram Equalization

#include <wb.h>

#define HISTOGRAM_LENGTH 256

#define uint8_t unsigned char
#define bd blockDim.x
#define bx blockIdx.x
#define tx threadIdx.x


__global__ void FloatToUChar(float* float_img_in, uint8_t* char_img_out, int imgsize){
    int index = (bd * bx) + tx;
    // int imgSize = height * width * channels;
    if(index < imgSize){
        char_img_out[index] = (uint8_t)((HISTOGRAM_LENGTH - 1) * float_img_in[index]);
    }
}

__global__ void UCharToGray(uint8_t* imgIn, uint8_t* imgOut, int width, int height, int channels){
    int index = bd * bx + tx;
    if(index < width * height) {

        uint8_t r = imgIn[index * 3];
        uint8_t g = imgIn[index * 3 + 1];
        uint8_t b = imgIn[index * 3 + 2];
        imgOut[index] = (uint8_t) (0.21*r + 0.71*g + 0.07*b);
    }
}

__global__ void Histcalcer(uint8_t* imgIn, unsigned int* histOut, int height, int width){
    __shared__ unsigned int histgram[HISTOGRAM_LENGTH];

    int imgSize = height * width;

    int index = bd * bx + tx;

    if(tx < HISTOGRAM_LENGTH){
        histgram[tx] = 0;
    }
    __syncthreads();

    if(index < imgSize){
        atomicAdd(&histgram[imgIn[index]], 1);
    }
    __syncthreads();


    if(tx < HISTOGRAM_LENGTH){
        atomicAdd(&histOut[tx], histgram[tx]);
    }

}

__global__ void HistCDF(unsigned int* histIn, float* CDFOut, int height, int width){
    int scan_int = 2 * HISTOGRAM_LENGTH;
    int imgSize = height * width;
    __shared__ float adder[2 * HISTOGRAM_LENGTH];

    int index = bd * bx + tx;

    if(index < HISTOGRAM_LENGTH){
        adder[tx] = histIn[index];
    }

    for(unsigned int i = 1; i <= HISTOGRAM_LENGTH; i *= 2){
        __syncthreads();
        unsigned int j = (2 * i * (1 + tx)) - 1;
        if(j < HISTOGRAM_LENGTH && j < scan_int){
            adder[j] = adder[j] + adder[j - i];
        }
    }

    for(unsigned int i = HISTOGRAM_LENGTH / 2; i > 0; i /= 2){
        __syncthreads();
        unsigned int j = (2 * i * (1 + tx)) - 1;
        if(j + i < HISTOGRAM_LENGTH && j + i < scan_int){
            adder[j + i] = adder[j + i] + adder[j];
        }
    }
    __syncthreads();

    if(index < HISTOGRAM_LENGTH){
        CDFOut[index] = adder[tx] / imgSize;
    }
}

__global__ void CDF_min(float* floatImg, uint8_t* ucharImg, float* cdf, int height, int width, int Channels){
    int index = bd * bx + tx;

    if(index < height * width * Channels){
        float convert_char = 255 * (cdf[ucharImg[index]] - cdf[0]) / (1 - cdf[0]) / (HISTOGRAM_LENGTH - 1);

        floatImg[index] = (float) min(max(convert_char, 0.0), 255.0);
    }
}



int main(int argc, char **argv) {
  wbArg_t args;
  int imageWidth;
  int imageHeight;
  int imageChannels;
  wbImage_t inputImage;
  wbImage_t outputImage;
  float *hostInputImageData;
  float *hostOutputImageData;
  const char *inputImageFile;

  //@@ Insert more code here
    float *deviceInput;
    uint8_t *deviceUchar;
    uint8_t *deviceGray;
    unsigned int *deviceHist;
    float* deviceCDF;
    float* deviceOutput;


  args = wbArg_read(argc, argv); /* parse the input arguments */

  inputImageFile = wbArg_getInputFile(args, 0);

  wbTime_start(Generic, "Importing data and creating memory on host");
  inputImage = wbImport(inputImageFile);
  imageWidth = wbImage_getWidth(inputImage);
  imageHeight = wbImage_getHeight(inputImage);
  imageChannels = wbImage_getChannels(inputImage);
  outputImage = wbImage_new(imageWidth, imageHeight, imageChannels);
  hostInputImageData = wbImage_getData(inputImage);
  hostOutputImageData = wbImage_getData(outputImage);
  wbTime_stop(Generic, "Importing data and creating memory on host");

  //@@ insert code here
    cudaMalloc((void **)&deviceInput, imageWidth*imageHeight*imageChannels*sizeof(float));
    cudaMalloc((void **)&deviceUchar, imageWidth*imageHeight*imageChannels*sizeof(uint8_t));
    cudaMalloc((void **)&deviceGray, imageWidth*imageHeight*sizeof(uint8_t));
    cudaMalloc((void **)&deviceHist, HISTOGRAM_LENGTH*sizeof(unsigned int));
    cudaMalloc((void **)&deviceCDF, HISTOGRAM_LENGTH*sizeof(float));
    cudaMalloc((void **)&deviceOutput, imageWidth*imageHeight*imageChannels*sizeof(float));


    cudaMemset((void *) deviceHist, 0, HISTOGRAM_LENGTH * sizeof(unsigned int));
    cudaMemset((void *) deviceCDF, 0, HISTOGRAM_LENGTH * sizeof(float));
    cudaMemcpy(deviceInput, hostInputImageData, imageWidth*imageHeight*imageChannels*sizeof(float), cudaMemcpyHostToDevice);

    dim3 block(HISTOGRAM_LENGTH);
    dim3 grid(((imageWidth*imageHeight*imageChannels) - 1) / HISTOGRAM_LENGTH + 1);

    FloatToUChar<<<grid, block>>>(deviceInput, deviceUchar, imageHeight * imageWidth * imageChannels);
    UCharToGray<<<grid, block>>>(deviceUchar, deviceGray, imageWidth, imageHeight, imageChannels);
    Histcalcer<<<grid, block>>>(deviceGray, deviceHist, imageHeight, imageWidth);
    HistCDF<<<grid, block>>>(deviceHist, deviceCDF, imageHeight, imageWidth);
    CDF_min<<<grid, block>>>(deviceOutput, deviceUchar, deviceCDF, imageHeight, imageWidth, imageChannels);

    cudaMemcpy(hostOutputImageData, deviceOutput, imageWidth * imageHeight * imageChannels * sizeof(float), cudaMemcpyDeviceToHost);

    wbImage_setData(outputImage, hostOutputImageData);

  wbSolution(args, outputImage);

  //@@ insert code here

    cudaFree(deviceInput);
    cudaFree(deviceUchar);
    cudaFree(deviceGray);
    cudaFree(deviceHist);
    cudaFree(deviceCDF);
    cudaFree(deviceOutput);
  return 0;
}
