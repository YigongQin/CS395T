#include <stdio.h>

#include <cuda.h>
#include <cuda_runtime.h>
#include <driver_functions.h>


#include "CycleTimer.h"


extern float toBW(int bytes, float sec);


/* Helper function to round up to a power of 2. 
 */
static inline long long int nextPow2(long long int n)
{
    n--;
    n |= n >> 1;
    n |= n >> 2;
    n |= n >> 4;
    n |= n >> 8;
    n |= n >> 16;
    n++;
    return n;
}

// Device code

__global__ void
upsweep(long long int N, int dim, int twod, int twod1, float* output) {

    int index = blockIdx.x * blockDim.x + threadIdx.x;

    if (index < N && ( (index/dim) % twod1 ==0) )
       {output[index+ dim*(twod1 -1)] += output[index+ dim*(twod -1)];}
}

__global__ void
downsweep(long long int N, int dim, int twod, int twod1, float* output) {

    int index = blockIdx.x * blockDim.x + threadIdx.x;

    if ( (0< index) && (index < N) && ( (index/dim) % twod1 ==0) ){
         output[index+ dim*(twod-1)] += output[index- dim*1];}
}

__global__ void
sum_const(long long int N, int dim, float* const_array, float* output){

    // here N is the total length n*dim
    int index = blockIdx.x * blockDim.x + threadIdx.x;

    int dim_id = index%dim; 

    if ( index < N ){
         output[index] += const_array[dim_id];}

}



void inclusive_scan(long long int N, long long int lens, int dim, float* device_result){

    long long int blocksize = 512;
    long long int num_blocks = (N+blocksize-1)/blocksize;
    printf("N=%lld,block size = %lld, number of blocks %lld \n",N,blocksize,num_blocks); 
    for (int twod =1; twod <lens; twod *=2){
        int twod1 = twod*2;
            upsweep<<< num_blocks, blocksize  >>>(N, dim, twod, twod1, device_result);        
    }

    for (int twod = lens/4; twod >=1; twod /=2){
        int twod1 = twod*2;
            downsweep<<< num_blocks, blocksize  >>>(N, dim, twod, twod1, device_result);
    }
}

void plus_const(long long int N, int dim, float* const_array, float* device_result){
    // here N is the total length n*dim
    long long int blocksize = 512;
    long long int num_blocks = (N+blocksize-1)/blocksize;
    sum_const<<< num_blocks, blocksize  >>>(N, dim, const_array, device_result);

}


// cuda scan
double cudaScan(float* inarray, long long int N, int dim)
{
    // N is the total length of the array N = n*dim;
    // round N to the next power of 2 to call recursive kernels;
    long long int lens = N/dim; printf("N %lld, lens %lld\n",lens); 
    long long int round_lens = nextPow2(lens); printf("round lens %d\n",round_lens);
    long long int rounded_length=dim*round_lens; printf("the rounded length %lld,blocks %lld\n",rounded_length,rounded_length/512);
    float* device_result;

    cudaMalloc((void **)&device_result, sizeof(float) * rounded_length);

    cudaMemcpy(device_result, inarray, N * sizeof(float), 
               cudaMemcpyHostToDevice);

    double startTime = CycleTimer::currentSeconds();

    inclusive_scan(rounded_length, round_lens, dim, device_result);

    // Wait for any work left over to be completed.
    cudaDeviceSynchronize();
    double endTime = CycleTimer::currentSeconds();
    double overallDuration = endTime - startTime;
    
    cudaMemcpy(inarray, device_result, N * sizeof(float),
               cudaMemcpyDeviceToHost);

    cudaFree(device_result);
    return overallDuration;
}

// cuda parallel sum
double cudaSum(long long int N, int dim, float* sum0, float* resultarray)
{   // here N is the total length n*dim
    float* device_result;
    float* device_sum;
    cudaMalloc((void **)&device_result, sizeof(float) * N);
    cudaMalloc((void **)&device_sum, sizeof(float) * dim)
;
    cudaMemcpy(device_result, resultarray, N * sizeof(float),
               cudaMemcpyHostToDevice);
    cudaMemcpy(device_sum, sum0, dim * sizeof(float),
               cudaMemcpyHostToDevice);

    double startTime = CycleTimer::currentSeconds();
    //printf("the length %d", rounded_length);
    plus_const(N, dim, device_sum,  device_result);//exclusive_scan(device_input, end - inarray, device_result);

    // Wait for any work left over to be completed.
    cudaDeviceSynchronize();
    double endTime = CycleTimer::currentSeconds();
    double overallDuration = endTime - startTime;

    cudaMemcpy(resultarray, device_result, N * sizeof(float),
               cudaMemcpyDeviceToHost);
    cudaFree(device_result);
    cudaFree(device_sum);
    //for (int i = 0; i < rounded_length; i++){printf("%d ",resultarray[i]);}
    return overallDuration;
}




void printCudaInfo()
{
    // for fun, just print out some stats on the machine

    int deviceCount = 0;
    cudaError_t err = cudaGetDeviceCount(&deviceCount);

    printf("---------------------------------------------------------\n");
    printf("Found %d CUDA devices\n", deviceCount);

    for (int i=0; i<deviceCount; i++)
    {
        cudaDeviceProp deviceProps;
        cudaGetDeviceProperties(&deviceProps, i);
        printf("Device %d: %s\n", i, deviceProps.name);
        printf("   SMs:        %d\n", deviceProps.multiProcessorCount);
        printf("   Global mem: %.0f MB\n",
               static_cast<float>(deviceProps.totalGlobalMem) / (1024 * 1024));
        printf("   CUDA Cap:   %d.%d\n", deviceProps.major, deviceProps.minor);
    }
    printf("---------------------------------------------------------\n"); 
}
