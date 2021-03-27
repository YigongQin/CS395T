#include <stdio.h>
#include <stdlib.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <driver_functions.h>
#include <curand.h>
#include <curand_kernel.h>
#include "CycleTimer.h"


extern float toBW(int bytes, float sec);

__managed__ bool end_flag;
__managed__ bool color_flag;
// Device code

//based on edges
__global__ void
find_min(int num_vert, long long int num_edges,  int* d_flag, int* d_flag2, int* d_row, int* d_column,  long long int* d_random){

  // compare the two numbers of edges and update flag of larger one to -2
  int index = blockIdx.x * blockDim.x + threadIdx.x;
  if (index < num_edges){
    int left_vert = d_row[index]; 
    int right_vert = d_column[index];
    if (  (left_vert!=right_vert) && (d_flag2[ left_vert ]==0) && (d_flag2[ right_vert ]==0) ){
      // compare random nubmer
      long long int left_rand = d_random[ left_vert ];
      long long int right_rand = d_random[ right_vert ];
      if ( left_rand < right_rand ) { d_flag[ right_vert ] = -2;}
                                    //  printf("edge %d , set vertex %d to -2\n", index, right_vert );    }
      else if ( left_rand > right_rand )  { d_flag[ left_vert ] = -2;}
                                     // printf("edge %d , set vertex %d to -2\n", index, left_vert );    }
      else if ( left_rand == right_rand ){ 
                d_flag[ left_vert ] = -2; d_flag[ right_vert ] = -2 ; 
                printf("find two vertices have the same random number\n");}
      else{printf("bug exists, %lld, %lld\n", left_rand, right_rand);}
    }
  }
}

//based on edges, remove the other side of min
__global__ void
find_nbrs(int num_vert, long long int num_edges,  int* d_flag, int* d_flag2, int* d_row, int* d_column){

  // find neighbors first, set flag from -2 to -1
  int index = blockIdx.x * blockDim.x + threadIdx.x;
  // coalesced mem of the same column
  if (index < num_edges){
    if ( (d_flag2[ d_column[index] ]==0) && (d_flag2[ d_row[index] ] == -2)  ) 
        { d_flag[ d_row[index] ] = -1;} 
         //  printf("edge %d , set vertex %d to -1 because of vertex %d\n", index, d_row[index], d_column[index]);}
    if ( (d_flag2[ d_row[index] ]==0) && (d_flag2[ d_column[index] ] == -2)  )
        { d_flag[ d_column[index] ] = -1;}
           //printf("edge %d , set vertex %d to -1 because of vertex %d\n", index, d_column[index], d_row[index]);}
  }
}


__global__ void
copy_flag(int num_vert, int* d_flag, int* d_flag2){
  int index = blockIdx.x * blockDim.x + threadIdx.x;
  if (index < num_vert){
   d_flag2[index] = d_flag[index];}

}





//based on vertices, update status
__global__ void
update_status(int colorid, int num_vert, int* d_flag, int* d_flag2){

  //set 0 to the colorid, set remained -2 to 0,
  int index = blockIdx.x * blockDim.x + threadIdx.x;
  if (index < num_vert){
    if ( d_flag[index] ==0 ) { d_flag[index] = colorid; } 
    if ( d_flag[index] ==-2 ) { d_flag[index] = 0;
                                end_flag = false; }
    d_flag2[index] = d_flag[index];
  }
}

__global__ void
random_init(int num_vert, long long int* d_random){

    // generate random numbers in the range [0, RAND_MAX^2]
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned int seed = 1 ;//time(NULL);

    if ( index < num_vert ){
        curandState_t state;
        curand_init(seed+index, 0, 0, &state);
        d_random[index] = curand(&state);}
       // printf(" vertex %d, the random number %lld \n", index, d_random[index]);}

}


// activate the -1 to 0 for coloring problem
__global__ void
activate(int num_vert, int* d_flag, int* d_flag2){

    int index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index < num_vert){
       if ( d_flag[index] ==-1 ) { d_flag[index] = 0;
                                    color_flag = false;}
        d_flag2[index] = d_flag[index]; 
        // if there is still -1, means coloring not finished yet
     }
}






void MIS(int colorid, int num_vert, long long int num_edges, int* d_flag, int* d_flag2, int* d_row, int* d_column, long long int* d_random){

     int blocksize = 512;
     int num_block1 = (num_vert+blocksize-1)/blocksize;
     int num_block2 = (num_edges+blocksize-1)/blocksize;
    // random number initialization for d_random, flags all to 0 (active)
     random_init<<< num_block1, blocksize >>> (num_vert, d_random);
     end_flag = false;
     int iter_mis = 0;
     while ( !end_flag ){
     //for (int i =0; i<5;i++){
    // compare the two vertices at one edge, the bigger one (or equal one) set to -2  
    // map (1,-1, 0) to (1, -1 , 0, -2)
     find_min<<< num_block2, blocksize >>>( num_vert, num_edges, d_flag, d_flag2, d_row, d_column, d_random ); 
    // set the neighbor of min (zeros) to -1
    // (1, -1 , 0, -2) to map (1 ,-1, 0, -1, -2)
     copy_flag<<< num_block1, blocksize >>>(num_vert, d_flag, d_flag2); 
     find_nbrs<<< num_block2, blocksize >>>( num_vert, num_edges, d_flag, d_flag2, d_row, d_column);
    // update status of flags
     end_flag = true;
     update_status<<< num_block1, blocksize >>>( colorid, num_vert, d_flag, d_flag2); 
    // if still active points, repeat
     cudaDeviceSynchronize(); // most important
     //printf(end_flag ? "true" : "false"); printf("\n");
     //copy_flag<<< num_block1, blocksize >>>(num_vert, d_flag, d_flag2);
     iter_mis +=1;
     }
     printf("iterations for one MIS: %d\n", iter_mis);
     //find_min<<< num_block2, blocksize >>>( num_vert, num_edges, d_flag, d_row, d_column, d_random );


}


void coloring(int colorid, int num_vert, long long int num_edges, int* d_flag, int* d_flag2, int* d_row, int* d_column, long long int* d_random){


    // add a kernel that reset -1 to 0 after color one MIS
     int blocksize = 512;
     int num_block1 = (num_vert+blocksize-1)/blocksize;
     int num_block2 = (num_edges+blocksize-1)/blocksize;
    // random number initialization for d_random, flags all to 0 (active)
     random_init<<< num_block1, blocksize >>> (num_vert, d_random);
     color_flag = false;
     while( !color_flag ){
       end_flag = false;
       while ( !end_flag ){
         // compare the two vertices at one edge, the bigger one (or equal one) set to -2
         find_min<<< num_block2, blocksize >>>( num_vert, num_edges, d_flag, d_flag2, d_row, d_column, d_random );
         // set the neighbor of min (zeros) to -1
         copy_flag<<< num_block1, blocksize >>>(num_vert, d_flag, d_flag2);
         find_nbrs<<< num_block2, blocksize >>>( num_vert, num_edges, d_flag, d_flag2, d_row, d_column);
         // update status of flags
         end_flag = true;
         update_status<<< num_block1, blocksize >>>( colorid, num_vert, d_flag, d_flag2);
         cudaDeviceSynchronize(); // most important
         //printf("one color end ");printf(end_flag ? "true" : "false"); printf("\n");
         //copy_flag<<< num_block1, blocksize >>>(num_vert, d_flag, d_flag2);  
       // if still active points, repeat
       }
       color_flag = true;
       activate<<< num_block1, blocksize >>>( num_vert, d_flag, d_flag2);
       colorid +=1; // finish one color, colorid plus one;
       cudaDeviceSynchronize();
       //printf("all color end");printf(color_flag ? "true" : "false"); 
     }
     printf("the number of colors: %d\n",colorid-1);
}




// cuda MIS
// flag status: inactive: -1; active: 0; MIS: colorid 
double cudaMIS(int colorid, int* flag, int num_vert, long long int num_edges, int* row, int* column)
{
    int* d_flag;    
    int* d_flag2; // a copy of flag array to avoid racing condition 
    int* d_row; 
    int* d_column;
    long long int* d_random;    

    cudaMalloc((void **)&d_flag, sizeof(int) * num_vert);
    cudaMalloc((void **)&d_flag2, sizeof(int) * num_vert);
    cudaMalloc((void **)&d_random, sizeof(long long int) * num_vert);

    cudaMalloc((void **)&d_row, sizeof(int) * num_edges);
    cudaMalloc((void **)&d_column, sizeof(int) * num_edges);

    cudaMemcpy(d_flag, flag, num_vert * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_flag2, flag, num_vert * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_row, row, num_edges * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_column, column, num_edges * sizeof(int), cudaMemcpyHostToDevice);

    double startTime = CycleTimer::currentSeconds();

    MIS(colorid, num_vert, num_edges, d_flag, d_flag2, d_row, d_column, d_random);
    // Wait for any work left over to be completed.
    cudaDeviceSynchronize();
    double endTime = CycleTimer::currentSeconds();
    double overallDuration = endTime - startTime;
    
    cudaMemcpy(flag, d_flag, num_vert * sizeof(int), cudaMemcpyDeviceToHost);

    cudaFree(d_flag); cudaFree(d_flag2); cudaFree(d_row); cudaFree(d_column); cudaFree(d_random);
    return overallDuration;
}

// cuda coloring
double cudaColoring(int colorid, int* flag, int num_vert, long long int num_edges, int* row, int* column)
{
    int* d_flag;
    int* d_flag2; // a copy of flag array to avoid racing condition
    int* d_row; 
    int* d_column;
    long long int* d_random;

    cudaMalloc((void **)&d_flag, sizeof(int) * num_vert);
    cudaMalloc((void **)&d_flag2, sizeof(int) * num_vert);
    cudaMalloc((void **)&d_random, sizeof(long long int) * num_vert);

    cudaMalloc((void **)&d_row, sizeof(int) * num_edges);
    cudaMalloc((void **)&d_column, sizeof(int) * num_edges);

    cudaMemcpy(d_flag, flag, num_vert * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_flag2, flag, num_vert * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_row, row, num_edges * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_column, column, num_edges * sizeof(int), cudaMemcpyHostToDevice);

    double startTime = CycleTimer::currentSeconds();

    coloring(colorid, num_vert, num_edges, d_flag, d_flag2, d_row, d_column, d_random);
    // Wait for any work left over to be completed.
    cudaDeviceSynchronize();
    double endTime = CycleTimer::currentSeconds();
    double overallDuration = endTime - startTime;

    cudaMemcpy(flag, d_flag, num_vert * sizeof(int), cudaMemcpyDeviceToHost);

    cudaFree(d_flag); cudaFree(d_row); cudaFree(d_column); cudaFree(d_random);
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
