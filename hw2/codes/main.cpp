#include <stdlib.h>
#include <stdio.h>
#include <getopt.h>
#include <string>
#include <cstring>
#include <omp.h>
#include "CycleTimer.h"
#include <mpi.h>
// #include "ompscan.h"
#define NUM_THREADS 2
double cpu_plus_const(long long int N, int dim, float* sum0, float* resultarray);
double cudaSum(long long int N, int dim, float* sum0, float* resultarray);
double cudaScan(float* start, long long int N, int dim);
double ompScan(float* start, long long int N, int dim);
void printCudaInfo();

static inline int nextPow2(int n)
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

void usage(const char* progname) {
    printf("Usage: %s [options]\n", progname);
    printf("Program Options:\n");
    printf("  -m  --test <TYPE>      Run specified function on input.  Valid tests are: cuda, omp_cpu\n"); 
    printf("  -i  --input <NAME>     Run test on given input type. Valid inputs  are: test1, random\n");
    printf("  -n  --arraysize <INT>  length of array in one dimension\n");
    printf("  -d  --arraydim <INT>   Dimensions of arrays\n");
    printf("  -?  --help             This message\n");
}


// sequential code
void cpu_inclusive_scan(float* start, int rank, int N, int dim)
{
    float*output = new float[N];
    memmove(output, start, N*sizeof(float)); 
    int length = N/dim;
    for (int j = 0; j < dim; j++) {output[j] = start[j] ;}
    for (int i = 1; i < length; i++)
    {
        for (int j = 0; j < dim; j++)
           output[i*dim+j] = output[ (i-1)*dim +j] + start[ i*dim +j];
    }
   //printf("%d uuuu",N*rank);
   for (int i = 0; i < N; i++) output[i] += N*rank;
   memmove(start, output, N*sizeof(float));
   delete[] output;
}

void initialization(std::string input, int N,  float* inarray, float* checkarray){

    if (input.compare("random") == 0) {

        srand(time(NULL));
        //srand(1);
        // generate random array
        for (int i = 0; i < N; i++) {
            float val = 1.0* rand() / RAND_MAX;
            inarray[i] = val;
            checkarray[i] = val;
        }
    } else if (input.compare("test1") == 0){
        // default array of ones for debugging
        for(int i = 0; i < N; i++){ 
            inarray[i] = 1.0;
            checkarray[i] = 1.0;}
    }else
      {for(int i = 0; i < N; i++) {inarray[i] = 1.0;}
    }

}

int main(int argc, char** argv)
{
    int rank, size;
    MPI_Comm comm = MPI_COMM_WORLD;
    MPI_Init(&argc, &argv);
    MPI_Comm_rank(comm, &rank);
    MPI_Comm_size(comm, &size);
    long long int N = 64;
    int dim =1;
    std::string test; 
    std::string input;
    int itrs = 10;
    // parse commandline options ////////////////////////////////////////////
    int opt;
    static struct option long_options[] = {
        {"test",       1, 0, 'm'},
        {"arraysize",  1, 0, 'n'},
        {"input",      1, 0, 'i'},
        {"help",       0, 0, '?'},
        {"dimension",     1, 0, 'd'},
        {0 ,0, 0, 0}
    };

    while ((opt = getopt_long(argc, argv, "m:n:i:d:?t", long_options, NULL)) != EOF) {
        switch (opt) {
        case 'm':
            test = optarg; 
            break;
        case 'n':
            N = atoll(optarg);
            break;
        case 'i':
            input = optarg;
            break;
        case 'd':
            dim = atoi(optarg);
            break;
        case '?':
        default:
            usage(argv[0]);
            return 1;
        }
    }
    // end parsing of commandline options //////////////////////////////////////
    if (rank==0)
       printf("the length, dimensions and total size of the array: %lld, %d, %lld \n\n", N, dim, N*dim);
    long long int Nall = N*dim;
    N= (Nall+size-1)/size; // total size of the array;
    if (rank==0) printf("each proccesor has size %lld\n", N); 
    float* inarray = new float[N];
    float* checkarray = new float[N];
    float* sum0 = new float[dim]; 
    float* recvbuf = new float[dim];

    double cudaTime=0,ompTime=0;

    for (int ite=0; ite<itrs; ite++) {
     printf("\n");printf("%d iteration",ite+1);printf("\n");
     initialization(input, N, inarray, checkarray);
    if (test.compare("cuda") == 0) { 
        // run CUDA implementation
        //printCudaInfo();
                cudaTime += cudaScan(inarray, N, dim);
    } else if (test.compare("omp_cpu") == 0) { // Test find_repeats
        printf("rank %d, uses number of threads: %d\n", rank, NUM_THREADS); 
                ompTime += ompScan(inarray, N, dim);
    } else { 
        usage(argv[0]); 
        exit(1); 
    }
    if (size >1){

     // printf("MPI communication ----------------------------------------\n");
      MPI_Barrier( comm );
      double startTime = CycleTimer::currentSeconds();
      {for (int i = 0; i<dim; i++) {sum0[i] = inarray[i+N-dim];}}
    // communication between cpus of the last element. 
      //MPI_Barrier( comm ); //printf("rank %d ready to communicate\n",rank);

      if (rank !=0){ MPI_Recv(recvbuf, dim, MPI_FLOAT, rank-1 , rank-1, comm, MPI_STATUS_IGNORE);
                  printf("rank %d recv message, tag %d \n",rank,rank-1);
                  for (int i = 0; i<dim; i++) {sum0[i] += recvbuf[i];}}
      if (rank!=size-1) {MPI_Send(sum0, dim, MPI_FLOAT, rank+1 , rank, comm);
                  printf("rank %d send message, tag %d \n",rank,rank);}
     MPI_Barrier( comm );
    double endTime = CycleTimer::currentSeconds(); 
    if (rank==0) printf("MPI communication time  %.3f ms\n", 1000.f *(endTime-startTime));
     // printf("MPI communication ----------------------------------------\n");
    //printf("rank %d, the summation from previous rank %f \n", rank, recvbuf[0]);
    // add sum0 to every processor
    if (test.compare("cuda") == 0) {
        if (rank !=0)  cudaTime += cudaSum(N, dim, recvbuf, inarray); 
    }else if (test.compare("omp_cpu") == 0) { // Test find_repeats
        if (rank !=0) ompTime += cpu_plus_const(N, dim, recvbuf, inarray);
    } else { usage(argv[0]);exit(1);}

    }}

    if (test.compare("cuda") == 0) {
        if (rank ==0) printf("GPU_time: %.3f ms\n", 1000.f * cudaTime/itrs);
    }else if (test.compare("omp_cpu") == 0) { // Test find_repeats
        if (rank ==0) printf("CPU_time: %.3f ms\n", 1000.f * ompTime/itrs);
    } else { usage(argv[0]);exit(1);}

    if (((input.compare("test1") == 0) | (input.compare("test1") == 0) )){
     // check with sequential code.
     cpu_inclusive_scan(checkarray, rank, N, dim);
    // printf("rank %d result \n",rank);for (int i = 0; i < 10; i++){printf("%f ",inarray[i]);}
    // printf("\n");
    // printf("rank %d check \n",rank);for (int i = 0; i < 10; i++){printf("%f ",checkarray[i]);}
    // printf("\n");
     for (int i = 0; i < N; i++)
        {   //printf("real value = %d\n",checkarray[i]);
            if( abs(checkarray[i] - inarray[i])/abs(checkarray[i])>1e-6 )
            {
                fprintf(stderr,
                        "Error: Device inclusive_scan outputs incorrect result."
                        " A[%d] = %f, expecting %f.\n",
                        i, inarray[i], checkarray[i]);
                exit(1);
            }
        }
     printf("Scan outputs are correct!\n");
    }

    delete[] inarray;
    delete[] checkarray;
    return 0;
}

double ompScan( float* start, long long int length, int dim ){

  int lens = length/dim;
  int round_lens = nextPow2(lens);
  int N=dim*round_lens;
  float* output = new float[N];  
  memmove(output, start, length*sizeof(float)); 
  omp_set_num_threads(NUM_THREADS);
  double startTime = CycleTimer::currentSeconds();
 
    // upsweep phase  
    for (int twod = 1; twod < round_lens; twod*=2){
        int twod1 = twod*2;
        int intvl = dim*twod1;
        #pragma omp parallel for
        for (int i = 0; i < N; i += intvl){ 
           for (int j = 0; j < dim; j++){
             output[i+(twod1-1)*dim +j] += output[i+(twod-1)*dim +j];}}
       // implicit synchronization after omp parallel for 
    }

    // downsweep phase
    for (int twod = round_lens/4; twod >= 1; twod /= 2){
        int twod1 = twod*2;
        int intvl = dim*twod1;
        #pragma omp parallel for
        for (int i=intvl; i < N; i += intvl){
          for (int j = 0; j < dim; j++){
             output[i+(twod-1)*dim +j] += output[i-1*dim +j];}}
    }
    #pragma omp barrier
   
    double endTime = CycleTimer::currentSeconds();
    double overallDuration = endTime - startTime;

   memmove(start, output, length*sizeof(float));
   delete[] output;
   return overallDuration;
}

double cpu_plus_const(long long int N, int dim, float* sum0, float* resultarray){

  int dim_id;
  double startTime = CycleTimer::currentSeconds();

     #pragma omp for
     for (int i=0; i<N; i++){
          dim_id = i%dim;
          resultarray[i] += sum0[dim_id]; }

  double endTime = CycleTimer::currentSeconds();
  double overallDuration = endTime - startTime;
   return overallDuration;
}







