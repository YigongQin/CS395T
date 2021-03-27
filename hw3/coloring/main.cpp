#include <stdlib.h>
#include <stdio.h>
#include <getopt.h>
#include <string>
#include <cstring>
#include "CycleTimer.h"
#include <iostream>
#include <fstream>
#include <sstream>  
double cudaMIS(int colorid, int* flag, int num_vert, long long int num_edges, int* row, int* column);
double cudaColoring(int colorid, int* flag, int num_vert, long long int num_edges, int* row, int* column);
void printCudaInfo();

void usage(const char* progname) {
    printf("Usage: %s [options]\n", progname);
    printf("Program Options:\n");
    printf("  -m  --test <TYPE>      Run specified function on input.  Valid tests are: mis, color\n"); 
    printf("  -i  --input <NAME>     Run test on given input matrix. Valid inputs  are: check_graph1, ... YOUR_GRAPH \n");
    printf("  -n  --arraysize <INT>  length of array in one dimension\n");
    printf("  -c  --correctness <INT>   check correctness, Valid input: check\n");
    printf("  -?  --help             This message\n");
}

// squeantially check if the two points of an edge has the same colorid, for MIS, -1 for inactive
void check_list(int colorid, int* flag, int num_vert, long long int num_edges, int* row, int* column){
    int num_colored=0;

    for (int i = 0; i < num_vert; i++ ){
        if ( flag[ i ] ==0 ){fprintf(stderr,
                        "Error: vertex %d not colored\n", i);
                exit(1);
             }
        if ( flag[ i ] == colorid ) { num_colored+=1;}
    }
    printf(" colored vertices %d\n",num_colored);
    for (int i = 0; i < num_edges; i++ ){
        if ( row[i]!=column[i] ){
             if ( (flag[ row[i] ] == flag[ column[i] ]) && ( flag[ row[i] ]>0) ){fprintf(stderr,
                        "Error: Find edge %d (vertices: %d, %d) the same color %d\n",
                        i, row[i], column[i], flag[ row[i] ]);
                exit(1);
             }
         }
    }  

    // check maximum
    int repeats=0;
    for (int i = 0; i < num_vert; i++ ){
       if ( flag[ i ] == -1 ){
             flag[ i ] = colorid;
              for (int j = 0; j < num_edges; j++ ){
                   if ( (flag[ row[j] ] == flag[ column[j] ]) && ( flag[ row[j] ]>0) ) repeats+=1;}

       }
    }
    if (repeats==0){fprintf(stderr,"Error: Not maximum\n");exit(1);}
    printf("the results are correct!!! it is MIS!!!\n");
}



int main(int argc, char** argv)
{
    int N,dim;
    std::string test; 
    std::string input;
    std::string check;
    // parse commandline options ////////////////////////////////////////////
    int opt;
    static struct option long_options[] = {
        {"test",       1, 0, 'm'},
        {"arraysize",  1, 0, 'n'},
        {"input",      1, 0, 'i'},
        {"help",       0, 0, '?'},
        {"correctness",1, 0, 'c'},
        {0 ,0, 0, 0}
    };

    while ((opt = getopt_long(argc, argv, "m:n:i:c:?t", long_options, NULL)) != EOF) {
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
        case 'c':
            check = optarg;
            break;
        case '?':
        default:
            usage(argv[0]);
            return 1;
        }
    }
    // end parsing of commandline options //////////////////////////////////////

    // read the sparse matrix
    int num_lines = 0;
    int num_vert;
    long long int num_edges;
    std::string line;
    std::ifstream graph(input);

    while (std::getline(graph, line) ){
           std::stringstream ss(line);
           if (line[0]!='%'){
                ss >> num_vert >> num_vert >> num_edges;
                break;} }
    //std::cout << line << std::endl;
    //graph >> num_vert >> num_vert >> num_edges;
   
    std::cout << "vertices  " <<num_vert << ";  edges  " <<num_edges << std::endl;
    int* row = new int[num_edges];
    int* column = new int[num_edges];
    float* weights = new float[num_edges];
    int* flag = new int[num_vert];

    while (std::getline(graph, line))
        { 
           std::stringstream ss(line);
           ss >> row[num_lines] >> column[num_lines] >> weights[num_lines];
           //std::cout << row[num_lines] << "  "<< column[num_lines] << "  " << weights[num_lines]<< std::endl;
           num_lines +=1;}//ss >> a >>b >>c; } //++num_lines;
    std::cout << "number of lines of input file: " << num_lines << std::endl;

  //  for (int i = 0; i < num_vert; i++){ flag[i]=0; }
    for (int i = 0; i < num_edges; i++){ row[i]=row[i]-1; column[i]=column[i]-1; }
    //for (int i = 0; i < num_edges; i++){printf("%f ", weights[i]);}
    //printf("\n");

    double cudaTime=0;
    int itrs=10;


    for (int i = 0; i < itrs; i++){
        if (test.compare("mis") == 0){
              for (int j = 0; j < num_vert; j++){ flag[j]=0; } 
              cudaTime += cudaMIS(1, flag, num_vert, num_edges, row, column);
        }else if(test.compare("color") == 0){
              for (int j = 0; j < num_vert; j++){ flag[j]=0; }
              cudaTime += cudaColoring(1, flag, num_vert, num_edges, row, column); 
        }else{ usage(argv[0]);
                exit(1);
        }
    }

//    printf("\nflags of vertices: \n");
 //   for (int i = 0; i < num_vert; i++){printf("%d ", flag[i]);}
//    printf("\nvertices got colored: \n");
//    for (int i = 0; i < num_vert; i++){
//           if (flag[i] >0 ) printf("%d ", i+1);}
//    printf("\n"); 
    printf("GPU_time: %.3f ms\n", 1000.f * cudaTime/itrs);
    if ( (test.compare("mis") == 0) && (check.compare("check") == 0)) {check_list(1, flag, num_vert, num_edges, row, column);}
  
    delete[] row;
    delete[] column;
    delete[] weights;
    delete[] flag;

    return 0;
}







