compile: make    
executable: genericScan     
command line option:   
   -m [method] (omp_cpu/cuda)   
   -n [length of array]   
   -d [dimension of array]  
   -i [input] (test1: initialize array with all ones)   

MPI (for both cpu and gpu)  
    ibrun -np size ./genericScan [options]   
    (not gpu-aware implementation)   
    
examples:  
    1. run cpu with openmp, 1d vector, length 1000000: ./genericScan -m omp_cpu -n 1000000 -d 1  
    2. run cuda version, 3d vector, check the correctness with array of ones: ./genericScan -m cuda -n 1000000 -d 3 -i test1  
    3. MPI: ibrun -np 4 ./genericScan -m cuda -n 1000000

