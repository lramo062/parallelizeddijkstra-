#include "header.h"

// ================== Function: dijkstra ====================
// Funtion that implements Dijkstra's single source shortest path algorithm
// for a graph represented using array representation
__global__ void gpu_setUpGraph(float *graph, bool *sptSet, float *result) {

     // Initialize all distances as INFINITE and stpSet[] as false
     int index = threadIdx.x + blockIdx.x * blockDim.x;

     if(index == ((blockDim.x * blockIdx.x) + blockIdx.x))
       result[index] = 0; // distance to itself is always 0

     else result[index] = INT_MAX; // else initialize infinite
     
     sptSet[index] = false; // is shortest
     __syncthreads();
}

__global__ void gpu_findMinDistance(float* graph, bool* sptSet, float* result) {

    int min = INT_MAX;
    int index = threadIdx.x + blockIdx.x * blockDim.x;
    if(sptSet[index] == false && result[index] < min) {
        min = result[index];
        sptSet[index] = true;
    }
    __syncthreads(); // sync the threads, is Neccessary!
}


__global__ void gpu_updateResult(float* graph, bool* sptSet, float* result) {

    int index = threadIdx.x + blockIdx.x * blockDim.x;
    if(!sptSet[index] && graph[index] && result[index] != INT_MAX &&
       result[index] + graph[index] < result[index]) {
        
        result[index] = result[index] + graph[index];
    }

    __syncthreads();

};