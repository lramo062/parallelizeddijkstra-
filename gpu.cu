#include "header.h"

// ================== Function: dijkstra ====================
// Funtion that implements Dijkstra's single source shortest path algorithm
// for a graph represented using array representation
__global__ void gpu_dijkstra(float *graph, bool *sptSet, float *result) {

     // Initialize all distances as INFINITE and stpSet[] as false
     int index = threadIdx.x + blockIdx.x * blockDim.x;
    

     if(index == ((blockDim.x * blockIdx.x) + blockIdx.x))
       result[index] = 0; // distance to itself is always 0

     else result[index] = INT_MAX; // else initialize infinite
     
     sptSet[index] = false; // is shortest
     

     /* COMMON CODE ENDS */
     __syncthreads();


     

     // HAVENT IMPLEMENTED IN PARALLEL
     int min, min_index;
     
     for (int count = 0; count < blockDim.x-1; count++) {
       if (sptSet[index] == false && result[index] <= min)
           min = result[index], min_index = index;

       // Mark the picked vertex as processed
       sptSet[min_index] = true;
  
       // Update dist value of the adjacent vertices of the picked vertex.
       for (int v = 0; v < blockDim.x; v++) {
           // Update dist[v] only if is not in sptSet, there is an edge from 
         // u to v, and total weight of path from src to  v through u is 
         // smaller than current value of dist[v]
         if (!sptSet[v] && graph[(min_index*blockDim.x) + index] && result[min_index] != INT_MAX
             && result[min_index]+graph[(min_index*blockDim.x) + index] < result[index])
           result[index] = result[min_index] + graph[(min_index*blockDim.x) + index];

         else printf("can not find min");
         __syncthreads();
       }
       __syncthreads();
     }
}