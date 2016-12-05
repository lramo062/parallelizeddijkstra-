#include "header.h"

// ================== Function: gpu_setUpGraph ====================
// initializes all the unvisted vertices as infinity
// marks all vertices not visited
// sets the weight of distance to itself as 0
// all done in multiple cores / threads
__global__ void gpu_setUpGraph(float *result, bool *visited) {

     // Initialize all distances as INFINITE and stpSet[] as false
     int index = threadIdx.x + blockIdx.x * blockDim.x;

     visited[index] = false;
     
     if(index == ((blockDim.x * blockIdx.x) + blockIdx.x))
       result[index] = 0; // distance to itself is always 0

     else result[index] = INT_MAX;
}


// ================== Function: gpu_dijkstra ====================
// performs dijkstra's algorithm for every vertice in the graph in separate cores
__global__ void gpu_dijkstra(float *graph, float *result, bool* visited, int V) {

    // Find shortest path for all vertices
    for (int count = 0; count < V-1; count++)
    {
        // Pick the minimum distance vertex from the set of vertices not
        // yet processed.
        int min = INT_MAX, u;
        for (int v = 0; v < V; v++)
            if (visited[(V * blockIdx.x) + v] == false && result[(V *blockIdx.x) +  v] <= min)
                min = result[(V * blockIdx.x) + v], u = v;
  
        // Mark the picked vertex as processed
        visited[(V * blockIdx.x) + u] = true;
  
        // Update the wieght value 
        for (int v = 0; v < V; v++) {
  
            // Update only if is not in visited, there is an edge from 
            // u to v, and total weight of path from src to  v through u is 
            // smaller than current value
            if (!visited[(V * blockIdx.x) + v] && graph[(u*V) + v] && result[(V * blockIdx.x) + u] != INT_MAX
                && result[(V * blockIdx.x) + u] + graph[(u*V) + v] < result[(V * blockIdx.x) + v])
                result[(V * blockIdx.x) + v] = result[(V*blockIdx.x) + u] + graph[(u*V) + v];
        }
    }
}


// // ================== Function: gpu_dijkstra_mutli_threaded ==================== (NOT WORKING)
// // performs dijkstra's algorithm for every vertice in the graph in separate cores
// __global__ void gpu_dijkstra_multi_threaded(float *graph, float *result, bool* visited, int V) {

//     // Find shortest path for all vertices
//     for (int count = 0; count < V-1; count++)
//     {
//         // Pick the minimum distance vertex from the set of vertices not
//         // yet processed.
//         int min = INT_MAX, u;
//         if (visited[(V * blockIdx.x) + threadIdx.x] == false && result[(V *blockIdx.x) +  threadIdx.x] <= min)
//                 min = result[(V * blockIdx.x) + threadIdx.x], u = threadIdx.x;
  
//         // Mark the picked vertex as processed
//         visited[(V * blockIdx.x) + u] = true;

//         __syncthreads();
//         // Update the wieght value 
//         // Update only if is not in visited, there is an edge from 
//         // u to v, and total weight of path from src to  v through u is 
//         // smaller than current value
//         if (!visited[(V * blockIdx.x) + threadIdx.x] && graph[(u*V) + threadIdx.x] && result[(V * blockIdx.x) + u] != INT_MAX
//             && result[(V * blockIdx.x) + u] + graph[(u*V) + threadIdx.x] < result[(V * blockIdx.x) + threadIdx.x])
//             result[(V * blockIdx.x) + threadIdx.x] = result[(V*blockIdx.x) + u] + graph[(u*V) + threadIdx.x];

//         __syncthreads();
//     }
   
// }