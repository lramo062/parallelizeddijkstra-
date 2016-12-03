#include "header.h"

// ================== Function: dijkstra ====================
// Funtion that implements Dijkstra's single source shortest path algorithm
// for a graph represented using array representation
__global__ void gpu_dijkstra(float *graph, int *src, bool *sptSet, int V, float *result)
{

     int *min_index;
     
     // Initialize all distances as INFINITE and stpSet[] as false
     for (int i = 0; i < V; i++)
        result[i] = INT_MAX, sptSet[i] = false;
  
     // Distance of source vertex from itself is always 0
     result[*src] = 0;
  
     // Find shortest path for all vertices
     for (int count = 0; count < V-1; count++)
     {
       // Pick the minimum distance vertex from the set of vertices not
       // yet processed. u is always equal to src in first iteration.
       float min = INT_MAX;
  
       for (int v = 0; v < V; v++)
         if (sptSet[v] == false && result[v] <= min)
           min = result[v], *min_index = v;
       // Mark the picked vertex as processed
       sptSet[*min_index] = true;
  
       // Update dist value of the adjacent vertices of the picked vertex.
       for (int v = 0; v < V; v++) {
  
         // Update dist[v] only if is not in sptSet, there is an edge from 
         // u to v, and total weight of path from src to  v through u is 
         // smaller than current value of dist[v]
           if (!sptSet[v] && graph[(*min_index * V) + v] && result[*min_index] != INT_MAX
               && result[*min_index]+graph[(*min_index * V) + v] < result[v])
               result[v] = result[*min_index] + graph[(*min_index * V) + v];
       }
     }
}