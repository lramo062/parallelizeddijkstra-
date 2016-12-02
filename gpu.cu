#include "header.h"


// __global__ int minDistance(float *dist, bool *sptSet, int *V)
// {
//    // Initialize min value
//    float min = INT_MAX, min_index;
  
//    for (int v = 0; v < V; v++)
//      if (sptSet[v] == false && dist[v] <= min)
//          min = dist[v], min_index = v;
  
//    return min_index;
// }

// // ================== Function: dijkstra ====================
// // Funtion that implements Dijkstra's single source shortest path algorithm
// // for a graph represented using array representation
// __global__ void dijkstra(float *graph, int *src, int *V, float *result)
// {
  
//      bool sptSet[V]; // sptSet[i] will true if vertex i is included in shortest
//                      // path tree or shortest distance from src to i is finalized
  
//      // Initialize all distances as INFINITE and stpSet[] as false
//      for (int i = 0; i < V; i++)
//         result[i] = INT_MAX, sptSet[i] = false;
  
//      // Distance of source vertex from itself is always 0
//      result[src] = 0;
  
//      // Find shortest path for all vertices
//      for (int count = 0; count < V-1; count++)
//      {
//        // Pick the minimum distance vertex from the set of vertices not
//        // yet processed. u is always equal to src in first iteration.
//          int u = minDistance(result, sptSet, V);
  
//        // Mark the picked vertex as processed
//        sptSet[u] = true;
  
//        // Update dist value of the adjacent vertices of the picked vertex.
//        for (int v = 0; v < V; v++) {
  
//          // Update dist[v] only if is not in sptSet, there is an edge from 
//          // u to v, and total weight of path from src to  v through u is 
//          // smaller than current value of dist[v]
//            if (!sptSet[v] && graph[(u*V) + v] && result[u] != INT_MAX
//                && result[u]+graph[(u*V) + v] < result[v])
//                result[v] = result[u] + graph[(u*V) + v];
//        }
       
           
//      }
// }
