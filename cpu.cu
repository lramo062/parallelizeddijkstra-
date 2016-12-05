#include "header.h"
// ================== Function: minDistance ====================
// A utility function to find the vertex with minimum distance value, from
// the set of vertices not yet included in shortest path tree
int minDistance(float *dist, bool *sptSet, int V)
{
   // Initialize min value
   float min = INT_MAX, min_index;
  
   for (int v = 0; v < V; v++)
     if (sptSet[v] == false && dist[v] <= min)
         min = dist[v], min_index = v;
  
   return min_index;
}

// ================== Function: printSolution ====================
// A utility function to print the constructed distance array
void printSolution(int src, float *dist, int V) { 

    printf("\nVertex   Distance from Source: %d\n", src);
    for (int i = 0; i < V; i++) {
        printf("%d \t\t %.1f\n", i, dist[i]);
    }
}

// ================== Function: dijkstra ====================
// Funtion that implements Dijkstra's single source shortest path algorithm
// for a graph represented using array representation
void dijkstra(float *graph, int src, int V, float *result)
{
  
     bool sptSet[V]; // sptSet[i] will true if vertex i is included in shortest
                     // path tree or shortest distance from src to i is finalized
  
     // Initialize all distances as INFINITE and stpSet[] as false
     for (int i = 0; i < V; i++)
        result[i] = INT_MAX, sptSet[i] = false;
  
     // Distance of source vertex from itself is always 0
     result[src] = 0;
  
     // Find shortest path from src
     for (int count = 0; count < V-1; count++)
     {
       // Pick the minimum distance vertex from the set of vertices not
       // yet processed. u is always equal to src in first iteration.
         int u = minDistance(result, sptSet, V);
  
       // Mark the picked vertex as processed
       sptSet[u] = true;
  
       // Update dist value of the adjacent vertices of the picked vertex.
       for (int v = 0; v < V; v++) {
  
         // Update dist[v] only if is not in sptSet, there is an edge from 
         // u to v, and total weight of path from src to  v through u is 
         // smaller than current value of dist[v]
           if (!sptSet[v] && graph[(u*V) + v] && result[u] != INT_MAX
               && result[u]+graph[(u*V) + v] < result[v])
               result[v] = result[u] + graph[(u*V) + v];
       }
       
           
     }
  
     // print the constructed distance array
     // printSolution(dist, V); <--- NOT PRINTING ANYMORE
}

// ================== Function: createGraph ====================
// creates a graph and stores it in array representation
// toggle commented line for a symmetric graph
void createGraph(float *arr, int N) {

    time_t t; // used for randomizing values
    int col; 
    int row;
    int maxWeight = 100; // limit the weight an edge can have

    srand((unsigned) time(&t)); // generate random

    for (col = 0; col < sqrt(N); col++) { 
	for(row = 0; row < sqrt(N); row++) {
            if( col != row){
                arr[(int)(row*sqrt(N)) + col] = rand() % maxWeight; // assign random

                // have a symmetric graph
                arr[(int)(col*sqrt(N)) + row] = arr[(int)(row*sqrt(N)) + col];
            }
            else
                arr[(int)(row*sqrt(N)) + col] = 0; // NO LOOPS
        }
    }
};

// ================== Function: printGraph ====================
// prints the graph as it would look in array representation
void printGraph(float *arr, int size) {
    int index;
    printf("\nGraph:\n");
    for(index = 0; index < size; index++) {
        if(((index + 1) % (int)sqrt(size)) == 0) {
            printf("%5.1f\n", arr[index]);
        }
        else {
            printf("%5.1f ", arr[index]);
        }
    }
    printf("\n");
}
