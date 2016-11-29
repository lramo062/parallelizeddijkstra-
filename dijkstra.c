#include <stdio.h>
#include <stdlib.h>
#include <stdbool.h> // FOR BOOLEAN TYPES
#include <limits.h> // FOR INT_MAX IN DIJKSTRA
#include <sys/time.h> // FOR RANDOMIZE GRAPH IN CREATE GRAPH
#include <math.h> // FOR SQRT

// ================== Function: minDistance ====================
// A utility function to find the vertex with minimum distance value, from
// the set of vertices not yet included in shortest path tree
int minDistance(int dist[], bool sptSet[], int V)
{
   // Initialize min value
   int min = INT_MAX, min_index;
  
   for (int v = 0; v < V; v++)
     if (sptSet[v] == false && dist[v] <= min)
         min = dist[v], min_index = v;
  
   return min_index;
}

// ================== Function: printSolution ====================
// A utility function to print the constructed distance array
void printSolution(int dist[], int V) { 
   printf("Vertex   Distance from Source\n");
   for (int i = 0; i < V; i++)
      printf("%d \t\t %d\n", i, dist[i]);
}

// ================== Function: dijkstra ====================
// Funtion that implements Dijkstra's single source shortest path algorithm
// for a graph represented using adjacency matrix representation
void dijkstra(int graph[], int src, int V)
{
    // initialize the array that holds the distance to each vertex.
     int dist[V];     // The output array.  dist[i] will hold the shortest
                      // distance from src to i
  
     bool sptSet[V]; // sptSet[i] will true if vertex i is included in shortest
                     // path tree or shortest distance from src to i is finalized
  
     // Initialize all distances as INFINITE and stpSet[] as false
     for (int i = 0; i < V; i++)
        dist[i] = INT_MAX, sptSet[i] = false;
  
     // Distance of source vertex from itself is always 0
     dist[src] = 0;
  
     // Find shortest path for all vertices
     for (int count = 0; count < V-1; count++)
     {
       // Pick the minimum distance vertex from the set of vertices not
       // yet processed. u is always equal to src in first iteration.
         int u = minDistance(dist, sptSet, V);
  
       // Mark the picked vertex as processed
       sptSet[u] = true;
  
       // Update dist value of the adjacent vertices of the picked vertex.
       for (int v = 0; v < V; v++) {
  
         // Update dist[v] only if is not in sptSet, there is an edge from 
         // u to v, and total weight of path from src to  v through u is 
         // smaller than current value of dist[v]
           if (!sptSet[v] && graph[(u*V) + v] && dist[u] != INT_MAX
               && dist[u]+graph[(u*V) + v] < dist[v])
               dist[v] = dist[u] + graph[(u*V) + v];
       }
       
           
     }
  
     // print the constructed distance array
     printSolution(dist, V);
}

// ================== Function: createGraph ====================
// creates a graph and stores it in array representation
// toggle commented line for a symmetric graph
void createGraph(int *a, int N) {

    time_t t; // used for randomizing values
    int col; 
    int row;
    int maxWeight = 100; // limit the weight an edge can have

    srand((unsigned) time(&t)); // generate random

    for (col = 0; col < sqrt(N); col++) { 
	for(row = 0; row < sqrt(N); row++) {
            if( col != row){
                a[(int)(row*sqrt(N)) + col] = rand() % maxWeight; // assign random

                // have a symmetric graph
                //  a[(int)(col*sqrt(N)) + row] = a[(int)(row*sqrt(N)) + col];
            }
            else
                a[(int)(row*sqrt(N)) + col] = 0; // NO LOOPS
        }
    }
};

// ================== Function: printGraph ====================
// prints the graph as it would look in adjacency matrix representation
void printGraph(int arr[], int size) {
    int index;
    printf("Graph:\n");
    for(index = 0; index < size; index++) {
        if(((index + 1) % (int)sqrt(size)) == 0) {
            printf("%3d\n", arr[index]);
        }
        else {
            printf("%3d ", arr[index]);
        }
    }
}

// ================== Function: main ====================
// driver function for the program
int main(int argc, char *argv[]) {
    
    const int numOfVertices =  atoi(argv[1]);    
    const int arrayLength = numOfVertices * numOfVertices;
    
    int arr[arrayLength]; // Initialize array that will hold graph

    createGraph(arr, arrayLength); // Generate the graph & store in array
    printGraph(arr, arrayLength); // Print the array

    dijkstra(arr, 1, numOfVertices);

    return 0;
}
