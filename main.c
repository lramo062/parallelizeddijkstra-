#include "header.h"
#include "cpu.c"
#include "gpu.c"

// ================== Function: main ====================
// driver function for the program
int main() {

    // FETCH INPUT FROM USER
    int numOfVertices; // # of vertices in the graph
    printf("Please enter the number of vertices for graph: ");
    scanf("%d", &numOfVertices);
    const int arrayLength = numOfVertices * numOfVertices;

    float* result = (float *) malloc(sizeof(float) * numOfVertices);
    // HOST COPY
    float* graph = (float *) malloc(arrayLength * sizeof(float));

    // DEVICE COPY
    // float* d_graph;
    // allocate memory for the graph
    // cudaMalloc((void **) &d_graph, (arrayLength * sizeof(float)));

    // FUNCTION CALLS (CPU)
    createGraph(graph, arrayLength); // Generate the graph & store in array
    printGraph(graph, arrayLength); // Print the array
    dijkstra(graph, 1, numOfVertices, result); 
    printSolution(result, numOfVertices);
    
    // FUNCTION CALLS (GPU)
    /* cudaMemcpy(d_graph, &graph, (arrayLength * sizeof(float)), cudaMemcpyHostToDevice); */
    /* cudaFree(d_graph); */
    return 0;
}
   
