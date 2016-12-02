#include "header.h"
#include "cpu.cu"
#include "gpu.cu"

// ================== Function: main ====================
// driver function for the program
int main() {

    // FETCH INPUT FROM USER
    int *numOfVertices = (int *) malloc(sizeof(int)); // # of vertices in the graph
    int *arrayLength = (int *) malloc(sizeof(int));

    printf("Please enter the number of vertices for graph: ");
    scanf("%d", numOfVertices);

    *arrayLength = *numOfVertices * *numOfVertices;

    float* graph = (float *) malloc(*arrayLength * sizeof(float));
    float* result = (float *) malloc(sizeof(float) * *numOfVertices);
    
    

    // FUNCTION CALLS (CPU)
    createGraph(graph, *arrayLength); // Generate the graph & store in array
    printGraph(graph, *arrayLength); // Print the array
    dijkstra(graph, 1, *numOfVertices, result); 
    printSolution(result, *numOfVertices);
    

    // FUNCTION CALLS (GPU)
    int *d_numOfVertices, *d_arrayLength;
    float* d_graph; // GRAPH DEVICE COPY

    cudaMalloc((void **) &d_numOfVertices, sizeof(int));
    cudaMalloc((void **) &d_arrayLength, sizeof(int));
    cudaMalloc((void **) &d_graph, (*arrayLength * sizeof(float))); // allocate mem

    cudaMemcpy(d_numOfVertices, &numOfVertices, (*arrayLength * sizeof(float)), cudaMemcpyHostToDevice);
    cudaMemcpy(d_arrayLength, &arrayLength, (*arrayLength * sizeof(float)), cudaMemcpyHostToDevice);
    cudaMemcpy(d_graph, &graph, (*arrayLength * sizeof(float)), cudaMemcpyHostToDevice);
    
    
    // GPU Dijkstra Call

    cudaFree(d_graph);
    cudaFree(d_numOfVertices);
    cudaFree(d_arrayLength);


    free(numOfVertices);
    free(arrayLength);
    free(graph);
    free(result);
    
    return 0;
}
