#include "header.h"
#include "cpu.cu"
#include "gpu.cu"

// ================== Function: main ====================
// driver function for the program
int main() {

    int *numOfVertices = (int *) malloc(sizeof(int)); // # of vertices in the graph
    int *arrayLength = (int *) malloc(sizeof(int));

    // PROMPT USER FOR # OF VERTICES
    printf("Enter the number of vertices for graph: ");
    scanf("%d", numOfVertices);

    *arrayLength = *numOfVertices * *numOfVertices;





    
    // ALLOCATE CPU MEMORY
    float* graph = (float *) malloc(*arrayLength * sizeof(float));
    float* result = (float *) malloc(*arrayLength * sizeof(float));
        
    // FUNCTION CALLS (CPU)
    createGraph(graph, *arrayLength); // Generate the graph & store in array
    printGraph(graph, *arrayLength); // Print the array

    dijkstra(graph, 0, *numOfVertices, result); 
    printSolution(result, *numOfVertices);
    
    dijkstra(graph, 1, *numOfVertices, result); 
    printSolution(result, *numOfVertices);

    dijkstra(graph, 2, *numOfVertices, result); 
    printSolution(result, *numOfVertices);


    
    // ALLOCATE GPU MEMORY 
    float *d_graph, *d_result;
    bool *d_sptSet;

    cudaMalloc((void **) &d_sptSet, (*arrayLength * sizeof(bool)));
    cudaMalloc((void **) &d_graph, (*arrayLength * sizeof(float)));
    cudaMalloc((void **) &d_result, (*arrayLength * sizeof(float)));
    
    // COPY CPU MEM --> GPU MEM
    cudaMemcpy(d_graph, &graph, (*arrayLength * sizeof(float)), cudaMemcpyHostToDevice);
    
    gpu_dijkstra<<<*numOfVertices,*numOfVertices>>>(d_graph, d_sptSet,d_result);
      
    // COPY GPU MEM GPU --> CPU
    cudaMemcpy(result, d_result, (*arrayLength * sizeof(float)), cudaMemcpyDeviceToHost); 
    printGraph(result, *arrayLength);


    
    // FREE GPU MEM
    cudaFree(d_graph);
    cudaFree(d_result);
 
    // FREE CPU MEM
    free(numOfVertices);
    free(arrayLength);
    free(graph);
    free(result);
    
    return 0;
}
