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
    bool* sptSet = (bool *) malloc(*arrayLength * sizeof(bool));
    
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
    cudaMemcpy(d_graph, graph, (*arrayLength * sizeof(float)), cudaMemcpyHostToDevice);
    
    gpu_setUpGraph<<<*numOfVertices,*numOfVertices>>>(d_graph, d_sptSet, d_result);


    
    for(int i =0; i<*numOfVertices; i++) {
        // has to be done in a for-loop
        gpu_findMinDistance<<<*numOfVertices,*numOfVertices>>>(d_graph,d_sptSet,d_result);
        gpu_updateResult<<<*numOfVertices, *numOfVertices>>>(d_graph, d_sptSet, d_result);
    }

    
    cudaMemcpy(result, d_result, (*arrayLength * sizeof(float)), cudaMemcpyDeviceToHost);
    printGraph(result, *arrayLength);
    
   
    // FREE GPU MEM
    cudaFree(d_graph);
    cudaFree(d_result);
    cudaFree(d_sptSet);
 
    // FREE CPU MEM
    free(numOfVertices);
    free(arrayLength);
    free(graph);
    free(result);
    free(sptSet);
    
    return 0;
}
