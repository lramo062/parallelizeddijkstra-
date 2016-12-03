#include "header.h"
#include "cpu.cu"
#include "gpu.cu"

// ================== Function: main ====================
// driver function for the program
int main() {

    // CPU MEMORY ALLOCATION
    int *numOfVertices = (int *) malloc(sizeof(int)); // # of vertices in the graph
    int *arrayLength = (int *) malloc(sizeof(int));

    // PROMPT USER FOR # OF VERTICES
    printf("Enter the number of vertices for graph: ");
    scanf("%d", numOfVertices);

    *arrayLength = *numOfVertices * *numOfVertices;

    // CPU MEMORY
    float* graph = (float *) malloc(*arrayLength * sizeof(float));
    float* result = (float *) malloc(*arrayLength * sizeof(float));
        
    // FUNCTION CALLS (CPU)
    createGraph(graph, *arrayLength); // Generate the graph & store in array
    printGraph(graph, *arrayLength); // Print the array
    dijkstra(graph, 1, *numOfVertices, result); 
    printSolution(result, *numOfVertices);
    

    // GPU MEMORY ALLOCATION
    int *d_numOfVertices, *d_arrayLength;
    float *d_graph, *d_result;
    bool *d_sptSet;

    cudaMalloc((void **) &d_numOfVertices, sizeof(int));
    cudaMalloc((void **) &d_arrayLength, sizeof(int));
    cudaMalloc((void **) &d_sptSet, (*numOfVertices * sizeof(bool)));
    cudaMalloc((void **) &d_graph, (*arrayLength * sizeof(float)));
    cudaMalloc((void **) &d_result, (*numOfVertices * sizeof(float)));

    // COPY CPU MEM --> GPU MEM
    cudaMemcpy(d_numOfVertices, &numOfVertices, (*arrayLength * sizeof(float)), cudaMemcpyHostToDevice);
    cudaMemcpy(d_arrayLength, &arrayLength, (*arrayLength * sizeof(float)), cudaMemcpyHostToDevice);
    cudaMemcpy(d_graph, &graph, (*arrayLength * sizeof(float)), cudaMemcpyHostToDevice);
    
   
    //gpu_dijkstra<<<1,1>>>(d_graph,index,d_sptSet,*numOfVertices,d_result);
      
      
    // COPY GPU MEM GPU --> CPU
    //cudaMemcpy(result, &d_result, (*arrayLength * sizeof(float)), cudaMemcpyDeviceToHost); 
    
    // FREE GPU MEM
    cudaFree(d_graph);
    cudaFree(d_numOfVertices);
    cudaFree(d_arrayLength);


    // FREE CPU MEM
    free(numOfVertices);
    free(arrayLength);
    free(graph);
    free(result);
    
    return 0;
}
