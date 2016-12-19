#include "header.h"
#include "cpu.cu"
#include "gpu.cu"


int main() {

    /**************************** TAKE USER INPUT *****************************/
    
    int *numOfVertices = (int *) malloc(sizeof(int));
    int *arrayLength = (int *) malloc(sizeof(int));

    // PROMPT USER FOR # OF VERTICES
    printf("\nEnter the number of vertices for graph: ");
    scanf("%d", numOfVertices);

    // WILL BE AN ARRAY REPRESENTATION OF A MATRIX
    *arrayLength = *numOfVertices * *numOfVertices;


    /***************************** CPU DIJKSTRA  ******************************/

    // to measure dijkstra on the cpu
    clock_t start_cpu, end_cpu;
    double total_time_cpu;

    // allocate cpu memory
    float* graph = (float *) malloc(*arrayLength * sizeof(float));
    float* result = (float *) malloc(*arrayLength * sizeof(float));

    createGraph(graph, *arrayLength); // Generate the graph & store in array
    printGraph(graph, *arrayLength); // Print the array
    
    for(int vertex = 0; vertex<*numOfVertices; vertex++) {
        start_cpu = clock(); // start time
        dijkstra(graph, vertex, *numOfVertices, result);
        end_cpu = clock() - start_cpu; // end time

        total_time_cpu += ((double) end_cpu / CLOCKS_PER_SEC);
        printSolution(vertex, result, *numOfVertices);
    }
    //printf("TOTAL TIME CPU: %f\n", total_time_cpu);
    
    /***************************** GPU DIJKSTRA  ******************************/

    // to measure time on the gpu
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    
    // initialize the varibles needed in the gpu
    float *d_graph, *d_result;
    bool *d_visited;

    // allocate memory in the gpu for our variables
    cudaMalloc((void **) &d_graph, (*arrayLength * sizeof(float)));
    cudaMalloc((void **) &d_result, (*arrayLength * sizeof(float)));
    cudaMalloc((void **) &d_visited, (*arrayLength * sizeof(bool)));
    
    // copy graph generated in the cpu to the gpu
    cudaMemcpy(d_graph, graph, (*arrayLength * sizeof(float)), cudaMemcpyHostToDevice);

    // set up the graph using multiple cores & threads
    gpu_setUpGraph<<<*numOfVertices,*numOfVertices>>>(d_result, d_visited);

    cudaEventRecord(start);
    // perform dijstra on ALL vertices as src vertex using multiple cores
    gpu_dijkstra<<<*numOfVertices,1>>>(d_graph,d_result, d_visited, *numOfVertices);

    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    float milliseconds = 0;
    cudaEventElapsedTime(&milliseconds, start, stop);
    // copy the results back to cpu
    cudaMemcpy(result, d_result, (*arrayLength * sizeof(float)), cudaMemcpyDeviceToHost);
    printGraph(result, *arrayLength);
    //printf("TOTAL TIME GPU: %.10f\n", milliseconds * .0001);

    
    // free the gpu memory
    cudaFree(d_graph);
    cudaFree(d_result);
    cudaFree(d_visited);

    
    // free the cpu memory
    free(numOfVertices);
    free(arrayLength);
    free(graph);
    free(result);
  
    return 0;
}
