#include <stdio.h>
#include <stdlib.h>
#include <stdbool.h> // FOR BOOLEAN TYPES
#include <limits.h> // FOR INT_MAX IN DIJKSTRA
#include <sys/time.h> // FOR RANDOMIZE GRAPH IN CREATE GRAPH
#include <math.h> // FOR SQRT
#include <time.h>


int minDistance(float *dist, bool *sptSet, int V);
void printSolution(float *dist, int V);
void dijkstra(float *graph, int src, int V, float *result);
void createGraph(float *arr, int N);
void printGraph(float *arr, int size);
