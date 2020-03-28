#include <stdio.h>
#include <stdbool.h>
#include <stdlib.h>
#include <limits.h>
#include <fstream>

// Number of vertices
int N = 0;

void printGraph(int *arr, int size) {
    for (int i = 0; i < size; i++) {
        for (int j = 0; j < size; j++) {
            printf("%d ", arr[i * size + j]);
        }
        printf("\n");
    }
}

__global__ void initializeVisited(int *result, bool *visited) {
    int index = threadIdx.x + blockIdx.x * blockDim.x;
    visited[index] = false;
    
    if ( index == ((blockDim.x * blockIdx.x) + blockIdx.x)) {
        result[index] = 0;
    } else {
        else result[index] = INT_MAX;
    }
}

__global__ void dijkstra(int *graph, int *result, bool* visited, int N) {
    
   for (int i = 0; i < N-1; i++) {
       // Get vertex with minimum distance
       int blockIndex1D = N  * blockIdx.x;
       int minDistance = INT_MAX;
       int minVertex;

       for (int vertex = 0; vertex < N; vertex++) {
           if (!visited[blockIndex1D + vertex] && result[blockIndex1D +  vertex] <= minDistance) {
                minDistance = result[blockIndex1D + vertex];
                minVertex = vertex;
           }
       }

       visited[blockIndex1D + minVertex] = true;
       int minBlockIndex1D = N * minVertex;
       
       for (int vertex = 0; vertex < N; vertex++) {
           if (!visited[blockIndex1D + vertex] &&
                graph[minBlockIndex1D + vertex] &&
                result[blockIndex1D + minVertex] != INT_MAX &&
                result[blockIndex1D + minVertex] + graph[minBlockIndex1D + vertex] < result[blockIndex1D + vertex]) {
                    result[blockIndex1D + vertex] = result[blockIndex1D + minVertex] + graph[minBlockIndex1D + vertex];
                }
       }
   }
}

int main(int argc, char *argv[]) {

    // Get matrix size from argument vector in , convert to int
    N = strtol(argv[1], NULL, 10);
    printf("N: %d\n ", N);

    int* cpuGraph = (int *) malloc(sizeof(int) * N * N);
    int* result = (int *) malloc(N * N * sizeof(int));
    
    srand(13517115);
    // Fill the matrix with rand() function
    for (int i = 0; i < N; i++) {
        for (int j = 0; j < N; j++) {
            cpuGraph[i * N + j] = rand() % 1000;
            if (i == j) {
                cpuGraph[i * N + j] = 0;
            }
        }
    }

    // Variable declaration for measuring time
    float totalTime = 0;
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    int *gpuGraph, *gpuResult;
    bool *gpuVisited;

    cudaMalloc((void **) &gpuGraph, (sizeof(int) * N * N));
    cudaMalloc((void **) &gpuVisited, (sizeof(bool) * N * N));
    cudaMalloc((void **) &gpuResult, (sizeof(int) * N * N));
    
    cudaEventRecord(start);
    // Copy from cpuGraph to gpuGraph (transfer from cpu to gpu!)
    cudaMemcpy(gpuGraph, cpuGraph, (sizeof(int) * N * N), cudaMemcpyHostToDevice);

    // Initialize visited graph: dimGrid = N, dimBlock = N
    initializeVisited<<<N, N>>>(gpuResult, gpuVisited);
    // Do the dijkstra: dimGrid = N, dimBlock = 1 (only 1 thread per block)
    dijkstra<<<N, 1>>>(gpuGraph, gpuResult, gpuVisited, N);
    cudaEventRecord(stop);

    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&totalTime, start, stop);

    // Copy from gpuGraph to cpuGraph (transfer from gpu to cpu!)
    cudaMemcpy(result, gpuResult, (sizeof(int) * N * N), cudaMemcpyDeviceToHost);
    // Print elapsed time in microsecs
    printf("%f µs\n", totalTime * 1000);

    char filename[100];
    snprintf(filename, sizeof(char) * 100, "output-%i.txt", N);
    FILE *f = fopen(filename, "w");
    for (int i = 0; i < N; i++) {
        for (int j = 0; j < N; j++) {
            fprintf(f, "%d ", result[i * N + j]);
        }
        fprintf(f, "\n");
    }
    fclose(f);

    cudaFree(gpuResult);
    cudaFree(gpuVisited);
    cudaFree(gpuGraph);
    free(cpuGraph);
    free(result);

    return 0;
}
