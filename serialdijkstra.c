#include <stdio.h>
#include <stdbool.h>
#include <stdlib.h>
#include <limits.h>
#include <time.h>

int N = 0;

int getmin_index(long **graph, bool pickedVertices[N], int sourceVertex) {
    int minDistance = INT_MAX;
    int min_index = -1;

    for (int j = 0; j < N; j++) {
        if (!pickedVertices[j] && graph[sourceVertex][j] <= minDistance) {
            minDistance = graph[sourceVertex][j];
            min_index = j;
        }
    }
    return min_index;
}

void print(long **graph){
    printf("Matrix: \n");
    for (int i = 0; i < N; i++) {
        for (int j = 0; j < N; j++) {
            printf("%ld ", graph[i][j]);
        }
        printf("\n");
    }
}

void dijkstra(long** graph, int sourceVertex) {

    // Distance from single source to all of the nodes
    bool pickedVertices[N];

    for (int vertex = 0; vertex < N; vertex++) {
        pickedVertices[vertex] = false;
    }

    for (int i = 0; i < N - 1; i++) {
        // Get minimum distance
        int min_index = getmin_index(graph, pickedVertices, sourceVertex);

        // Mark the vertice as picked
        pickedVertices[min_index] = true;

        // Update distance value
        for (int vertex = 0; vertex < N; vertex++) {
            if ((!pickedVertices[vertex]) && 
                (graph[min_index][vertex]) && 
                (graph[sourceVertex][min_index] != INT_MAX) &&
                (graph[sourceVertex][min_index] + graph[min_index][vertex] < graph[sourceVertex][vertex])) {
                
                graph[sourceVertex][vertex] = graph[sourceVertex][min_index] + graph[min_index][vertex];
            }
        }
    }
    return;
}

int main(int argc, char *argv[]) {
	
    // Get matrix size from argument vector in , convert to int
    N = strtol(argv[1], NULL, 10);

    long** graph;
    graph = (long**) malloc(sizeof(long*) * N);
    for (int i = 0; i < N; ++i)
    {
        graph[i] = (long*) malloc(sizeof(long) * N);
    }

    // int numtasks, rank, dest, source, rc, count, tag=1;
    // double start_time, end_time, total_time;

    srand(13517109);
	// Fill the matrix with rand() function
    for (int i = 0; i < N; i++) {
        for (int j = 0; j < N; j++) {
            graph[i][j] = rand() % 1000;
            if (i == j) {
                graph[i][j] = 0;
            }
        }
    }
    graph[0][0] = 0;
    graph[0][1] = 411;
    graph[0][2] = 712;
    graph[0][3] = 657;
    graph[0][4] = 327;

    graph[1][0] = 353;
    graph[1][1] = 0;
    graph[1][2] = 516;
    graph[1][3] = 965;
    graph[1][4] = 525;

    graph[2][0] = 682;
    graph[2][1] = 957;
    graph[2][2] = 0;
    graph[2][3] = 300;
    graph[2][4] = 926;

    graph[3][0] = 42;
    graph[3][1] = 135;
    graph[3][2] = 943;
    graph[3][3] = 0;
    graph[3][4] = 378;

    graph[4][0] = 897;
    graph[4][1] = 919;
    graph[4][2] = 43;
    graph[4][3] = 188;
    graph[4][4] = 0;

    // 0 411 712 657 327
    // 353 0 516 965 525
    // 682 957 0 300 926
    // 42 135 943 0 378
    // 897 919 43 188 0

    // Assign with infinity
    // for (int i = 0; i < N; i++) {
    //     for (int j = 0; j < N; j++) {
    //         if (!(i == j || graph[i][j])){
    //             graph[i][j] = INT_MAX;
    //         }
    //     }
    // }
    // for (int i = 0; i < N; i++) {
    //     for (int j = 0; j < N; j++) {
    //         if (i == j){
    //             graph[i][j] = 0;
    //         }
    //     }
    // }

    clock_t time;
    time = clock();

    FILE *fbefore = fopen("output-before.txt", "w");
    for (int i = 0; i < N; i++) {
        for (int j = 0; j < N; j++) {
            fprintf(fbefore, "%ld ", graph[i][j]);
        }
        fprintf(fbefore, "\n");
    }
    fclose(fbefore);
    
    for (int i = 0; i < N; i++ ) {
        dijkstra(graph, i);
    }

    time = clock() - time;
    double time_taken = ((double)time); // in seconds

    printf("%f µs\n", time_taken);
  

    FILE *f = fopen("output-after.txt", "w");
    for (int i = 0; i < N; i++) {
        for (int j = 0; j < N; j++) {
            fprintf(f, "%ld ", graph[i][j]);
        }
        fprintf(f, "\n");
    }
    fclose(f);


    for (int i = 0; i < N; ++i) {
        free(graph[i]);
    }
    free(graph);

    return 0;
}