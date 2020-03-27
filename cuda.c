    #include <stdio.h>
    #include <stdbool.h>
    #include <stdlib.h>
    #include <limits.h>

    #include <omp.h>
    #include <time.h>

    int N = 0;

    int getmin_index(long *graph, bool pickedVertices[N], int sourceVertex) {
        int minDistance = INT_MAX;
        int min_index = -1;

        for (int j = 0; j < N; j++) {
            if (!pickedVertices[j] && graph[sourceVertex*N+j] <= minDistance) {
                minDistance = graph[sourceVertex*N+j];
                min_index = j;
            }
        }
        return min_index;
    }

    void print(long *graph){
        printf("Matrix: \n");
        for (int i = 0; i < N; i++) {
            for (int j = 0; j < N; j++) {
                printf("%ld ", graph[i*N+j]);
            }
            printf("\n");
        }
    }

    void dijkstra(long* graph, int sourceVertex) {

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
                    (graph[min_index*N+vertex]) && 
                    (graph[sourceVertex*N+min_index] != INT_MAX) &&
                    (graph[sourceVertex*N+min_index] + graph[min_index*N+vertex] < graph[sourceVertex*N+vertex])) {
                    
                    graph[sourceVertex*N+vertex] = graph[sourceVertex*N+min_index] + graph[min_index*N+vertex];
                }
            }
        }
        return;
    }


    int main(int argc, char *argv[]) {
        
        // Get matrix size from argument vector in , convert to int
        int thread_count = strtol(argv[1], NULL, 10);
        N = strtol(argv[2], NULL, 10);

        long* graph;
        graph = (long*) malloc(sizeof(long) * N*N);

        srand(13517115);
        // Fill the matrix with rand() function
        for (int i = 0; i < N; i++) {
            for (int j = 0; j < N; j++) {
                graph[i*N+j] = rand();
            }
        }

        for (int i = 0; i < N; i++) {
            for (int j = 0; j < N; j++) {
                if (i == j){
                    graph[i*N+j] = 0;
                }
            }
        }

        // Assign with infinity
        for (int i = 0; i < N; i++) {
            for (int j = 0; j < N; j++) {
                if (!(i == j || graph[i*N+j])){
                    graph[i*N+j] = INT_MAX;
                }
            }
        }

        printf("start\n");

        // Calculate start time
        double start = omp_get_wtime();
        int idx;
        #pragma omp parallel for num_threads (thread_count)
        for (idx = 0; idx < N; ++idx)
        {
            dijkstra(graph, idx);
        }

        double end = omp_get_wtime();
        double total = end - start;
        printf("%f µs\n", total*1000000);
        printf("end\n");

        char filename[100];
        snprintf(filename, sizeof(char) * 100, "output-%i.txt", N);
        // Now print the output matrix
        FILE *f = fopen(filename, "w");
        for (int i = 0; i < N; i++) {
            for (int j = 0; j < N; j++) {
                fprintf(f, "%ld ", graph[i * N +j]);
            }
            fprintf(f, "\n");
        }
        fclose(f);

        free(graph);

        return 0;
    }
