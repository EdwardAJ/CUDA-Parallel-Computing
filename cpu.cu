#include "header.h"
// ================== Function: minDistance ====================
// A utility function to find the vertex with minimum distance value, from
// the set of vertices not yet included in shortest path tree
int minDistance(int *dist, bool *sptSet, int V)
{
   // Initialize min value
   int min = INT_MAX, min_index;
  
   for (int v = 0; v < V; v++)
     if (sptSet[v] == false && dist[v] <= min)
         min = dist[v], min_index = v;
  
   return min_index;
}

// ================== Function: printSolution ====================
// A utility function to print the constructed distance array
void printSolution(int src, int *dist, int V) { 

    printf("\nVertex   Distance from Source: %d\n", src);
    for (int i = 0; i < V; i++) {
        printf("%d \t\t %d\n", i, dist[i]);
    }
}

// ================== Function: dijkstra ====================
// Funtion that implements Dijkstra's single source shortest path algorithm
// for a graph represented using array representation
void dijkstra(int *graph, int src, int V, int *result)
{
  
     bool sptSet[V]; // sptSet[i] will true if vertex i is included in shortest
                     // path tree or shortest distance from src to i is finalized
  
     // Initialize all distances as INFINITE and stpSet[] as false
     for (int i = 0; i < V; i++)
        result[i] = INT_MAX, sptSet[i] = false;
  
     // Distance of source vertex from itself is always 0
     result[src] = 0;
  
     // Find shortest path from src
     for (int count = 0; count < V-1; count++)
     {
       // Pick the minimum distance vertex from the set of vertices not
       // yet processed. u is always equal to src in first iteration.
         int u = minDistance(result, sptSet, V);
  
       // Mark the picked vertex as processed
       sptSet[u] = true;
  
       // Update dist value of the adjacent vertices of the picked vertex.
       for (int v = 0; v < V; v++) {
  
         // Update dist[v] only if is not in sptSet, there is an edge from 
         // u to v, and total weight of path from src to  v through u is 
         // smaller than current value of dist[v]
           if (!sptSet[v] && graph[(u*V) + v] && result[u] != INT_MAX
               && result[u]+graph[(u*V) + v] < result[v])
               result[v] = result[u] + graph[(u*V) + v];
       }
       
           
     }
  
     // print the constructed distance array
     // printSolution(dist, V); <--- NOT PRINTING ANYMORE
}

// ================== Function: createGraph ====================
// creates a graph and stores it in array representation
// toggle commented line for a symmetric graph
void createGraph(int *graph, int N) {
    srand(13517115);
    // Fill the matrix with rand() function
    for (int i = 0; i < N; i++) {
        for (int j = 0; j < N; j++) {
            graph[i*N+j] = rand() % 1000;
            if (i == j) {
                graph[i*N+j] = 0;
            }
        }
    }
};


// ================== Function: printGraph ====================
// prints the graph as it would look in array representation
void printGraph(int *arr, int size) {
    // int index;
    // printf("\nGraph:\n");
    // for(index = 0; index < size; index++) {
    //     if(((index + 1) % (int)sqrt(size)) == 0) {
    //         printf("%d\n", arr[index]);
    //     }
    //     else {
    //         printf("%d ", arr[index]);
    //     }
    // }
    // printf("\n");
    for (int i = 0; i < size; i++) {
        for (int j = 0; j < size; j++) {
            printf("%d ", arr[i * size + j]);
        }
        printf("\n");
    }
}
