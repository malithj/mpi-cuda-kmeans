#include <stdio.h>
#include <stdlib.h>
#include <mpi.h>
#include <assert.h>
#include "include/txtproc.h"

#define THRESHOLD 0.0001

int get_cluster_id(const float * local_vector, const float * centoids, const int k, const int d);
float vector_distance(const float * v1, const float * v2, const int d);
float sum_vectors(const float * v, float * sum, const int d);
float * create_rand_nums(const int num_elements);

extern void do_cluster_on_gpu(const float * local_vectors, const float * centroids, const int vectors_per_proc,
                              const int number, const int dimension, int * count, float * sum); 

/* variables local to the process */
float * local_vectors,
      * local_sum, 
      * local_centroids;

int   * local_count,
      * local_labels,
      * displacements,       /* displacements */
      * scounts;             /* sub array counts */

/* variables managed by root */
float * global_vectors,
      * global_sum;
          
int   * global_count,
      * global_labels;


int main() {
    int N = 0;   /* Total number of samples */
    int k = 8;   /* Number of clusters */
    int d = 0;   /* Dimensions of data */
    
    MPI_Init(NULL, NULL);   /* Initialize MPI */
    
    int rank,     /* rank of the MPI process */
        tasks;    /* number of total processes */

    double start, /* start time of MPI */
           end;   /* end time of MPI */

    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &tasks);
   
    MPI_Barrier(MPI_COMM_WORLD); /* IMPORTANT */

    /* initialize the process */
    if (rank == 0) {
        /* init global vector */
        //global_vectors = create_rand_nums(d * N);
        
        DocumentObject * dom = get_document_matrix(0.5, 2);
        N = dom->document_count;
        d = dom->vocab_count;
        global_vectors = dom->document_matrix; 
        local_centroids = (float *)malloc(k * d * sizeof(float));
        /* assign the first k as centroids */
        for (int i = 0; i < k * d; i ++) {
            local_centroids[i] = global_vectors[i];
        }
        /* initialize global variables */
        global_sum       = (float *)malloc(k * d * sizeof(float));
        global_count     = (int *)malloc(k * sizeof(int));
        global_labels    = (int *)malloc(N * d * sizeof(int));
    }
    /* Broadcast vector dimensions to all MPI processes */
    MPI_Bcast(&N, 1 , MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Bcast(&d, 1 , MPI_INT, 0, MPI_COMM_WORLD);
   
    /* start timing */ 
    start = MPI_Wtime();

    int vectors_per_process = N / tasks;
    local_vectors   = (float *)malloc(vectors_per_process * d * sizeof(float));
    local_sum       = (float *)malloc(k * d * sizeof(float));
    local_count     = (int *)malloc(k * sizeof(int));
    local_centroids = (float *)malloc(k * d * sizeof(float));
    local_labels    = (int *)malloc(vectors_per_process * sizeof(int));
    if (rank != 0) {
        local_centroids = (float *)malloc(k * d * sizeof(float));
    }
    
   /* scatter the numbers while handling perfectly indivisible cases */
    displacements        = (int *)malloc(tasks * sizeof(int)); 
    scounts              = (int *)malloc(tasks * sizeof(int));

    for (int i = 0; i < tasks; i ++) {
        displacements[i] = i * d * vectors_per_process;
        scounts[i] = d * vectors_per_process;
    }

    if (N % tasks != 0) {
        scounts[tasks - 1] += d * (N % tasks); 
    }

    if (rank == tasks - 1) {
        vectors_per_process += (N % tasks);
        free(local_vectors);
        free(local_labels);
        local_vectors = (float *)malloc(vectors_per_process * d * sizeof(float));
        local_labels = (int *)malloc(vectors_per_process * sizeof(int));
    }
 
    /* Scatter the values to each MPI process */
    MPI_Scatterv(global_vectors, scounts, displacements, MPI_FLOAT, 
                local_vectors, d * vectors_per_process, MPI_FLOAT,
                0, MPI_COMM_WORLD);
    
    float diff = 1;
    int count = 0;
    while (diff > THRESHOLD) {
        count ++;
        /* Broadcast centroids from root to other nodes */
        MPI_Bcast(local_centroids, k * d, MPI_FLOAT, 0, MPI_COMM_WORLD);

        /* reinitialize local sum and count */
        for (int i = 0; i < k * d; i ++) local_sum[i] = 0.0;
        for (int i = 0; i < k; i ++)     local_count[i] = 0;
        
        /* run gpu code */
        do_cluster_on_gpu(local_vectors, local_centroids, vectors_per_process,
                          k, d, local_count, local_sum);

        /* Gather count and sum at root node */
        MPI_Reduce(local_sum, global_sum, k * d, MPI_FLOAT, MPI_SUM, 0, MPI_COMM_WORLD);
        MPI_Reduce(local_count, global_count, k, MPI_INT, MPI_SUM, 0, MPI_COMM_WORLD);
 
        /* compute new cluster centroids */
        if (rank == 0) {
            for (int i = 0; i < k; i ++) {
                for (int j = 0; j < d; j ++) {
                    global_sum[d * i + j] /= global_count[d * i + j];
                }
            }
 
            diff = vector_distance(global_sum, local_centroids, d * k);

            /* copy the new centroids from global sum */
            for (int i = 0; i < k * d; i ++) {
                local_centroids[i] = global_sum[i];
            }
        }

        /* Broadcast the diff for testing */ 
        MPI_Bcast(&diff, 1, MPI_FLOAT, 0, MPI_COMM_WORLD);
    }  


    /* compute final labels */
    for (int i = 0; i < vectors_per_process; i ++) {
        local_labels[i] = get_cluster_id(&local_vectors[i * d], local_centroids, k, d);
    }

    MPI_Gatherv(local_labels, vectors_per_process, MPI_INT,
                global_labels, scounts, displacements, MPI_INT,
                0, MPI_COMM_WORLD);

    MPI_Barrier(MPI_COMM_WORLD); /* IMPORTANT */
    end = MPI_Wtime();

    if (rank == 0) {
        /* for (int i = 0; i < N; i ++) {
            for (int j = 0; j < d; j ++) {
                printf("%f ", global_vectors[i * d + j]);
            }
            printf("%4d\n", global_labels[i]); 
        }*/
        printf("Execution time: %f seconds\n", end - start);
    }
    MPI_Finalize();
}


int get_cluster_id(const float * local_vector, const float * centroids, const int k, const int d) {
    int cluster_id = 0;

    /* first assignment has to be done to find minimum */
    float min_distance = vector_distance(local_vector, &centroids[0], d);
    
    /* for each centroid, compute the distance and find min */ 
    for (int i = 1; i < k; i ++) {
        float distance = vector_distance(local_vector, &centroids[i], d);
        if (distance < min_distance) {
            min_distance = distance;
            cluster_id = i;
        }        
    }
    return cluster_id;
}

float vector_distance(const float * v1, const float * v2, const int d) {
    float distance = 0.0;

    for (int i = 0; i < d; i ++) {
        float diff = v1[i] - v2[i];
        distance += diff * diff;
    }
    return distance;
}

float sum_vectors(const float * v, float * sum, const int d) {
    for (int i = 0; i < d; i ++) {
        sum[i] += v[i];
    }
}

float * create_rand_nums(const int num_elements) {
  float * rand_nums = (float *)malloc(sizeof(float) * num_elements);
  for (int i = 0; i < num_elements; i++) {
    rand_nums[i] = rand() / (float)RAND_MAX;
  }
  return rand_nums;
}
