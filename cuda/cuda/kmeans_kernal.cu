#include <stdio.h>

#define THRESHOLD 0.0001

__device__ float d_vector_distance(const float * v1, const float * v2, const int d) {
    float distance = 0.0;

    for (int i = 0; i < d; i ++) {
        float diff = v1[i] - v2[i];
        distance += diff * diff;
    }
    return distance;
}


__device__ int d_get_cluster_id(const float * local_vector, const float * centroids, const int k, const int d) {
    int cluster_id = 0;

    /* first assignment has to be done to find minimum */
    float min_distance = d_vector_distance(local_vector, &centroids[0], d);

    /* for each centroid, compute the distance and find min */
    for (int i = 1; i < k; i ++) {
        float distance = d_vector_distance(local_vector, &centroids[i], d);
        if (distance < min_distance) {
            min_distance = distance;
            cluster_id = i;
        }
    }
    return cluster_id;
}


__device__ void d_sum_vectors(const float * v, float * sum, const int d) {
    for (int i = 0; i < d; i ++) {
        atomicAdd(&sum[i], v[i]);
    }
}


__global__ void assign_clusters(float * d_local_vectors, float * d_centroids, const int vectors_per_proc, 
                                const int number, const int dimension, int * d_count, float * d_sum) {
    int i = blockIdx.x * blockDim.x + threadIdx.x; 
    
    if (i < vectors_per_proc) {
        int cluster_id = d_get_cluster_id(&d_local_vectors[i * dimension], d_centroids, number, dimension);    
        atomicAdd(&d_count[cluster_id], 1);
        d_sum_vectors(&d_local_vectors[i * dimension], &d_sum[cluster_id * dimension], dimension);
    } 
}

float h_vector_distance(const float * v1, const float * v2, const int d) {
    float distance = 0.0;

    for (int i = 0; i < d; i ++) {
        float diff = v1[i] - v2[i];
        distance += diff * diff;
    }
    return distance;
}

__global__ void compute_updated_means(float * centroids, float * sum, int * count, const int d) {
    const int cluster = threadIdx.x;
    for (int i = 0; i < d; i ++) {
        sum[d * cluster + i] /= count[cluster];
        centroids[d * cluster + i] = sum[d * cluster + i];
    }
}

extern "C" void do_cluster_on_gpu(const float * global_vectors, const int total_number,
                                  const int num_of_clusters, const int dimension) {
    
    float * centroids,
          * updated_centroids,
          * sum;

    float * d_global_vectors,
          * d_centroids,
          * d_sum;

    int   * count;
    int   * d_count;

    cudaEvent_t start, stop;
    float       elapsedTime;

    /* begin timing */
    cudaEventCreate(&start);
    cudaEventRecord(start, 0);

    /* malloc on CPU */
    centroids         = (float *)malloc(sizeof(float) * num_of_clusters * dimension);
    updated_centroids = (float *)malloc(sizeof(float) * num_of_clusters * dimension);
    sum               = (float *)malloc(sizeof(float) * num_of_clusters * dimension);
    count             = (int   *)malloc(sizeof(int) * num_of_clusters);

    /* initialize centroids */
    for (int i = 0; i < num_of_clusters * dimension; i ++) {
        centroids[i] = global_vectors[i];
        updated_centroids[i] = global_vectors[i];
        sum[i] = 0;
    }

    for (int i = 0; i < num_of_clusters; i ++) {
        count[i] = 0;
    }

    /* perform memory copy to be accessible from CPU or GPU */
    cudaMalloc(&d_global_vectors, sizeof(float) * total_number * dimension);
    cudaMalloc(&d_centroids, sizeof(float) * num_of_clusters * dimension);
    cudaMalloc(&d_sum, sizeof(float) * num_of_clusters * dimension);
    cudaMalloc(&d_count, sizeof(int) * num_of_clusters);
    
    cudaMemcpy(d_global_vectors, global_vectors, sizeof(float) * total_number * dimension, cudaMemcpyHostToDevice);
    cudaMemcpy(d_centroids, centroids, sizeof(float) * num_of_clusters * dimension, cudaMemcpyHostToDevice);
    cudaMemcpy(d_sum, sum, sizeof(float) * num_of_clusters * dimension, cudaMemcpyHostToDevice);
    cudaMemcpy(d_count, count, sizeof(int) * num_of_clusters, cudaMemcpyHostToDevice);


    /* define dimensions of the kernal */
    dim3 dimGrid((total_number + 255) / 256);
    dim3 dimBlock(256);

    float diff = 1;

    while (diff > THRESHOLD) {    
        cudaMemset(d_count, 0, sizeof(int) * num_of_clusters);
        cudaMemset(d_sum, 0, sizeof(float) * num_of_clusters * dimension);
        /* execute the kernel */
        assign_clusters <<<dimGrid, dimBlock>>>(d_global_vectors, d_centroids, total_number,
                                                num_of_clusters, dimension, d_count, d_sum);
        cudaDeviceSynchronize();
        compute_updated_means <<<1, num_of_clusters>>>(d_centroids, d_sum, d_count, dimension);
        cudaDeviceSynchronize();
        cudaMemcpy(updated_centroids, d_centroids, sizeof(float) * num_of_clusters * dimension, cudaMemcpyDeviceToHost); 
        diff = h_vector_distance(centroids, updated_centroids, num_of_clusters * dimension); 
        /* copy centroids */
        for (int i = 0; i < num_of_clusters * dimension; i ++) {
            centroids[i] = updated_centroids[i];
        }
    }

    /* end timing */
    cudaEventCreate(&stop);
    cudaEventRecord(stop, 0);
    cudaEventSynchronize(stop);

    cudaEventElapsedTime(&elapsedTime, start, stop);
    printf("Execution time: %f seconds\n", elapsedTime / 1000);

    free(centroids);
    free(updated_centroids);
    free(sum);
    free(count);
    cudaFree(d_global_vectors);
    cudaFree(d_centroids);
    cudaFree(d_sum);
    cudaFree(d_count);
}

