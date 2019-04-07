#include <stdio.h>


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


__global__ void kernel(float * d_local_vectors, float * d_centroids, const int vectors_per_proc, 
                       const int number, const int dimension, int * d_count, float * d_sum) {
    int i = blockIdx.x * blockDim.x + threadIdx.x; 
    
    if (i < vectors_per_proc) {
        int cluster_id = d_get_cluster_id(&d_local_vectors[i * dimension], d_centroids, number, dimension);    
        atomicAdd(&d_count[cluster_id], 1);
        d_sum_vectors(&d_local_vectors[i * dimension], &d_sum[cluster_id * dimension], dimension);
    } 
}

extern "C" void do_cluster_on_gpu(const float * local_vectors, const float * centroids, const int vectors_per_proc,
                              const int number, const int dimension, int * count, float * sum) {
    float * d_local_vectors,
          * d_centroids,
          * d_sum;

    int   * d_count;

    int N = vectors_per_proc;

    /* perform memory copy to be accessible from CPU or GPU */
    cudaMalloc(&d_local_vectors, sizeof(float) * vectors_per_proc * dimension);
    cudaMalloc(&d_centroids, sizeof(float) * number * dimension);
    cudaMalloc(&d_sum, sizeof(float) * number * dimension);
    cudaMalloc(&d_count, sizeof(int) * number);
    cudaMemcpy(d_local_vectors, local_vectors, sizeof(float) * vectors_per_proc * dimension, cudaMemcpyHostToDevice);
    cudaMemcpy(d_centroids, centroids, sizeof(float) * number * dimension, cudaMemcpyHostToDevice);
    cudaMemcpy(d_sum, sum, sizeof(float) * number * dimension, cudaMemcpyHostToDevice);
    cudaMemcpy(d_count, count, sizeof(int) * number, cudaMemcpyHostToDevice);

    /* define dimensions of the kernal */
    dim3 dimGrid((N + 255) / 256);
    dim3 dimBlock(256);

    
    /* execute the kernel */
    kernel <<<dimGrid, dimBlock>>>(d_local_vectors, d_centroids, N,
                                   number, dimension, d_count, d_sum);
    
    /* copy back the results to the cpu */
    cudaMemcpy(sum, d_sum, sizeof(float) * number * dimension, cudaMemcpyDeviceToHost);
    cudaMemcpy(count, d_count, sizeof(int) * number, cudaMemcpyDeviceToHost);
    cudaFree(d_local_vectors);
    cudaFree(d_centroids);
    cudaFree(d_sum);
    cudaFree(d_count);
}
