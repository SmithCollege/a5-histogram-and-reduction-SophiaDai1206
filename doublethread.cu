#include <stdio.h>
#include <cuda_runtime.h>
#include <chrono>
#include <iostream>

__global__ void sumReduction(int *input, int *output, int n) {
    extern __shared__ int sharedData[];

    unsigned int tid = threadIdx.x;
    unsigned int i = blockIdx.x * blockDim.x * 2 + tid;
    
    sharedData[tid] = (i < n) ? input[i] : 0;
    if (i + blockDim.x < n) sharedData[tid] += input[i + blockDim.x];
    __syncthreads();

    for (unsigned int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) {
            sharedData[tid] += sharedData[tid + s];
        }
        __syncthreads();
    }

    if (tid == 0) output[blockIdx.x] = sharedData[0];
}

__global__ void maxReduction(int *input, int *output, int n) {
    extern __shared__ int sharedData[];
    unsigned int tid = threadIdx.x;
    unsigned int i = blockIdx.x * blockDim.x * 2 + tid;

    sharedData[tid] = (i < n) ? input[i] : INT_MIN;
    if (i + blockDim.x < n) sharedData[tid] = max(sharedData[tid], input[i + blockDim.x]);
    __syncthreads();

    for (unsigned int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) {
            sharedData[tid] = max(sharedData[tid], sharedData[tid + s]);
        }
        __syncthreads();
    }

    if (tid == 0) output[blockIdx.x] = sharedData[0];
}

// Kernel for min
__global__ void minReduction(int *input, int *output, int n) {
    extern __shared__ int sharedData[];
    unsigned int tid = threadIdx.x;
    unsigned int i = blockIdx.x * blockDim.x * 2 + tid;

    sharedData[tid] = (i < n) ? input[i] : INT_MAX;
    if (i + blockDim.x < n) sharedData[tid] = min(sharedData[tid], input[i + blockDim.x]);
    __syncthreads();

    for (unsigned int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) {
            sharedData[tid] = min(sharedData[tid], sharedData[tid + s]);
        }
        __syncthreads();
    }

    if (tid == 0) output[blockIdx.x] = sharedData[0];
}

__global__ void productReduction(int *input, int *output, int n) {
    extern __shared__ int sharedData[];
    unsigned int tid = threadIdx.x;
    unsigned int i = blockIdx.x * blockDim.x * 2 + tid;

    sharedData[tid] = (i < n) ? input[i] : 1;
    if (i + blockDim.x < n) sharedData[tid] *= input[i + blockDim.x]; 
    __syncthreads();

    // Perform reduction
    for (unsigned int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) {
            sharedData[tid] *= sharedData[tid + s]; 
        }
        __syncthreads();
    }

    if (tid == 0) output[blockIdx.x] = sharedData[0];
}

int main() {
    int n = 1000000;  
    int *h_input, *d_input, *d_output;

    h_input = (int*) malloc(n * sizeof(int));
    
    for (int i = 0; i < n; i++) {
        h_input[i] = i + 1; 
    }

    cudaMalloc((void**)&d_input, n * sizeof(int));
    cudaMalloc((void**)&d_output, sizeof(int));

    cudaMemcpy(d_input, h_input, n * sizeof(int), cudaMemcpyHostToDevice);

    int blockSize = 512;
    int gridSize = (n + blockSize * 2 - 1) / (blockSize * 2);

    int *d_intermediate;
    cudaMalloc((void**)&d_intermediate, gridSize * sizeof(int));

    // Sum Reduction
    auto startsum = std::chrono::high_resolution_clock::now();
    sumReduction<<<gridSize, blockSize, blockSize * sizeof(int)>>>(d_input, d_intermediate, n);
    auto endsum = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> elapsedsum = endsum - startsum;
    std::cout << "Time taken for sum: " << elapsedsum.count() << " seconds" << std::endl;

    
    if (gridSize > 1) {
        sumReduction<<<1, blockSize, blockSize * sizeof(int)>>>(d_intermediate, d_output, gridSize);
    } else {
        cudaMemcpy(d_output, d_intermediate, sizeof(int), cudaMemcpyDeviceToDevice);
    }
    int sum_result;
    cudaMemcpy(&sum_result, d_output, sizeof(int), cudaMemcpyDeviceToHost);
    printf("Sum: %d\n", sum_result);

    // Max Reduction
    auto startmax = std::chrono::high_resolution_clock::now();
    maxReduction<<<gridSize, blockSize, blockSize * sizeof(int)>>>(d_input, d_intermediate, n);
    auto endmax = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> elapsedmax = endmax- startmax;
    std::cout << "Time taken for max: " << elapsedmax.count() << " seconds" << std::endl;
    
    if (gridSize > 1) {
        maxReduction<<<1, blockSize, blockSize * sizeof(int)>>>(d_intermediate, d_output, gridSize);
    } else {
        cudaMemcpy(d_output, d_intermediate, sizeof(int), cudaMemcpyDeviceToDevice);
    }
    int max_result;
    cudaMemcpy(&max_result, d_output, sizeof(int), cudaMemcpyDeviceToHost);
    printf("Max: %d\n", max_result);

    // Min Reduction
    auto startmin = std::chrono::high_resolution_clock::now();
    minReduction<<<gridSize, blockSize, blockSize * sizeof(int)>>>(d_input, d_intermediate, n);
    auto endmin = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> elapsedmin = endmin- startmin;
    std::cout << "Time taken for min: " << elapsedmin.count() << " seconds" << std::endl;

    if (gridSize > 1) {
        minReduction<<<1, blockSize, blockSize * sizeof(int)>>>(d_intermediate, d_output, gridSize);
    } else {
        cudaMemcpy(d_output, d_intermediate, sizeof(int), cudaMemcpyDeviceToDevice);
    }
    int min_result;
    cudaMemcpy(&min_result, d_output, sizeof(int), cudaMemcpyDeviceToHost);
    printf("Min: %d\n", min_result);

    // Product Reduction
    auto startproduct = std::chrono::high_resolution_clock::now();
    productReduction<<<gridSize, blockSize, blockSize * sizeof(int)>>>(d_input, d_intermediate, n);
    auto endproduct = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> elapsedproduct = endproduct- startproduct;
    std::cout << "Time taken for product: " << elapsedproduct.count() << " seconds" << std::endl;

    if (gridSize > 1) {
        productReduction<<<1, blockSize, blockSize * sizeof(int)>>>(d_intermediate, d_output, gridSize);
    } else {
        cudaMemcpy(d_output, d_intermediate, sizeof(int), cudaMemcpyDeviceToDevice);
    }
    int product_result;
    cudaMemcpy(&product_result, d_output, sizeof(int), cudaMemcpyDeviceToHost);
    printf("Product: %d\n", product_result);

    // Cleanup
    cudaFree(d_input);
    cudaFree(d_output);
    cudaFree(d_intermediate);
    free(h_input);

    return 0;
}
