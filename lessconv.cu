#include <iostream>
#include <cuda_runtime.h>
#include <cfloat>
#include <chrono>
#include <iostream>

// Kernel for reduction with SUM operator
__global__ void reduce_sum(float *input, float *output, int n) {
    extern __shared__ float sdata[];
    unsigned int tid = threadIdx.x;
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;

    sdata[tid] = (idx < n) ? input[idx] : 0.0f;
    __syncthreads();

    for (unsigned int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (tid < stride) {
            sdata[tid] += sdata[tid + stride];
        }
        __syncthreads();
    }


    if (tid == 0) output[blockIdx.x] = sdata[0];
}

// Kernel for reduction with PRODUCT operator
__global__ void reduce_product(float *input, float *output, int n) {
    extern __shared__ float sdata[];
    unsigned int tid = threadIdx.x;
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;

    sdata[tid] = (idx < n) ? input[idx] : 1.0f;
    __syncthreads();


    for (unsigned int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (tid < stride) {
            sdata[tid] *= sdata[tid + stride];
        }
        __syncthreads();
    }

    if (tid == 0) output[blockIdx.x] = sdata[0];
}

// Kernel for reduction with MIN operator
__global__ void reduce_min(float *input, float *output, int n) {
    extern __shared__ float sdata[];
    unsigned int tid = threadIdx.x;
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;

    sdata[tid] = (idx < n) ? input[idx] : FLT_MAX;
    __syncthreads();

    for (unsigned int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (tid < stride) {
            sdata[tid] = min(sdata[tid], sdata[tid + stride]);
        }
        __syncthreads();
    }

    if (tid == 0) output[blockIdx.x] = sdata[0];
}

__global__ void reduce_max(float *input, float *output, int n) {
    extern __shared__ float sdata[];
    unsigned int tid = threadIdx.x;
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;

    sdata[tid] = (idx < n) ? input[idx] : -FLT_MAX;
    __syncthreads();

    for (unsigned int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (tid < stride) {
            sdata[tid] = max(sdata[tid], sdata[tid + stride]);
        }
        __syncthreads();
    }

    if (tid == 0) output[blockIdx.x] = sdata[0];
}

// Helper function to launch kernel
void launch_reduction_kernel(float *input, float *output, int n, int operatorType) {
    const int threads = 256;
    const int blocks = (n + threads - 1) / threads;
    const int sharedMemSize = threads * sizeof(float);

    switch (operatorType) {
        case 0: // Sum
            reduce_sum<<<blocks, threads, sharedMemSize>>>(input, output, n);
            break;
        case 1: // Product
            reduce_product<<<blocks, threads, sharedMemSize>>>(input, output, n);
            break;
        case 2: // Min
            reduce_min<<<blocks, threads, sharedMemSize>>>(input, output, n);
            break;
        case 3: // Max
            reduce_max<<<blocks, threads, sharedMemSize>>>(input, output, n);
            break;
    }
}


int main() {
    const int n = 1000000; 
    float *h_input = new float[n];
    float *h_output = new float[(n + 255) / 256];
    float *d_input, *d_output;


    for (int i = 0; i < n; i++) {
        h_input[i] = static_cast<float>(i + 1);
    }

    cudaMalloc(&d_input, n * sizeof(float));
    cudaMalloc(&d_output, (n + 255) / 256 * sizeof(float));

    cudaMemcpy(d_input, h_input, n * sizeof(float), cudaMemcpyHostToDevice);

    
    
    auto start = std::chrono::high_resolution_clock::now();
    launch_reduction_kernel(d_input, d_output, n, 1); // Change operatorType (0 = Sum, 1 = Product, 2 = Min, 3 = Max)
    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> elapsed = end- start;
    std::cout << "Time taken: " << elapsed.count() << " seconds" << std::endl;
    cudaMemcpy(h_output, d_output, (n + 255) / 256 * sizeof(float), cudaMemcpyDeviceToHost);

    float result = 0.0f;
    for (int i = 0; i < (n + 255) / 256; i++) {
        result += h_output[i];
    }

    std::cout << "Result: " << result << std::endl;

    // Clean up
    delete[] h_input;
    delete[] h_output;
    cudaFree(d_input);
    cudaFree(d_output);

    return 0;
}
