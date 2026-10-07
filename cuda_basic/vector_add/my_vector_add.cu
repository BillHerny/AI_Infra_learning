#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <cuda_runtime.h>
#include <cfloat>

#define CUDA_CHECK(call)                                        \
do{                                                             \
    cudaError_t err = (call);                                   \
    if (err != cudaSuccess)                                     \
    {                                                           \
        fprintf(stderr, "CUDA error:%s,%d:%s",                  \
                   __FILE__, __LINE__, cudaGetErrorString(err));\
        exit(EXIT_FAILURE);                                     \
    }                                                           \
}while(0)

__global__ void vectorAdd(const float* __restrict__ a, 
                          const float* __restrict__ b,
                          float* __restrict__ c,
                          int n)
{
    int idx = blockDim.x * blockIdx.x + threadIdx.x;
    for(int i =idx ; i < n ; i += blockDim.x * gridDim.x)
    {
        c[i] = a[i] + b[i];
    }
}

void vectorAddHost(const float* a, const float* b, float* c, int n)
{
    for(int i = 0 ; i < n ; i++)
    {
        c[i] = a[i] + b[i];
    }
}

int main(int argc, char** argv)
{
    int n = (argc > 1) ? atoi(argv[1]) : 1 << 26;
    int blockSize = (argc > 2) ? atoi(argv[2]) : 256;
    size_t bytes_per_vector = (size_t)n * sizeof(float);
    int gridSize = (n + blockSize - 1) / blockSize;
    printf("N = %d, grid number = %d, thread_per_block = %d, each array = %.1f MiB\n",
           n, gridSize, blockSize, bytes_per_vector / 1024.0 / 1024.0);
    
    float* h_a = (float*)malloc(bytes_per_vector);
    float* h_b = (float*)malloc(bytes_per_vector);
    float* h_c = (float*)malloc(bytes_per_vector);
    float* ref = (float*)malloc(bytes_per_vector);
    for(int i = 0 ; i < n ; i++) {h_a[i] = 1.0f; h_b[i] = 2.0f;}

    float *d_a, *d_b, *d_c;
    CUDA_CHECK(cudaMalloc(&d_a, bytes_per_vector));
    CUDA_CHECK(cudaMalloc(&d_b, bytes_per_vector));
    CUDA_CHECK(cudaMalloc(&d_c, bytes_per_vector));
    
    cudaEvent_t e_HTD_start, e_HTD_stop, start_kernel, stop_kernel, e_DTH_start, e_DTH_stop;
    CUDA_CHECK(cudaEventCreate(&e_HTD_start));
    CUDA_CHECK(cudaEventCreate(&e_HTD_stop));
    CUDA_CHECK(cudaEventCreate(&start_kernel));
    CUDA_CHECK(cudaEventCreate(&stop_kernel));
    CUDA_CHECK(cudaEventCreate(&e_DTH_start));
    CUDA_CHECK(cudaEventCreate(&e_DTH_stop));


    
    CUDA_CHECK(cudaEventRecord(e_HTD_start));
    CUDA_CHECK(cudaMemcpy(d_a, h_a, bytes_per_vector, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_b, h_b, bytes_per_vector, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaEventRecord(e_HTD_stop));

    

    //warm-up
    vectorAdd<<<gridSize, blockSize>>>(d_a, d_b, d_c, n);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    
    

    const int iter = 20;
    //float best_time = 1e30f;//此处可用哨兵模式,三种写法
    //float best_time = -1.0f;
    //     .....
    //     if(best_time < 0.0f || ms < best_time )best_time = ms;

    float kernel_ms = 0.0f,time_HTD = 0.0f, time_DTH = 0.0f;
    
    CUDA_CHECK(cudaEventElapsedTime(&time_HTD, e_HTD_start, e_HTD_stop));
    

    float best_time = FLT_MAX;
    for(int i = 0 ; i <iter ; i++)
    {
        CUDA_CHECK(cudaEventRecord(start_kernel));
        vectorAdd<<<gridSize, blockSize>>>(d_a, d_b, d_c, n);
        CUDA_CHECK(cudaEventRecord(stop_kernel));
        CUDA_CHECK(cudaEventSynchronize(stop_kernel));
        CUDA_CHECK(cudaEventElapsedTime(&kernel_ms, start_kernel, stop_kernel));

        
        if(kernel_ms < best_time) best_time = kernel_ms;
        
    }

    CUDA_CHECK(cudaEventRecord(e_DTH_start));
    CUDA_CHECK(cudaMemcpy(h_c, d_c, bytes_per_vector, cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaEventRecord(e_DTH_stop));
    CUDA_CHECK(cudaEventSynchronize(e_DTH_stop));
    CUDA_CHECK(cudaEventElapsedTime(&time_DTH, e_DTH_start, e_DTH_stop));
    
    vectorAddHost(h_a, h_b, ref, n);

    int bad = 0;
    for(int i = 0 ; i < n ; i++)
    {
        if(fabsf(h_c[i] - ref[i]) > 1e-5f)
        {
            if(bad < 5)
            {
                printf("mismathch @%d,gpu = %f, cpu = %f\n",i, h_c[i], ref[i]);
                bad++;
            }
        }
    }

    printf("verify: %s (%d mismatches)\n",bad ? "FAILURE": "PASS",bad);
    
    double gb = 3.0 * bytes_per_vector / 1e9;
    double bw_kernel = gb / (best_time / 1e3);
    printf("HostToDevice copy : %.4f ms\n", time_HTD);
    printf("kernel best       : %.4f ms\n", best_time);
    printf("DeviceToHost copy : %.4f ms\n", time_DTH);
    double bw_HTD = (2.0 * bytes_per_vector / 1e9) / (time_HTD / 1e3);   // 搬了 a、b 两份
    double bw_DTH = (1.0 * bytes_per_vector / 1e9) / (time_DTH / 1e3);   // 只搬了 c 一份
    printf("kernel bandwith: %.2f GB/s\n",bw_kernel);
    printf("HostToDevice copy bandweight: %.2f GB/s\n",bw_HTD);
    printf("DeviceToHost copy bandweight: %.2f GB/s\n",bw_DTH);
    CUDA_CHECK(cudaFree(d_a));
    CUDA_CHECK(cudaFree(d_b));
    CUDA_CHECK(cudaFree(d_c));
    free(h_a);
    free(h_b);
    free(h_c);
    free(ref);
    CUDA_CHECK(cudaEventDestroy(e_HTD_start));
    CUDA_CHECK(cudaEventDestroy(e_HTD_stop));
    CUDA_CHECK(cudaEventDestroy(e_DTH_start));
    CUDA_CHECK(cudaEventDestroy(e_DTH_stop));
    CUDA_CHECK(cudaEventDestroy(start_kernel));
    CUDA_CHECK(cudaEventDestroy(stop_kernel));

    return 0;
}
