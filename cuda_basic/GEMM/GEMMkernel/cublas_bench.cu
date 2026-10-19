// cuBLAS 对照基准：FP32（SGEMM）与 TF32（Tensor Core）各测一遍
// 编译：nvcc -O3 -std=c++14 -arch=sm_89 cublas_bench.cu -lcublas -o cublas_bench
// 运行：./cublas_bench M N K iters
#include <cstdio>
#include <cstdlib>
#include <cuda_runtime.h>
#include <cublas_v2.h>

#define CK(x) do { cudaError_t _e = (x); if (_e != cudaSuccess) { \
    printf("CUDA err line %d: %s\n", __LINE__, cudaGetErrorString(_e)); exit(1); } } while (0)
#define CB(x) do { cublasStatus_t s = (x); if (s != CUBLAS_STATUS_SUCCESS) { \
    printf("cuBLAS err: %d\n", (int)s); exit(1); } } while (0)

__global__ void fill_kernel(float* p, size_t n, float v)
{
    for (size_t i = blockIdx.x * (size_t)blockDim.x + threadIdx.x;
         i < n; i += (size_t)gridDim.x * blockDim.x)
        p[i] = v;
}

int main(int argc, char** argv)
{
    int M = (argc > 1) ? atoi(argv[1]) : 2048;
    int N = (argc > 2) ? atoi(argv[2]) : 2048;
    int K = (argc > 3) ? atoi(argv[3]) : 2048;
    int iters = (argc > 4) ? atoi(argv[4]) : 20;

    int ndev = 0;
    cudaError_t derr = cudaGetDeviceCount(&ndev);
    printf("devices=%d (%s)\n", ndev, cudaGetErrorString(derr));

    size_t aN = (size_t)M * K, bN = (size_t)K * N, cN = (size_t)M * N;
    float *A = nullptr, *B = nullptr, *C = nullptr;
    CK(cudaMalloc(&A, aN * sizeof(float)));
    CK(cudaMalloc(&B, bN * sizeof(float)));
    CK(cudaMalloc(&C, cN * sizeof(float)));
    fill_kernel<<<256, 256>>>(A, aN, 0.25f);
    fill_kernel<<<256, 256>>>(B, bN, 0.50f);
    fill_kernel<<<256, 256>>>(C, cN, 0.00f);
    CK(cudaDeviceSynchronize());

    cublasHandle_t h;
    CB(cublasCreate(&h));
    float alpha = 1.0f, beta = 0.0f;
    double flops = 2.0 * M * N * K;

    // column-major：C(M×N) = A(M×K) * B(K×N)
    auto bench = [&](const char* name, auto launch)
    {
        launch();
        CK(cudaDeviceSynchronize());
        cudaEvent_t evS, evE;
        CK(cudaEventCreate(&evS));
        CK(cudaEventCreate(&evE));
        float best = 1e30f;
        for (int i = 0; i < iters; ++i)
        {
            CK(cudaEventRecord(evS));
            launch();
            CK(cudaEventRecord(evE));
            CK(cudaEventSynchronize(evE));
            float ms = 0.0f;
            CK(cudaEventElapsedTime(&ms, evS, evE));
            if (ms < best) best = ms;
        }
        CK(cudaEventDestroy(evS));
        CK(cudaEventDestroy(evE));
        printf("%-8s best = %9.3f ms, %8.1f GFLOPS\n", name, best, flops / (best * 1e6));
    };

    printf("cuBLAS  %d x %d x %d, iters=%d\n", M, N, K, iters);

    bench("SGEMM", [&] {
        CB(cublasSgemm(h, CUBLAS_OP_N, CUBLAS_OP_N,
                       M, N, K, &alpha,
                       A, M, B, K, &beta, C, M));
    });

    bench("TF32", [&] {
        CB(cublasGemmEx(h, CUBLAS_OP_N, CUBLAS_OP_N,
                        M, N, K, &alpha,
                        A, CUDA_R_32F, M,
                        B, CUDA_R_32F, K,
                        &beta,
                        C, CUDA_R_32F, M,
                        CUBLAS_COMPUTE_32F_FAST_TF32,
                        CUBLAS_GEMM_DEFAULT_TENSOR_OP));
    });

    CB(cublasDestroy(h));
    CK(cudaFree(A));
    CK(cudaFree(B));
    CK(cudaFree(C));
    return 0;
}
