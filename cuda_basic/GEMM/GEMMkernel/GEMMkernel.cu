#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <cuda_runtime.h>

#define CUDA_CHECK(call)                                         \
    do{                                                          \
        cudaError_t err = (call);                                \
        if(err != cudaSuccess){                                  \
            fprintf(stderr, "CUDA ERROR:%s:%d:%s\n",             \
                    __FILE__, __LINE__, cudaGetErrorString(err));\
            exit(EXIT_FAILURE);                                  \
        }                                                        \
    }while(0)

#define TILE_DIM 32

__global__ void GEMMkernel_v1_native(const float* __restrict__ m1,
                                     const float* __restrict__ m2,
                                     int M, int N, int K, 
                                     float* __restrict__ result)
{
    int i = blockIdx.y * blockDim.y + threadIdx.y;
    int j = blockIdx.x * blockDim.x + threadIdx.x;
    if(i < M && j < N)
    {   
        float acc = 0.0f;
        for(int k = 0;k < K; ++k)
        {
            acc += m1[i * K + k] * m2[k * N + j];
        }
        result[i * N + j] = acc;
    }
}

using KernelFn = void(*)(const float*, const float*, int, int, int, float*);

float BestTime(KernelFn kernel, 
            const float* dm1, const float*dm2, float* dOut,
            dim3 grid, dim3 block,
            int m, int n, int k, int iters)
{
    kernel<<<grid, block>>>(dm1, dm2, m, n, k, dOut);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    cudaEvent_t start, stop;
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));

    float best = 1e30f;
    for(int i = 0;i < iters; ++i)
    {
        CUDA_CHECK(cudaEventRecord(start));
        kernel<<<grid, block>>>(dm1, dm2, m, n, k, dOut);
        CUDA_CHECK(cudaEventRecord(stop));
        CUDA_CHECK(cudaEventSynchronize(stop));

        float ms = 0.0f;
        CUDA_CHECK(cudaEventElapsedTime(&ms, start,stop));
        if(ms < best) best = ms;
    }
    CUDA_CHECK(cudaEventDestroy(start));
    CUDA_CHECK(cudaEventDestroy(stop));
    return best;
}

int main(int argc,char **argv)
{
    int M     = (argc>1) ? atoi(argv[1]) : 2048;
    int N     = (argc>2) ? atoi(argv[2]) : 2048;
    int K     = (argc>3) ? atoi(argv[3]) : 2048;
    int iters = (argc>4) ? atoi(argv[4]) : 20;

    size_t m1_bytes = (size_t)M * K * sizeof(float);
    size_t m2_bytes = (size_t)K * N * sizeof(float);
    size_t out_bytes = (size_t)M * N * sizeof(float);
    size_t total_bytes = m1_bytes + m2_bytes + out_bytes;

    printf("M1 is a %d x %d matrix, M2 is a %d x %d matrix,out is a %d x %d matrix\n",
         M, K, K, N, M, N);
    printf("Total memory required is %.1f MB\n", 
        total_bytes / (1024.0 * 1024.0));

    float *h_m1  = (float*)malloc(m1_bytes);
    float *h_m2  = (float*)malloc(m2_bytes);
    float *h_ref = (float*)calloc((size_t)M * N, sizeof(float));   // calloc(n, size) 申请的是 n×size 字节
    float *h_out = (float*)malloc(out_bytes);


    // ================= 2. 造数据（host） =================
    // ⚠ 不能照抄 transpose 的「下标当值」：GEMM 要把 K 个数累加起来，
    //   值域一大，浮点舍入误差就会把真假错误一起淹没。
    //   改用「0.25 步长的小整数」（值域 [-2, 2]）：
    //   乘积都是 1/16 的整数倍，K ≤ 4096 时所有部分和 < 2^24 → float 能精确表示
    for(size_t i = 0;i < (size_t)M * K; ++i) h_m1[i] = (float)((int)((i *  7) % 17) - 8) * 0.25f;
    for(size_t i = 0;i < (size_t)K * N; ++i) h_m2[i] = (float)((int)((i * 11) % 17) - 8) * 0.25f;
    
    
    // ---- CPU 参考：三重循环 ----
    // 循环顺序写成 i→k→j（而不是教科书的 i→j→k）：
    // 内层 j 连续扫过 hRef 和 hB 的连续内存，CPU 缓存友好；
    // 对固定的 (i,j) 来说 k 仍是递增累加的 —— 数学上和教科书完全一样
    for(int i = 0;i < M ; ++i)
        for(int k = 0;k < K; ++k)
            for(int j = 0;j < N; ++j)
                h_ref[(size_t)i * N + j] += h_m1[(size_t)i * K + k] * h_m2[(size_t)k * N + j];

    float *d_m1  = nullptr;
    float *d_m2  = nullptr;
    float *d_out = nullptr;
    CUDA_CHECK(cudaMalloc(&d_m1, m1_bytes));
    CUDA_CHECK(cudaMalloc(&d_m2, m2_bytes));
    CUDA_CHECK(cudaMalloc(&d_out, out_bytes));

    CUDA_CHECK(cudaMemcpy(d_m1, h_m1, m1_bytes, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_m2, h_m2, m2_bytes, cudaMemcpyHostToDevice));

    // ================= 4. 启动尺寸 =================
    // ⚠ grid 的 x 方向盖 N（列 j）、y 方向盖 M（行 i）——和 kernel 里的映射对应。
    //   （方阵下写反了照样跑过；一测长方形就露馅）
    dim3 block(TILE_DIM, TILE_DIM);
    dim3 grid((N + TILE_DIM - 1) / TILE_DIM,
              (M + TILE_DIM - 1) / TILE_DIM);
    printf("Grid size: %u x %u, Block size: %u x %u\n", grid.x, grid.y, block.x, block.y);

    float best = BestTime(GEMMkernel_v1_native, d_m1, d_m2, d_out, grid, block, M, N, K, iters);
    
    // ---- 成绩单：GFLOPS + 有效带宽 ----
    // 最小流量 = A 读 1 遍 + B 读 1 遍 + C 写 1 遍 —— 任何实现至少要搬这么多
    // 有效带宽 = 最小流量 / 实测时间（衡量「这题每秒至少要多少数据」，不是硬件真实流量）
    double flops     = 2.0 * M * N * K;
    double min_bytes = 4.0 * ((double)M * K + (double)K * N + (double)M * N);
    double eff_bw    = min_bytes / (best * 1e6);         // GB/s（best 单位是 ms）

    printf("%-12s best = %9.3f ms, %8.1f GFLOP/S, 有效带宽 = %6.2f GB/s\n",
           "GEMM V1 native", best, flops / (best * 1e6), eff_bw);

    double dram_floor_ms = min_bytes / 256e9 * 1e3;      // 假设打满 256 GB/s 需要的时间下限
    printf("算术强度 %.0f flop/B；DRAM 下限 %.3f ms（占实测 %.2f%%）\n",
           flops / min_bytes, dram_floor_ms, dram_floor_ms / best * 100.0);
    
    CUDA_CHECK(cudaMemcpy(h_out, d_out, out_bytes, cudaMemcpyDeviceToHost));
    size_t bad = 0, firstBad = 0;
    for (size_t i = 0; i < (size_t)M * N; ++i)
    {
        float diff = fabsf(h_out[i] - h_ref[i]);
        float tol  = 1e-3f * fabsf(h_ref[i]) + 1e-3f;   // GEMM 要用容差，不能用 ==
        if (diff > tol) { if (bad == 0) firstBad = i; ++bad; }
    }
    printf("verify: %s\n", bad ? "FAIL" : "PASS");
    if (bad)
        printf("  第一个对不上：h_out[%zu] 得到 %.4f，期望 %.4f\n",
            firstBad, h_out[firstBad], h_ref[firstBad]);
    
    free(h_m1);
    free(h_m2);
    free(h_ref);
    free(h_out);
    CUDA_CHECK(cudaFree(d_m1));
    CUDA_CHECK(cudaFree(d_m2));
    CUDA_CHECK(cudaFree(d_out));
    return 0;
}