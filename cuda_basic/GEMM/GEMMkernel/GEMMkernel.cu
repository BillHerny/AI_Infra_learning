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

__global__ void GEMMkernel_v2_smem(const float* __restrict__ m1,
                                   const float* __restrict__ m2,
                                   int M, int N, int K,
                                   float* __restrict__ result)
{
    __shared__ float As[TILE_DIM][TILE_DIM];
    __shared__ float Bs[TILE_DIM][TILE_DIM];

    int i = blockIdx.y * blockDim.y + threadIdx.y;
    int j = blockIdx.x * blockDim.x + threadIdx.x;
    
    float acc = 0.0f;
    for (int kt = 0; kt < K; kt += TILE_DIM)
    {
        As[threadIdx.y][threadIdx.x] = 
            (i < M && kt + threadIdx.x < K) ? m1[i * K + kt + threadIdx.x] : 0.0f;
        Bs[threadIdx.y][threadIdx.x] = 
            (j < N && kt + threadIdx.y < K) ? m2[(kt + threadIdx.y) * N + j] : 0.0f;
        __syncthreads();

        for(int k = 0; k < TILE_DIM; ++k)
            acc += As[threadIdx.y][k] * Bs[k][threadIdx.x];
        __syncthreads();
    }
    if ( i < M && j < N )
        result[i * N + j] = acc;
}

// ==================== V3：寄存器分块（每线程 4×4 输出） ====================
// 一个 block：16×16 = 256 线程，认领 C 的 64×64 小块
// 一个线程：4×4 = 16 个输出（16 个寄存器累加器）
// 沿 K 每 16 一段：载入 As[64][16] + Bs[16][64]（共 8 KB）→ sync → 计算 → sync
// 每个 k 步：读 4(A) + 4(B) = 8 次，做 16 次 FMA → 2 B/FMA（V2 是 8 B/FMA）
#define BM 64
#define BN 64
#define BK 16
#define TM 4
#define TN 4
#define V3_BLOCK 16                    // 线程块 16×16 = 256

__global__ void __launch_bounds__(V3_BLOCK * V3_BLOCK)
GEMMkernel_v3_register(const float* __restrict__ m1,
                       const float* __restrict__ m2,
                       int M, int N, int K,
                       float* __restrict__ result)
{
    __shared__ float As[BM][BK + 1];       
    // __shared__ float As[BM][BK]  // A 的 64 行 × 16 个 k  //bank conflict
    __shared__ float Bs[BK][BN];       // 16 个 k × B 的 64 列

    const int tx  = threadIdx.x;       // 0..15
    const int ty  = threadIdx.y;       // 0..15
    const int tid = ty * V3_BLOCK + tx;

    const int row0 = blockIdx.y * BM;  // 本块负责的 C 行起点
    const int col0 = blockIdx.x * BN;  // 本块负责的 C 列起点

    float acc[TM][TN];
#pragma unroll
    for (int i = 0; i < TM; ++i)
#pragma unroll
        for (int j = 0; j < TN; ++j)
            acc[i][j] = 0.0f;

    for (int kt = 0; kt < K; kt += BK)
    {
        // ---- ① 载入 A tile：64×16 = 1024 个元素，256 线程每人搬 4 个 ----
        // 连续 tid → 先沿 k（16 列）连续，再换行
#pragma unroll
        for (int n = 0; n < BM * BK / (V3_BLOCK * V3_BLOCK); ++n)
        {
            int idx = tid + n * V3_BLOCK * V3_BLOCK;
            int r   = idx / BK;        // 0..63
            int c   = idx % BK;        // 0..15
            As[r][c] = (row0 + r < M && kt + c < K)
                     ? m1[(row0 + r) * K + kt + c]
                     : 0.0f;
        }

        // ---- ② 载入 B tile：16×64 = 1024 个元素，每人搬 4 个 ----
        // 连续 tid → 同一行 64 个连续列，global 读满合并（128 B）
#pragma unroll
        for (int n = 0; n < BK * BN / (V3_BLOCK * V3_BLOCK); ++n)
        {
            int idx = tid + n * V3_BLOCK * V3_BLOCK;
            int r   = idx / BN;        // 0..15
            int c   = idx % BN;        // 0..63
            Bs[r][c] = (kt + r < K && col0 + c < N)
                     ? m2[(kt + r) * N + col0 + c]
                     : 0.0f;
        }
        __syncthreads();

        // ---- ③ 计算：每个 k 读 4+4 个操作数，做 16 次 FMA ----
        // As[ty*4+i][k] → warp 内只有两个不同地址（ty=0/1），广播，无冲突
        // Bs[k][tx*4+j] → 16 个地址步长 4，2-way bank conflict（可接受）
#pragma unroll
        for (int k = 0; k < BK; ++k)
        {
            float a[TM], b[TN];
#pragma unroll
            for (int i = 0; i < TM; ++i) a[i] = As[ty * TM + i][k];
#pragma unroll
        //    for (int j = 0; j < TN; ++j) b[j] = Bs[k][tx * TN + j];
        for (int j = 0; j < TN; ++j) b[j] = Bs[k][tx + V3_BLOCK * j];

#pragma unroll
            for (int i = 0; i < TM; ++i)
#pragma unroll
                for (int j = 0; j < TN; ++j)
                    acc[i][j] += a[i] * b[j];   // 16 次 FMA 共用这 8 个操作数
        }
        __syncthreads();
    }

    // ---- ④ 写回：全部 k 段跑完后才写，每个线程 16 个结果 ----
    // （固定 j 时相邻线程地址步长 16 B，store 效率只有 25%，
    //   但 C 的流量只有 GB/s 级，完全不是瓶颈）
#pragma unroll
    for (int i = 0; i < TM; ++i)
    {
        int row = row0 + ty * TM + i;
#pragma unroll
        for (int j = 0; j < TN; ++j)
        {
            //int col = col0 + tx * TN + j; bank conflict
            int col = col0 + tx + V3_BLOCK * j; // bank conflict
            //int col = col0 + tx * TN + j + (ty * TM + i) * N; // bank conflict
            //int col = col0 + tx * TN + j + (ty * TM + i) * N + (row0 + ty * TM + i) * N; // bank conflict
            if (row < M && col < N)
                result[row * N + col] = acc[i][j];
        }
    }
}

// ==================== V4：float4 向量化（每线程 4×4 输出） ====================
// 分块与 V3 完全相同（16×16 线程 / 64×64 block tile / BK=16），
// 只把「每 k 步 8 条 LDS.32」压成「2 条 LDS.128」：
//   A：转置存进 As_t[BK][BM+4]，读 As_t[k][4ty..4ty+3]   → 1 条 LDS.128
//   B：Bs[k][4tx..4tx+3]                                 → 1 条 LDS.128
// 对齐：As_t 行 stride 68 字（272 B），Bs 行 stride 64 字（256 B），均为 16 B 倍数。
// ⚠ As_t 的 +4 padding 是为了让行 stride 保持 16 B 对齐（float4 要求），
//   不是为了消 bank conflict（广播访问本来不冲突）。
#define V4_PAD 4

__global__ void __launch_bounds__(V3_BLOCK * V3_BLOCK)
GEMMkernel_v4_float4(const float* __restrict__ m1,
                     const float* __restrict__ m2,
                     int M, int N, int K,
                     float* __restrict__ result)
{
    __shared__ __align__(16) float As_t[BK][BM + V4_PAD];  // 16×68 字 = 4352 B
    __shared__ __align__(16) float Bs[BK][BN];             // 16×64 字 = 4096 B

    const int tx  = threadIdx.x;        // 0..15
    const int ty  = threadIdx.y;        // 0..15
    const int tid = ty * V3_BLOCK + tx;

    const int row0 = blockIdx.y * BM;
    const int col0 = blockIdx.x * BN;

    float acc[TM][TN];
#pragma unroll
    for (int i = 0; i < TM; ++i)
#pragma unroll
        for (int j = 0; j < TN; ++j)
            acc[i][j] = 0.0f;

    for (int kt = 0; kt < K; kt += BK)
    {
        // ---- ① 载入 A：global 读法不变，写 smem 时转置 ----
        // As_t[c][r] = A[row0 + r][kt + c]；行 stride 68 字
#pragma unroll
        for (int n = 0; n < BM * BK / (V3_BLOCK * V3_BLOCK); ++n)
        {
            int idx = tid + n * V3_BLOCK * V3_BLOCK;
            int r   = idx / BK;        // 0..63
            int c   = idx % BK;        // 0..15
            As_t[c][r] = (row0 + r < M && kt + c < K)
                       ? m1[(row0 + r) * K + kt + c]
                       : 0.0f;
        }

        // ---- ② 载入 B：与 V3 完全一致 ----
#pragma unroll
        for (int n = 0; n < BK * BN / (V3_BLOCK * V3_BLOCK); ++n)
        {
            int idx = tid + n * V3_BLOCK * V3_BLOCK;
            int r   = idx / BN;        // 0..15
            int c   = idx % BN;        // 0..63
            Bs[r][c] = (kt + r < K && col0 + c < N)
                     ? m2[(kt + r) * N + col0 + c]
                     : 0.0f;
        }
        __syncthreads();

        // ---- ③ 计算：每 k 步 2 条 LDS.128 + 16 条 FFMA ----
        // av = A[row0 + 4ty .. +3][kt + k]（4 行）
        // bv = B[kt + k][col0 + 4tx .. +3]（4 列）
#pragma unroll
        for (int k = 0; k < BK; ++k)
        {
            const float4 av = *reinterpret_cast<const float4*>(&As_t[k][ty * TM]);
            const float4 bv = *reinterpret_cast<const float4*>(&Bs[k][tx * TN]);

            acc[0][0] += av.x * bv.x;   acc[0][1] += av.x * bv.y;
            acc[0][2] += av.x * bv.z;   acc[0][3] += av.x * bv.w;
            acc[1][0] += av.y * bv.x;   acc[1][1] += av.y * bv.y;
            acc[1][2] += av.y * bv.z;   acc[1][3] += av.y * bv.w;
            acc[2][0] += av.z * bv.x;   acc[2][1] += av.z * bv.y;
            acc[2][2] += av.z * bv.z;   acc[2][3] += av.z * bv.w;
            acc[3][0] += av.w * bv.x;   acc[3][1] += av.w * bv.y;
            acc[3][2] += av.w * bv.z;   acc[3][3] += av.w * bv.w;
        }
        __syncthreads();
    }

    // ---- ④ 写回：列恢复连续的 4 个（tx*4 + j） ----
#pragma unroll
    for (int i = 0; i < TM; ++i)
    {
        int row = row0 + ty * TM + i;
#pragma unroll
        for (int j = 0; j < TN; ++j)
        {
            int col = col0 + tx * TN + j;
            if (row < M && col < N)
                result[row * N + col] = acc[i][j];
        }
    }
}

// ==================== V5：8×8 寄存器分块 + float4（每线程 8×8 输出） ====================
// 线程块 16×16 = 256，每线程 8×8 → block tile 128×128；BK 仍为 16。
// 每 k 步：A 读 2 条 LDS.128、B 读 2 条 LDS.128，做 64 次 FMA
// → 内存指令 0.0625 条/FMA（V4 是 0.125），数据 1 B/FMA（V4 是 2）。
//
// ⚠ 2026-10-19 bank conflict 修正（原版代码保留在下方注释中）：
//   原版 B 读用 tx*8 + {0,4}，16 个 lane 的步长是 8 字（32 B），
//   只落在 4 个 bank 组上、每个被 4 个 lane 复用 → 4-way conflict
//   （ncu：33,554,432 次冲突，每条 LDS.128 平均 5 个 wavefront）。
//   现改为「两段 4 列」：前 64 列 + 后 64 列，lane 步长 4 字（16 B）
//   → 2-way conflict，与 V4 相同。
#define V5_TM 8
#define V5_TN 8
#define V5_BM (V3_BLOCK * V5_TM)      // 128
#define V5_BN (V3_BLOCK * V5_TN)      // 128
#define V5_PAD 4

__global__ void __launch_bounds__(V3_BLOCK * V3_BLOCK, 2)
GEMMkernel_v5_reg8x8(const float* __restrict__ m1,
                     const float* __restrict__ m2,
                     int M, int N, int K,
                     float* __restrict__ result)
{
    __shared__ __align__(16) float As_t[BK][V5_BM + V5_PAD];  // 16×132 字 = 8448 B
    __shared__ __align__(16) float Bs[BK][V5_BN];             // 16×128 字 = 8192 B

    const int tx  = threadIdx.x;        // 0..15
    const int ty  = threadIdx.y;        // 0..15
    const int tid = ty * V3_BLOCK + tx;

    const int row0 = blockIdx.y * V5_BM;
    const int col0 = blockIdx.x * V5_BN;

    float acc[V5_TM][V5_TN];
#pragma unroll
    for (int i = 0; i < V5_TM; ++i)
#pragma unroll
        for (int j = 0; j < V5_TN; ++j)
            acc[i][j] = 0.0f;

    for (int kt = 0; kt < K; kt += BK)
    {
        // ---- ① 载入 A：128×16 = 2048 个元素，256 线程每人 8 个 ----
        // 写 smem 时转置：As_t[c][r] = A[row0 + r][kt + c]
#pragma unroll
        for (int n = 0; n < V5_BM * BK / (V3_BLOCK * V3_BLOCK); ++n)   // 8
        {
            int idx = tid + n * V3_BLOCK * V3_BLOCK;
            int r   = idx / BK;        // 0..127
            int c   = idx % BK;        // 0..15
            As_t[c][r] = (row0 + r < M && kt + c < K)
                       ? m1[(row0 + r) * K + kt + c]
                       : 0.0f;
        }

        // ---- ② 载入 B：16×128 = 2048 个元素，每人 8 个 ----
#pragma unroll
        for (int n = 0; n < BK * V5_BN / (V3_BLOCK * V3_BLOCK); ++n)   // 8
        {
            int idx = tid + n * V3_BLOCK * V3_BLOCK;
            int r   = idx / V5_BN;     // 0..15
            int c   = idx % V5_BN;     // 0..127
            Bs[r][c] = (kt + r < K && col0 + c < N)
                     ? m2[(kt + r) * N + col0 + c]
                     : 0.0f;
        }
        __syncthreads();

        // ---- ③ 计算：每 k 步 4 条 LDS.128 + 64 条 FFMA ----
#pragma unroll
        for (int k = 0; k < BK; ++k)
        {
            const float4 a0 = *reinterpret_cast<const float4*>(&As_t[k][ty * V5_TM + 0]);
            const float4 a1 = *reinterpret_cast<const float4*>(&As_t[k][ty * V5_TM + 4]);

            // ---- bank conflict 修正：原版代码保留作对照 ----
            // 原版（每线程连续 8 列）：lane 步长 8 字（32 B）→ 4-way conflict
            // const float4 b0 = *reinterpret_cast<const float4*>(&Bs[k][tx * V5_TN + 0]);
            // const float4 b1 = *reinterpret_cast<const float4*>(&Bs[k][tx * V5_TN + 4]);
            // 修正（两段 4 列：前 64 列 / 后 64 列）：lane 步长 4 字（16 B）→ 2-way
            const float4 b0 = *reinterpret_cast<const float4*>(&Bs[k][tx * 4 + 0]);
            const float4 b1 = *reinterpret_cast<const float4*>(&Bs[k][V5_BN / 2 + tx * 4 + 0]);

            const float av[8] = {a0.x, a0.y, a0.z, a0.w, a1.x, a1.y, a1.z, a1.w};
            const float bv[8] = {b0.x, b0.y, b0.z, b0.w, b1.x, b1.y, b1.z, b1.w};

#pragma unroll
            for (int i = 0; i < V5_TM; ++i)
#pragma unroll
                for (int j = 0; j < V5_TN; ++j)
                    acc[i][j] += av[i] * bv[j];
        }
        __syncthreads();
    }

    // ---- ④ 写回 64 个结果 ----
#pragma unroll
    for (int i = 0; i < V5_TM; ++i)
    {
        int row = row0 + ty * V5_TM + i;
#pragma unroll
        for (int j = 0; j < V5_TN; ++j)
        {
            // 原版（与「连续 8 列」配套；bank conflict 修正前的映射）：
            // int col = col0 + tx * V5_TN + j;
            // 修正后：j=0..3 用前 64 列，j=4..7 用后 64 列
            int col = col0 + (j < 4 ? tx * 4 + j
                                    : V5_BN / 2 + tx * 4 + (j - 4));
            if (row < M && col < N)
                result[row * N + col] = acc[i][j];
        }
    }
}

// ==================== V6：cp.async 双缓冲（在 V5 基础上加流水线） ====================
// 分块与 V5 完全相同（16×16 线程 / 8×8 输出 / 128×128 tile / BK=16），
// 唯一变化：把「载入 → 同步 → 计算 → 同步」改成两段流水：
//
//   预取 stage 0
//   for 每一段：
//       异步把下一段搬进另一个 buffer   ← 不等待
//       等当前段就绪（cp.async.wait_group 1）
//       计算当前段
//
// 这样全局读的延迟被当前段的计算盖住，而不是每段都裸露一次。
// smem 从 16.25 KB 变成 2×16.25 = 32.5 KB；2 块/SM 时 65 KB，仍然放得下。
#define V6_STAGES 2

__device__ __forceinline__ void cp_async_4(void* smem_dst, const void* gmem_src, int src_bytes)
{
    unsigned saddr = (unsigned)__cvta_generic_to_shared(smem_dst);
    // cp-size = 4 B；src_bytes = 0 时硬件写 0（越界零填充）
    asm volatile("cp.async.ca.shared.global [%0], [%1], 4, %2;\n"
                 :: "r"(saddr), "l"(gmem_src), "r"(src_bytes));
}

__device__ __forceinline__ void cp_async_commit()
{
    asm volatile("cp.async.commit_group;\n" ::);
}

__device__ __forceinline__ void cp_async_wait_1()
{
    asm volatile("cp.async.wait_group 1;\n" ::);
}

__device__ __forceinline__ void cp_async_wait_0()
{
    asm volatile("cp.async.wait_group 0;\n" ::);
}

// 把第 kt 段搬进 As/Bs（异步，不等待）；越界处用 src_bytes=0 零填充
__device__ __forceinline__ void v6_load_stage(
    float (*As)[V5_BM + V5_PAD], float (*Bs)[V5_BN],
    const float* __restrict__ m1, const float* __restrict__ m2,
    int M, int N, int K, int row0, int col0, int tid, int kt)
{
    // A：128×16 = 2048 个元素，每人 8 个
    // 全局地址不保证 16 B 对齐（K 可能不是 4 的倍数），所以用 4 B 的 cp.async
#pragma unroll
    for (int n = 0; n < V5_BM * BK / (V3_BLOCK * V3_BLOCK); ++n)
    {
        int idx = tid + n * V3_BLOCK * V3_BLOCK;
        int r   = idx / BK;        // 0..127
        int c   = idx % BK;        // 0..15
        bool ok = (row0 + r < M) && (kt + c < K);
        cp_async_4(&As[c][r],
                   ok ? &m1[(row0 + r) * K + kt + c] : m1,
                   ok ? 4 : 0);
    }
    // B：16×128 = 2048 个元素，每人 8 个
#pragma unroll
    for (int n = 0; n < BK * V5_BN / (V3_BLOCK * V3_BLOCK); ++n)
    {
        int idx = tid + n * V3_BLOCK * V3_BLOCK;
        int r   = idx / V5_BN;     // 0..15
        int c   = idx % V5_BN;     // 0..127
        bool ok = (kt + r < K) && (col0 + c < N);
        cp_async_4(&Bs[r][c],
                   ok ? &m2[(kt + r) * N + col0 + c] : m2,
                   ok ? 4 : 0);
    }
    cp_async_commit();
}

__global__ void __launch_bounds__(V3_BLOCK * V3_BLOCK, 2)
GEMMkernel_v6_double_buffer(const float* __restrict__ m1,
                            const float* __restrict__ m2,
                            int M, int N, int K,
                            float* __restrict__ result)
{
    __shared__ __align__(16) float As_t[V6_STAGES][BK][V5_BM + V5_PAD];
    __shared__ __align__(16) float Bs[V6_STAGES][BK][V5_BN];

    const int tx  = threadIdx.x;        // 0..15
    const int ty  = threadIdx.y;        // 0..15
    const int tid = ty * V3_BLOCK + tx;

    const int row0 = blockIdx.y * V5_BM;
    const int col0 = blockIdx.x * V5_BN;

    float acc[V5_TM][V5_TN];
#pragma unroll
    for (int i = 0; i < V5_TM; ++i)
#pragma unroll
        for (int j = 0; j < V5_TN; ++j)
            acc[i][j] = 0.0f;

    // 预取第 0 段
    v6_load_stage(As_t[0], Bs[0], m1, m2, M, N, K, row0, col0, tid, 0);

    int stage = 0;
    for (int kt = 0; kt < K; kt += BK, stage ^= 1)
    {
        const int  nxt      = stage ^ 1;
        const bool has_next = (kt + BK < K);

        if (has_next)
            v6_load_stage(As_t[nxt], Bs[nxt], m1, m2, M, N, K, row0, col0, tid, kt + BK);

        // 当前段就绪；如果还有下一段在飞，允许 1 组未完成
        if (has_next) cp_async_wait_1();
        else          cp_async_wait_0();
        __syncthreads();

        // ---- 计算当前段（与 V5 相同）----
#pragma unroll
        for (int k = 0; k < BK; ++k)
        {
            const float4 a0 = *reinterpret_cast<const float4*>(&As_t[stage][k][ty * V5_TM + 0]);
            const float4 a1 = *reinterpret_cast<const float4*>(&As_t[stage][k][ty * V5_TM + 4]);
            const float4 b0 = *reinterpret_cast<const float4*>(&Bs[stage][k][tx * 4 + 0]);
            const float4 b1 = *reinterpret_cast<const float4*>(&Bs[stage][k][V5_BN / 2 + tx * 4 + 0]);

            const float av[8] = {a0.x, a0.y, a0.z, a0.w, a1.x, a1.y, a1.z, a1.w};
            const float bv[8] = {b0.x, b0.y, b0.z, b0.w, b1.x, b1.y, b1.z, b1.w};

#pragma unroll
            for (int i = 0; i < V5_TM; ++i)
#pragma unroll
                for (int j = 0; j < V5_TN; ++j)
                    acc[i][j] += av[i] * bv[j];
        }
        __syncthreads();   // 所有线程读完当前段，下一轮才能覆盖它
    }

    // ---- 写回（与 V5 相同：两段 4 列）----
#pragma unroll
    for (int i = 0; i < V5_TM; ++i)
    {
        int row = row0 + ty * V5_TM + i;
#pragma unroll
        for (int j = 0; j < V5_TN; ++j)
        {
            int col = col0 + (j < 4 ? tx * 4 + j
                                    : V5_BN / 2 + tx * 4 + (j - 4));
            if (row < M && col < N)
                result[row * N + col] = acc[i][j];
        }
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
    dim3 block_v1(TILE_DIM, TILE_DIM);                 // V1 / V2：32×32 = 1024 线程
    dim3 grid_v1((N + TILE_DIM - 1) / TILE_DIM,
                 (M + TILE_DIM - 1) / TILE_DIM);

    dim3 block_v3(V3_BLOCK, V3_BLOCK);                 // V3 / V4：16×16 = 256 线程
    dim3 grid_v3((N + BM - 1) / BM,
                 (M + BM - 1) / BM);                   // 64×64 的 C 小块

    dim3 block_v5(V3_BLOCK, V3_BLOCK);                 // V5：16×16 = 256 线程
    dim3 grid_v5((N + V5_BN - 1) / V5_BN,
                 (M + V5_BM - 1) / V5_BM);             // 128×128 的 C 小块

    printf("V1/V2: Grid %u x %u, Block %u x %u\n",
           grid_v1.x, grid_v1.y, block_v1.x, block_v1.y);
    printf("V3/V4: Grid %u x %u, Block %u x %u\n",
           grid_v3.x, grid_v3.y, block_v3.x, block_v3.y);
    printf("V5   : Grid %u x %u, Block %u x %u\n",
           grid_v5.x, grid_v5.y, block_v5.x, block_v5.y);

    // 计时 + 对拍统一放进下面的 report lambda：每个 kernel 各跑各的、各验各的
    
    // ---- 成绩单：GFLOPS + 有效带宽 ----
    // 最小流量 = A 读 1 遍 + B 读 1 遍 + C 写 1 遍 —— 任何实现至少要搬这么多
    // 有效带宽 = 最小流量 / 实测时间（衡量「这题每秒至少要多少数据」，不是硬件真实流量）
    double flops     = 2.0 * M * N * K;
    double min_bytes = 4.0 * ((double)M * K + (double)K * N + (double)M * N);
    double dram_floor_ms = min_bytes / 256e9 * 1e3;      // 假设打满 256 GB/s 需要的时间下限

    // 跑一个 kernel：计时 → 拷回 → 对拍 → 打印，返回最快时间
    // ⚠ 对拍必须紧跟它自己那次 BestTime：d_out 只有一块，会被下一个 kernel 覆盖
    auto report = [&](const char* name, KernelFn kernel, dim3 g, dim3 b) -> float
    {
        float best = BestTime(kernel, d_m1, d_m2, d_out, g, b, M, N, K, iters);

        CUDA_CHECK(cudaMemcpy(h_out, d_out, out_bytes, cudaMemcpyDeviceToHost));
        size_t bad = 0, firstBad = 0;
        for (size_t i = 0; i < (size_t)M * N; ++i)
        {
            float diff = fabsf(h_out[i] - h_ref[i]);
            float tol  = 1e-3f * fabsf(h_ref[i]) + 1e-3f;   // GEMM 要用容差，不能用 ==
            if (diff > tol) { if (bad == 0) firstBad = i; ++bad; }
        }

        printf("%-12s best = %9.3f ms, %8.1f GFLOP/S, 有效带宽 = %6.2f GB/s, verify: %s\n",
               name, best, flops / (best * 1e6), min_bytes / (best * 1e6), bad ? "FAIL" : "PASS");
        if (bad)
            printf("  第一个对不上：h_out[%zu] 得到 %.4f，期望 %.4f\n",
                   firstBad, h_out[firstBad], h_ref[firstBad]);
        return best;
    };

    float t1 = report("GEMM V1 native", GEMMkernel_v1_native,   grid_v1, block_v1);
    float t2 = report("GEMM V2 smem",   GEMMkernel_v2_smem,     grid_v1, block_v1);
    float t3 = report("GEMM V3 reg",    GEMMkernel_v3_register, grid_v3, block_v3);
    float t4 = report("GEMM V4 f4",     GEMMkernel_v4_float4,   grid_v3, block_v3);
    float t5 = report("GEMM V5 8x8",    GEMMkernel_v5_reg8x8,   grid_v5, block_v5);
    float t6 = report("GEMM V6 dbuf",   GEMMkernel_v6_double_buffer, grid_v5, block_v5);


    printf("V2 相对 V1 加速比 = %.4fx（>1 表示 V2 更快）\n", t1 / t2);
    printf("V3 相对 V1 加速比 = %.4fx（>1 表示 V3 更快）\n", t1 / t3);
    printf("V4 相对 V1 加速比 = %.4fx（>1 表示 V4 更快）\n", t1 / t4);
    printf("V5 相对 V1 加速比 = %.4fx（>1 表示 V5 更快）\n", t1 / t5);
    printf("V6 相对 V1 加速比 = %.4fx（>1 表示 V6 更快）\n", t1 / t6);
    printf("V4 相对 V3 加速比 = %.4fx（>1 表示 V4 更快）\n", t3 / t4);
    printf("V5 相对 V4 加速比 = %.4fx（>1 表示 V5 更快）\n", t4 / t5);
    printf("V6 相对 V5 加速比 = %.4fx（>1 表示 V6 更快）\n", t5 / t6);

    printf("算术强度 %.0f flop/B；DRAM 下限 %.3f ms（V1 占实测 %.2f%%，V2 占实测 %.2f%%，V3 占实测 %.2f%%，V4 占实测 %.2f%%，V5 占实测 %.2f%%，V6 占实测 %.2f%%）\n",
           flops / min_bytes, dram_floor_ms,
           dram_floor_ms / t1 * 100.0,
           dram_floor_ms / t2 * 100.0,
           dram_floor_ms / t3 * 100.0,
           dram_floor_ms / t4 * 100.0,
           dram_floor_ms / t5 * 100.0,
           dram_floor_ms / t6 * 100.0);
    
    // （对拍已在上面的 report lambda 里完成：V1~V6 各自跑完立刻验证）
    
    free(h_m1);
    free(h_m2);
    free(h_ref);
    free(h_out);
    CUDA_CHECK(cudaFree(d_m1));
    CUDA_CHECK(cudaFree(d_m2));
    CUDA_CHECK(cudaFree(d_out));
    return 0;
}

/*########################################################################
  GEMMkernel.cu　V3（寄存器分块）逐行详解
  —— 代码在上，讲解全部在下（与 reducesum*.cu 同一风格）

  前置：
    这份讲解只针对 GEMMkernel_v3_register 这一个 kernel。
    如果还不明白「2 B/FMA」和「一个线程算 16 个输出」，
    先看 readme.md 的 §4（V2 为什么失败）和 §5（V3 实测）。

  行号说明：
    本文按加入本注释块之前的文件行号标注；上方代码若有改动，
    请以「宏名 / 变量名 / 代码原文」为准，不要死记行号。

  目录
    第 0 步  全景：V3 到底改了什么
    第 1 步  数据流：从 global 到 FMA 的三级复用
    第 2 步  宏定义（第 69–74 行）
    第 3 步  函数签名（第 76–80 行）
    第 4 步  两块共享内存（第 82–83 行）
    第 5 步  线程坐标与块坐标（第 85–90 行）
    第 6 步  16 个累加器（第 92–97 行）
    第 7 步  沿 K 的分段循环（第 99–100 行）
    第 8 步  载入 A tile（第 103–112 行）
    第 9 步  载入 B tile（第 116–125 行）
    第 10 步 第一次 __syncthreads（第 126 行）
    第 11 步 计算：8 次读取 → 16 次 FMA（第 131–145 行）
    第 12 步 第二次 __syncthreads（第 146 行）
    第 13 步 写回 C（第 152–163 行）
    第 14 步 三个坑与调参约束
    第 15 步 自测题


========================================================================
第 0 步　全景：V3 到底改了什么
========================================================================

  V1 / V2 的每个线程只算 1 个 C 元素：

      for k:  acc += A[i][k] * B[k][j]
      每做 1 次 FMA 要读 2 个数  →  8 B/FMA

  V3 的每个线程算 4×4 = 16 个 C 元素：

      每个 k 步读 A 的 4 行 + B 的 4 列 = 8 个数
      这 8 个数两两相乘  →  16 次 FMA
      → 8 次读取 / 16 次 FMA = 0.5 次/FMA = 2 B/FMA

  四个关键数字：

      项            值        说明
      ----------    ------    ------------------------------------------
      线程块        16×16     256 线程，每 SM 可驻 4 块（上限 1536 线程）
      块 tile       64×64     16 线程 × 每线程 4 格
      每线程输出    4×4       16 个寄存器累加器
      BK            16        每次载入 8 KB smem，K=2048 时 128 段

  实测（Release）：
      2048³             5.105 ms     3365.5 GFLOPS（3.39× V1）
      8864×8864×7931    381.607 ms   3265.9 GFLOPS（4.00× V1，仅 −3.0%）


========================================================================
第 1 步　数据流：从 global 到 FMA 的三级复用
========================================================================

      global A / B
          │   每块每段只搬 (64+64)×16 = 2048 个 float = 8 KB
          ▼
      smem  As[64][16] / Bs[16][64]      ← 每个元素被 16 个线程各读 1 次
          │   每线程每 k 步读 4 + 4 个
          ▼
      寄存器 a[4], b[4]                   ← 每个读出的值参与 4 次 FMA
          │
          ▼
      acc[4][4]（16 个寄存器）

  每个全局元素的「一生」：
      1 次 global 搬运 → 16 次 smem 读 → 64 次 FMA 使用

  对比 V2：
      V2 每线程算 1 格，smem 值读出来只用 1 次 → 8 B/FMA
      V3 每线程算 16 格，同样的 smem 值被乘 4 次 → 2 B/FMA
      ★ smem 没变，变的是「读出来之后怎么用」。


========================================================================
第 2 步　宏定义（第 69–74 行）
========================================================================

  #define BM 64
      一个 block 负责的 C 行数。
      host 端 grid_v3 = (N+BM-1)/BM, (M+BM-1)/BM —— 两边必须一致。

  #define BN 64
      一个 block 负责的 C 列数。

  #define BK 16
      沿 K 每次前进多少。它决定三件事：
        1) smem 大小：(64+64)×16×4 = 8 KB
        2) 段数：ceil(K/16)（K=2048 → 128 段；K=337 → 22 段）
        3) 每段的乘加量：64×64×16 = 65536 次

  #define TM 4 / #define TN 4
      每个线程算 4 行 × 4 列。
      ⚠ 硬约束：BM == V3_BLOCK * TM，BN == V3_BLOCK * TN（16×4 = 64）。
        只改 TM 不改 BM → 行覆盖对不上，结果错或越界；
        要改成 8×8，必须同时把 BM/BN 改成 128，并同步改 host 的 grid 公式。

  #define V3_BLOCK 16
      线程块边长，16×16 = 256 线程。
      1536 / 256 = 6 是线程数上限；实际受寄存器限制约 4 块/SM。
      选 16 而不是 32：保证每 SM 能驻多块，__syncthreads 时还有别的块在算。


========================================================================
第 3 步　函数签名（第 76–80 行）
========================================================================

  __global__ void __launch_bounds__(V3_BLOCK * V3_BLOCK)
      __launch_bounds__(256) 告诉编译器「每块最多 256 线程」，
      编译器会据此限制寄存器用量，避免为了 ILP 把寄存器吃到占用率掉档。
      如果以后想强制「至少 4 块/SM」，可以写 __launch_bounds__(256, 4)，
      但要注意寄存器不够时会发生 spill，反而更慢。

  GEMMkernel_v3_register(const float* __restrict__ m1,
                         const float* __restrict__ m2,
                         int M, int N, int K,
                         float* __restrict__ result)
      参数顺序必须和 using KernelFn 完全一致，否则 report lambda 传不进去：
          using KernelFn = void(*)(const float*, const float*, int, int, int, float*);

      __restrict__ 向编译器保证三块内存互不重叠：
        - 编译器可以把 m1/m2 的值缓存在寄存器里
        - 可以提前发 load、重排指令
      去掉它不会算错，但内层循环可能明显变慢（编译器必须假设
      「写 result 可能改到 m1/m2」）。


========================================================================
第 4 步　两块共享内存（第 82–83 行）
========================================================================

  __shared__ float As[BM][BK];        // 64 行 × 16 个 k = 4 KB
  __shared__ float Bs[BK][BN];        // 16 个 k × 64 列 = 4 KB

  - As 装的是：本块 C 的 64 行 × 当前 k 段的 16 个 k。
  - Bs 装的是：当前 k 段的 16 个 k × 本块 C 的 64 列。
  - C 不进 smem：16 个累加器全程在寄存器里，smem 只负责「操作数共享」。
  - 合计 8 KB/块；4 块/SM 时 32 KB，L1/smem 完全放得下。
  - 两块都是行主序，为了「载入时连续写、计算时按需读」。


========================================================================
第 5 步　线程坐标与块坐标（第 85–90 行）
========================================================================

  const int tx  = threadIdx.x;        // 0..15
  const int ty  = threadIdx.y;        // 0..15
  const int tid = ty * V3_BLOCK + tx; // 0..255，只用于载入分工

  const int row0 = blockIdx.y * BM;   // 本块 C 的行起点
  const int col0 = blockIdx.x * BN;   // 本块 C 的列起点

  ⚠ blockIdx.x 对应 N（列），blockIdx.y 对应 M（行），
    与 host 端 dim3 grid_v3((N+BM-1)/BM, (M+BM-1)/BM) 严格对应。
    写反的话：方阵照样 PASS，256×512×256 这种长方形必错。

  计算时不用 tid：线程负责哪些输出，只由 (tx, ty) 决定。


========================================================================
第 6 步　16 个累加器（第 92–97 行）
========================================================================

  float acc[TM][TN];

  #pragma unroll
  for (int i = 0; i < TM; ++i)
  #pragma unroll
      for (int j = 0; j < TN; ++j)
          acc[i][j] = 0.0f;

  - 16 个 float，必须待在寄存器里。
  - 两个 #pragma unroll 是生死线：只有循环全展开、i/j 变成编译期常量，
    编译器才肯把 acc 分配成 16 个寄存器。
  - 否则 acc 会落到 local memory（本质是显存），每次累加都读写内存，
    性能直接塌方（常见现象：慢 5~10 倍）。
  - 清零发生在 k 循环之前；之后跨所有 kt 段一直累加，最后只写回一次 C。


========================================================================
第 7 步　沿 K 的分段循环（第 99–100 行）
========================================================================

  for (int kt = 0; kt < K; kt += BK)
  {
      ... 载入 → sync → 计算 → sync ...
  }

  - 段数 = ceil(K / 16)：
        K=2048 → 128 段
        K=337  → 22 段（最后一段只有 1/16 有效，靠零填充）
        K=7931 → 496 段
  - 不能把整个 K 搬进 smem：64×K 个 float，K=7931 时是 2 MB，远超 smem。
  - 每段的内核工作完全相同，只是 kt 在前进 —— 这就是所谓的「分块」。


========================================================================
第 8 步　载入 A tile（第 103–112 行）
========================================================================

  #pragma unroll
  for (int n = 0; n < BM * BK / (V3_BLOCK * V3_BLOCK); ++n)   // 1024/256 = 4
  {
      int idx = tid + n * V3_BLOCK * V3_BLOCK;
      int r   = idx / BK;        // 0..63（行）
      int c   = idx % BK;        // 0..15（k 偏移）
      As[r][c] = (row0 + r < M && kt + c < K)
               ? m1[(row0 + r) * K + kt + c]
               : 0.0f;
  }

  逐行：

  1) 循环上界 4
        A tile 有 64×16 = 1024 个元素，256 个线程 → 每人搬 4 个。

  2) idx = tid + n*256
        第 n 轮，线程 tid 搬第 tid + n×256 号元素（标准的一维铺开）。

  3) r = idx / 16，c = idx % 16
        线性编号还原成 (行, k 偏移)。
        连续 tid → 先在同一行里走完 16 列，再换下一行。

  4) 合并性
        一个 warp = 32 个 lane = 2 行 × 16 列。
        每行 16 个 float = 64 B，所以一条 warp 指令发出两段 64 B 的 global 读，
        字节全部用上（效率 100%，只是事务数比 128 B 多一倍）。

  5) 守卫 row0 + r < M
        行越界（M 不是 64 的倍数时出现）→ 填 0。

  6) 守卫 kt + c < K
        最后一段 k 不足 16 → 填 0。

  7) 为什么填 0 而不是跳过
        零填充让越界乘加贡献 0，计算段就不需要任何 if，
        所有线程走同一条路径（无 divergence）。

  8) As[r][c] 行主序写入
        后面计算时按列取（As[row][k]）——这正是 A 读取会产生
        bank conflict 的根源，详见第 11 步。


========================================================================
第 9 步　载入 B tile（第 116–125 行）
========================================================================

  #pragma unroll
  for (int n = 0; n < BK * BN / (V3_BLOCK * V3_BLOCK); ++n)   // 4
  {
      int idx = tid + n * V3_BLOCK * V3_BLOCK;
      int r   = idx / BN;        // 0..15（k 偏移）
      int c   = idx % BN;        // 0..63（列偏移）
      Bs[r][c] = (kt + r < K && col0 + c < N)
               ? m2[(kt + r) * N + col0 + c]
               : 0.0f;
  }

  逐行：

  1) 同样是每人 4 个，但编号方式不同：
        B tile 是 16 行 × 64 列，所以 r = idx / 64，c = idx % 64。

  2) 合并性更好
        一个 warp 的 32 个 tid 落在同一行、连续 32 列
        → 一条 128 B 的满合并事务。

  3) 守卫 kt + r < K（k 方向）和 col0 + c < N（列方向），同样是零填充。

  4) 布局是 Bs[k][col] 行主序，正好对应计算时 b[j] = Bs[k][...] 的读法。


========================================================================
第 10 步　第一次 __syncthreads（第 126 行）
========================================================================

  __syncthreads();

  语义：块内 256 个线程全部到达这里才继续，
        并且之前的 smem 写对之后所有线程可见。

  缺了它会怎样：
        A/B 还没写完，某些 warp 就已经开始读 tile
        → 读到旧数据或半个 tile → 结果随机错
        → 典型症状：小矩阵偶尔 PASS，大矩阵必 FAIL。

  ★ 这就是为什么 V3 仍然需要 smem：寄存器分块解决「读几次」，
    smem 解决「同一个数怎么让 256 个线程共享」。


========================================================================
第 11 步　计算：8 次读取 → 16 次 FMA（第 131–145 行）
========================================================================

  #pragma unroll
  for (int k = 0; k < BK; ++k)
  {
      float a[TM], b[TN];
  #pragma unroll
      for (int i = 0; i < TM; ++i) a[i] = As[ty * TM + i][k];
  #pragma unroll
      for (int j = 0; j < TN; ++j) b[j] = Bs[k][tx * TN + j];

  #pragma unroll
      for (int i = 0; i < TM; ++i)
  #pragma unroll
          for (int j = 0; j < TN; ++j)
              acc[i][j] += a[i] * b[j];
  }

  逐行：

  1) for k = 0..15 + 全展开
        展开后 k 是常量，smem 地址和寄存器分配都能定死。

  2) a[i] = As[4ty + i][k]（4 条 LDS）
        线程负责的 4 行 A。a[0] 会被 b[0..3] 各乘一次。

  3) b[j] = Bs[k][4tx + j]（4 条 LDS）
        b[0] 会被 a[0..3] 各乘一次。

  4) acc[i][j] += a[i] * b[j]（16 条 FFMA）
        a 的每个值用 4 次，b 的每个值用 4 次 —— 8 次读取服务 16 次乘加。
        这就是 2 B/FMA 的来源，也是 V3 比 V2 快 4 倍的根本原因。

  5) a[] / b[] 是局部数组
        循环全展开后它们同样变成寄存器，不会进 local memory。

  ----------------------------------------------------------------------
  5.1 一个必须纠正的说法：这两处都有 2-way bank conflict
  ----------------------------------------------------------------------

    代码原注释写着「As ... 广播，无冲突」，这不准确：

      As 读取：warp 内 ty 只有 0 和 1 两个值，读的是第 i 行和第 4+i 行。
               行 stride = 16 个字，两行相差 4×16 = 64 个字，
               64 mod 32 = 0 → 两个地址落在同一个 bank → 2-way conflict。

      Bs 读取：16 个 lane 的地址步长是 4（4tx），
               映射到 8 个 bank、每个 bank 2 个地址 → 2-way conflict。

    两个都是「2 个不同地址落进同一 bank」，
    不是广播冲突，也不是无冲突。

  ----------------------------------------------------------------------
  5.2 性能账：为什么实测 3.37 TFLOPS 是「合理」的
  ----------------------------------------------------------------------

    每 SM 每时钟：
      FFMA 吞吐 128 条 → 一个 warp 的 16 条 FFMA 占 16/4 = 4 个周期
      LDS 吞吐 128 B   → 一个 warp 的一条 LDS 占 1 个周期

    每 k 步（每 warp）：
      16 条 FFMA → 4 周期
      8 条 LDS  → 8 周期（无冲突）或 16 周期（2-way 冲突）

    → 无冲突上限 = 4/8 = 50% 峰值 = 8 TFLOPS
    → 2-way 冲突上限 = 4/16 = 25% 峰值 = 4 TFLOPS
    → 实测 3.37 TFLOPS ≈ 正好贴着 4 TFLOPS 这个上限

    换句话说：V3 现在不是被 DRAM 卡住，也不是被 FMA 卡住，
    而是被「smem 的 bank 冲突 + LDS 指令数」卡住。

  ----------------------------------------------------------------------
  5.3 下一版的修法（预期直接翻倍）
  ----------------------------------------------------------------------

    1) As 加 padding：
         __shared__ float As[BM][BK + 1];
         stride 变成 17，行差 4 → 68 mod 32 = 4，不再同 bank。

    2) Bs 改成列交错映射：
         b[j] = Bs[k][tx + 16 * j];          // 而不是 tx*4 + j
         写回时 col = col0 + tx + 16 * j;
         → LDS 无冲突，而且 C 写回从 25% 效率变成满合并，一箭双雕。

    预期：LDS 从 16 周期降到 8 周期，吞吐上限从 4 TFLOPS 抬到 8 TFLOPS。
    这比直接上 8×8 寄存器分块更便宜（不用多 48 个累加器）。


========================================================================
第 12 步　第二次 __syncthreads（第 146 行）
========================================================================

  __syncthreads();

  作用与第一次不同：保证所有线程把当前 tile「用完」之后，
  才允许下一轮覆盖 As / Bs。

  缺了它会怎样：
        跑得快的 warp 已经把下一段 tile 写进 smem，
        跑得慢的 warp 还在读上一段 → 结果错乱。

  ★ 两个 sync 的分工：
        第一个：写完 → 再读
        第二个：读完 → 再写


========================================================================
第 13 步　写回 C（第 152–163 行）
========================================================================

  #pragma unroll
  for (int i = 0; i < TM; ++i)
  {
      int row = row0 + ty * TM + i;
  #pragma unroll
      for (int j = 0; j < TN; ++j)
      {
          int col = col0 + tx * TN + j;
          if (row < M && col < N)
              result[row * N + col] = acc[i][j];
      }
  }

  逐行：

  1) row = 块行起点 + 线程行偏移 + 线程内第 i 行（覆盖 0..63）
  2) col = 块列起点 + 线程列偏移 + 线程内第 j 列
  3) 守卫 row < M && col < N：越界区域参与了计算（零填充），但不写回。
  4) 只在全部 k 段结束后写一次，每个线程 16 个结果。

  5) 写合并性
        固定 j 时，相邻 tx 的地址步长是 4 个 float = 16 B，
        一条 32 B sector 只用 8 B → store 效率约 25%。
        但 C 的总流量只有 GB/s 级（2048³ 全程 154 MB / 5 ms ≈ 31 GB/s），
        即使放大 4 倍也是零头，所以不影响成绩。
        改成 5.3 的列交错映射后，这里会顺带变成满合并。

  6) 大矩阵提醒
        row * N + col 是 int 运算；元素数超过 2^31（约 46341²）会溢出。
        本练习的尺寸都安全；想写得通用可以改成 (size_t)row * N + col。


========================================================================
第 14 步　三个坑与调参约束
========================================================================

  坑 1：#pragma unroll 不能删
      删了之后 acc 很可能落到 local memory，慢一个数量级。

  坑 2：两个 __syncthreads 缺一不可
      一个管「写完再读」，一个管「读完再写」。

  坑 3：grid 的 x/y 不能写反
      x 盖 N（列）、y 盖 M（行）。方阵会骗你，长方形才露馅。

  调参约束：
      1) BM = V3_BLOCK * TM，BN = V3_BLOCK * TN，改一个就要改全套。
      2) 改 BM/BN 后，host 端 grid_v3 的公式必须同步改。
      3) BK 增大 → smem 变大、段数变少；BK×2 时 smem 16 KB。
      4) __launch_bounds__ 与占用率：8 KB smem/块、约 40~60 个寄存器；
         实际约 4 块/SM（1024 线程），别为了 ILP 把寄存器吃到 3 块以下。


========================================================================
第 15 步　自测题
========================================================================

  1. 每线程每个 k 步只读 8 个数，凭什么能算 16 个输出？
     这 16 个输出分别用到哪几个数？

  2. 把 TM 从 4 改成 8、BM 仍保持 64，会发生什么？
     正确改法是什么？

  3. 第二个 __syncthreads 能不能省？省了会怎样？

  4. As[4ty+i][k] 这 4 条 LDS 到底有没有 bank conflict？
     为什么？怎么修？

  5. 337³ 时 V3 只有 1.83×，而 2048³ 有 3.39×，原因是什么？

  6. 为什么 V3 在 8864×8864×7931 上只掉 3.0%，
     而 V1 掉了 17.9%？

  ── 参考答案 ──
  1. 因为 a[0..3] 与 b[0..3] 交叉相乘正好产生 16 个乘积：
     acc[i][j] += a[i] * b[j]。a[i] 被 4 个 b 乘，b[j] 被 4 个 a 乘，
     8 次读取 → 16 次 FMA。
  2. 行覆盖变成 V3_BLOCK × TM = 16×8 = 128 > BM=64：
     第 8~15 个 ty 组会算到别的块的区域，结果错或越界。
     正确改法：BM 同时改成 128（BN 同理），并同步改 host 的 grid 公式；
     另外 64 个累加器 + 16 个操作数约 90+ 个寄存器，占用率会明显下降。
  3. 不能。下一轮的 As/Bs 写入会覆盖当前轮还在被其他 warp 读取的 tile，
     典型症状是偶发 FAIL。两次 sync 分别管「写完再读」和「读完再写」。
  4. 有，2-way。warp 内 ty=0/1 读第 i 行和第 4+i 行，
     行 stride 16 个字，相差 64 个字，64 mod 32 = 0 → 同一 bank。
     修法：As[BM][BK+1]（stride 17，行差 68 mod 32 = 4）。
  5. 64×64 的块盖到 6×6 = 384×384，而实际只有 337×337，
     约 23% 的乘加做在越界区域（零填充照算、只是不写回）。
     256×512×256 和 2048³ 都能被 64 整除，所以是完整的 2.25× / 3.39×。
  6. V3 每 FMA 的取数从 8 B 降到 2 B、全局读从 8 flop/B 升到 16 flop/B，
     对 L2/DRAM 延迟的依赖大幅下降；V1 的读混在 FMA 循环里、靠 L1 命中，
     工作集一旦超过 L2，延迟就藏不住。实测跌幅 17.9% / 7.4% / 3.0%
     正好随「取数延迟敏感度」单调下降。

########################################################################*/