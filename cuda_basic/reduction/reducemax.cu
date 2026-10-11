#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <cfloat>            // FLT_MAX —— 求 max 的「单位元」在这里
#include <cuda_runtime.h>

#define CUDA_CHECK(call)                                          \
    do{                                                           \
        cudaError_t err = (call);                                 \
        if(err != cudaSuccess){                                   \
            fprintf(stderr, "CUDA error %s:%d: %s\n",             \
                   __FILE__, __LINE__, cudaGetErrorString(err));  \
            exit(EXIT_FAILURE);                                   \
        }                                                         \
    }while(0)

/*########################################################################
  求最大值归约（max reduction）

  和 reducesumV3.cu 是**同一套骨架**，只换了三处：
    ① 单位元（identity）：0.0f      → -FLT_MAX
    ② 合并运算（op）：    a + b     → fmaxf(a, b)
    ③ 主机端最终归约：    sum += x  → best = fmaxf(best, x)

  为什么要强调「单位元」？
    归约的骨架是「用单位元填空白，再用合并运算两两撞下来」。
    求和时 0 是单位元（x + 0 = x），所以越界线程补 0 无害；
    求 max 时 0 是**毒药**（max(x, 0) 会把所有负数抹成 0），
    必须补 -FLT_MAX（-∞）。

    ⚠ 本文件用的是 grid-stride 循环，所以「越界」不靠补零处理 ——
      但单位元仍然出现在两处：
        · 每线程的累加器初值       float m = -FLT_MAX;
        · 最后一步「多余的 lane」  (lane < nWarps) ? ... : -FLT_MAX
      如果把这两处写成 0.0f，全负数数组会算出 0，且结果看起来「很合理」。
########################################################################*/

// ============ KERNEL : 一个 block 归约出一段数据的最大值 ============
// 前提：blockSize 必须是 32 的倍数且 >= 32（本文件默认 256）
__global__ void reduceMaxWarp(const float* __restrict__ in,
                              float* __restrict__ out,
                              int n)
{
    // ---- ① 寄存器内归约（同 reducesumV3，只换了运算和单位元）----
    //   grid-stride：每个线程先在寄存器里把多个元素「撞」成一个最大值
    //   初值必须是 -FLT_MAX，不能是 0.0f
    float m = -FLT_MAX;
    for (int i = blockIdx.x * blockDim.x + threadIdx.x;
         i < n;
         i += blockDim.x * gridDim.x)
    {
        m = fmaxf(m, in[i]);
    }

    // ---- ② warp 内归约：__shfl_down_sync ----
    //   与求和版唯一的区别：把 `sum += x` 换成 `m = fmaxf(m, x)`
    //   ⚠ 不能写 `m += __shfl_down_sync(...)` —— 那是加法，不是取大
    for (int offset = 16; offset > 0; offset >>= 1)
        m = fmaxf(m, __shfl_down_sync(0xffffffffu, m, offset));

    // ---- ③ 每个 warp 的结果由 lane 0 写进共享内存 ----
    //   静态共享内存：大小编译期定死 32 个 float，所以启动时不用第三个参数
    __shared__ float warpMax[32];       // 一个 block 最多 32 个 warp（1024 线程）

    int lane = threadIdx.x & 31;        // 等价于 threadIdx.x % 32
    int wid  = threadIdx.x >> 5;        // 等价于 threadIdx.x / 32

    if (lane == 0) warpMax[wid] = m;
    __syncthreads();                    // 全 block 唯一的栅栏

    // ---- ④ 由 0 号 warp 归约这至多 32 个 warp 部分最大值 ----
    if (wid == 0)
    {
        int nWarps = blockDim.x >> 5;
        // 多余的 lane 必须补「单位元」-FLT_MAX，不能补 0
        m = (lane < nWarps) ? warpMax[lane] : -FLT_MAX;

        for (int offset = 16; offset > 0; offset >>= 1)
            m = fmaxf(m, __shfl_down_sync(0xffffffffu, m, offset));

        if (lane == 0) out[blockIdx.x] = m;    // 每个 block 一个部分最大值
    }
}

int main(int argc, char** argv)
{
    int n              = (argc > 1) ? atoi(argv[1]) : (1 << 26);
    int blockSize      = (argc > 2) ? atoi(argv[2]) : 256;
    int elemsPerThread = (argc > 3) ? atoi(argv[3]) : 4;   // 每个线程大约处理几个元素

    size_t bytes = (size_t)n * sizeof(float);

    // grid 不再「精确覆盖」，而是「封顶」：按 elemsPerThread 把 block 数缩小，
    // 剩下的元素交给 kernel 里的 grid-stride 循环按轮次吃掉。
    int grid    = (n + blockSize - 1) / blockSize;                       // 上限
    int gridCap = (n + blockSize * elemsPerThread - 1) /
                  (blockSize * elemsPerThread);                          // 封顶值
    if (grid > gridCap) grid = gridCap;
    if (grid < 1) grid = 1;

    // ---- host 数据 ----
    //   用确定性伪随机（LCG）填一片**全负**的数据（[-132, -1]），再埋一个已知最大值。
    //
    //   ⚠ 为什么刻意全取负值？
    //     因为「单位元写错成 0.0f」这个 bug 只在**真最大值 < 0** 时才暴露：
    //       · 数据全正 → fmaxf(0.0f, v) 仍然等于 v，结果碰巧正确 → 测试 PASS（bug 潜伏）
    //       · 数据全负 → fmaxf(0.0f, v) 变成 0.0f，答案错成 0 → 测试 FAIL（bug 现形）
    //     所以一个「只用正数」的测试是抓不到这个 bug 的，那正是它危险的地方。
    float* h_a = (float*)malloc(bytes);
    unsigned seed = 12345u;
    for (int i = 0; i < n; i++)
    {
        seed = seed * 1664525u + 1013904223u;                       // 线性同余发生器
        h_a[i] = (float)((seed >> 8) & 0xFFFFu) * -0.002f - 1.0f;   // 落在 [-132.07, -1.0]
    }
    const float planted = -0.001f;                            // 埋一个必然最大的值（> -1.0）
    int plantedIdx = n / 3;
    h_a[plantedIdx] = planted;

    // ---- CPU 参考实现：这才是「标准答案」----
    //   ⚠ 注意初值同样是 -FLT_MAX；用 0.0f 做初值会导致全负数时算出 0
    float refMax = -FLT_MAX;
    for (int i = 0; i < n; i++) refMax = fmaxf(refMax, h_a[i]);

    // 实际每线程元素数 = n / (grid * blockSize)
    double actualEpt = (double)n / ((double)grid * blockSize);
    printf("N = %d (%.1f MiB), grid = %d, blockSize = %d, elemsPerThread = %d (actual %.1f)\n",
           n, bytes / 1024.0 / 1024.0, grid, blockSize, elemsPerThread, actualEpt);
    printf("planted max = %f at index %d\n", planted, plantedIdx);

    // ---- device 内存 ----
    float *d_a, *d_out;
    CUDA_CHECK(cudaMalloc(&d_a, bytes));
    CUDA_CHECK(cudaMalloc(&d_out, (size_t)grid * sizeof(float)));
    CUDA_CHECK(cudaMemcpy(d_a, h_a, bytes, cudaMemcpyHostToDevice));

    // ---- warm-up ----
    reduceMaxWarp<<<grid, blockSize>>>(d_a, d_out, n);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    // ---- 计时：best-of-20 ----
    cudaEvent_t start, stop;
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));

    const int iters = 20;
    float best = 1e30f;
    for (int k = 0; k < iters; k++)
    {
        CUDA_CHECK(cudaEventRecord(start));
        reduceMaxWarp<<<grid, blockSize>>>(d_a, d_out, n);
        CUDA_CHECK(cudaEventRecord(stop));
        CUDA_CHECK(cudaEventSynchronize(stop));

        float ms = 0.0f;
        CUDA_CHECK(cudaEventElapsedTime(&ms, start, stop));
        if (ms < best) best = ms;
    }

    // ---- 取回部分最大值，host 端继续「撞」----
    //   ⚠ 这一步也必须用 fmaxf，不能把部分最大值加起来
    float* h_partial = (float*)malloc((size_t)grid * sizeof(float));
    CUDA_CHECK(cudaMemcpy(h_partial, d_out, (size_t)grid * sizeof(float),
                          cudaMemcpyDeviceToHost));

    float gpuMax = -FLT_MAX;
    for (int i = 0; i < grid; i++) gpuMax = fmaxf(gpuMax, h_partial[i]);

    // ---- 验证 ----
    //   求 max 与求和最大的不同：**结果是精确的**。
    //   max 的结果必然「恰好等于」某个输入元素（不存在舍入），
    //   所以这里可以直接用 == 比较，不需要 1e-5 之类的容差。
    //   （对比 reducesumV3：求和必须用容差，因为浮点加法有舍入）
    printf("reference (cpu)    = %.6f\n", refMax);
    printf("gpu max            = %.6f\n", gpuMax);
    printf("verify: %s\n", (gpuMax == refMax) ? "PASS" : "FAIL");
    printf("  - hits planted value: %s\n", (gpuMax == planted) ? "yes" : "NO");
    printf("best time = %.4f ms\n", best);
    printf("bandwidth = %.1f GB/s\n", (bytes / 1e9) / (best / 1e3));

    free(h_a);
    free(h_partial);
    CUDA_CHECK(cudaFree(d_a));
    CUDA_CHECK(cudaFree(d_out));
    CUDA_CHECK(cudaEventDestroy(start));
    CUDA_CHECK(cudaEventDestroy(stop));
    return 0;
}

/*########################################################################
  对拍/调试建议
########################################################################

  1. 故意把单位元写错，看会发生什么：
       把 kernel 里的**两处** -FLT_MAX 都改成 0.0f，重新编译运行 →
       本文件的数据会给出 `gpu max = 0.000000` → **FAIL**（bug 现形）。

     再做反面实验：把 `h_a[i] = ... * -0.002f - 1.0f;` 里的负号去掉
     （数据变成全正），同样写错单位元 → **结果反而 PASS**。
     两次实验对照，就能亲眼看到「单位元 bug 只在真最大值 < 0 时才暴露」。
     ⚠ 所以「只用正数测 max」是不充分的 —— 这是最危险的一类静默 bug。

     【实测记录】用临时副本改单位元后实际跑过，结果与上面的推理完全一致
       （N = 2^22，blockSize = 256，elemsPerThread = 4）：

                                    kernel 单位元    gpu max        verify   hits planted
         全负数据（本文件默认）      -FLT_MAX         -0.001000      PASS        yes
         全负数据                     0.0f  ❌        0.000000       FAIL        NO
         全正数据                     0.0f  ❌      200.000000       PASS        yes  ← 潜伏！

       注意第三行：**同一处错误，换成全正数据就测不出来了**。

     补充：两处单位元分别是
       · 第 ① 步 `float m = -FLT_MAX;`
       · 第 ④ 步 `(lane < nWarps) ? warpMax[lane] : -FLT_MAX;`
     两处都要改对。只改第一处、漏掉第二处，在 blockSize=256（8 个 warp）时
     同样会算错 —— 因为 nWarps = 8 < 32，第 8~31 号 lane 的初值会顺着
     shuffle-down 一路汇到 lane 0（offset = 16,8,4,2,1 每轮都会把它们加进来），
     它们补的 0.0f 会把负的最大值抹掉。

  2. 检验「精确比较」这个论断：
       本文件用 `gpuMax == refMax` 而不是容差比较。
     连续跑 10 次，结果应当**每次都逐位相同**（max 不会像 sum 那样波动）。

  3. 一个容易错的写法：
       m += __shfl_down_sync(0xffffffffu, m, offset);   // ❌ 这是加法！
       m  = fmaxf(m, __shfl_down_sync(...));            // ✅
     从求和版复制粘贴过来时，这一处最容易漏改，而且编译能过、结果很离谱。

  4. 主机端那一行同样易错：
       sum += h_partial[i];                    // ❌ 把部分最大值加起来
       gpuMax = fmaxf(gpuMax, h_partial[i]);   // ✅

  5. 关于 NaN（本文件未处理）：
       fmaxf(a, NaN) 返回 a，也就是说 **NaN 会被静默忽略**。
     如果你的数据可能含 NaN，必须显式处理（用 isnan 判断 / 用 _mm_max_ps 之类的语义），
     否则「最大值」会悄悄跳过所有 NaN。

  6. 想扩展成 argmax（返回最大值所在的下标）：
     只需要把「shuffle 一个 float」改成「shuffle 一个 {value, index} 结构」：
       两个 __shfl_down_sync（一个传值、一个传下标），
       比较时若值更大就同时更新下标。
     这是理解「归约不只能归约标量」的最好练习。

  7. 用规模扫一遍，看 L2 的效应（实测，blockSize=256，elemsPerThread=4）：

         N       数据量      min time     带宽       峰值占比   说明
         ----    --------    ---------    --------   --------   ------------
         2^22    16 MiB      0.0369 ms    455.1 GB/s   177% ❗   16 MiB < L2(24 MiB)
         2^24    64 MiB      0.2764 ms    242.8 GB/s    95%      DRAM 受限
         2^26    256 MiB     1.2360 ms    217.2 GB/s    85%      DRAM 受限（低时钟档）

     ⚠ 第一行的 455 GB/s **超过了本机 DRAM 峰值 256.3 GB/s**。
       这不是代码 bug —— 而是因为工作集（16 MiB）能整个装进 L2（24 MiB），
       此时跑的就不再是 DRAM 带宽，而是 L2 带宽。

       这也正呼应了 README 里那条警告的**适用范围**：
         「超过硬件峰值的数字 = 代码 bug」的前提是**工作集远大于 L2**。
         一旦数据能装进 cache，带宽超过 DRAM 峰值反而是**物理正确**的。
       判断方法：看规模。N 小到能装进 L2 时，就必须拿 L2 带宽（而非 DRAM 峰值）去判读。

########################################################################*/
