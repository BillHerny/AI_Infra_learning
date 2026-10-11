#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <cfloat>            // FLT_MAX  —— 值的单位元
#include <climits>           // INT_MAX  —— 下标的「无效」哨兵
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
  argmax：找出「最大值所在的下标」

  和 reducemax.cu 的关系：
    max    返回「最大的值是多少」        → float
    argmax 返回「最大的值在哪个位置」    → int
    argmax 顺便也就得到了 max（最大值是副产品），所以一起返回。

  骨架完全没变，变的是「归约的元素」：
    求和    归约的是 1 个 float
    求 max  归约的是 1 个 float
    argmax  归约的是 1 个 **(值, 下标) 对** ← 这是本文件唯一的本质区别

  ⚠ 不能分两遍做（先求 max，再扫一遍找它在哪）—— 那要读两遍数据，慢一倍。
    必须让「值」和「下标」绑定着一起往下传。

  ──────────────────────────────────────────────────────────────
  三个新的坑（全是 max 版没有的）
  ──────────────────────────────────────────────────────────────

  ① 单位元从 1 个变成 2 个
       值的单位元  -FLT_MAX
       下标的单位元 INT_MAX   ← 表示「我这个线程压根没摊到数据」
     两者必须成套出现；只改一个就会出现「值为空但下标有效」的鬼状态。

  ② 下标是「全局下标 i」，不是 threadIdx.x
       每个线程在 grid-stride 里会处理多个元素，下标分别是 i0, i1, i2...
       写成 keepBetter(in[i], threadIdx.x, ...) 编译能过、小 N 能过、大 N 必错。

  ③ 并列（tie）时取哪个 —— 这是 argmax 唯一真正复杂的地方
       CPU 顺序扫描时，用 `>` 会自然地取到「第一个」出现的位置。
       但 GPU 是树形归约：两两合并的顺序和「下标顺序」无关，
       所以朴素写法（只比较值）在并列时**可能给出和 CPU 不同的下标**。
       两者都是正确答案（都满足 a[idx] == max），但如果你的下游依赖
       「必须和 CPU 一致」，就必须在比较里显式编码并列规则。

       本文件用 TIE_BREAK_BY_SMALLEST_INDEX 开关控制：
         打开 → 并列时取下标最小的（与 CPU 顺序扫描一致，可对拍）
         关闭 → 只比较值（少一条指令，但并列时下标不保证）
########################################################################*/

// 并列时是否取「下标最小」的那个。
//   1 = 与 CPU 顺序扫描一致（可以对拍下标）
//   0 = 只比较值，并列时下标取决于归约树的合并顺序
#define TIE_BREAK_BY_SMALLEST_INDEX 1

// ========= 「更优」的比较：值和下标必须一起更新 =========
//   只有把两个字段写在一起，才能保证不会出现「值取了新的、下标还是旧的」
__device__ __forceinline__
void keepBetter(float ov, int oi, float& m, int& mi)
{
#if TIE_BREAK_BY_SMALLEST_INDEX
    if (ov > m || (ov == m && oi < mi)) { m = ov; mi = oi; }
#else
    if (ov > m)                         { m = ov; mi = oi; }
#endif
}

// ============ KERNEL : 一个 block 求出 (最大值, 其下标) ============
// 注意输出是**两个数组**：outVal 存值，outIdx 存下标。
// 前提：blockSize 必须是 32 的倍数且 >= 32（本文件默认 256）
__global__ void argmaxWarp(const float* __restrict__ in,
                           float* __restrict__ outVal,
                           int*   __restrict__ outIdx,
                           int n)
{
    // ---- ① 寄存器内归约：每线程先在自己的多个元素里找最优 ----
    float m  = -FLT_MAX;    // 值的单位元
    int   mi = INT_MAX;     // 下标的单位元（哨兵：表示「还没有任何有效数据」）
    for (int i = blockIdx.x * blockDim.x + threadIdx.x;
         i < n;
         i += blockDim.x * gridDim.x)
    {
        keepBetter(in[i], i, m, mi);   // ⚠ 下标是 i（全局），不是 threadIdx.x
    }

    // ---- ② warp 内归约：__shfl_down_sync ----
    //   ⚠ 与 reducemax 的区别：这里要 shuffle「两份」数据
    //     一次传值、一次传下标，然后整对比较
    for (int offset = 16; offset > 0; offset >>= 1)
    {
        float ov = __shfl_down_sync(0xffffffffu, m,  offset);
        int   oi = __shfl_down_sync(0xffffffffu, mi, offset);
        keepBetter(ov, oi, m, mi);
    }

    // ---- ③ 每个 warp 的结果由 lane 0 写进共享内存（同样要两份）----
    __shared__ float warpVal[32];
    __shared__ int   warpIdx[32];

    int lane = threadIdx.x & 31;        // 等价于 threadIdx.x % 32
    int wid  = threadIdx.x >> 5;        // 等价于 threadIdx.x / 32

    if (lane == 0) { warpVal[wid] = m; warpIdx[wid] = mi; }
    __syncthreads();                    // 全 block 唯一的栅栏

    // ---- ④ 由 0 号 warp 归约这至多 32 个 warp 的部分结果 ----
    if (wid == 0)
    {
        int nWarps = blockDim.x >> 5;
        // 多余的 lane 必须补「两个单位元」，缺一不可
        m  = (lane < nWarps) ? warpVal[lane] : -FLT_MAX;
        mi = (lane < nWarps) ? warpIdx[lane] : INT_MAX;

        for (int offset = 16; offset > 0; offset >>= 1)
        {
            float ov = __shfl_down_sync(0xffffffffu, m,  offset);
            int   oi = __shfl_down_sync(0xffffffffu, mi, offset);
            keepBetter(ov, oi, m, mi);
        }

        if (lane == 0)
        {
            outVal[blockIdx.x] = m;     // 每个 block 交出一对
            outIdx[blockIdx.x] = mi;
        }
    }
}

int main(int argc, char** argv)
{
    int n              = (argc > 1) ? atoi(argv[1]) : (1 << 26);
    int blockSize      = (argc > 2) ? atoi(argv[2]) : 256;
    int elemsPerThread = (argc > 3) ? atoi(argv[3]) : 4;

    size_t bytes = (size_t)n * sizeof(float);

    int grid    = (n + blockSize - 1) / blockSize;                       // 上限
    int gridCap = (n + blockSize * elemsPerThread - 1) /
                  (blockSize * elemsPerThread);                          // 封顶值
    if (grid > gridCap) grid = gridCap;
    if (grid < 1) grid = 1;

    // ---- host 数据 ----
    //   全负的伪随机数据 + **刻意埋两个并列的最大值**（下标 3 和 5）。
    //
    //   为什么选 3 和 5？
    //     这两个下标在任何合理配置下都属于 **block 0 / warp 0 / lane 3 和 lane 5**，
    //     也就是必然落在同一个 warp 内 —— 这样「并列怎么裁决」这个问题
    //     就一定会在 warp 0 的 shuffle 树里发生，与 blockSize / elemsPerThread 无关。
    float* h_a = (float*)malloc(bytes);
    unsigned seed = 12345u;
    for (int i = 0; i < n; i++)
    {
        seed = seed * 1664525u + 1013904223u;                       // 线性同余发生器
        h_a[i] = (float)((seed >> 8) & 0xFFFFu) * -0.002f - 1.0f;   // 落在 [-132.07, -1.0]
    }
    const float planted  = -0.001f;   // 比 [-132.07, -1.0] 里的任何值都大
    const int   tieA     = 3;
    const int   tieB     = 5;
    h_a[tieA] = planted;              // 两个并列的最大值
    h_a[tieB] = planted;

    // ---- CPU 参考实现 ----
    //   ⚠ 初值同样是 -FLT_MAX；用 `>` 做比较 → 并列时留下「第一个」出现的位置
    float refVal = -FLT_MAX;
    int   refIdx = -1;
    for (int i = 0; i < n; i++)
        if (h_a[i] > refVal) { refVal = h_a[i]; refIdx = i; }

    // 统计有多少个元素等于最大值（用来解释「为什么并列会影响下标」）
    int tieCount = 0;
    for (int i = 0; i < n; i++) if (h_a[i] == refVal) tieCount++;

    double actualEpt = (double)n / ((double)grid * blockSize);
    printf("N = %d (%.1f MiB), grid = %d, blockSize = %d, elemsPerThread = %d (actual %.1f)\n",
           n, bytes / 1024.0 / 1024.0, grid, blockSize, elemsPerThread, actualEpt);
    printf("planted max = %f at indices %d and %d   (并列!)\n", planted, tieA, tieB);
    printf("TIE_BREAK_BY_SMALLEST_INDEX = %d\n\n", TIE_BREAK_BY_SMALLEST_INDEX);

    // ---- device 内存：输出是「值 + 下标」两个数组 ----
    float *d_a, *d_outVal;
    int   *d_outIdx;
    CUDA_CHECK(cudaMalloc(&d_a, bytes));
    CUDA_CHECK(cudaMalloc(&d_outVal, (size_t)grid * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_outIdx, (size_t)grid * sizeof(int)));
    CUDA_CHECK(cudaMemcpy(d_a, h_a, bytes, cudaMemcpyHostToDevice));

    // ---- warm-up ----
    argmaxWarp<<<grid, blockSize>>>(d_a, d_outVal, d_outIdx, n);
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
        argmaxWarp<<<grid, blockSize>>>(d_a, d_outVal, d_outIdx, n);
        CUDA_CHECK(cudaEventRecord(stop));
        CUDA_CHECK(cudaEventSynchronize(stop));

        float ms = 0.0f;
        CUDA_CHECK(cudaEventElapsedTime(&ms, start, stop));
        if (ms < best) best = ms;
    }

    // ---- 取回部分结果，host 端继续「整对比」----
    //   ⚠ 主机端也必须把值和下标配着比较，不能只取 max
    float* h_partVal = (float*)malloc((size_t)grid * sizeof(float));
    int*   h_partIdx = (int*)malloc((size_t)grid * sizeof(int));
    CUDA_CHECK(cudaMemcpy(h_partVal, d_outVal, (size_t)grid * sizeof(float),
                          cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(h_partIdx, d_outIdx, (size_t)grid * sizeof(int),
                          cudaMemcpyDeviceToHost));

    float gpuVal = -FLT_MAX;
    int   gpuIdx = INT_MAX;
    for (int i = 0; i < grid; i++)
    {
#if TIE_BREAK_BY_SMALLEST_INDEX
        if (h_partVal[i] > gpuVal ||
            (h_partVal[i] == gpuVal && h_partIdx[i] < gpuIdx))
        { gpuVal = h_partVal[i]; gpuIdx = h_partIdx[i]; }
#else
        if (h_partVal[i] > gpuVal)
        { gpuVal = h_partVal[i]; gpuIdx = h_partIdx[i]; }
#endif
    }

    // ---- 验证：分三层 ----
    //   ⭐ 关键：并列时「下标」不是唯一的，所以**不能**把 index 相等当成主要判据。
    //     真正该验证的是「这个下标确实指向最大值」——它对并列天然宽容。
    int okVal   = (gpuVal == refVal);                       // 值必须完全相等
    int okIdxVld= (gpuIdx >= 0 && gpuIdx < n &&
                   h_a[gpuIdx] == refVal);                  // 下标有效：a[idx] == max
    int okIdxEq = (gpuIdx == refIdx);                       // 下标与 CPU 一致（仅在开关打开时保证）

    printf("reference (cpu) = %.6f @ idx %d     (CPU 顺序扫描，取第一个)\n", refVal, refIdx);
    printf("gpu argmax      = %.6f @ idx %d\n", gpuVal, gpuIdx);
    printf("# of elements equal to max = %d\n", tieCount);
    printf("verify (value)                          : %s\n", okVal    ? "PASS" : "FAIL");
    printf("verify (index validity: a[idx] == max)  : %s\n", okIdxVld ? "PASS" : "FAIL");
    printf("verify (index == cpu's)                 : %s%s\n",
           okIdxEq ? "PASS" : "FAIL",
           (!okIdxEq && okIdxVld) ? "   <- 并列导致的差异，两个下标都是正确答案" : "");
    printf("best time = %.4f ms\n", best);
    printf("bandwidth = %.1f GB/s\n", (bytes / 1e9) / (best / 1e3));

    free(h_a);
    free(h_partVal);
    free(h_partIdx);
    CUDA_CHECK(cudaFree(d_a));
    CUDA_CHECK(cudaFree(d_outVal));
    CUDA_CHECK(cudaFree(d_outIdx));
    CUDA_CHECK(cudaEventDestroy(start));
    CUDA_CHECK(cudaEventDestroy(stop));
    return 0;
}

/*########################################################################
  实验与调试建议
########################################################################

  1. ⭐ 亲手看一次「并列导致下标不同」：
       把上面的 `#define TIE_BREAK_BY_SMALLEST_INDEX` 改成 **0**，重新编译运行。

     实测结果（N = 2^22，blockSize = 256，elemsPerThread = 4）：

       开关   GPU 下标   CPU 下标   verify(value)   verify(idx==cpu)
       ----   -------   -------   ------------    ---------------
         1       3         3         PASS            PASS
         0       5         3         PASS            FAIL  ← 并列裁决不同

     注意 `verify(value)` **两次都是 PASS** —— 因为 5 号位置的值确实是最大值。
     这就是 argmax 和 max 最本质的区别：**max 的答案是唯一的，argmax 的
     （并列时）不是唯一的。**

  2. 为什么朴素写法（开关=0）会给出 5 而不是 3？
       下标 3 和 5 都在 warp 0 的 lane 3 / lane 5 上。shuffle-down 的合并顺序是：

         offset=4:   lane 1 ← lane 5    (lane 1 先吸收掉 5)
         offset=2:   lane 3 → lane 1    (lane 1 再和 lane 3 比，此时并列 → 保留自己)
         offset=1:   lane 0 ← lane 1

       lane 1 是「先吸收了 5、后遇到 3」，而比较是「并列时保留自己」，
       于是 5 活到了最后。**归约树的合并顺序与下标顺序无关**，这就是根因。

       加上 `|| (ov == m && oi < mi)` 之后，并列时会显式比较下标 → 总是取最小。

  3. ⚠ 一个必查的写法错误：
       keepBetter(in[i], threadIdx.x, m, mi);   // ❌ 下标写成 block 内编号
       keepBetter(in[i], i,           m, mi);   // ✅ 必须是全局下标
     这个错误在小 N（一个线程只处理 1 个元素）时会「碰巧」看不出问题，
     只有 elemsPerThread > 1 时才开始错，而且错得毫无规律。

  4. ⚠ 另一个必查点：单位元是**两个**，必须成套出现。
       第 ① 步： float m = -FLT_MAX;  int mi = INT_MAX;
       第 ④ 步： (lane < nWarps) ? warpVal[lane] : -FLT_MAX;
                 (lane < nWarps) ? warpIdx[lane] : INT_MAX;
     只补值的单位元、把下标留成 0 或 -1，就会出现「值是真的、下标是假的」——
     结果 verify(value) PASS、verify(index validity) FAIL。

  5. 性能：与 reducemax **基本持平**（实测，blockSize=256，elemsPerThread=4）

         N       数据量     min time    带宽          说明
         ----    --------   ---------   -----------   --------------------
         2^22    16 MiB     0.0430 ms   389.8 GB/s    装进 L2，跑的是 L2 带宽
         2^24    64 MiB     0.3163 ms   212.2 GB/s    DRAM 受限
         2^26    256 MiB    1.2401 ms   216.5 GB/s    DRAM 受限（低时钟档）

       同条件对照 reducemax（N=2^26）：1.2349 ms / 217.4 GB/s
         → 差 0.4%，在噪声范围内。**argmax 比 max 一点没慢。**

     按理说 argmax 多出来的开销有三项：
         · 每个 shuffle 步骤多一次 __shfl_down_sync（传下标）
         · 多一个 int 的共享内存数组（128 字节）
         · 比较从 1 次 fmaxf 变成 1~2 次浮点比较 + 1 次整数比较
     但它们全部被 DRAM 带宽掩盖了 —— 这就是「访存受限」的好处：
     只要计算量不越过内存这条线，加多少计算都不涨时间。

        ⚠ 反过来说，如果实测明显慢于 reducemax，先怀疑 elemsPerThread 太小
          （块内归约占比过高），而不是怀疑这几次额外比较。

  6. 用 ncu 对比：
       ncu --set full -k argmaxWarp --launch-count 1 ./build/bin/argmax $((1<<26)) 256 4
     预期 DRAM Throughput ≈ 96%，Mem Pipes Busy 与 reducemax 同量级。

  7. 扩展方向：
       · argmin —— 把比较反向（用 `<` / `-FLT_MAX` 换成 `+FLT_MAX`），其余不动
       · top-2 / top-k —— 把「一对」扩成「k 个候选」，用插入排序式的合并
       · 找最小距离 —— argmin 的常见变体（最近邻搜索）

########################################################################*/
