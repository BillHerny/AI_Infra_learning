#include <cstdio>
#include <cstdlib>
#include <cmath>
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

// ============ KERNEL : 一个 block 归约出一段数据的和 ============
// V3 相对 V2 只改了「块内归约」这一段：
//   V2：数据搬进共享内存，再逐轮减半 → 3 次 __syncthreads，约 44 个 L1TEX wavefront
//   V3：块内归约全部用 __shfl_down_sync 在「寄存器之间」完成
//       共享内存只剩「每个 warp 一个格子」（32 个 float），栅栏只剩 1 次
//
//   前提：blockSize 必须是 32 的倍数且 >= 32（本文件默认 256）
__global__ void reduceSumWarp(const float* __restrict__ in,
                              float* __restrict__ out,
                              int n)
{
    // ---- ① 寄存器内累加（与 V2 相同）----
    //   grid-stride：每个线程先把多个元素累加到局部变量 sum 里，
    //   越界由循环条件天然处理，不需要补单位元（sum 初值 0.0f 就是单位元）
    float sum = 0.0f;
    for (int i = blockIdx.x * blockDim.x + threadIdx.x;
         i < n;
         i += blockDim.x * gridDim.x)
    {
        sum += in[i];
    }

    // ---- ② warp 内归约：__shfl_down_sync ----
    //   第 k 轮让 lane i 吃掉 lane (i + offset) 的「寄存器值」。
    //   offset = 16, 8, 4, 2, 1 共 5 轮之后，lane 0 手里就是整个 warp 的和。
    //   全程只动寄存器：不碰共享内存、不需要任何栅栏。
    //
    //   ⚠ 当 lane + offset >= 32 时，返回的是「自己」的值（相当于自我累加），
    //     那些 lane 的结果是垃圾值 —— 但我们只读 lane 0，所以不影响正确性。
    for (int offset = 16; offset > 0; offset >>= 1)
        sum += __shfl_down_sync(0xffffffffu, sum, offset);

    // ---- ③ 每个 warp 的结果由 lane 0 写进共享内存 ----
    //   注意这里是「静态」共享内存：一个block最多1024个线程，大小在编译期就定死为 32 个 float，
    //   所以启动 kernel 时不再需要 <<<grid, block, shmem>>> 的第三个参数。
    __shared__ float warpSums[32];      // 一个 block 最多 32 个 warp（1024 线程）

    int lane = threadIdx.x & 31;        // 等价于 threadIdx.x % 32
    int wid  = threadIdx.x >> 5;        // 等价于 threadIdx.x / 32

    if (lane == 0) warpSums[wid] = sum;
    __syncthreads();                    // 全 block 唯一的栅栏（V2 有 3 次）

    // ---- ④ 由 0 号 warp 归约这至多 32 个 warp 部分和 ----
    if (wid == 0)
    {
        int nWarps = blockDim.x >> 5;
        sum = (lane < nWarps) ? warpSums[lane] : 0.0f;   // 多余的 lane 补单位元

        for (int offset = 16; offset > 0; offset >>= 1)
            sum += __shfl_down_sync(0xffffffffu, sum, offset);

        if (lane == 0) out[blockIdx.x] = sum;            // 每个 block 一个部分和
    }
}

int main(int argc, char** argv)
{
    int n              = (argc > 1) ? atoi(argv[1]) : (1 << 26);
    int blockSize      = (argc > 2) ? atoi(argv[2]) : 256;
    int elemsPerThread = (argc > 3) ? atoi(argv[3]) : 4;   // 每个线程大约处理几个元素

    size_t bytes = (size_t)n * sizeof(float);

    // grid 不再「精确覆盖」，而是「封顶」：
    //   先算出「一线程一元素」需要多少 block（grid 的上限），
    //   再按 elemsPerThread 把它缩小，剩下的元素交给 kernel 里
    //   的 grid-stride 循环按轮次吃掉。
    int grid    = (n + blockSize - 1) / blockSize;                       // 上限
    int gridCap = (n + blockSize * elemsPerThread - 1) /
                  (blockSize * elemsPerThread);                          // 封顶值
    if (grid > gridCap) grid = gridCap;
    if (grid < 1) grid = 1;

    // ---- host 数据：全填 1.0f，精确和就是 n ----
    float* h_a = (float*)malloc(bytes);
    for (int i = 0; i < n; i++) h_a[i] = 1.0f;

    double ref = 0.0;
    for (int i = 0; i < n; i++) ref += h_a[i];

    // 实际每线程元素数 = n / (grid * blockSize)
    double actualEpt = (double)n / ((double)grid * blockSize);
    printf("N = %d (%.1f MiB), grid = %d, blockSize = %d, elemsPerThread = %d (actual %.1f)\n",
           n, bytes / 1024.0 / 1024.0, grid, blockSize, elemsPerThread, actualEpt);

    // ---- device 内存 ----
    float *d_a, *d_out;
    CUDA_CHECK(cudaMalloc(&d_a, bytes));
    CUDA_CHECK(cudaMalloc(&d_out, (size_t)grid * sizeof(float)));
    CUDA_CHECK(cudaMemcpy(d_a, h_a, bytes, cudaMemcpyHostToDevice));

    // 注意：V3 没有动态共享内存，不需要 shmem 变量，也不传第三个启动参数

    // ---- warm-up ----
    reduceSumWarp<<<grid, blockSize>>>(d_a, d_out, n);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    // ---- 计时 ----
    cudaEvent_t start, stop;
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));

    const int iters = 20;
    float best = 1e30f;
    for (int k = 0; k < iters; k++)
    {
        CUDA_CHECK(cudaEventRecord(start));
        reduceSumWarp<<<grid, blockSize>>>(d_a, d_out, n);
        CUDA_CHECK(cudaEventRecord(stop));
        CUDA_CHECK(cudaEventSynchronize(stop));

        float ms = 0.0f;
        CUDA_CHECK(cudaEventElapsedTime(&ms, start, stop));
        if (ms < best) best = ms;
    }

    // ---- 取回部分和，host 用 double 累加 ----
    float* h_partial = (float*)malloc((size_t)grid * sizeof(float));
    CUDA_CHECK(cudaMemcpy(h_partial, d_out, (size_t)grid * sizeof(float),
                          cudaMemcpyDeviceToHost));
    double sum = 0.0;
    for (int i = 0; i < grid; i++) sum += (double)h_partial[i];

    printf("reference (double) = %.0f\n", ref);
    printf("gpu sum            = %.0f\n", sum);
    printf("verify: %s\n", fabs(sum - ref) <= 1e-3 ? "PASS" : "FAIL");
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
  reducesumV3.cu 逐行详解
  —— 代码在上，讲解全部在下（与 reducesum.cu / reducesumV2.cu 同一风格）

  前置建议：
    这份讲解假定你已经读完 reducesumV2.cu 底部的注释块。
    如果对「grid-stride」和「单位元」还不熟，先回去看那两个概念。

  目录
    第 0 步  从 V2 到 V3：一句话说清改了什么
    第 1 步  先补一个概念：warp 和 lane
    第 2 步  __shfl_down_sync 到底在干什么（本文最重要的一节）
    第 3 步  kernel 逐行
    第 4 步  main 里变了的几处
    第 5 步  V2 vs V3 对照表（设计 + 实测）
    第 6 步  ⚠ 一个必须纠正的说法：shuffle 并不是「免费」的
    第 7 步  自测题


========================================================================
第 0 步　从 V2 到 V3：一句话说清改了什么
========================================================================

  V2 和 V3 的「装载阶段」**一模一样**（都是 grid-stride 寄存器内累加）。
  变的是「块内归约」这一段：

      V2：把 256 个局部和搬进【共享内存】，再逐轮减半
          → 3 次 __syncthreads，约 44 个 L1TEX/MIO wavefront

      V3：块内归约改用【warp shuffle】在寄存器之间直接换
          → 1 次 __syncthreads，共享内存只剩 32 个 float

  一个直观的对比：

                        V2（共享内存）                V3（shuffle）
      ------------------  --------------------------  --------------------------
      数据几次落"地"      256 个值搬进 sdata          全程在寄存器/交换网络里
      栅栏次数            3                            1
      动态共享内存        1024 字节                     0 字节
      静态共享内存        0                             128 字节（warpSums[32]）
      启动参数            <<<grid, block, shmem>>>      <<<grid, block>>>
      需要 volatile       需要（展开最后 warp 那段）    不需要
      blockSize 约束      >= 64                        >= 32

  ⚠ 「1 次栅栏」「0 字节动态共享内存」是 V3 最直观的两个标志。


========================================================================
第 1 步　先补一个概念：warp 和 lane
========================================================================

  GPU 硬件调度的**最小单位不是线程，而是 warp** —— 固定 32 个线程一组。

      block（256 线程） = 8 个 warp

        ┌──── warp 0 ────┐┌──── warp 1 ────┐        ┌──── warp 7 ────┐
        T0 T1 T2 ... T31   T32 T33 ... T63    ...     T224 ... T255

  两个术语：

      术语      含义                       怎么算出来
      ------    -----------------------    -----------------------------
      wid       我在第几个 warp             wid  = threadIdx.x >> 5   （除以 32）
      lane      我在本 warp 内的编号         lane = threadIdx.x & 31   （模 32）

  ⚠ 代码里的这两个位运算不是"炫技"，而是因为 32 是 2 的幂：
      除以 32  = 右移 5 位
      模 32    = 与 31  （0b11111）
    这两句在 CUDA 代码里是标准写法，见到就能立刻读出来。

  为什么 V3 要按 warp 来写？因为 __shfl_down_sync 只能在**同一个 warp 内部**交换数据。
  跨 warp 是做不到的 —— 这一点是理解 V3 后面"为什么还要共享内存"的关键。


========================================================================
第 2 步　__shfl_down_sync 到底在干什么
========================================================================

  ★ 这是全文最重要的一节。看懂它，V3 就没有难点。

------------------------------------------------------------------------
2.1 函数签名和一句话语义
------------------------------------------------------------------------
      T __shfl_down_sync(unsigned mask, T var, unsigned delta)

      **一句话：返回「lane 号比我大 delta 的那个 lane」的 var 值。**

  三个参数：

      mask    哪些 lane 参与这次交换。0xffffffff = 全部 32 个 lane 都参与
      var     要交换的那份数据（这里是寄存器变量 sum）
      delta   往下看几个 lane

  默认还有一个没写的参数 width = 32（warp 的大小），本文件用默认值。

  ⚠ 关键前提：mask 里标为"参与"的 lane **必须都处于活跃（converged）状态**。
    如果某个 lane 已经因为分支退出去了，行为是未定义的。
    所以第 ② 步的 shuffle 写在**没有分支**的循环里 —— 32 个 lane 必然都在。

------------------------------------------------------------------------
2.2 把 5 轮交换画出来（以 offset = 16 为例）
------------------------------------------------------------------------

      自己的值：  lane  0    1    2  ...  15 | 16   17  ...  31
                        v0   v1   v2  ... v15| v16  v17 ...  v31

      offset=16 拿到的：
                         v16  v17  v18 ... v31| v16  v17 ...  v31
                                              ↑
                                    16~31 号越界（32 号不存在）
                                    → 返回的是"自己的值"

      累加之后：
                         v0+  v1+  v2+ ... v15+ | v16+  v17+ ...
                         v16  v17  v18    v31  | v16   v17
                                                  ↑
                                        这些是垃圾值 —— 但没人去读它们

  ⚠ 关于"越界返回自己的值"这条语义：
    它是**为了安全**（不会崩、不会读到别人），而不是"正确"。
    16~31 号 lane 的结果是错的（相当于把自己算了两遍），
    但我们**最终只读 lane 0**，而 lane 0 在每一轮的来源都合法 → 不影响正确性。

------------------------------------------------------------------------
2.3 为什么是 5 轮？—— 每一步覆盖的 lane 范围
------------------------------------------------------------------------

      轮次   offset   这一轮过后 lane 0 手里累计覆盖的 lane
      ----   ------   ------------------------------------------
        1      16     {0, 16}
        2       8     {0, 16, 8, 24}
        3       4     {0, 16, 8, 24, 4, 20, 12, 28}
        4       2     + {2, 18, 10, 26, 6, 22, 14, 30}          共 16 个
        5       1     + {1, 3, 5, 7, 9, 11, 13, 15,
                          17, 19, 21, 23, 25, 27, 29, 31}      共 32 个 ✅

  看出来了吗：**每多跑一轮，覆盖的 lane 数翻倍**。
      1 → 2 → 4 → 8 → 16 → 32
  所以要覆盖 32 个 lane，只需要 log2(32) = 5 轮。

  这就是它和 V2 的 `for (int s = ...; s >>= 1)` 的对应关系：

      V2 每轮把【存活的数字个数】减半  →  8 轮
      V3 每轮把【覆盖的 lane 数】翻倍  →  5 轮

  两者都是 2 的幂次收敛，只是方向表述不同。

------------------------------------------------------------------------
2.4 和共享内存版本的本质区别
------------------------------------------------------------------------

      V2：sdata[tid] += sdata[tid + s];
           · 2 次读共享内存 + 1 次写共享内存 + 1 次加
           · 因为改了共享内存，必须考虑"别人什么时候能看到" → 栅栏 / volatile

      V3：sum += __shfl_down_sync(0xffffffffu, sum, offset);
           · 1 次 lane 间交换 + 1 次加
           · 是"我主动去取别人的值"，取回来的东西**是明确的**

  ⚠ 这个区别带来一个重要的后果：
    共享内存版本里那个 `volatile` 是"防止编译器把别人的改动缓存掉"，
    而 shuffle 是**显式的传值**，编译器完全看得懂 ——
    所以 V3 **根本不需要 volatile**。

------------------------------------------------------------------------
2.5 ⚠ 一个最容易被忽略的细节：mask 与"全部 lane 参与"
------------------------------------------------------------------------

  `mask = 0xffffffff` 的意思是"32 个 lane 全都参与"。

  在第 ② 步里，所有线程都会走到这个循环（没有分支），
  所以 32 个 lane 一定都在 → mask 正确。

  在第 ④ 步里，shuffle 写在 `if (wid == 0)` 里面，
  但进到这个分支的是**整个 warp 0 的 32 个 lane**（条件对 warp 0 是"全真"），
  所以 32 个 lane 依然都在 → mask 依然正确。

  ⚠ 反过来：如果你把某个 shuffle 写在 `if (lane < 8)` 这种**真正会发散**的分支里，
    mask 写 0xffffffff 就是错的（有 lane 不在场），必须改成 0xff。


========================================================================
第 3 步　kernel 逐行
========================================================================

------------------------------------------------------------------------
3.1 函数签名
------------------------------------------------------------------------
      __global__ void reduceSumWarp(const float* __restrict__ in,
                                    float* __restrict__ out,
                                    int n)

  和 V2 完全一样，只是函数名换了（叫 Warp 是为了和共享内存版区分）。

------------------------------------------------------------------------
3.2 ① 寄存器内累加（与 V2 逐字相同）
------------------------------------------------------------------------
      float sum = 0.0f;
      for (int i = blockIdx.x * blockDim.x + threadIdx.x;
           i < n;
           i += blockDim.x * gridDim.x)
      {
          sum += in[i];
      }

  这一段在 reducesumV2.cu 的注释块第 2 步详细讲过，这里不重复。

  ⚠ 只提醒一点：这里的 `sum` 住在**寄存器**里，而且它是**每线程私有**的。
    V3 后面做的事情，就是把这 8388608 个私有的 sum 一层层合起来：

        warp 内 32 个 sum  →  1 个（靠 __shfl_down_sync，第 ② 步）
        block 内 8 个 warp  →  1 个（靠共享内存 + 栅栏，第 ③④ 步）

------------------------------------------------------------------------
3.3 ② warp 内归约
------------------------------------------------------------------------
      for (int offset = 16; offset > 0; offset >>= 1)
          sum += __shfl_down_sync(0xffffffffu, sum, offset);

  就两行，但它是 V3 的灵魂。展开写就是：

      sum += __shfl_down_sync(0xffffffffu, sum, 16);   // 5 轮
      sum += __shfl_down_sync(0xffffffffu, sum,  8);
      sum += __shfl_down_sync(0xffffffffu, sum,  4);
      sum += __shfl_down_sync(0xffffffffu, sum,  2);
      sum += __shfl_down_sync(0xffffffffu, sum,  1);

  语义见第 2 步。执行完之后：**每个 warp 的 lane 0 手里就是该 warp 32 个 sum 的和。**

  ⚠ 两个容易踩的点：

    (a) 别把 `sum +=` 写成别的。因为要和别的 lane 的**值**相加，
        所以是 `sum += 取回来的值`。求 max 时这里就要写
        `m = fmaxf(m, __shfl_down_sync(...))`（见 reducemax.cu）。

    (b) 这里**不需要也不能**写 `__syncthreads()` ——
        它只在 warp 内部交换，跨 warp 管不着。

------------------------------------------------------------------------
3.4 ③ 把每个 warp 的结果写进共享内存
------------------------------------------------------------------------
      __shared__ float warpSums[32];

      int lane = threadIdx.x & 31;        // = threadIdx.x % 32
      int wid  = threadIdx.x >> 5;        // = threadIdx.x / 32

      if (lane == 0) warpSums[wid] = sum;
      __syncthreads();

  (a) 为什么这里**还需要**共享内存？

      因为 __shfl_down_sync 只能在**一个 warp 内部**交换数据。
      而一个 block 有 8 个 warp，要把 8 个 warp 的结果合成 1 个，
      **必须有一样东西 8 个 warp 都能看到** —— 那就是共享内存。

      所以 V3 并没有"消灭共享内存"，只是把它从"装 256 个值"
      缩减成"装 8 个值"（每个 warp 一个）。

          256 个值 →  8 个值      共享内存流量降了 32 倍
          3 次栅栏 →  1 次栅栏

  (b) 为什么是 `__shared__ float warpSums[32]`（静态），而不是 V2 的
      `extern __shared__ float sdata[]`（动态）？

      因为大小是**编译期就能确定**的：一个 block 最多 1024 线程 = 32 个 warp，
      所以 32 个格子永远够用，不需要按 blockSize 动态分配。

      这带来三个好处：
        · 启动时**不用传第三个参数** → `<<<grid, blockSize>>>`，
          顺手消灭了"忘记传 shmem"这个经典坑
        · 每个 block 的共享内存占用从 1024 字节降到 128 字节
        · 不需要在 main 里算 shmem

  (c) ⚠ 这里的 `__syncthreads()` 为什么**必须**有？

      因为 warpSums[wid] 是由 **8 个不同的 warp** 写的，
      而下一步由 **warp 0** 去读。warp 之间是独立调度的，
      没有任何顺序保证 —— 不加栅栏，warp 0 可能读到别人还没写的格子。

      ⚠ 注意这和 V2 里的情况不同：
        · V2 的主循环里，栅栏是"防同一批线程内部乱序"
        · V3 这里的栅栏是"防 8 个 warp 之间乱序"
        两者都需要，但 V3 只需要这一道。

  (d) 为什么 `if (lane == 0)` 就能保证每格只写一次？

      因为每个 warp 恰好有 1 个 lane 0，8 个 warp 就有 8 个 lane 0，
      分别写 warpSums[0] 到 warpSums[7] —— 互不重叠。

------------------------------------------------------------------------
3.5 ④ 由 warp 0 归约这 8 个值
------------------------------------------------------------------------
      if (wid == 0)
      {
          int nWarps = blockDim.x >> 5;                  // 256 / 32 = 8
          sum = (lane < nWarps) ? warpSums[lane] : 0.0f;

          for (int offset = 16; offset > 0; offset >>= 1)
              sum += __shfl_down_sync(0xffffffffu, sum, offset);

          if (lane == 0) out[blockIdx.x] = sum;
      }

  逐行：

  (a) `if (wid == 0)` —— 只让 0 号 warp 干这活。其他 7 个 warp 直接结束。

      ⚠ 这个条件对 warp 0 来说是**全真**的（32 个 lane 的 wid 都是 0），
        所以 warp 0 内部没有发散 → 后面的 shuffle 用 mask=0xffffffff 是安全的。

  (b) `int nWarps = blockDim.x >> 5;` —— 本 block 有几个 warp。
      blockSize=256 → 8 个。

  (c) `sum = (lane < nWarps) ? warpSums[lane] : 0.0f;`

      **★ 这是 V3 里唯一需要「单位元」的地方，也是全文件最容易写错的一行。**

      为什么需要这个 `: 0.0f`？

        warpSums 只有 8 个有效值（下标 0~7），但 shuffle 要跑满 32 个 lane。
        lane 0 最终的累加结果会覆盖 **全部 32 个 lane**（见第 2.3 节），
        所以 lane 8~31 手里**必须拿着一个"无害"的数** —— 也就是加法的单位元 0。

        如果把 `: 0.0f` 漏掉（直接写 `sum = warpSums[lane];`）：
          · lane 8~31 会读到 warpSums[8..31] —— 那块内存**从没被写过**，
            里面是上一个 kernel 留下的垃圾
          · 这些垃圾会顺着 shuffle 汇进 lane 0 → **结果直接错**

        ⚠ 而且这个 bug 是**间歇性**的：垃圾值恰好是 0 时就 PASS，
          不是 0 时就 FAIL。所以一旦看到"偶尔 PASS 偶尔 FAIL"，先查这里。

        ⚠ 求 max 时这一行要改成 `: -FLT_MAX`（见 reducemax.cu）；
          写成 0.0f 就会把负数全抹掉。

  (d) 第二个 shuffle 循环 —— 和 ② 完全一样，只是这次归约的是 8 个 warp 的部分和。

  (e) `if (lane == 0) out[blockIdx.x] = sum;` —— 每个 block 交出一个部分和。

      V3 里**没有 `volatile`**，因为这里不需要：
      sum 是寄存器变量，shuffle 是显式的值传递，不存在"编译器缓存了共享内存"的问题。

------------------------------------------------------------------------
3.6 blockSize 的约束变了
------------------------------------------------------------------------

      V2：必须 >= 64（因为展开最后 warp 时硬编码了 v[tid + 32]）
      V3：必须 >= 32 且是 32 的倍数

  为什么 V3 放宽了？因为 V3 没有任何硬编码的下标。
  为什么还是必须 >= 32 / 32 的倍数？

    · 若 blockSize < 32：一个 warp 只有一部分 lane 存在，
      但 __shfl_down_sync 的 mask 写的是 0xffffffff（要 32 个 lane 都在）
      → 未定义行为
    · 若 blockSize 不是 32 的倍数：最后一个 warp 的 lane 数不足 32，
      同样是 mask 不匹配

  ⚠ 和 V2 一样，本文件**没有**对 blockSize 做校验（属于已知遗留问题）。
    blockSize = 3 或 100 会静默出错/崩溃。


========================================================================
第 4 步　main 里变了的几处
========================================================================

------------------------------------------------------------------------
4.1 少了 shmem 这个变量
------------------------------------------------------------------------
      V2:  size_t shmem = (size_t)blockSize * sizeof(float);
           reduceSum<<<grid, blockSize, shmem>>>(...);

      V3:  // 没有 shmem
           reduceSumWarp<<<grid, blockSize>>>(...);

  因为 V3 的共享内存是静态的（编译期定死 128 字节），不需要在启动时传大小。

  ⚠ 别忘了：**启动参数少一个，不是"可以省"，而是"本来就不需要"。**
    这和 V2 里"忘了传 shmem 就崩"是完全不同的情况。

------------------------------------------------------------------------
4.2 其他都没变
------------------------------------------------------------------------
  参数解析、grid 封顶、数据填充、参考值、warm-up、计时、取回、验证、释放
  —— 全部与 V2 逐字相同。

  默认 elemsPerThread 有一点区别：
      V2 默认 8，V3 默认 4
  原因是实测下来 elems=4 是甜点（见第 5 步）。


========================================================================
第 5 步　V2 vs V3 对照表
========================================================================

------------------------------------------------------------------------
5.1 设计与 ncu 指标（elems=1，grid 都是 262144，单变量对比）
------------------------------------------------------------------------

      指标                              V2（共享内存）   V3（shuffle）    变化
      ------------------------------    --------------   -------------   --------
      DRAM Throughput                        52.01%         58.34%      ↑ 12.2%
      Memory Throughput                     133.18 GB/s    149.78 GB/s   ↑ 12.5%
      Elapsed Cycles                      3,129,540      2,789,148      ↓ 10.9%
      Mem Pipes Busy（L1TEX/MIO）            64.23%         55.70%      ↓ 13.3%
      Compute (SM) Throughput                64.23%         55.70%      ↓ 13.3%
      **Executed Instructions**         104,071,168     78,643,200      ↓ 24.4%  ⭐
      **Branch Instructions**            17,039,360      8,912,896      ↓ 47.7%  ⭐
      **Dynamic Shared Memory / block**      1,024 B            0 B      −100%
      Registers Per Thread                       16             18      +2
      Achieved Occupancy                      72.46%         72.12%      ≈
      Grid Size / Waves Per SM      262144/1820.44  262144/1820.44       同

------------------------------------------------------------------------
5.2 端到端实测（正反交替交叉测量，8 轮）
------------------------------------------------------------------------

      配置                V2                V3              差异
      ----------------    --------------    -------------   --------
      elems = 1           152.9 GB/s        177.2 GB/s      **+15.9%**
      elems = 4           248.0 GB/s        247.8 GB/s      ≈ 0

  **★ 这两行是本文件最重要的实验结论：**

    · elems=1 时，块内归约的开销在总时间里占比最大 → V3 的收益看得见（+15.9%）
    · elems=4 时，瓶颈已经转移到 DRAM（96% 峰值）→ V3 的优势**完全消失**

  ⚠ 得到一个通用的判断准则：

      **优化的收益取决于瓶颈在哪里。**
      先确定瓶颈，再决定优化方向 —— 反过来的话，
      在 elems=4 上测 V3，就会误以为"shuffle 没用"。


========================================================================
第 6 步　⚠ 一个必须纠正的说法：shuffle 并不是「免费」的
========================================================================

  很多人（包括很多教程）会说"shuffle 比共享内存快，因为它不碰内存"。
  这句话**只对了一半**。

  看第 5.1 节的 Mem Pipes Busy：**64.23% → 55.70%，只降了 13%，不是数量级。**

  为什么？因为 **warp shuffle 执行在 MIO（Memory Input/Output）流水线上 ——
  和共享内存是同一个功能单元。**它并不是绕开访存，只是把
  「读 + 写 + 读 + 写」压缩成了「一次交换」。

  把账算清楚（blockSize = 256，每个 block）：

      V2 的块内归约：
          8 次共享内存 store（装载）
        + 约 36 次 wavefront（8 轮树形，每轮 read+read+write）
        = 约 44 个 L1TEX/MIO wavefront

      V3 的块内归约：
          8 warps × 5 次 shuffle        = 40 次 MIO 操作
        + 8 次共享内存 store + 1 次 load =  9 次
        ≈ 49 次 MIO 操作

  **数量级相当！**

  那 V3 为什么还是快了 15.9%？真正的收益来自这三处：

      ① Executed Instructions  ↓ 24.4%
         （sdata[tid] += sdata[tid+s] 要 2 读 + 1 写 + 1 加；
           shuffle 只要 1 交换 + 1 加）
      ② Branch Instructions    ↓ 47.7%
         （V2 的循环带条件分支 + 栅栏；V3 是 5 条无分支语句）
      ③ 栅栏 3 次 → 1 次

  ⚠ 一句话记住：
      **shuffle 的收益来自「操作次数变少」，而不是「换了一条更快的路」。**


========================================================================
第 7 步　自测题
========================================================================

  1. `__shfl_down_sync(0xffffffffu, sum, 8)` 返回的是什么？
     如果我在 lane 30，它返回谁的值？

  2. 为什么 warp 内归约只要 5 轮（offset = 16,8,4,2,1），
     而 V2 的共享内存版要 8 轮？

  3. V3 已经用 shuffle 做完 warp 内归约了，为什么第 ③ 步还要用共享内存？
     能不能把 8 个 warp 的结果也 shuffle 掉？

  4. `sum = (lane < nWarps) ? warpSums[lane] : 0.0f;` 里的 `: 0.0f`
     为什么不能省？如果省了，什么情况下会 PASS、什么情况下会 FAIL？

  5. V3 为什么不需要 `volatile`，而 V2 需要？

  6. 把 elemsPerThread 从 1 改成 4，V3 相对 V2 的优势会变大还是消失？为什么？

  ── 参考答案 ──
  1. 返回「lane 号比我大 8」的那个 lane 的 sum 值。
     在 lane 30：30 + 8 = 38 >= 32，越界 → 返回**自己**的值（等于自我累加，
     是个垃圾值，但因为最终只读 lane 0，无影响）。
  2. warp 内部只有 32 个 lane，log2(32) = 5 轮就够了。
     V2 要在 256 个值上做归约，log2(256) = 8 轮。
  3. 不能。__shfl_down_sync **只能在一个 warp 内部**交换数据，跨 warp 做不到。
     所以必须有一个"8 个 warp 都能看到"的地方 —— 那就是共享内存（哪怕只装 8 个值）。
  4. 因为 lane 0 最终会累加全部 32 个 lane，而 warpSums 只有 8 个有效格子，
     lane 8~31 必须拿单位元 0 填充。省掉的话它们会读到**未初始化**的共享内存：
     恰好是 0 就 PASS，不是 0 就 FAIL（间歇性 bug）。
  5. V2 在"展开最后 warp"那段里，同一个 warp 的多个 lane 直接读写共享内存
     且没有栅栏，编译器可能把它们缓存进寄存器 → 读到过期值，所以要 volatile。
     V3 这一段改用 shuffle（显式传值），编译器完全看得懂，不存在缓存问题。
  6. **消失**。elems=4 时瓶颈已经转移到 DRAM（96% 峰值），
     块内归约不再是关键路径 —— 实测两者打平（248.0 vs 247.8 GB/s）。

########################################################################*/
