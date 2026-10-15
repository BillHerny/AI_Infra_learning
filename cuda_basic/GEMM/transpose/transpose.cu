// ============================================================================
// transpose.cu —— 0 号练习：共享内存分块矩阵转置
//
//   目的：在进入 GEMM 之前，先单独练两样 GEMM 必备的手艺：
//         ① 2D 线程块 + 2D tile 坐标（第一次用 threadIdx.y / blockIdx.y）
//         ② 共享内存 bank conflict 与 padding（"+1" 的魔法）
//
//   三个版本跑同一份数据、互相对拍、横向比时间：
//         V1 naive     读合并 / 写不合并（直接搬，不过共享内存）
//         V2 tiled     借道共享内存（读、写全救回来），但 tile 行宽=32 → bank conflict
//         V3 padded    行宽=33，冲突消除（与 V2 只差一个 "+1"）
//
//   运行：  ./build/bin/transpose [W] [iters]      默认 W=4096, iters=20
//   讲解：  本文件底部的「逐行详解」注释块（代码在上、讲解在下）
// ============================================================================

#include <cstdio>
#include <cstdlib>
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

// ======================= 尺寸约定 =======================
// TILE_DIM = 32：block 是 32x32（= 1024 线程，正好塞满一个 block）
// W 不必是 32 的倍数：kernel 里都有边界判断，grid 用「向上取整」覆盖
#define TILE_DIM 32

// ============ V1 : naive —— 直接搬，不经过共享内存 ============
// 读：warp 内 j 连续 → in[i*W + j] 地址连续 → 合并 ✓
// 写：warp 内 j 连续 → out[j*W + i] 地址步长 = W → 不合并 ✗
// （详解见底部「第 2 步」）
__global__ void transposeNaive(const float* __restrict__ in,
                               float* __restrict__ out,
                               int W)
{
    int i = blockIdx.y * blockDim.y + threadIdx.y;   // in 的行号（= out 的列号）
    int j = blockIdx.x * blockDim.x + threadIdx.x;   // in 的列号（= out 的行号）
    if (i < W && j < W)
        out[j * W + i] = in[i * W + j];
}

// ============ V2 : 共享内存 tile（行宽 = 32，故意保留 bank conflict） ============
// 两个阶段：读 in 的 tile 进共享内存（读合并 ✓）→ 转置着写出去（写合并 ✓）
// （详解见底部「第 3、4 步」）
__global__ void transposeTiled(const float* __restrict__ in,
                               float* __restrict__ out,
                               int W)
{
    __shared__ float tile[TILE_DIM][TILE_DIM];       // 行宽 = 32 个字

    int tx = threadIdx.x;
    int ty = threadIdx.y;

    // ---- 阶段 1：把 in 的一个 tile 读进来 ----
    int x = blockIdx.x * TILE_DIM + tx;              // in 的列号
    int y = blockIdx.y * TILE_DIM + ty;              // in 的行号
    if (x < W && y < W)
        tile[ty][tx] = in[y * W + x];                // tile 与 in 同朝向存放
    __syncthreads();

    // ---- 阶段 2：把 tile 转置着写出去 ----
    // ⚠ 下面三行里 x / y 的含义与上面相反（“换角色”，见第 4 步）
    x = blockIdx.y * TILE_DIM + tx;                  // out 的列号（= in 的行号）
    y = blockIdx.x * TILE_DIM + ty;                  // out 的行号（= in 的列号）
    if (x < W && y < W)
        out[y * W + x] = tile[tx][ty];               // ⚠ 交换下标读共享内存
}

// ============ V3 : 同上 + 1 个 padding（消除 bank conflict） ============
// 与 V2 的唯一区别：共享内存行宽 32 → 33（见第 5 步）
__global__ void transposeTiledPadded(const float* __restrict__ in,
                                     float* __restrict__ out,
                                     int W)
{
    __shared__ float tile[TILE_DIM][TILE_DIM + 1];   // 行宽 = 33 个字  ← 差异只有这一处

    int tx = threadIdx.x;
    int ty = threadIdx.y;

    int x = blockIdx.x * TILE_DIM + tx;
    int y = blockIdx.y * TILE_DIM + ty;
    if (x < W && y < W)
        tile[ty][tx] = in[y * W + x];
    __syncthreads();

    x = blockIdx.y * TILE_DIM + tx;
    y = blockIdx.x * TILE_DIM + ty;
    if (x < W && y < W)
        out[y * W + x] = tile[tx][ty];
}

// ============ 统一的「跑一遍 + 计时」小工具 ============
// 三个 kernel 的签名完全相同（in, out, W），所以可以用函数指针统一调用
using KernelFn = void (*)(const float*, float*, int);

// ----------------------------- 详细注释：KernelFn -----------------------------
// 【这是什么】
//   using 名字 = 类型; 是 C++ 给类型起外号的语法（等价于老式的 typedef）。
//   右边 void (*)(const float*, float*, int) 要从中间往外读：
//       void            → 这种函数"不返回东西"
//       (*)             → 这是一把"指针"
//       ( ... )         → 它指向的函数收三个参数：输入表、输出表、边长
//   连起来：KernelFn = "指向这样一种函数的指针"的类型名。
//
// 【为什么需要它】
//   V1/V2/V3 三个 kernel 的"插头"长得一模一样 → 在这里立一个 USB 插座标准。
//   于是计时器（timeBest）和成绩单（report）只用写一份：
//   谁来测就把谁插上去 —— 不用给每个 kernel 各抄一遍计时代码。
//
// ⚠ 使用时的坑：传它要写 transposeNaive（不带括号）= "把函数本身"当值；
//   写成 transposeNaive(...) 就变成"调用它"了，类型对不上、编译报错。
// ------------------------------------------------------------------------------

float timeBest(KernelFn kernel, const float* dIn, float* dOut,
               dim3 grid, dim3 block, int W, int iters)
{
    // ---------------------------- 详细注释：参数逐个说 ----------------------------
    //   KernelFn kernel     要测的那个 kernel（"USB 设备"插进来）
    //   const float* dIn    显存里的输入表（main 里已分配好，这里只借用）
    //   float* dOut         显存里的输出表（同上）
    //   dim3 grid / block   启动尺寸（派多少班组 × 每班组多少线程）
    //   int W               方阵边长
    //   int iters           正式计时的次数（最后取最快的一次）
    //
    // 【返回值 float】最快一次的耗时（毫秒），交给调用者去算加速比。
    //
    // 【职责边界】本函数只管"跑 + 掐表"：不分配内存、不造数据、不对拍 ——
    //   那些脏活都在 main 里完成，于是三个 kernel 能共用同一份计时逻辑。
    // ----------------------------------------------------------------------------

    // warm-up：首次启动的模块加载 / 指令缓存等一次性开销不算进结果
    kernel<<<grid, block>>>(dIn, dOut, W);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    cudaEvent_t start, stop;
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));

    float best = 1e30f;                  // 取「最快的一次」而不是平均：
    for (int it = 0; it < iters; ++it)   // 调度抖动只会让某次变慢，不会变快
    {
        CUDA_CHECK(cudaEventRecord(start));
        kernel<<<grid, block>>>(dIn, dOut, W);
        CUDA_CHECK(cudaEventRecord(stop));
        CUDA_CHECK(cudaEventSynchronize(stop));

        float ms = 0.0f;
        CUDA_CHECK(cudaEventElapsedTime(&ms, start, stop));
        if (ms < best) best = ms;
    }

    CUDA_CHECK(cudaEventDestroy(start));
    CUDA_CHECK(cudaEventDestroy(stop));
    return best;
}

int main(int argc, char** argv)
{
    int W     = (argc > 1) ? atoi(argv[1]) : 4096;   // 方阵边长
    int iters = (argc > 2) ? atoi(argv[2]) : 20;     // 每个 kernel 计时次数

    size_t elems = (size_t)W * (size_t)W;
    size_t bytes = elems * sizeof(float);

    // 规模建议：两阵总量 >> 24 MiB(L2) 时测的是 DRAM（W = 4096 起步）；
    //           故意调小到两阵能进 L2（如 W = 1024）会看到「虚高」的带宽
    printf("W = %d  方阵 %dx%d（单阵 %.1f MiB，两阵共 %.1f MiB）\n",
           W, W, W, bytes / 1024.0 / 1024.0, 2.0 * bytes / 1024.0 / 1024.0);

    // ---- host 端数据 ----
    // 每个元素填一个唯一的值（就是它的下标）：
    //   转置一旦把哪一格和哪一格搞混，对拍时必然暴露
    //   W=4096 时最大下标 16.7M < 2^24 → float 能精确表示，不引入舍入
    float* hIn  = (float*)malloc(bytes);
    float* hRef = (float*)malloc(bytes);   // CPU 参考结果
    float* hOut = (float*)malloc(bytes);   // 回读区
    for (size_t i = 0; i < elems; ++i) hIn[i] = (float)i;

    for (int i = 0; i < W; ++i)
        for (int j = 0; j < W; ++j)
            hRef[(size_t)j * W + i] = hIn[(size_t)i * W + j];   // out[j][i] = in[i][j]

    // ---- device 端数据 ----
    float *dIn = nullptr, *dOut = nullptr;
    CUDA_CHECK(cudaMalloc(&dIn,  bytes));
    CUDA_CHECK(cudaMalloc(&dOut, bytes));
    CUDA_CHECK(cudaMemcpy(dIn, hIn, bytes, cudaMemcpyHostToDevice));

    dim3 block(TILE_DIM, TILE_DIM);                                   // 32 x 32
    dim3 grid((W + TILE_DIM - 1) / TILE_DIM,
              (W + TILE_DIM - 1) / TILE_DIM);                         // 向上取整

    // ------------------------------ 详细注释 ------------------------------
    // 【dim3 是什么】
    //   CUDA 自带的小结构体，里面装着 x / y / z 三个数（可以只填 1~3 个）。
    //   填两个就代表"二维"：第一个数是 x 方向，第二个数是 y 方向。
    //
    // 【block —— 一个班组有多少线程】
    //   block(TILE_DIM, TILE_DIM) = (32, 32) → 每班组 32×32 = 1024 个线程。
    //   1024 正好是硬件允许的单 block 线程数上限，一个班组刚好塞满。
    //   块内排布：线性线程号 = threadIdx.y * 32 + threadIdx.x
    //   → 一个 warp（32 人小队）= 同一行：ty 固定、tx 从 0 到 31。
    //
    // 【grid —— 一共派多少个班组】
    //   要盖住 W×W 的大表，而一个班组只干 32×32 的范围，
    //   所以每个方向需要 W/32 个班组（横向几块 × 纵向几块），即：
    //
    //       (W + TILE_DIM - 1) / TILE_DIM    ←「整数除法向上取整」的经典写法
    //
    //   为什么加 (TILE_DIM - 1)？因为 C++ 的整数除法是向下取整：
    //       W=4097 时 4097/32 = 128（余 1 被丢掉）→ 盖不满！
    //       先补到 4097+31 = 4128，再除：4128/32 = 129 → 够盖了。
    //   这个技巧对整除的情况也无害（多的 31 会被除法截掉）：
    //       W=4096：(4096+31)/32 = 4127/32 = 128 ✓
    //
    //   代入本程序跑过的几个 W：
    //       W = 4096 → grid = (128, 128)   每块刚好 32 格，无浪费
    //       W = 2048 → grid = ( 64,  64)   同上
    //       W = 1024 → grid = ( 32,  32)   同上
    //       W = 4097 → grid = (129, 129)   最后一圈班组只有 1 列 / 1 行有效
    //
    // ⚠ 为什么不能"向下取整"（少派）？少派 = 表盖不满：
    //   例如 W=4097 若只派 128 块，最右一列、最下一行的格子没人干，
    //   结果直接是错的（对拍必然 FAIL，而且错得很隐蔽）。
    //   向上取整只是"多派"，多出来的边缘线程靠 kernel 里的
    //   if (x < W && y < W) 拦住 —— 够不着格子的线程不许动。
    //
    // 【这两个变量最后去哪了】
    //   被原样传进 kernel 的启动语法 <<<grid, block>>>（在 timeBest 里），
    //   三个 kernel 共用同一套尺寸 → 对比才公平。
    // --------------------------------------------------------------------

    printf("grid = (%u, %u), block = (%u, %u)\n",
           grid.x, grid.y, block.x, block.y);

    const double trafficGB = 2.0 * (double)bytes / 1e9;   // 读一遍 + 写一遍

    // 跑一个 kernel：计时 → 对拍 → 打印一行，返回最快时间
    auto report = [&](const char* name, KernelFn kernel) -> float
    {
        // ---------------------------- 详细注释：这一行在说什么 ----------------------------
        // 【一句话】这一行 = "在 main 里现定义一个小函数，起名叫 report"。
        //   （这种"就地定义的小函数"叫 lambda；它没有名字，只能挂在变量上。）
        //
        // 【句型模板】以后见到的 lambda 都长这样：
        //     auto 名字 = [捕获](参数表) -> 返回类型 { 函数体 };
        //
        // 【逐块拆这一行】
        //   auto                  → lambda 的真实类型是编译器临时生成的（写不出来），
        //                           所以只能用 auto 让编译器自己推断。
        //   report                → 给这个小函数起的名；下面三次调用都靠它。
        //   [&]                   → 捕获列表："外面所有变量我直接借用（按引用）"。
        //                           于是函数体里能直接用 hOut / hRef / dIn / dOut /
        //                           grid / block / W / iters / trafficGB / elems……
        //                           换成普通函数就得把这些一个个当参数传，太啰嗦。
        //   (const char* name,    → 参数表，和普通函数一样：
        //    KernelFn kernel)        name  ：打印成绩时显示的版本名（如 "V1 naive"）
        //                            kernel：要测的那个 kernel（KernelFn 函数指针）
        //   -> float              → 返回类型：它会 return 最快耗时（毫秒），
        //                           给 main 用来算加速比（t1 / t2 那几行）。
        //
        // ⚠ 这一行只负责"声明"；真正干的四件事（跑 + 计时 / 拷回 / 对拍 / 打印）
        //   都在下面的大括号里。
        // ---------------------------------------------------------------------------------

        float best = timeBest(kernel, dIn, dOut, grid, block, W, iters);

        CUDA_CHECK(cudaMemcpy(hOut, dOut, bytes, cudaMemcpyDeviceToHost));
        size_t bad = 0, firstBad = 0;
        for (size_t i = 0; i < elems; ++i)
            if (hOut[i] != hRef[i]) { if (bad == 0) firstBad = i; ++bad; }
        // ⚠ 转置只是「搬数据」，没有任何浮点运算 → 可以要求精确相等，不需要容差

        printf("  %-12s best = %8.4f ms   bandwidth = %7.1f GB/s   verify: %s\n",
               name, best, trafficGB / (best / 1e3), bad ? "FAIL" : "PASS");
        if (bad)
            printf("      对不上 %zu 个元素，第一个在 hOut[%zu]：得到 %.0f，期望 %.0f\n",
                   bad, firstBad, hOut[firstBad], hRef[firstBad]);
        return best;
    };

    printf("--- 三种 kernel 各跑 %d 次、取最快 ---\n", iters);
    float t1 = report("V1 naive",  transposeNaive);
    float t2 = report("V2 tiled",  transposeTiled);
    float t3 = report("V3 padded", transposeTiledPadded);

    printf("\n加速比（相对 V1）：V2 = %.2fx，V3 = %.2fx\n", t1 / t2, t1 / t3);
    printf("V3 相对 V2（padding 的收益）= %.2fx\n", t2 / t3);

    free(hIn); free(hRef); free(hOut);
    CUDA_CHECK(cudaFree(dIn));
    CUDA_CHECK(cudaFree(dOut));
    return 0;
}

/*########################################################################
  transpose.cu 逐行详解
  —— 代码在上，讲解全部在下（与 reducesum*.cu 同一风格）

  这是什么：GEMM 之前的「0 号」过渡练习，只做一件事 —— WxW 方阵转置，
  而且要求「读、写两头都合并访问」。为此你会第一次用到 GEMM 的两大零件：
    ① 2D 线程块 / 2D tile 坐标
    ② 共享内存的 bank conflict 与 padding（"+1" 的魔法）

  运行：
    cmake --build build --target transpose && ./build/bin/transpose 4096
    第二个参数是计时次数（默认 20）

  目录
    第 0 步  为什么先练转置（和 GEMM 的关系）
    第 1 步  先想清楚：转置为什么「天生」不合并
    第 2 步  V1 naive：读合并、写不合并
    第 3 步  V2 tiled：借道共享内存，把「写」也救回来
    第 4 步  ⚠ 换角色（role swap）：本文件最容易读错的三行
    第 5 步  V3 padded：bank conflict 与「+1」的魔法
    第 6 步  main：计时 / 对拍 / 命令行
    第 7 步  实测数据与解读
    第 8 步  可改进点与下一步（GEMM）
    第 9 步  自测题 + 参考答案


========================================================================
第 0 步　为什么先练转置（和 GEMM 的关系）
========================================================================

  GEMM 的一个 block 干的活可以概括成两件事：
    (a) 把 A 的一块、B 的一块从显存搬进共享内存（tile 化）
    (b) 在共享内存里反复取用，做乘加

  转置练习把 (b) 抽掉，只留 (a)：纯搬运、没有计算。
  但它保留了 GEMM tiling 的全部套路：
    · 一个 block 负责二维的一块；
    · 进出共享内存时，读 / 写方向是「横」还是「竖」；
    · 共享内存会不会 32 路 bank conflict，怎么用 padding 治。

  另外转置本身也是真实算子：attention、卷积实现里经常出现
  x.transpose(...) 这种搬运，写得好坏直接影响性能。

  所以：先花 20 分钟把这张「搬运地图」画清楚，GEMM 会讲得快很多。


========================================================================
第 1 步　先想清楚：转置为什么「天生」不合并
========================================================================

  两个矩阵都按行优先（row-major）存放：
    in  第 i 行、第 j 列  →  地址 i*W + j   （同一行内地址连续）
    out 第 j 行、第 i 列  →  地址 j*W + i

  转置做的事：out[j][i] = in[i][j]。

  一个 warp 的 32 个线程同时发访存请求。硬件最喜欢的：
    32 个线程的地址连成一段（128 B，正好 1 条 cache line）→ 1 次事务
  最怕的：
    地址东一个西一个 → 拆成 32 次事务

  现在看转置：
    读 in ：线程沿 j 方向排 → i*W + j 里 j 连续 → 合并 ✓
    写 out：线程还是沿 j 方向排 → j*W + i 里 j 的步长是 W → 不合并 ✗

  ⚠ 只要「直接读、直接写」，就不可能两头都合并：
    in 连续的方向是「行」，out 连续的方向是「列」，两者天然垂直。
    不管你怎么排线程，总有一边是跨步的。

  结论：想把不合并的那一边救回来，必须**借道共享内存** ——
  先在「合并的方向」把数据搬进片上，再从片上按需要转向搬出去。
  这，就是 tiling 的动机。


========================================================================
第 2 步　V1 naive：读合并、写不合并
========================================================================

    int i = blockIdx.y * blockDim.y + threadIdx.y;   // in 的行号
    int j = blockIdx.x * blockDim.x + threadIdx.x;   // in 的列号
    if (i < W && j < W)
        out[j * W + i] = in[i * W + j];

  先认 2D 索引（本工程第一次用 .y）：

    块内：32x32 的 block = 1024 线程 = 32 个 warp

        线性线程号 = threadIdx.y * 32 + threadIdx.x
        → 每个 warp = 同一行（ty 固定、tx 从 0 到 31）

              tx →   0    1    2   ...   31
           ty
            ↓
            0      [========= warp 0 =========]
            1      [========= warp 1 =========]
           ...
           31      [========= warp 31 ========]

  所以一个 warp 内：tx 变化、ty 固定。

    读 in[i*W + j]：warp 内 j 连续 → 地址连续 → 合并 ✓
      一个 warp 读 128 B 连续数据，1 次事务搞定。

    写 out[j*W + i]：warp 内 j 连续 → 地址 = j*W + i，
      相邻线程相差 W*4 字节（4096 * 4 = 16 KiB）
      → 32 个线程落在 32 条完全不同的 cache line → 32 次事务 ✗

  后果（以 W = 4096 估算）：
    读  16.7M * 4 B                      ≈ 64 MiB
    写  16.7M * 4 B * 放大倍数            ← 每个 4 B 只用了 1/8 条 32 B sector
    实际 DRAM 流量约 600 MB，是理想值（128 MiB）的 4~5 倍。

  ⚠ 记住这个手感：naive 版的慢不是「算得慢」，是「搬得浪费」。


========================================================================
第 3 步　V2 tiled：借道共享内存，把「写」也救回来
========================================================================

  骨架（两个阶段 + 中间一道栅栏）：

    ┌── 阶段 1 ────────────────────────────────────────────┐
    │ warp 沿「连续方向」读 in 的一个 32x32 tile            │
    │ → 写进共享内存                                      │  读合并 ✓
    └──────────────────────────────────────────────────────┘
                     __syncthreads()
    ┌── 阶段 2 ────────────────────────────────────────────┐
    │ 从共享内存里「转置着」取出                            │
    │ → 按「连续方向」写 out 的 tile                       │  写合并 ✓
    └──────────────────────────────────────────────────────┘

  阶段 1 的三行：
      int x = blockIdx.x * TILE_DIM + tx;      // in 的列号
      int y = blockIdx.y * TILE_DIM + ty;      // in 的行号
      if (x < W && y < W) tile[ty][tx] = in[y * W + x];

    · 块 (bx, by) 负责 in 的第 [by*32, by*32+32) 行、
      第 [bx*32, bx*32+32) 列的那个 tile；
    · tile 在共享内存里保持和 in 一样的「行优先」朝向：
        tile 第 r 行第 c 列 = in 的 (by*32 + r, bx*32 + c)

  阶段 2 的三行：
      x = blockIdx.y * TILE_DIM + tx;          // ⚠ 含义与阶段 1 相反
      y = blockIdx.x * TILE_DIM + ty;          // ⚠
      if (x < W && y < W) out[y * W + x] = tile[tx][ty];

    · 写 out 的地址 = y*W + x，warp 内 tx 连续 → x 连续 → 合并 ✓
    · 读共享内存用 tile[tx][ty]：warp 内 ty 固定、tx 连续
      → 地址步长 = 行宽 = 32 个字 → 全落在同一个 bank ✗（第 5 步细说）

  栅栏只需要 1 道：阶段 2 会读到「别人写的格子」，
  必须在全员写完后再开始（对比 reduction V1 树形归约里的多道栅栏）。


========================================================================
第 4 步　⚠ 换角色（role swap）：本文件最容易读错的三行
========================================================================

  先立一条搬运 kernel 的总原则：

    ★ 一个 warp 的 32 个线程，永远让 tx 去扫「内存里连续的那一维」。

  阶段 1（搬 in 的 tile）：
    in 的连续维 = 列 → tx 管 in 的列 ✓（代码里 x = ... + tx，已经对了）

  阶段 2（写 out 的 tile）：
    out 的连续维 = 列 = in 的行 = tile 的【第 1 个】下标
    out 的行     = in 的列 = tile 的【第 2 个】下标
    想让 tx 去管 out 的列 → 就得让 tx 去当 tile 的【第 1 个】下标
    → 所以读的是 tile[tx][ty]（下标交换！）

  同时，block 偏移也整体对调（bx ↔ by），边界判断里 x / y 的含义
  也整体对调。方阵 W 下两个 if 长得一模一样、看不出玄机 ——
  但这正是「换角色」留下的痕迹（非方阵时两边条件就不一样了，见第 8 步）。

  拿 4x4 的小例子过一遍（真实代码是 32x32，原理相同）：

    in（行优先）           out = in 的转置
      0  1  2  3             0  4  8 12
      4  5  6  7             1  5  9 13
      8  9 10 11             2  6 10 14
     12 13 14 15             3  7 11 15

    阶段 1 之后，共享内存里的 tile 就是左边那张表（与 in 同朝向）。
    阶段 2 里，线程 (ty, tx) 负责写 out 的【第 ty 行、第 tx 列】，
    但取的值来自 tile 的「转置位置」：tile[tx][ty]。

      例：线程 (ty=0, tx=1) 写 out[0][1] = 4，取的是 tile[1][0] = 4 ✓

    「让 tx 扫 out 的列」和「从 tile[tx][ty] 取值」是同一件事的两面：
    想两头合并，就必须在片上把行列转一次 —— 代价，就是第 5 步的 bank conflict。


========================================================================
第 5 步　V3 padded：bank conflict 与「+1」的魔法
========================================================================

  共享内存的物理组织：
    · 每 4 字节一个「字」；共 32 个 bank，字地址 mod 32 = bank 号
    · 一个 warp 的 32 个线程同时访问共享内存：
        32 个地址分散在 32 个不同 bank → 1 个周期完成
        多个地址挤进同一个 bank       → 串行（几路冲突 = 几个周期）

  阶段 2 读 tile[tx][ty]（warp 内 ty 固定、tx = 0..31）：

    行宽 32 时：
      线程 tx:   0     1     2    ...   31
      字地址:    0    32    64    ...   992
      bank:      0     0     0    ...     0      ← 32 路冲突！

    行宽 33（tile[32][33]）时：
      字地址:    0    33    66    ...  1023
      bank:      0     1     2    ...    31      ← 每个线程一个 bank ✓

  为什么 +1 就够了？
    地址 = tx*33 + ty。对它 mod 32：因为 33 = 32 + 1，
      bank = (tx*33 + ty) % 32 = (tx + ty) % 32
    tx 从 0 到 31 时，(tx+ty) % 32 恰好把 0..31 每个值走一遍
    → 无冲突。（33 与 32 互素，步长 33 的序列“转一圈”不重复。）

  代价：每行浪费 1 个字 → 整个 tile 多 32 个字 = 128 B
        （共享内存共 32*33*4 = 4224 B）。一点片上空间换 32 倍访问速度，极划算。

  ⚠ 注意冲突只发生在「阶段 2 的共享内存读」：
    阶段 1 写共享内存是 tile[ty][tx]，warp 内地址连续，本来就没冲突。
    （所以 V2 慢，不是慢在「用了共享内存」，而是慢在「读共享内存的方式」。）

  想亲眼看到冲突计数（选做）：
    ncu --kernel-name regex:transpose --launch-skip 20 --launch-count 3 \
        --metrics l1tex__data_bank_conflicts_pipe_lsu_mem_shared_op_ld.sum \
        ./build/bin/transpose 4096


========================================================================
第 6 步　main：计时 / 对拍 / 命令行
========================================================================

  · 命令行：./build/bin/transpose [W] [iters]，默认 4096 与 20。
  · 数据：hIn[i] = i（值即下标，唯一）→ 任何一格搬错都会对拍失败；
    W=4096 时最大下标 16.7M < 2^24，float 能精确表示。
  · 对拍用【精确相等】：转置没有任何浮点运算，不搬家就是错，
    不需要像 sum 那样放容差（这一点和求 max 一样，比求和“干净”）。
  · grid 用向上取整：(W+31)/32 —— 所以 W 不必是 32 的倍数。
  · timeBest：warm-up 1 次（模块加载、缓存冷启动不参与计时）+
    正式 20 次取【最快】。取最快而不是平均：调度抖动只会让某次变慢。
  · 三个 kernel 签名相同 → 用函数指针 KernelFn 统一调用，少抄三遍。


========================================================================
第 7 步　实测数据与解读（RTX 4060 Laptop，W = 4096，iters = 20）
========================================================================

  两遍独立运行（同配置两遍差 ~10% —— 和 reduction 那课一样，
  单次数字别当真理；看三者的相对关系，要严谨就交叉测量）：

                 第 1 遍                       第 2 遍
    V1 naive     1.7961 ms /  74.7 GB/s       1.6916 ms /  79.3 GB/s     PASS
    V2 tiled     0.9779 ms / 137.2 GB/s       0.9042 ms / 148.4 GB/s     PASS
    V3 padded    0.7382 ms / 181.8 GB/s       0.6554 ms / 204.8 GB/s     PASS

    加速比（相对 V1）：V2 ≈ 1.85x，V3 ≈ 2.5x；V3 相对 V2 ≈ 1.35x

  解读：
    ① V1 → V2（~1.85x）：消灭了不合并的写。
       没有想象中 4~5 倍那么夸张 —— 因为 L2 会把相邻 warp 写出的
       32 B sector 部分攒起来再回写 DRAM，实际放大比理论小。
       ⚠ 但机理不变：naive 的写永远是浪费的主要来源。
    ② V2 → V3（~1.35x）：store 阶段 32 路 bank conflict 被 "+1" 抹掉。
       这 1.35x 就是「读共享内存的方式」值多少钱的直接证据。
    ③ V3 约 205 GB/s（约 80% 峰值），没到 ~250 GB/s 的天花板：
       转置的访存模式（读 128 B / 写 128 B 交替 + 栅栏依赖）本身
       就比纯流式难喂饱 DRAM，属正常现象，不是写法写错了。
    ④ 边界：W = 4097（不是 32 的倍数）全 PASS —— 边界判断经得起非整除。

  彩蛋（自测题 7 的答案）：W = 1024（两阵共 8 MiB，塞得进 L2）时
    V3 跑到 327.7 GB/s —— 超过 256 GB/s 的 DRAM 峰值，因为跑的是 L2。
    而且 V2/V3 差距拉大（V3 快 1.80x）：数据不再被 DRAM 瓶颈掩盖后，
    bank conflict 的代价看得更清楚。


========================================================================
第 8 步　可改进点与下一步（GEMM）
========================================================================

  可改进点：
    ① 换成 32x8 的 block + 每线程负责 4 行：读时用循环凑成 128 B 事务、
       且给每线程更多独立访存（ILP），是经典 CUDA sample 的做法；
    ② 用 float4 向量化：读、写一次搬 4 个 float，事务数再减；
    ③ 非方阵（M x N）：两个阶段的边界条件会互换，动手改一遍最能
       检验自己有没有真懂第 4 步（⚠ 改错时小矩阵可能碰巧 PASS，拿大矩阵测）；
    ④ swizzle（XOR 折叠下标）代替 padding：节省那 128 B 的场景更优雅。

  下一步 GEMM 时回看这里的三个零件：
    · 阶段 1 的两行 = GEMM 里装填 A、B tile 的写法（边界判断完全同款）；
    · __syncthreads 之后进计算循环 = GEMM 里沿 K 维滑动的主循环；
    · tile[tx][ty] 的取用方式 = GEMM 里从 smem 读 A / B 时同样要盯
      bank conflict（GEMM 里常用 padding 或寄存器分块规避）。


========================================================================
第 9 步　自测题 + 参考答案
========================================================================

  1. V1 里为什么「读合并、写不合并」？如果把两个 kernel 里的 i、j 完全
     对调（线程负责另一头），会变成「读不合并、写合并」吗？本质原因？

  2. 32x32 的 block 里，一个 warp 覆盖的是哪 32 个线程？
     为什么 V2 的 load 阶段没有 bank conflict、store 阶段却有 32 路？

  3. 为什么行宽从 32 改成 33 就能消除冲突？33 这个数字特殊在哪？
     （提示：33 mod 32 = 1。）

  4. padding 浪费了多少共享内存？为什么不在阶段 2 改用 tile[ty][tx]
     直接读（不加 padding 也不换下标）？

  5. 阶段 2 里 blockIdx.x / blockIdx.y 为什么要交换？

  6. 为什么对拍可以要求「精确相等」，而 reducesum 里必须放容差？

  7. W 改成 1024（两个矩阵共 8 MiB，能塞进 24 MiB 的 L2）跑一遍，
     带宽数字会变成多少？为什么这时候数字会「虚高」？
     （回忆 reducemax 里 16 MiB 跑出 455 GB/s 的那一课。）

  ── 参考答案 ──
  1. in 的连续维是行（j 方向），out 的连续维是它的列 —— 两者垂直。
     对调后会变成「读不合并、写合并」，但不可能两头都好：
     只要直接读、直接写，总有一边要跨步，因为转置本来就把连续方向换了。
  2. 线性线程号 = ty*32 + tx → 一个 warp = 同一行（ty 固定、tx 0..31）。
     load 阶段写 tile[ty][tx]：地址 = ty*32 + tx，tx 连续 → bank 连续 → 无冲突；
     store 阶段读 tile[tx][ty]：地址 = tx*32 + ty，tx 变化时步长 32 →
     32 个地址全是同一 bank → 32 路冲突。
  3. 33 mod 32 = 1，所以地址 tx*33+ty 的 bank 号 = (tx+ty) mod 32，
     随 tx 走遍 0..31，互不相同。32 与 33 互素是关键（32 不行）。
  4. 每行多 1 个字 → 32 行多 32 字 = 128 B。
     若改用 tile[ty][tx] 来「不换下标」，那么每个线程都直接写自己刚存的
     那格（tile[ty][tx] 就是自己写的），值是对的，但写 out 的地址变成
     y*W + x（y 含 ty、x 含 tx）——其实这就是「不换角色」的写法，会导致
     warp 内写 out 的地址跨步（不合并），等于退回 V1 的写问题。
     合并与冲突，只能把代价从全局挪到共享内存，再用 +1 抹掉。
  5. out 的行号来自 in 的列号、out 的列号来自 in 的行号；
     块一级自然也要换成 (bx, by) 对调，让这个 block 写 out 的正确 tile。
  6. 转置只搬数据、没有任何算术，浮点值原样搬运 →
     正确性要求「一个 bit 都不差」，可以用 != 精确比较。
     求和有浮点结合律问题（加法顺序不同结果不同），才需要容差。
  7. 会显著高于 DRAM 峰值（跑的是 L2 而不是显存）——
     这类数字只反映 L2 带宽，不能拿来对比「优化是否更快」。


     ./build/bin/transpose 1024
W = 1024  方阵 1024x1024（单阵 4.0 MiB，两阵共 8.0 MiB）
grid = (32, 32), block = (32, 32)
--- 三种 kernel 各跑 20 次、取最快 ---
  V1 naive     best =   0.1096 ms   bandwidth =    76.6 GB/s   verify: PASS
  V2 tiled     best =   0.0470 ms   bandwidth =   178.5 GB/s   verify: PASS
  V3 padded    best =   0.0256 ms   bandwidth =   327.7 GB/s   verify: PASS

加速比（相对 V1）：V2 = 2.33x，V3 = 4.28x
V3 相对 V2（padding 的收益）= 1.84x

     ./build/bin/transpose 2048
W = 2048  方阵 2048x2048（单阵 16.0 MiB，两阵共 32.0 MiB）
grid = (64, 64), block = (32, 32)
--- 三种 kernel 各跑 20 次、取最快 ---
  V1 naive     best =   0.4230 ms   bandwidth =    79.3 GB/s   verify: PASS
  V2 tiled     best =   0.1741 ms   bandwidth =   192.8 GB/s   verify: PASS
  V3 padded    best =   0.0901 ms   bandwidth =   372.4 GB/s   verify: PASS

加速比（相对 V1）：V2 = 2.43x，V3 = 4.69x
V3 相对 V2（padding 的收益）= 1.93x

########################################################################*/
