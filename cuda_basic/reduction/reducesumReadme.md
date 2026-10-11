# reducesum —— CUDA 归约（Reduction）算子优化手记

> 记录时间：2026-10-11
> 对应源码：`reduction/reducesum.cu`（V1）、`reduction/reducesumV2.cu`（V2）、`reduction/reducesumV3.cu`（V3）
> 环境：RTX 4060 Laptop（Ada / CC 8.9）、CUDA 12.x、Nsight Compute 2022.4.1
> 写作形式：**代码构建 → 调试 → 优化思路 → 实验 → 结果笔记**
> 阅读建议：只看第 0 节能拿到结论；想复现实验看第 1、5 节；想看"思路怎么被数据推翻"看第 4、6 节。

---

## 目录

- [0. TL;DR](#0-tldr)
- [1. 环境与基准](#1-环境与基准)
- [2. 代码构建](#2-代码构建)
- [3. 调试](#3-调试)
- [4. 优化思路](#4-优化思路)
- [5. 实验](#5-实验)
- [6. ncu Profile 对比](#6-ncu-profile-对比)
- [7. 结果笔记](#7-结果笔记)
- [8. 后续方向](#8-后续方向)

---

## 0. TL;DR

要解决的问题：把 N = 2²⁶（67,108,864）个 float 求和，输出 1 个数。

| 版本 | 核心做法 | grid | 耗时 | 带宽 | 峰值占比 |
|---|---|---|---|---|---|
| **V1** `reducesum.cu` | 一线程一元素 + 共享内存树形归约（3 次栅栏） | 262,144 | 1.7930 ms | 149.7 GB/s | 58.4% |
| **V2** `reducesumV2.cu` | 每线程先在**寄存器**里累加 4 个元素，再做同样的树形归约 | 65,536 | **1.0803 ms** | **248.5 GB/s** | **97.0%** |
| **V3** `reducesumV3.cu` | 同 V2，但块内归约改用 **`__shfl_down_sync`**（1 次栅栏、0 字节动态共享内存） | 65,536 | 1.0803 ms | 248.5 GB/s | 97.0% |

**V1 → V2 提速 1.66×，摸到 DRAM 天花板**（对照 `vector_add` 的 245.8 GB/s / 95.9%）。
**V3 在 `elems=4` 时与 V2 打平**（瓶颈已转移到 DRAM），但在 `elems=1` 时快 **15.9%**（152.9 → 177.2 GB/s）—— 这个差值就是「块内归约本身的开销」。

四条最有价值的结论：

1. **V1 的瓶颈不是 DRAM，是 SM 内部的访存路径（L1TEX）**。ncu 显示 V1 的 DRAM Throughput 只有 52.01%，而 L1TEX / Compute 却有 64.23%。看着像"计算和存储很均衡"，实际是"DRAM 被饿着，SM 在忙着做无用功"。
2. **根因：共享内存树形归约的开销 ∝ block 数，而不是 ∝ 数据量**。V1 每个 block 只搬 1 KiB 数据，却要做一遍完整的 256 元素树（约 44 次共享内存 wavefront，对比全局访存只有 8 次）。262,144 个 block 把这份固定开销重复了 26 万遍。
3. **所以修法不是"调参"，而是"提高每 block 的载荷"**。一个改动同时改善三件事：每 block 载荷 ×4、grid ÷4、总栅栏数 ÷4。

4. **`__shfl_down_sync` 的收益是「有限且可量化」的**：它把块内归约的指令数降了 24.4%、分支数降了 47.7%，但 `Mem Pipes Busy` 只降了 13% —— 因为 **shuffle 和共享内存共用 MIO 流水线**。收益只在 `elems=1`（块内归约占比最大）时显现。

一条方法学结论：**单次运行的数字不可信**（同一二进制同一参数能给出 217 和 248 两种结果，差 13%），对比多个配置必须**交叉（interleaved）测量**；而且**顺序要正反交替**，否则位置本身会引入系统性偏置。

---

## 1. 环境与基准

### 1.1 硬件（判读所有数字的基准线）

| 项目 | 数值 |
|---|---|
| GPU | NVIDIA GeForce RTX 4060 Laptop（CC 8.9 / Ada） |
| SM 数量 | 24 |
| 每 SM 最大常驻线程 | 1536 |
| 每 SM 最大常驻 warp | 48 |
| 每 SM 最大常驻 block | 24 |
| 每 SM 共享内存 | ~100 KiB |
| L2 缓存 | ~24 MiB |
| 显存 | 8 GB GDDR6 128-bit @ 8.01 GHz |
| **峰值带宽** | **≈ 256.3 GB/s** |
| FP32 峰值 | ≈ 9.5 TFLOP/s |
| PCIe | 4.0 ×8，实测 ~12 GB/s |

### 1.2 测量规模

```
N = 2^26 = 67,108,864 个 float = 256 MiB
```

数据全部填 `1.0f`，所以**精确和就是 N**，且树形归约的中间值全是 2 的幂，在 float 里精确可表示 → 不存在舍入误差干扰判读。

### 1.3 构建与运行

```bash
cd /home/fate-whispering/projects/ai-infra/cuda_basic
cmake -S . -B build
cmake --build build --target reducesum reducesumV2 reducesumV3

./build/bin/reducesum   $((1<<26)) 256        # V1
./build/bin/reducesumV2 $((1<<26)) 256 4      # V2，第 3 个参数 = 每线程元素数
./build/bin/reducesumV3 $((1<<26)) 256 4      # V3，参数同 V2（默认 elems=4）
```

`reduction/CMakeLists.txt`：

```cmake
add_executable(reducesum reducesum.cu)
set_target_properties(reducesum PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON)

add_executable(reducesumV2 reducesumV2.cu)
set_target_properties(reducesumV2 PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON)

add_executable(reducesumV3 reducesumV3.cu)
set_target_properties(reducesumV3 PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON)
```

---

## 2. 代码构建

### 2.1 任务定义与它和 `vector_add` 的本质区别

```
vector_add:   a[i] ─┐
                     ├─ 线程 i ─→ c[i]      各写各的，零协作，输出 N 个数
              b[i] ─┘

reduction:    a[0] ─┐
              a[1] ─┤
              a[2] ─┼─ 分层合并（树形）─→ 1 个数   必须协作，输出 1 个数
               ... ─┤
            a[N-1] ─┘
```

`vector_add` 是"尴尬并行"：切 N 份活，谁也不用管谁。
reduction 不一样：**N 个输入只剩 1 个输出，线程之间必须汇合**。而线程的寄存器是私有的，要交换数据只能走三条路：

| 通路 | 速度 | 本算子是否用 |
|---|---|---|
| 共享内存 | 很快（片上） | ✅ 主归约用它 |
| warp shuffle | 最快（寄存器直接换） | ✅ 最后一个 warp 用它 |
| 全局内存 / 原子操作 | 慢（走 L2） | ❌ |

### 2.2 V1 的五步设计

```cpp
__global__ void reduceSum(const float* __restrict__ in,
                          float* __restrict__ out,
                          int n)
{
    extern __shared__ float sdata[];        // ① 动态共享内存

    int tid = threadIdx.x;
    int i   = blockIdx.x * blockDim.x + tid;

    sdata[tid] = (i < n) ? in[i] : 0.0f;    // ② 装载 + 越界补「单位元」
    __syncthreads();                        // ③ 竖第一道栅栏

    for (int s = blockDim.x >> 1; s > 32; s >>= 1)   // ④ 逐轮减半
    {
        if (tid < s) sdata[tid] += sdata[tid + s];
        __syncthreads();                    //    注意：同步必须在 if 外面
    }

    if (tid < 32)                           // ⑤ 展开最后一个 warp
    {
        volatile float* v = sdata;
        v[tid] += v[tid + 32];
        v[tid] += v[tid + 16];
        v[tid] += v[tid +  8];
        v[tid] += v[tid +  4];
        v[tid] += v[tid +  2];
        v[tid] += v[tid +  1];
    }

    if (tid == 0) out[blockIdx.x] = sdata[0];   // 每个 block 交出一个部分和
}
```

配上主机端：

```
a[0..N)  --每线程一元素-->  sdata  --树形归约-->  每 block 一个部分和 out[blockIdx]
                                                          |
                                          host 用 double 累加 grid 个部分和 --> sum
```

### 2.3 五个关键设计决策（以及为什么）

| # | 决策 | 为什么 |
|---|---|---|
| 1 | 共享内存下标用 `tid` 而不是全局的 `i` | `sdata` 只有 `blockDim.x` 长。用 `i` 会越界（block 1 的 tid 0 → i = 256 → `sdata[256]` 爆掉） |
| 2 | 越界线程补 `0.0f` 而不是 `return` | 补 0 是"加法的单位元"，`x + 0 = x`。若直接 `return`，`sdata[tid]` 留着上一个 kernel 的垃圾值参与加法 → 结果错。<br>⚠ 单位元因运算而异：求 max 要补 `-FLT_MAX`，求乘积补 `1.0f` |
| 3 | `__syncthreads()` 写在 `if` 外面 | 它是 block 级栅栏，要求**所有线程都到达**。写进 `if (tid < s)` 里就有一半线程永远到不了 → 未定义行为（常见表现是死锁） |
| 4 | `s > 32` 时退出循环，改用展开 | 归约到只剩 32 个数时，整件事落在一个 warp 内部。warp 的 32 个 lane 是硬件调度的最小单位、天然执行同一套指令 → 不需要栅栏。手工展开省掉 6 次 `__syncthreads()` |
| 5 | 展开段用 `volatile float*` | 编译器不知道别的线程会改共享内存，会把 `sdata[tid]` 缓存进寄存器，读到过期值。`volatile` 强制每次都真的读内存 |

### 2.4 为什么选择"一线程一元素"

这个选择在当时是**最自然的**，因为它和 `vector_add` 的写法一致：

```
vector_add:  grid = ceil(n / blockSize)，每个线程处理 1 个元素
V1:          grid = ceil(n / blockSize)，每个线程处理 1 个元素  ← 同一个惯性
```

**正是这个惯性（第 4 节会看到）成了 V1 的性能天花板。**

### 2.5 V2 的设计改动：只改"装载"，不动"归约"

V2 相对 V1 **只改了一处**：

```cpp
    int tid = threadIdx.x;

    // ---- ① 寄存器内累加（V2 唯一的改动）----
    float sum = 0.0f;
    for (int i = blockIdx.x * blockDim.x + tid;
         i < n;
         i += blockDim.x * gridDim.x)
    {
        sum += in[i];
    }

    // ---- ② 以下与 V1 完全一致 ----
    sdata[tid] = sum;            // ← 不再需要 (i<n) ? ... : 0
                                 //    grid-stride 循环天然处理了越界，
                                 //    sum 的初值 0.0f 就是「单位元」
    __syncthreads();
    // ...（树形归约 + 展开最后 warp，逐字未改）
```

配套的主机端改动：

```cpp
int elemsPerThread = (argc > 3) ? atoi(argv[3]) : 8;

int grid    = (n + blockSize - 1) / blockSize;                       // 上限
int gridCap = (n + blockSize * elemsPerThread - 1) /
              (blockSize * elemsPerThread);                          // 封顶值
if (grid > gridCap) grid = gridCap;
if (grid < 1) grid = 1;
```

设计要点：

- **grid 从"精确覆盖"变成"封顶"**：先算出"一线程一元素"需要多少 block，再按 `elemsPerThread` 缩小。
- **越界补零那一手消失了**：grid-stride 的循环条件 `i < n` 天然处理越界，`sum` 初值 `0.0f` 就是单位元。
- **归约部分一行没动**：这说明 V1 的归约设计是对的，错的只是"并行粒度的选择"。

### 2.6 V3 的设计改动：把「块内归约」从共享内存搬到寄存器

V3 相对 V2 **也只改了一处**：块内归约那一段。

```cpp
    // ---- ① 寄存器内累加（与 V2 完全相同）----
    float sum = 0.0f;
    for (int i = blockIdx.x * blockDim.x + threadIdx.x;
         i < n;
         i += blockDim.x * gridDim.x)
        sum += in[i];

    // ---- ② V3 的改动：warp 内归约，全部在寄存器之间完成 ----
    for (int offset = 16; offset > 0; offset >>= 1)
        sum += __shfl_down_sync(0xffffffffu, sum, offset);

    // ---- ③ 共享内存只剩「每个 warp 一个格子」----
    __shared__ float warpSums[32];      // 注意：静态共享内存，编译期定死

    int lane = threadIdx.x & 31;
    int wid  = threadIdx.x >> 5;

    if (lane == 0) warpSums[wid] = sum;
    __syncthreads();                    // 全 block 唯一的栅栏（V2 有 3 次）

    // ---- ④ 由 0 号 warp 归约这至多 32 个 warp 部分和 ----
    if (wid == 0)
    {
        int nWarps = blockDim.x >> 5;
        sum = (lane < nWarps) ? warpSums[lane] : 0.0f;

        for (int offset = 16; offset > 0; offset >>= 1)
            sum += __shfl_down_sync(0xffffffffu, sum, offset);

        if (lane == 0) out[blockIdx.x] = sum;
    }
```

`__shfl_down_sync(mask, var, offset)` 的语义：**让 lane i 拿到 lane (i + offset) 的寄存器值**。
所以 `sum += __shfl_down_sync(0xffffffffu, sum, offset)` 就是「把邻居的和加到自己头上」：
offset = 16, 8, 4, 2, 1 共 5 轮之后，`lane 0` 手里就是整个 warp 的和。

⚠ 一个细节：当 `lane + offset >= 32` 时，该 lane 拿到的是**自己的值**（相当于自我累加），结果是垃圾值。
但我们**只读 lane 0**，而 lane 0 的每一步来源都合法，所以不影响正确性。

三个附带好处：

| 项 | V2 | V3 |
|---|---|---|
| blockSize 约束 | 硬编码 `v[tid+32]` 展开 → 必须 ≥ 64 | 只要 ≥ 32 且是 32 的倍数 |
| `volatile` | 必须（防止编译器把共享内存缓存进寄存器） | **不需要**（shuffle 明确地传值） |
| 动态共享内存 | 1,024 B（要传第三个启动参数） | **0 B**（静态 128 B，`<<<grid, blockSize>>>`） |

---

## 3. 调试

### 3.1 构建期实际踩到的两个问题

| # | 现象 | 原因 | 修法 |
|---|---|---|---|
| 1 | `CMake Error at CMakeLists.txt:13 (add_subdirectory): The source directory .../reduction does not contain a CMakeLists.txt file.` | 顶层加了 `add_subdirectory(reduction)`，但子目录里还没有 `CMakeLists.txt` | 建 `reduction/CMakeLists.txt` |
| 2 | 构建报"找不到源文件" | 源文件从 `reduction.cu` 改名为 `reducesum.cu`，`add_executable` 里还是旧名字 | 同步改成 `reducesum.cu` |

教训：**CMake 的子目录约定是"目录 + 同名源 + 自己的 CMakeLists"三件套，缺一个就报错，而且报错信息指向的行号往往在顶层文件里**，容易让人误以为问题在顶层。

### 3.2 潜在坑清单（写代码时必须盯住的）

| # | 坑 | 症状 | 原因 |
|---|---|---|---|
| 1 | 动态共享内存**忘了第三个启动参数** | `an illegal memory access was encountered` | `<<<grid, block>>>` 相当于 `shmem = 0`，`sdata` 长度 0 字节。语法完全合法，运行时才炸 |
| 2 | `__syncthreads()` 写进了 `if` | 死锁 / 偶尔算错 | 不是所有线程都到达栅栏 → 未定义行为 |
| 3 | 越界线程直接 `return` | 结果偏小或乱 | `sdata[tid]` 留下垃圾值参与加法 |
| 4 | 漏了 `volatile` | 偶尔算错 | 编译器把共享内存值缓存进寄存器 |
| 5 | `blockSize < 64` | 越界 | 展开段要访问 `sdata[tid + 32]`，`sdata` 只有 `blockSize` 长 |
| 6 | 带宽流量算成 `3 × bytes`（从 `vector_add` 抄过来） | 带宽虚高 3 倍、超过硬件峰值 | reduction 只**读一遍**输入 → 乘 1 |
| 7 | `verify: PASS` 就以为万事大吉 | 假阳性 | kernel 没跑也可能 PASS（`d_out` 里是旧数据）。还要看带宽是否合理 |

### 3.3 精度相关的调试要点

**为什么 `ref` 必须用 `double` 累加？**

```
float 累加 2^26 个 1.0f：

  累加到 16,777,216（= 2^24）时，再加 1.0f：
  2^24 附近 float 的「最小间隔（ulp）」是 2
  → 16777217 落在两个可表示值正中间 → 就近舍偶 → 还是 16777216
  → 之后再怎么加 1.0f 都纹丝不动

  最终得到 16,777,216 而不是 67,108,864 —— 差 4 倍！
```

**那 GPU 里的 float 累加为什么不出错？** 因为用的是**树形归约**，相邻两数两两相加，所有中间结果都很小，且全是 2 的幂（精确可表示）：

```
V1: 每线程 1 个数 → block 内树形 → 部分和 256.0f（精确）→ host double 累加（精确）
V2: 每线程 4 个数 → 4.0f（精确）→ 部分和 1024.0f（精确）→ host double 累加（精确）
```

两个版本的误差都恰好为 0。这也是为什么验证能用 `1e-3` 这种"几乎等于要求相等"的容差。

⚠ 但这**不代表归约永远精确**。若数组是任意浮点数，树形归约仍有舍入误差，只是它把误差从 O(n) 降到 O(log n) —— 这是树形归约在数值上的第二个好处，不只是速度。

---

## 4. 优化思路

这一节记录**思路是怎么被数据一次次推翻的**，比最终的数字更重要。

### 4.1 第一次转变：从"怀疑代码"到"承认算法选错了并行粒度"

第一次跑 V1：

```
N = 67108864 (256.0 MiB), grid = 262144, blockSize = 256
verify: PASS
best time = 1.7162 ms
bandwidth = 156.4 GB/s
```

第一反应是"代码有 bug"。但排查后：

- `verify: PASS` 且和完全精确 → 逻辑没问题；
- 带宽 156 < 峰值 256 → 没有超越物理极限的"虚高"；
- 所以代码是**对的**，只是**慢**。

对照 `vector_add`：768 MiB / 3.28 ms = 245 GB/s（95.9% 峰值）。

两者对比：

| | 每线程的访存 | 栅栏 | 每 block 搬多少 |
|---|---|---|---|
| `vector_add` | 读 a + 读 b + 写 c，**无同步** | 0 | 3 × 1 KiB |
| V1 reduction | 只读 1 个 float，然后**立刻撞栅栏** | 3 | **1 KiB** |

> **思路转变 #1**：不是"代码写错了"，而是"**一线程一元素这个并行粒度，对归约来说是错的**"。
> 归约和流式算子（vector_add）不一样 —— 它每读一点数据就要做一次全局汇总。

### 4.2 第二次转变：blockSize 扫描把预测推翻 —— 发现"独立栅栏域"模型

**第一版假设（错的）**：每个 block 只有 1 KiB 数据却要做 3 次栅栏，说明"栅栏次数/grid 太大"。那把 blockSize 调大，block 数变少，栅栏总数也就变少 → 应该更快。

实验（V1，blockSize = 256 / 512 / 1024）：

| blockSize | 实测带宽 |
|---|---|
| 256 | 156.3 GB/s |
| 512 | 141.3 GB/s |
| **1024** | **95.4 GB/s** ← 反而最慢 |

**结论和预测完全相反。** 重新推导：

| blockSize | warp/block | 每 SM 能放几个 block | 占用率 | **独立栅栏域 / SM** | 实测带宽 |
|---|---|---|---|---|---|
| 256 | 8 | 1536/256 = **6** | 100% | **6** | **156.3** |
| 512 | 16 | 1536/512 = **3** | 100% | **3** | 141.3 |
| 1024 | 32 | 1536/1024 = **1** | **67%** | **1** | 95.4 |

带宽和"独立栅栏域数量" 6 → 3 → 1 完全单调对应。原因：

```
blockSize = 256（6 个互不相干的 block 挤在一个 SM 上）：

  block A: [发 load]──[barrier 等........]
  block B:           [发 load]──[barrier 等........]
  block C:                     [发 load]──[barrier 等........]   ← A 卡在栅栏时
  block D:                               [发 load]──[barrier]      B~F 正在发 load
  block E: ...                                                   → 内存流水线有活干 ✅
  block F: ...

blockSize = 1024（整个 SM 只有 1 个 block）：

  block A: [发 load]──────[barrier ──────── 全 SM 停摆 ────────]
                                     ↑
                          没有第二个 block 来填这个空档 → 内存流水线整段空闲 ❌
```

**`__syncthreads()` 是 block 级栅栏：一个 block 撞栅栏时它的全部 warp 一起停住，但不阻塞别的 block。**

所以"用更大的 block 减少栅栏次数"是错的：

```
栅栏的*个数*确实少了（262144 → 65536 个 block）
但每次栅栏的*代价*变大了：
    从"只停住 8 个 warp" → "停住整个 SM 的全部 48 个 warp"
```

顺带排除一个干扰项：256 和 512 的 `Waves Per SM` 完全相同（都是 1820.4），但带宽掉了 10% → **波数量不是原因，栅栏域才是**。

> **思路转变 #2**：从"减少栅栏的**数量**"转向"降低每次栅栏的**代价占比**"。
> 并且第一次意识到：**调 blockSize 这条路治不了这个病**，因为问题的量级不在参数上 ——
> 每个 block 只搬 **1 KiB** 数据，却要做 3 次栅栏 + 一次启动 + 一次退休。

### 4.3 第三次转变：ncu 揭穿真正的瓶颈 —— 不是 DRAM，是 L1TEX

这是整个过程中最关键的一步。跑 ncu：

```bash
ncu --set full -k reduceSum --launch-count 1 ./build/bin/reducesum $((1<<26)) 256
```

V1 的 `GPU Speed Of Light Throughput`：

```
Memory Throughput                   %        64.23
DRAM Throughput                     %        52.01     ← DRAM 只跑到一半！
L1/TEX Cache Throughput             %        64.28     ← 最高水位线在这里
L2 Cache Throughput                 %        17.87
Compute (SM) Throughput             %        64.23     ← 和 L1TEX 并列最高
```

ncu 给的是这条**误导性**提示：

```
INF   Compute and Memory are well-balanced: To reduce runtime, both computation and
      memory traffic must be reduced.
```

但真相是：**DRAM 只有 52%，说明不是 DRAM 喂不饱；L1TEX 路径 64% 才是天花板**。而"L1TEX 路径"包含了：

1. **全局 load**（每 4 字节数据一次 request，无向量化）
2. **共享内存访问**（树形归约）
3. **那个 4 字节的 store**（`out[blockIdx.x] = sdata[0]`）

把账算清楚（blockSize = 256，每个 block）：

```
全局访存：256 线程 / 32 = 8 个 warp，每个 warp 一次 128 B 的 load
          → 8 个 L1TEX wavefront

共享内存：① 装载 sdata[tid]            → 8 个 wavefront
          ② 树形归约 8 轮（s=128…1）   → 约 36 个 wavefront
                                        （每轮 read+read+write）
          → 合计约 44 个 wavefront

所以：L1TEX 总 wavefront ≈ 52 个，其中只有 8 个（15%）是"有用的"全局 load
      共享内存占了 85%
```

**根因：共享内存树形归约的开销 ∝ block 数，而不是 ∝ 数据量。**

V1 有 262,144 个 block，每个 block 都要把这份**固定大小的 256 元素树**完整做一遍 —— 不管它搬了多少数据。数据量不变，但"树"被重复了 26 万遍。

> **思路转变 #3**：瓶颈定位从"DRAM 带宽"修正为"**SM 内部的 L1TEX 吞吐**"，
> 再修正为"**共享内存归约开销 ∝ block 数**"。
> 这直接给出了正确的优化方向：**提高每 block 的载荷，从而减少 block 数。**

### 4.4 第四次转变：一个改动同时改善三个量

既然根因是"共享内存开销 ∝ block 数"，那么修法就是：**让每个 block 装更多数据**。

但**不能只把 block 变大**（4.2 已经证明那会更糟，因为栅栏域会变少）。正确做法是：**block 保持小，但让线程在寄存器里先累加多个元素**。

```
V1：  1 线程 ── 读 1 个 ── 撞栅栏                每 block 只搬 1 KiB
V2：  1 线程 ── 读 4 个，先在寄存器里累加 ── 撞栅栏   每 block 搬 4 KiB
```

一个改动，同时改善三件事：

| 量 | V1 | V2 (elems=4) | 变化 |
|---|---|---|---|
| 每 block 载荷 | 1 KiB | 4 KiB | ↑ 4× |
| grid（block 数） | 262,144 | 65,536 | ↓ 4× |
| 总栅栏次数 | 786,432 | 196,608 | ↓ 4× |
| **共享内存树的总量** | 262144 × 44 | 65536 × 44 | **↓ 4×** |
| 栅栏域 / SM | 6 | 6 | 不变 ✅ |

**保持 blockSize = 256 是关键**：这样"独立栅栏域 = 6"这个优势被完整保留，同时又砍掉了 3/4 的固定开销。

> **思路转变 #4**：优化动作从"调参"变成"**改变并行粒度**"。
> 并且认识到：**这个改动不牺牲任何已有的东西** —— 它不是在两个缺点之间权衡，而是把"每 block 载荷不足"这个唯一的缺点直接消掉了。

### 4.5 第五次转变（方法学）：测量本身也是设计的一部分

第一次扫 `elemsPerThread`：

```
elems=1   min=1.7551 ms  bw= 152.9 GB/s
elems=2   min=1.1684 ms  bw= 229.7 GB/s
elems=4   min=1.0916 ms  bw= 245.9 GB/s
elems=8   min=1.2349 ms  bw= 217.4 GB/s     ← 看起来"过了甜点就往下掉"
elems=16  min=1.2349 ms  bw= 217.4 GB/s
elems=32  min=1.2359 ms  bw= 217.2 GB/s
```

于是得出了"elems=4 是甜点，≥8 会退化到一个 217 GB/s 的平台"的结论。

**但重测同一组配置时，`elems=4` 变成了 217.2 GB/s**（之前是 245.9）。同一二进制、同一参数，差 13%。

而且注意那些"一模一样"的数字：`1.2349`、`1.2357`、`1.2359` —— 不同配置给出**逐位相同**的耗时，说明 GPU 在几个**离散的时钟档位**之间跳。

改成**交叉（interleaved）测量**（每一轮把所有配置都跑一遍）之后：

| 配置 | 交叉测量结果 |
|---|---|
| b128-e2 | 215.6 GB/s |
| b128-e4 | 248.5 GB/s |
| b128-e8 | 217.2 GB/s |
| b256-e2 | 213.0 GB/s |
| b256-e4 | 248.0 GB/s |
| b256-e8 | 248.0 GB/s |

最终用 6 轮交叉 A/B 得到干净、单调的数据（见 5.2）。“217 平台"是**测量假象**，不是真实的性能断崖。

> **思路转变 #5**：从"跑一次取一个数"转向"**交叉重复测量取极值**"。
> 这与 `vector_add` 笔记里"笔记本显卡时钟波动导致 13% 偏差"是同一个问题，
> 但在**对比多个配置**时，只有交叉测量才能把这个变量控制住。

### 4.6 第六次转变：量化「块内归约」本身的开销 —— 一个必须精心设计的对照实验

V2 把瓶颈推到 DRAM 之后，自然的问题是：**块内归约那段（共享内存树 + 栅栏）到底还占多少？**

这个问题**没法在 `elems=4` 的配置下回答** —— 那时 DRAM 已经 96.5%，块内归约被完全掩盖。

**实验设计**：把 `elemsPerThread` 设成 1（此时 grid-stride 循环恰好只跑一轮，退化成 V1 的行为），
块内归约的占比就被放到最大。然后对比 V2（共享内存版）和 V3（shuffle 版）：

```
elems=1：每线程只搬 4 字节 → 块内归约的开销占比最大 ← 差异在这里才看得见
elems=4：每线程搬 16 字节 → DRAM 成为瓶颈      ← 两者应该打平
```

**结果**：

| 配置 | V2（共享内存） | V3（shuffle） | 差异 |
|---|---|---|---|
| `elems=1` | 152.9 GB/s | **177.2 GB/s** | **+15.9%** |
| `elems=4` | 248.0 GB/s | 247.8 GB/s | **≈ 0** |

> **思路转变 #6**：优化的收益**不是恒定的，而是取决于瓶颈在哪里**。
> V3 比 V2 快 15.9%，但那是在「块内归约是瓶颈」的前提下；
> 一旦 DRAM 成为瓶颈（`elems=4`），这个优势**完全消失**。
>
> 由此得到一条实用的判断准则：**先确定瓶颈，再决定优化方向**。
> 如果先做了 V3 再看 `elems=4` 的数字，会误以为「shuffle 没用」。

⚠ 一个必须纠正的常见说法：**`__shfl_down_sync` 并不是「免费」的**。
ncu 显示 V3 的 `Mem Pipes Busy` 只从 64.23% 降到 55.70%（−13%），而不是降到接近 0。
原因是 **warp shuffle 走的是 MIO（内存输入输出）流水线 —— 和共享内存是同一条物理路径**。

```
共享内存：sdata[tid] += sdata[tid+s]   → 2 次 L1TEX 读 + 1 次 L1TEX 写
shuffle ：__shfl_down_sync(...)        → 1 次 MIO 交换操作（不碰内存）
```

收益来自**操作次数变少**（指令数 −24.4%，分支数 −47.7%），而不是「换了一条更快的路」。

### 4.7 为什么同一组 V2/V3 数据，在两个配置下会给出相反的结论

4.6 那张表里的两条结论，其实是**同一件事的两面**，值得单独展开 —— 因为这是整个手记里最能迁移的一条经验。

**（1）先算清楚「天花板」在哪**

```
N = 2^26，数据量 = 268,435,456 字节 = 268.4 MB
DRAM 峰值 = 256.3 GB/s
→ DRAM 理论最短时间 = 268.4 MB ÷ 256.3 GB/s = 1.0474 ms      ← 硬天花板
```

| | elems=1 | elems=4 |
|---|---|---|
| V2 实测 | 1.7930 ms | 1.0803 ms |
| 换算带宽 | 149.7 GB/s | 248.5 GB/s |
| **占 DRAM 理论上限** | **58.4%** | **96.9%** |
| **还剩多少空间** | **41.6%** | **3.1%** |

**看最后一行：3.1%。**

也就是说，在 `elems=4` 的配置下，**即使你把块内归约的开销全部免费抹掉（物理上不可能），整程序最多也只能快 3.1%。** 而 V3 只是「把归约做得更省一些」，远不是「完全消除」—— 所以那 3.1% 里它连边都摸不到。

**（2）为什么 `elems=1` 时块内归约占比最大**

关键在于一个容易忽略的事实：

> **块内归约的工作量只取决于 `blockSize`，和你搬了多少数据无关。**

用「工作单位」来量化（blockSize = 256，一个 block）：

| 阶段 | 工作单位 | 数量 |
|---|---|---|
| **装载** | 全局访存事务 | 8 个 warp × 1 次 = **8 次**（搬 1 KiB）|
| **块内归约（V2）** | 共享内存 wavefront | 装载 8 + 树形约 36 = **约 44 次** |
| | 栅栏 | **3 次** |

**装载只有 8 个工作单位，归约却有 44 个 —— 归约是装载的 5 倍多。**

因为 block 只搬了 1 KiB 数据，却要做完整的 256 元素树形归约。这个树的规模由 `blockSize` 决定，`elemsPerThread` 完全管不着。

`elems=4` 时发生了什么：

| | elems=1 | elems=4 |
|---|---|---|
| grid（block 数） | 262,144 | 65,536 |
| **全 kernel 的装载总量** | 67,108,864 个元素 | 67,108,864 个元素（**不变**）|
| **全 kernel 的归约总量** | 262,144 棵树 | 65,536 棵树（**少 4 倍**）|

装载总量不变，但归约总量随 block 数一起降了 4 倍 → **归约被「稀释」了**。

**（3）决定性的是「瓶颈换人了」，不是「稀释」**

单靠稀释还解释不了「从 +15.9% 变成 ±0%」这么大的落差。真正的原因是：

```
elems=1：DRAM 只用 58.4%  →  瓶颈在 SM（归约 / 栅栏）  →  优化 SM 有用 ✅
elems=4：DRAM 已用 96.9%  →  瓶颈在 DRAM              →  优化 SM 无用 ❌
```

> **你优化的那个部件，如果不是当下的瓶颈，优化它不会让程序变快。**

**（4）一个类比**

把 kernel 想成一个货运站：

| | 对应 |
|---|---|
| **装载** | 货车在公路上跑（DRAM，外部资源，24 个 SM 抢）|
| **块内归约** | 到站后卸货、对账（SM 内部，各干各的）|

- **elems=1**：车次多、每车只拉一点点，但对账会开得极频繁 → **一天到晚在开会，公路反而空着**。简化会议流程（V3）收益巨大。
- **elems=4**：车次少了 4 倍、每车拉满 → **公路已经堵死了**，会议再快也没用，货就是运不进来。

**（5）一条可以迁移到任何 kernel 的判断准则**

| 观察 | 结论 | 该往哪使劲 |
|---|---|---|
| 带宽利用率 ≥ 90% | 访存瓶颈 | **别在计算上花力气**，只能减少访存或提高带宽利用率 |
| 带宽利用率明显低于峰值 | 瓶颈在 SM 内部 | 用 ncu 找（L1TEX / 计算管线 / 栅栏 / 占用率）|

对照本项目：`vector_add` 是 245.8 GB/s（95.9%）→ 访存瓶颈，所以它再怎么优化 kernel 内部也没用。
V1@`elems=1` 是 58.4% → 瓶颈在 SM，所以有优化空间。

**（6）反过来会误判（这条最实用）**

同一个 V3，在不同实验配置下会得出**完全相反**的两个结论：

| 你只在哪个配置上比较 | 你的结论 | 对不对 |
|---|---|---|
| 只在 `elems=1` | 「V3 快 16%，值得用！」 | 都是对的，但**都只在自己的前提下成立** |
| 只在 `elems=4` | 「V3 没用，白折腾」 | |

两条实践原则：

1. **基准测试的配置要和你实际要跑的配置一致。** 真实推理场景里算子通常都是「每线程处理好几个元素」（`elems ≥ 4`），所以按 `elems=4` 的结论选型才是对的。
2. **别把「某个技术没用」和「某个技术在这个场景下用不上」混为一谈。** V3 的技术本身没问题（它确实少做了近一半的活），只是这个场景里**它不是瓶颈**。

---

## 5. 实验

### 5.1 实验 A：V1 的 blockSize 扫描（推翻第一版假设）

```bash
for b in 256 512 1024; do
  for i in 1 2 3; do ./build/bin/reducesum $((1<<26)) $b; done \
    | awk -v b=$b '/best time/{v=$(NF-1)+0; if(mint==""||v<mint)mint=v}
                   /bandwidth/{v=$(NF-1)+0; if(v>maxb)maxb=v}
                   END{printf "block=%-5s min=%.4f ms  bw=%.1f GB/s\n", b, mint, maxb}'
done
```

| blockSize | min time | 带宽 | 每 SM 栅栏域 |
|---|---|---|---|
| 256 | 1.7169 ms | 156.3 GB/s | 6 |
| 512 | 1.8996 ms | 141.3 GB/s | 3 |
| 1024 | 2.8128 ms | 95.4 GB/s | 1 |

**结论**：与"block 越大越好"的直觉相反，带宽随 block 增大而单调下降，与"独立栅栏域数量"完全对应 → 见 4.2。

### 5.2 实验 B：`elemsPerThread` 受控 A/B（核心实验）

**实验设计（重要）**：使用**同一个二进制** `reducesumV2`，只改第 3 个参数，这样就**只有一个变量**在动 —— 因为 `elems=1` 时 grid-stride 循环恰好只跑一轮，行为等价于 V1 的 kernel。

```bash
for r in 1 2 3 4 5 6; do                  # 6 轮
  for e in 1 2 4 8; do                    # 每轮把所有配置都跑一遍（交叉）
    ./build/bin/reducesumV2 $((1<<26)) 256 $e | awk -v e=$e '
      /best time/{t=$(NF-1)+0} /bandwidth/{printf "e%-3s %.4f %s\n", e, t, $(NF-1)}'
  done
done | awk '{k=$1; if(!(k in mt)||$2<mt[k])mt[k]=$2; if(!(k in mb)||$3>mb[k])mb[k]=$3}
            END{for(k in mt) printf "%-5s min_time=%.4f ms   max_bw=%6.1f GB/s\n", k, mt[k], mb[k]}' | sort
```

**结果**：

| elemsPerThread | grid | min time | max bw | 峰值占比 (256.3) | 相对 elems=1 |
|---|---|---|---|---|---|
| 1（= V1） | 262,144 | 1.7930 ms | 149.7 GB/s | 58.4% | 1.00× |
| 2 | 131,072 | 1.1684 ms | 229.7 GB/s | 89.6% | **1.53×** |
| **4** | 65,536 | **1.0803 ms** | **248.5 GB/s** | **97.0%** | **1.66×** |
| 8 | 32,768 | 1.0783 ms | 248.9 GB/s | 97.1% | 1.66× |

**判读**：

- `elems=1 → 2`：+53%，最大的一跳。说明"至少要有几个元素垫在寄存器里"是跨过门槛的关键。
- `elems=2 → 4`：+8%。到这里已经 97% 峰值。
- `elems=4 → 8`：+0.2%，**到顶了**。DRAM 已经满载，再加元素没有意义。
- **`elems=4` 是实用甜点**（`elems=8` 的收益在噪声范围内，但会浪费更多寄存器）。

### 5.3 实验 C：二维扫描（blockSize × elemsPerThread）

单次扫描（非交叉，仅用于看趋势）：

| blockSize \ elems | 2 | 4 | 8 |
|---|---|---|---|
| 128 | 243.0 | **248.3** | 217.2 |
| 256 | 229.7 | 217.2 | 217.4 |
| 512 | 192.2 | 216.5 | 217.2 |

趋势依然清晰：`elems ≥ 4` 时几乎所有 blockSize 都能到 217~248；`elems = 2` 时对 blockSize 更敏感。绝对值受时钟档位影响较大，不宜逐格对比（见 4.5）。

**结论**：`elemsPerThread` 是**一阶变量**，`blockSize` 在 128~256 范围内是**二阶变量**。

### 5.4 实验 D：证明"单次测量不可信"

同一个配置 `block=256 elems=4` 在不同时间点的结果：

| 运行批次 | 耗时 | 带宽 |
|---|---|---|
| 第 1 批 | 1.0916 ms | 245.9 GB/s |
| 第 2 批 | 1.2357 ms | 217.2 GB/s |

**差 13%，纯粹来自 GPU 时钟档位 / 温度。** 若没有交叉测量，就会错误地得出"elems=4 有 245.9，elems=8 有 217.4，所以 elems=4 更好"的结论（真相是两者都是 248）。

### 5.5 实验 E：V2 vs V3（`__shfl_down_sync` 的收益量化）

**实验设计（重要）**：这次是**两个不同的二进制**（`reducesumV2` / `reducesumV3`），
所以必须用**正反交替**的顺序跑，把「位置带来的热环境差异」消掉 ——
普通的交叉测量仍然会有系统性偏置（比如某个配置总是排在最后一个）。

```bash
seq_odd="V2:1 V3:1 V2:4 V3:4"        # 正向顺序
seq_even="V3:4 V2:4 V3:1 V2:1"       # 反向顺序（完全镜像）
for r in 1 2 3 4 5 6 7 8; do
  if [ $((r%2)) -eq 1 ]; then s="$seq_odd"; else s="$seq_even"; fi
  for item in $s; do
    prog=${item%:*}; e=${item#*:}
    case $prog in V2) bin=reducesumV2;; V3) bin=reducesumV3;; esac
    ./build/bin/$bin $((1<<26)) 256 $e | awk -v k="$prog-e$e" \
      '/bandwidth/{print k, $(NF-1)}'
  done
done | awk '{k=$1; if(!(k in m)||$2>m[k])m[k]=$2; c[k]++}
            END{for(k in m) printf "%-8s max_bw=%6.1f GB/s  (%d runs)\n", k, m[k], c[k]}' | sort
```

**结果**（8 轮 × 每配置 8 次）：

| 配置 | max bw | 峰值占比 |
|---|---|---|
| V2 `elems=1` | 152.9 GB/s | 59.7% |
| **V3 `elems=1`** | **177.2 GB/s** | **69.1%** |
| V2 `elems=4` | 248.0 GB/s | 96.8% |
| V3 `elems=4` | 247.8 GB/s | 96.7% |

**两条结论**：

1. `elems=1` 时 V3 快 **15.9%** → 这就是「块内归约本身的开销」的量化值。
2. `elems=4` 时两者**打平** → 瓶颈转移到 DRAM 后，块内归约不再是关键路径。

> ⚠ **实验设计教训**：第一次做这个对比时，循环顺序是固定的
> （`V2-e1, V3-e1, V2-e4, V3-e4, V2-e8, V3-e8`），
> 结果 `e8` 那一档**永远排在每轮最后**，总是在 GPU 最热的时候被测，
> 于是 `elems=8` 系统性地测出 217 而不是 248。
> **交叉测量必须配合「顺序正反交替」**，否则只是把「块间偏置」换成了「位置偏置」。

### 5.6 附：其他可做但未做的实验（留作后续）

```bash
# 让数据装进 L2（本机 L2 ≈ 24 MiB），验证"是否 DRAM 受限"
./build/bin/reducesumV2 $((1<<24)) 256 4      # 64 MiB，仍超 L2
./build/bin/reducesumV2 $((1<<22)) 256 4      # 16 MiB，能装进 L2 → 带宽会明显跳高

# 锁频消除噪声（笔记本可能不支持）
sudo nvidia-smi -lgc 1800
sudo nvidia-smi -rgc
```

---

## 6. ncu Profile 对比

### 6.1 采集命令

```bash
ncu --set full -k reduceSum     --launch-count 1 ./build/bin/reducesum   $((1<<26)) 256
ncu --set full -k reduceSum     --launch-count 1 ./build/bin/reducesumV2 $((1<<26)) 256 4
ncu --set full -k reduceSumWarp --launch-count 1 ./build/bin/reducesumV3 $((1<<26)) 256 1
```

⚠ 两个细节：
1. **kernel 名字不同**：V1/V2 都叫 `reduceSum`，V3 叫 `reduceSumWarp`，所以 `-k` 过滤条件要换。
2. **V3 用 `elems=1` 采集**：这样 V1 与 V3 的 **grid 都是 262,144**，可以做到单变量对比（见 6.7 / 6.8）；
   同时让「块内归约」在总时间里的占比最大，差异才看得出来。

⚠ 说明：ncu 默认在每次 replay 之间**冲刷 L2 缓存**，且 profiled 的是第一个（warm-up）launch，
所以 ncu 报告里的 `Duration` **绝对值不能直接和程序内计时对比**。
下面的分析以**百分比指标**（DRAM Throughput / Mem Pipes Busy 等）为主，那些是归一化的、可信的。

### 6.2 V1 报告（原文，`reducesum.cu`，grid = 262,144）

```
  reduceSum(const float *, float *, int) (262144, 1, 1)x(256, 1, 1), Context 1, Stream 7, Device 0, CC 8.9
    Section: GPU Speed Of Light Throughput
    ----------------------- ------------- ------------
    Metric Name               Metric Unit Metric Value
    ----------------------- ------------- ------------
    DRAM Frequency          cycle/nsecond         8.00
    SM Frequency            cycle/nsecond         1.55
    Elapsed Cycles                  cycle    3,129,540
    Memory Throughput                   %        64.23
    DRAM Throughput                     %        52.01
    Duration                      msecond         2.02
    L1/TEX Cache Throughput             %        64.28
    L2 Cache Throughput                 %        17.87
    SM Active Cycles                cycle 3,126,475.92
    Compute (SM) Throughput             %        64.23
    ----------------------- ------------- ------------

    INF   Compute and Memory are well-balanced: To reduce runtime, both computation and memory traffic must be reduced.

    Section: Compute Workload Analysis
    -------------------- ----------- ------------
    Metric Name          Metric Unit Metric Value
    -------------------- ----------- ------------
    Executed Ipc Active   inst/cycle         1.39
    Executed Ipc Elapsed  inst/cycle         1.39
    Issue Slots Busy               %        34.68
    Issued Ipc Active     inst/cycle         1.39
    SM Busy                        %        34.68
    -------------------- ----------- ------------

    Section: Memory Workload Analysis
    --------------------------- ------------ ------------
    Metric Name                  Metric Unit Metric Value
    --------------------------- ------------ ------------
    Memory Throughput           Gbyte/second       133.18
    Mem Busy                               %        35.70
    Max Bandwidth                          %        64.23
    L1/TEX Hit Rate                        %         1.42
    L2 Compression Success Rate            %            0
    L2 Compression Ratio                                0
    L2 Hit Rate                            %         3.14
    Mem Pipes Busy                         %        64.23
    --------------------------- ------------ ------------

    Section: Scheduler Statistics
    ---------------------------- ----------- ------------
    Metric Name                  Metric Unit Metric Value
    ---------------------------- ----------- ------------
    One or More Eligible                   %        34.77
    Issued Warp Per Scheduler                        0.35
    No Eligible                            %        65.23
    Active Warps Per Scheduler          warp         8.58
    Eligible Warps Per Scheduler        warp         0.64
    ---------------------------- ----------- ------------

    Section: Warp State Statistics
    ---------------------------------------- ----------- ------------
    Metric Name                              Metric Unit Metric Value
    ---------------------------------------- ----------- ------------
    Warp Cycles Per Issued Instruction             cycle        24.67
    Warp Cycles Per Executed Instruction           cycle        24.67
    Avg. Active Threads Per Warp                                31.61
    Avg. Not Predicated Off Threads Per Warp                    25.56
    ---------------------------------------- ----------- ------------

    WRN   On average, each warp of this kernel spends 9.8 cycles being stalled waiting for a scoreboard
          dependency on a L1TEX operation. This represents about 39.8% of the total average of 24.7 cycles
          between issuing two instructions.

    Section: Instruction Statistics
    ---------------------------------------- ----------- ------------
    Metric Name                              Metric Unit Metric Value
    ---------------------------------------- ----------- ------------
    Avg. Executed Instructions Per Scheduler        inst 1,084,074.67
    Executed Instructions                           inst  104,071,168
    Avg. Issued Instructions Per Scheduler          inst 1,084,185.67
    Issued Instructions                             inst  104,081,824
    ---------------------------------------- ----------- ------------

    WRN   This kernel executes 0 fused and 5767168 non-fused FP32 instructions. By converting pairs of
          non-fused instructions to their fused equivalent, the achieved FP32 performance could be
          increased by up to 50%.

    Section: Launch Statistics
    -------------------------------- --------------- ---------------
    Metric Name                          Metric Unit    Metric Value
    -------------------------------- --------------- ---------------
    Block Size                                                   256
    Function Cache Configuration                     CachePreferNone
    Grid Size                                                262,144
    Registers Per Thread             register/thread              16
    Shared Memory Configuration Size           Kbyte           32.77
    Driver Shared Memory Per Block       Kbyte/block            1.02
    Dynamic Shared Memory Per Block      Kbyte/block            1.02
    Static Shared Memory Per Block        byte/block               0
    Threads                                   thread      67,108,864
    Waves Per SM                                            1,820.44
    -------------------------------- --------------- ---------------

    Section: Occupancy
    ------------------------------- ----------- ------------
    Metric Name                     Metric Unit Metric Value
    ------------------------------- ----------- ------------
    Block Limit SM                        block           24
    Block Limit Registers                 block           16
    Block Limit Shared Mem                block           16
    Block Limit Warps                     block            6
    Theoretical Active Warps per SM        warp           48
    Theoretical Occupancy                     %          100
    Achieved Occupancy                        %        72.46
    Achieved Active Warps Per SM           warp        34.78
    ------------------------------- ----------- ------------

    Section: Source Counters
    ------------------------- ----------- ------------
    Metric Name               Metric Unit Metric Value
    ------------------------- ----------- ------------
    Branch Instructions Ratio           %         0.16
    Branch Instructions              inst   17,039,360
    Branch Efficiency                   %          100
    Avg. Divergent Branches                          0
    ------------------------- ----------- ------------

    WRN   The memory access pattern for stores from L1TEX to L2 is not optimal. The granularity of an
          L1TEX request to L2 is a 128 byte cache line, that is 4 consecutive 32-byte sectors per L2
          request. However, this kernel only accesses an average of 1.0 sectors out of the possible
          4 sectors per cache line.
```

### 6.3 V2 报告（原文，`reducesumV2.cu`，grid = 65,536，elemsPerThread = 4）

```
  reduceSum(const float *, float *, int) (65536, 1, 1)x(256, 1, 1), Context 1, Stream 7, Device 0, CC 8.9
    Section: GPU Speed Of Light Throughput
    ----------------------- ------------- ------------
    Metric Name               Metric Unit Metric Value
    ----------------------- ------------- ------------
    DRAM Frequency          cycle/nsecond         7.96
    SM Frequency            cycle/nsecond         1.54
    Elapsed Cycles                  cycle    1,681,573
    Memory Throughput                   %        96.51
    DRAM Throughput                     %        96.51
    Duration                      msecond         1.09
    L1/TEX Cache Throughput             %        37.70
    L2 Cache Throughput                 %        33.32
    SM Active Cycles                cycle 1,680,328.42
    Compute (SM) Throughput             %        37.68
    ----------------------- ------------- ------------

    INF   The kernel is utilizing greater than 80.0% of the available compute or memory performance of
          the device. To further improve performance, work will likely need to be shifted from the most
          utilized to another unit. Start by analyzing DRAM in the Memory Workload Analysis section.

    Section: Compute Workload Analysis
    -------------------- ----------- ------------
    Metric Name          Metric Unit Metric Value
    -------------------- ----------- ------------
    Executed Ipc Active   inst/cycle         1.04
    Executed Ipc Elapsed  inst/cycle         1.03
    Issue Slots Busy               %        25.89
    Issued Ipc Active     inst/cycle         1.04
    SM Busy                        %        25.89
    -------------------- ----------- ------------

    Section: Memory Workload Analysis
    --------------------------- ------------ ------------
    Metric Name                  Metric Unit Metric Value
    --------------------------- ------------ ------------
    Memory Throughput           Gbyte/second       245.80
    Mem Busy                               %        20.61
    Max Bandwidth                          %        96.51
    L1/TEX Hit Rate                        %         0.32
    L2 Compression Success Rate            %            0
    L2 Compression Ratio                                0
    L2 Hit Rate                            %         0.84
    Mem Pipes Busy                         %        37.68
    --------------------------- ------------ ------------

    Section: Scheduler Statistics
    ---------------------------- ----------- ------------
    Metric Name                  Metric Unit Metric Value
    ---------------------------- ----------- ------------
    One or More Eligible                   %        25.89
    Issued Warp Per Scheduler                        0.26
    No Eligible                            %        74.11
    Active Warps Per Scheduler          warp        10.38
    Eligible Warps Per Scheduler        warp         0.36
    ---------------------------- ----------- ------------

    Section: Warp State Statistics
    ---------------------------------------- ----------- ------------
    Metric Name                              Metric Unit Metric Value
    ---------------------------------------- ----------- ------------
    Warp Cycles Per Issued Instruction             cycle        40.09
    Warp Cycles Per Executed Instruction           cycle        40.10
    Avg. Active Threads Per Warp                                31.76
    Avg. Not Predicated Off Threads Per Warp                    27.59
    ---------------------------------------- ----------- ------------

    WRN   On average, each warp of this kernel spends 29.6 cycles being stalled waiting for a scoreboard
          dependency on a L1TEX operation. This represents about 73.9% of the total average of 40.1 cycles
          between issuing two instructions.

    Section: Instruction Statistics
    ---------------------------------------- ----------- ------------
    Metric Name                              Metric Unit Metric Value
    ---------------------------------------- ----------- ------------
    Avg. Executed Instructions Per Scheduler        inst   434,858.67
    Executed Instructions                           inst   41,746,432
    Avg. Issued Instructions Per Scheduler          inst   434,969.67
    Issued Instructions                             inst   41,757,088
    ---------------------------------------- ----------- ------------

    WRN   This kernel executes 0 fused and 3538944 non-fused FP32 instructions. By converting pairs of
          non-fused instructions to their fused equivalent, the achieved FP32 performance could be
          increased by up to 50%.

    Section: Launch Statistics
    -------------------------------- --------------- ---------------
    Metric Name                          Metric Unit    Metric Value
    -------------------------------- --------------- ---------------
    Block Size                                                   256
    Function Cache Configuration                     CachePreferNone
    Grid Size                                                 65,536
    Registers Per Thread             register/thread              16
    Shared Memory Configuration Size           Kbyte           32.77
    Driver Shared Memory Per Block       Kbyte/block            1.02
    Dynamic Shared Memory Per Block      Kbyte/block            1.02
    Static Shared Memory Per Block        byte/block               0
    Threads                                   thread      16,777,216
    Waves Per SM                                              455.11
    -------------------------------- --------------- ---------------

    Section: Occupancy
    ------------------------------- ----------- ------------
    Metric Name                     Metric Unit Metric Value
    ------------------------------- ----------- ------------
    Block Limit SM                        block           24
    Block Limit Registers                 block           16
    Block Limit Shared Mem                block           16
    Block Limit Warps                     block            6
    Theoretical Active Warps per SM        warp           48
    Theoretical Occupancy                     %          100
    Achieved Occupancy                        %         87.05
    Achieved Active Warps Per SM           warp        41.78
    ------------------------------- ----------- ------------
```

### 6.4 逐项对比与解读

| 指标 | V1 | V2 | 变化 | 解读 |
|---|---|---|---|---|
| **DRAM Throughput** | **52.01%** | **96.51%** | **↑ 85%** | ⭐ **核心结论：从"DRAM 半闲"到"DRAM 满载"** |
| Mem Pipes Busy（L1TEX） | 64.23% | 37.68% | ↓ 41% | SM 内部访存路径不再是瓶颈 |
| L1/TEX Cache Throughput | 64.28% | 37.70% | ↓ 41% | 与上一行基本同义 |
| Compute (SM) Throughput | 64.23% | 37.68% | ↓ 41% | **V1 的天花板就是它** |
| Elapsed Cycles | 3,129,540 | 1,681,573 | ↓ 46% | |
| Executed Instructions | 104,071,168 | 41,746,432 | ↓ 60% | 少做了一半以上的无用功 |
| Grid Size | 262,144 | 65,536 | ↓ 75% | 4× 少 |
| Threads | 67,108,864 | 16,777,216 | ↓ 75% | 4× 少 |
| **Waves Per SM** | **1,820.44** | **455.11** | **↓ 75%** | 4× 少，block 启动/退休开销摊薄 |
| Achieved Occupancy | 72.46% | 87.05% | ↑ | |
| Active Warps Per Scheduler | 8.58 | 10.38 | ↑ | 常驻 warp 更多 |
| Memory Throughput (GB/s) | 133.18 | 245.80 | ↑ 85% | 与 DRAM Throughput 一致 |
| L2 Hit Rate | 3.14% | 0.84% | ↓ | 都是纯流式访问，本就没有缓存复用 |
| L1/TEX Hit Rate | 1.42% | 0.32% | ↓ | 同上 |
| Branch Instructions | 17,039,360 | 6,356,992 | ↓ 63% | 循环/分支次数随 block 数一起降 |
| Registers Per Thread | 16 | 16 | — | **V2 没有多花寄存器**（4 个元素的累加用不到额外的寄存器） |
| Theoretical Occupancy | 100% | 100% | — | 共享内存 1 KiB/block 太小，从来不是占用率的限制因素 |
| Branch Efficiency | 100% | 100% | — | 顺序寻址确实做到了**零发散** |

### 6.5 三个"看起来变差、其实是变好"的指标（最容易误读的地方）

| 指标 | V1 | V2 | 单看会以为 | 真相 |
|---|---|---|---|---|
| `Warp Cycles Per Issued Instruction` | 24.67 | **40.09** | V2 更差（每条指令等更久） | V1 的 warp "发指令发得勤"，但发的**是共享内存访问** —— 在飞快地做无用功。V2 的 warp 大部分时间在等 DRAM 返回，这正是"把 DRAM 喂饱"的表现 |
| `Eligible Warps Per Scheduler` | 0.64 | **0.36** | V2 更差（可发射的 warp 更少） | 同上：warp 都在等 DRAM，"没得发"恰恰说明内存被压满了 |
| `Stall on L1TEX scoreboard` 占比 | 39.8% | **73.9%** | V2 更差（等内存更厉害） | 这是"访存密度提高"的直接后果 |

**判断准则（本算子最重要的一条经验）**：

```
看 DRAM Throughput，而不是看 stall 百分比。

  stall 高 + DRAM 满（96%）  = 好   ← V2：内存被喂饱，warp 在正常地等数据
  stall 低 + DRAM 半闲（52%）= 差   ← V1：warp 很忙，但忙的是无用功
```

孤立地看"stall 比例"会得出完全相反的结论。

### 6.6 ncu 两条建议的批判性解读

| ncu 的建议 | 该不该听 | 理由 |
|---|---|---|
| V1: "Compute and Memory are well-balanced: both computation and memory traffic must be reduced" | ❌ **误导** | 它看到 64.23% / 64.23% 就下结论"两者均衡"。但 DRAM 只有 52%，而 L1TEX 路径 64% —— 真正的天花板是后者，"减少计算"完全没抓到重点。**通用建议必须用自己的物理常识过滤** |
| V2: "utilizing greater than 80.0% ... Start by analyzing DRAM" | ✅ **正确** | 这次它说对了：DRAM 96.51% 确实是天花板。但也意味着**没有优化空间了** |
| 两者: "only accesses an average of 1.0 sectors out of the possible 4 sectors per cache line"（store 不合并） | ⚠️ 真实但收益有限 | 来自 `out[blockIdx.x] = sdata[0]`：每个 block 只写 4 字节，而 L1TEX→L2 的最小粒度是 32 字节 sector。V1：262144 × 32 B = 8 MiB 无效写流量（有效数据只有 1 MiB）；V2：65536 × 32 B = 2 MiB。相对 256 MiB 的读流量，V1 的额外开销约 +2.7% —— **不是瓶颈**。要修需要让一个 warp 一次写 32 个 block 的部分和，改动复杂、收益小，记录为已知问题 |
| 两者: "non-fused FP32 instructions ... could be increased by up to 50%" | ❌ **直接忽略** | roofline 显示 FP32 利用率只有 ~1%，离峰值差两个数量级。这是访存受限的 kernel，FP32 吞吐完全无关 |

### 6.7 V3 报告（原文，`reducesumV3.cu`，grid = 262,144，elemsPerThread = 1）

⚠ 用 `elems=1` 采集，是为了和 6.2 的 V1 报告**同 grid（262,144）**、同数据量，做到单变量对比；
且这样能让「块内归约」在总时间里的占比最大，差异才看得出来。

```
  reduceSumWarp(const float *, float *, int) (262144, 1, 1)x(256, 1, 1), Context 1, Stream 7, Device 0, CC 8.9
    Section: GPU Speed Of Light Throughput
    ----------------------- ------------- ------------
    Metric Name               Metric Unit Metric Value
    ----------------------- ------------- ------------
    DRAM Frequency          cycle/nsecond         8.02
    SM Frequency            cycle/nsecond         1.55
    Elapsed Cycles                  cycle    2,789,148
    Memory Throughput                   %        58.34
    DRAM Throughput                     %        58.34
    Duration                      msecond         1.80
    L1/TEX Cache Throughput             %        55.97
    L2 Cache Throughput                 %        20.28
    SM Active Cycles                cycle 2,771,154.71
    Compute (SM) Throughput             %        55.70
    ----------------------- ------------- ------------

    INF   The kernel is utilizing greater than 80.0% ...（未出现）

    Section: Memory Workload Analysis
    --------------------------- ------------ ------------
    Metric Name                  Metric Unit Metric Value
    --------------------------- ------------ ------------
    Memory Throughput           Gbyte/second       149.78
    Mem Busy                               %        29.14
    Max Bandwidth                          %        58.34
    L1/TEX Hit Rate                        %         1.64
    L2 Compression Success Rate            %            0
    L2 Compression Ratio                                0
    L2 Hit Rate                            %         3.01
    Mem Pipes Busy                         %        55.70
    --------------------------- ------------ ------------

    WRN   The memory access pattern for stores from L1TEX to L2 is not optimal ...
          this kernel only accesses an average of 1.0 sectors out of the possible 4 sectors per cache line.

    Section: Scheduler Statistics
    ---------------------------- ----------- ------------
    Metric Name                  Metric Unit Metric Value
    ---------------------------- ----------- ------------
    One or More Eligible                   %        30.29
    Issued Warp Per Scheduler                        0.30
    No Eligible                            %        69.71
    Active Warps Per Scheduler          warp         8.72
    Eligible Warps Per Scheduler        warp         0.49
    ---------------------------- ----------- ------------

    Section: Warp State Statistics
    ---------------------------------------- ----------- ------------
    Metric Name                              Metric Unit Metric Value
    ---------------------------------------- ----------- ------------
    Warp Cycles Per Issued Instruction             cycle        28.80
    Warp Cycles Per Executed Instruction           cycle        28.80
    Avg. Active Threads Per Warp                                31.45
    Avg. Not Predicated Off Threads Per Warp                    25.16
    ---------------------------------------- ----------- ------------

    Section: Instruction Statistics
    ---------------------------------------- ----------- ------------
    Metric Name                              Metric Unit Metric Value
    ---------------------------------------- ----------- ------------
    Avg. Executed Instructions Per Scheduler        inst   819,200.00
    Executed Instructions                           inst   78,643,200
    ---------------------------------------- ----------- ------------

    Section: Launch Statistics
    -------------------------------- --------------- ---------------
    Metric Name                          Metric Unit    Metric Value
    -------------------------------- --------------- ---------------
    Block Size                                                   256
    Grid Size                                                262,144
    Registers Per Thread             register/thread              18
    Shared Memory Configuration Size           Kbyte           32.77
    Driver Shared Memory Per Block       Kbyte/block            1.02
    Dynamic Shared Memory Per Block      byte/block               0     ← V2 是 1,024
    Static Shared Memory Per Block       byte/block             128     ← warpSums[32]
    Threads                                   thread      67,108,864
    Waves Per SM                                            1,820.44
    -------------------------------- --------------- ---------------

    Section: Occupancy
    ------------------------------- ----------- ------------
    Metric Name                     Metric Unit Metric Value
    ------------------------------- ----------- ------------
    Theoretical Occupancy                     %          100
    Achieved Occupancy                        %        72.12
    Achieved Active Warps Per SM           warp        34.62
    ------------------------------- ----------- ------------

    Section: Source Counters
    ------------------------- ----------- ------------
    Metric Name               Metric Unit Metric Value
    ------------------------- ----------- ------------
    Branch Instructions Ratio           %         0.11
    Branch Instructions              inst    8,912,896
    Branch Efficiency                   %          100
    Avg. Divergent Branches                          0
    ------------------------- ----------- ------------
```

### 6.8 V2 vs V3 逐项对比（均为 `elems=1`、grid = 262,144）

| 指标 | V2（共享内存） | V3（shuffle） | 变化 | 解读 |
|---|---|---|---|---|
| DRAM Throughput | 52.01% | **58.34%** | ↑ 12.2% | DRAM 喂得更饱了 |
| Memory Throughput (GB/s) | 133.18 | **149.78** | ↑ 12.5% | |
| Elapsed Cycles | 3,129,540 | **2,789,148** | ↓ 10.9% | |
| Mem Pipes Busy | 64.23% | 55.70% | ↓ 13.3% | ⚠ 只降 13%，不是数量级 → 见下方说明 |
| L1/TEX Cache Throughput | 64.28% | 55.97% | ↓ 12.9% | |
| Compute (SM) Throughput | 64.23% | **55.70%** | ↓ 13.3% | V2 的「天花板」被抬掉了 |
| **Executed Instructions** | 104,071,168 | **78,643,200** | **↓ 24.4%** | ⭐ 最直接的收益来源 |
| **Branch Instructions** | 17,039,360 | **8,912,896** | **↓ 47.7%** | 循环+条件+栅栏 → 5 条无分支语句 |
| **Dynamic Shared Memory / block** | 1,024 B | **0 B** | **−100%** | 只剩 128 B 静态共享内存 |
| Warp Cycles Per Issued Instruction | 24.67 | 28.80 | ↑ 16.7% | 又一次「看起来变差」（见 6.5 的判读准则） |
| Eligible Warps Per Scheduler | 0.64 | 0.49 | ↓ | 同上 |
| Achieved Occupancy | 72.46% | 72.12% | ≈ | |
| Active Warps Per Scheduler | 8.58 | 8.72 | ≈ | |
| Grid Size / Waves Per SM | 262,144 / 1,820.44 | 262,144 / 1,820.44 | — | 同（同 grid、同数据量） |
| L2 Hit Rate | 3.14% | 3.01% | ≈ | 都是纯流式 |
| Branch Efficiency | 100% | 100% | — | 两者都无发散 |
| Registers Per Thread | 16 | 18 | +2 | shuffle 需要额外的临时寄存器 |

**为什么 `Mem Pipes Busy` 只降 13%？**

因为 **warp shuffle 执行在 MIO（内存输入输出）流水线上 —— 和共享内存是同一个功能单元**。

```
V2 的块内归约（每 block，256 线程）：
    8 次共享内存 store（装载）
  + 约 36 次 wavefront（8 轮树形，每轮 read+read+write）
  = 约 44 个 L1TEX/MIO wavefront

V3 的块内归约（每 block）：
    8 warps × 5 次 shuffle        = 40 次 MIO 操作
  + 8 次共享内存 store + 1 次 load =  9 次
  ≈ 49 次 MIO 操作
```

**数量级相当！** 所以 shuffle 并没有「绕开」访存单元，它只是把
「读+写+读+写」换成了「一次交换」，同时把**栅栏从 3 次降到 1 次**、
把**指令数降了 24%**。真正的收益来自这三项，而不是「换了一条更快的路」。

⚠ 这是本算子最容易讲错的一点，值得单独记住。

---

## 7. 结果笔记

### 7.1 最终数据

**（a）V2 的 `elemsPerThread` 扫描**（同一二进制，6 轮交叉测量，N = 2²⁶，blockSize = 256）：

| 版本 | elemsPerThread | grid | min time | max bw | 峰值占比 | 相对 V1 |
|---|---|---|---|---|---|---|
| V1 等效 | 1 | 262,144 | 1.7930 ms | 149.7 GB/s | 58.4% | 1.00× |
| V2 | 2 | 131,072 | 1.1684 ms | 229.7 GB/s | 89.6% | 1.53× |
| **V2** | **4** | 65,536 | **1.0803 ms** | **248.5 GB/s** | **97.0%** | **1.66×** |
| V2 | 8 | 32,768 | 1.0783 ms | 248.9 GB/s | 97.1% | 1.66× |

**（b）V2 vs V3**（两个二进制，8 轮**正反交替**交叉测量）：

| 版本 | elems=1 | elems=4 |
|---|---|---|
| V2（共享内存，3 次栅栏） | 152.9 GB/s（59.7%） | **248.0 GB/s（96.8%）** |
| V3（shuffle，1 次栅栏） | **177.2 GB/s（69.1%）** | 247.8 GB/s（96.7%） |
| 差异 | **+15.9%** | **≈ 0** |

**（c）ncu 交叉验证**：

| 版本 | DRAM Throughput | Memory Throughput | Mem Pipes Busy |
|---|---|---|---|
| V1 / V2（elems=1） | 52.01% | 133.18 GB/s | 64.23% |
| V3（elems=1） | 58.34% | 149.78 GB/s | 55.70% |
| V2（elems=4） | **96.51%** | **245.80 GB/s** | 37.68% |

程序内计时（248.5 GB/s）与 ncu（245.80 GB/s）吻合。

对照 `vector_add`：245.8 GB/s / 95.9% —— **归约已经和纯流式算子打平**。

### 7.2 结论

1. **归约的"带宽"天然低于 `vector_add` 吗？不，是我们写错了并行粒度。** 修好之后两者一样快。
2. **瓶颈定位的正确顺序**：先确认正确性（verify PASS）→ 再排除内存布局/精度问题 → 用 ncu 区分"DRAM 受限"和"SM 受限" → 找到"开销 ∝ block 数"这个结构性根因 → 改并行粒度。
3. **`elemsPerThread = 4` 是甜点**。`2` 不够（89.6%），`4` 到位（97.0%），`8` 无增益。
4. **`blockSize` 在 128~256 之间是二阶变量**；真正的开关是 `elemsPerThread`。
5. **"block 越小越好"是错的**（4.2 的 1024 反例）—— 关键是"**很多个独立的小 block，每个 block 多搬数据**"。
6. **V1 的归约逻辑一行都没改**。V1 的错不在共享内存树、不在 `__syncthreads` 位置、不在 `volatile` —— 那些设计都是对的。错的只是**每线程只处理 1 个元素**。
7. **优化的收益取决于瓶颈在哪里**。V3 比 V2 快 15.9%，但只在 `elems=1`（块内归约是瓶颈）时；`elems=4` 时两者打平（DRAM 是瓶颈）。**先定位瓶颈，再选优化手段；反过来会误判「某个技术没用」。**
8. **`__shfl_down_sync` 不是「免费」的**。它和共享内存共用 MIO 流水线，`Mem Pipes Busy` 只降 13%。它的收益来自「操作次数变少」（指令 −24.4%、分支 −47.7%、栅栏 3→1、动态共享内存 1,024 B→0）。
9. **测量方法本身会影响结论**。交叉测量还不够，顺序必须**正反交替**，否则「位置偏置」会伪装成「配置差异」（`elems=8` 曾因此被系统性测成 217 而非 248）。

### 7.3 已知遗留问题

| # | 问题 | 影响 | 优先级 |
|---|---|---|---|
| 1 | `return 0` 不反映 `verify` 结果 | 自动化脚本无法通过退出码发现"结果不对" | 高（一行可修） |
| 2 | `blockSize` 没有校验：V1/V2 需 ≥64、32 的倍数、≤1024；V3 需 ≥32 且是 32 的倍数 | 传错参数时 kernel 静默失败或结果错 | 中 |
| 3 | `out` 的 store 未合并（ncu 有警告） | 约 +2.7% 多余流量 | 低 |
| 4 | `atoi` 不做错误检查 | `argv` 传非数字会变成 0 | 低 |
| 5 | 绝对容差 `1e-3` 只在 N=2²⁶ 时合理 | 换 N 会误报 | 低（应改相对误差） |

---

## 8. 后续方向

按"教学价值 / 性能收益"排序：

| # | 方向 | 状态 | 实际结果 / 说明 |
|---|---|---|---|
| 1 | ~~换 `__shfl_down_sync`~~ | ✅ **已完成（V3）** | `elems=1` 时 +15.9%；`elems=4` 时与 V2 打平 —— 见 4.6 / 5.5 |
| 2 | ~~用 ncu 对比 shuffle 版的 `Mem Pipes Busy`~~ | ✅ **已完成** | 64.23% → 55.70%（只降 13%，因为共用 MIO 流水线）—— 见 6.8 |
| 3 | 改成求 `max` / `min` | ⬜ 待做 | ≈0 性能收益，但能固化「单位元」概念（只需改 3 处） |
| 4 | `float4` 向量化（一次读 16 字节） | ⬜ 待做 | 预期收益小（已 97%）；主要用于学习「向量化访存」 |
| 5 | 第二个 kernel 归约 `out`（替代 host 端累加） | ⬜ 待做 | 预期 0 或负（grid 只有 65,536，再启一个 kernel 不划算） |
| 6 | 多 block 合并写部分和（修 ncu 的 store 警告） | ⬜ 待做 | ~1%，需要让一个 warp 一次写 32 个 block 的结果 |
| 7 | 把 `elems=1` 的 V3 当作「归约开销占比」的教学基线 | 💡 建议 | 目前最有教学价值的一档配置 |

**性能上 V1→V2 已经到顶（97% DRAM 峰值）；V1→V3 把块内归约的开销也降下来了。**
**剩下的方向性能收益都在 1% 量级，价值主要在概念固化（单位元、向量化、两级归约的取舍）。**

---

## 附录：完整实验命令速查

```bash
# 构建
cmake -S . -B build && cmake --build build --target reducesum reducesumV2 reducesumV3

# 单次运行
./build/bin/reducesum   $((1<<26)) 256        # V1
./build/bin/reducesumV2 $((1<<26)) 256 4      # V2
./build/bin/reducesumV3 $((1<<26)) 256 4      # V3
./build/bin/reducesumV3 $((1<<26)) 256 1      # V3 在「块内归约占比最大」的一档

# 交叉扫描 elemsPerThread（推荐做法）
for r in 1 2 3 4 5 6; do
  for e in 1 2 4 8; do
    ./build/bin/reducesumV2 $((1<<26)) 256 $e | awk -v e=$e '
      /best time/{t=$(NF-1)+0} /bandwidth/{printf "e%-3s %.4f %s\n", e, t, $(NF-1)}'
  done
done | awk '{k=$1; if(!(k in mt)||$2<mt[k])mt[k]=$2; if(!(k in mb)||$3>mb[k])mb[k]=$3}
            END{for(k in mt) printf "%-5s min_time=%.4f ms   max_bw=%6.1f GB/s\n", k, mt[k], mb[k]}' | sort

# ncu
ncu --set full -k reduceSum     --launch-count 1 ./build/bin/reducesum   $((1<<26)) 256
ncu --set full -k reduceSum     --launch-count 1 ./build/bin/reducesumV2 $((1<<26)) 256 4
ncu --set full -k reduceSumWarp --launch-count 1 ./build/bin/reducesumV3 $((1<<26)) 256 1

# V2 vs V3 正反交替交叉对比
seq_odd="V2:1 V3:1 V2:4 V3:4"; seq_even="V3:4 V2:4 V3:1 V2:1"
for r in 1 2 3 4 5 6 7 8; do
  if [ $((r%2)) -eq 1 ]; then s="$seq_odd"; else s="$seq_even"; fi
  for item in $s; do
    prog=${item%:*}; e=${item#*:}
    case $prog in V2) bin=reducesumV2;; V3) bin=reducesumV3;; esac
    ./build/bin/$bin $((1<<26)) 256 $e | awk -v k="$prog-e$e" '/bandwidth/{print k, $(NF-1)}'
  done
done | awk '{k=$1; if(!(k in m)||$2>m[k])m[k]=$2} END{for(k in m) printf "%-8s max_bw=%6.1f GB/s\n", k, m[k]}' | sort
```

最后附三句从本算子沉淀下来的话：

> **工具给的是通用建议，不是针对你的 —— 必须用自己的物理常识过滤。**（ncu 说 V1「计算和存储很均衡」，真相是 DRAM 被饿着）

> **优化的收益取决于瓶颈在哪里。** 先定位瓶颈，再选手段 —— 反过来会把「有效的技术」误判成「没用」。

> **测量方法本身会影响结论。** 交叉测量还不够，顺序必须正反交替。
