# 用 Nsight 诊断 CUDA 程序

## 先分清两个工具

| 工具 | 命令 | 粒度 | 解决什么问题 |
|---|---|---|---|
| **Nsight Systems** | `nsys` | 整个程序的时间线 | CPU/GPU 谁在等谁、kernel 占用多少时间、H2D/D2H 拷贝占比、同步开销 |
| **Nsight Compute** | `ncu` | 单个 kernel 的深度指标 | 该 kernel 是 memory-bound 还是 compute-bound、访存是否合并、occupancy 够不够、warp 为什么 stall |

诊断流程一般是**先 `nsys` 定位热点 kernel，再 `ncu` 深挖那一个 kernel**。

先确认装没装：

```bash
which nsys ncu
ncu --version
```

---

## 0. 编译准备（关键）

编译命令需要带 `-lineinfo`，这正好是分析所需的：

```bash
nvcc -g -lineinfo hello.cu -o hello
```

- `-lineinfo`：把 SASS 指令映射回 `.cu` 源码行，**性能分析必需**，对性能几乎无影响
- `-G`：完整调试信息，会关掉大部分优化，只在需要逐行调试时用
- 注意：分析时**不要**用过高的优化把 kernel 消掉，但正常 `-O3` 是要保留的，否则测的不是真实性能

CMake 里对应写法：

```cmake
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -lineinfo -O3")
```

---

## 1. Nsight Systems：整体时间线

```bash
# 采集 + 直接打印统计摘要
nsys profile -t cuda,nvtx,osrt --stats=true -o /tmp/hello_trace ./hello
```

`--stats=true` 会在采集结束后自动跑 `nsys stats`，直接看到：

- **CUDA API Summary**：`cudaMalloc` / `cudaMemcpy` / `cudaDeviceSynchronize` 各花了多久
- **GPU Kernel Summary**：每个 kernel 的次数、总耗时、平均耗时 ← **先看这张表找热点**
- **Memory Operation Summary**：拷贝吞吐

只想要统计不要 GUI：

```bash
nsys stats --report cuda_gpu_kern_sum /tmp/hello_trace.nsys-rep
```

`reduction/` 下那堆 `reducesum`、`reducesumV2`、`reducesumV3` 特别适合这么对比——同一问题不同实现的 kernel 耗时差距一目了然：

```bash
nsys profile -t cuda --stats=true -o /tmp/rs3 ./build/bin/reducesumV3
```

**`nvtx` 是分析时间线的好帮手**：在代码里打范围标记，时间线上就能看到"哪段是准备、哪段是计算"：

```cpp
#include <nvtx3/nvToolsExt.h>
nvtxRangePush("upload");
cudaMemcpy(...);
nvtxRangePop();
```

---

## 2. Nsight Compute：单 kernel 深度分析

```bash
# 全量采集（慢，但信息最全）
ncu --set full -o /tmp/hello_prof ./hello

# 只抓指定 kernel（推荐，先定位再精修）
ncu --kernel-name regex:reduce --launch-count 1 --set full -o /tmp/rs_prof ./build/bin/reducesum

# 快照式，直接终端输出
ncu --set full --print-summary per-kernel ./hello
```

### 最有用的几个 section

```bash
ncu --section SpeedOfLight \
    --section MemoryWorkloadAnalysis \
    --section Occupancy \
    --section ComputeWorkloadAnalysis \
    ./hello
```

- **Speed of Light**：Compute 和 Memory 各自达到峰值的百分比。谁先到 100% 谁就是瓶颈
- **Memory Workload Analysis**：L1/L2 命中率、**sector 利用率**（<100% 说明访存没合并，跨步访问）
- **Occupancy**：理论 vs 实际占用率，以及"被什么限制"（寄存器 / 共享内存 / block 大小）
- **Compute Workload Analysis**：ALU/FMA/LSU 各管线利用率
- **Warp State Statistics**：每个 warp 停在哪（`long scoreboard` = 等全局内存，`barrier` = 等 `__syncthreads()`，`MIO throttle` = 共享内存打满）
- **Source**：配合 `-lineinfo`，逐行看热点和对应 SASS

### 常用限定参数

```bash
--launch-skip 5 --launch-count 3   # 跳过 warmup，只测稳定期
--kernel-name regex:gemm           # 按 kernel 名过滤
-c 1                               # 等价 launch-count 1
--target-processes all             # 分析多进程
```

---

## 3. 权限坑（最常遇到的报错）

```
ERR_NVGPUCTRPERM: Permission issue with Performance Counters
```

`ncu` 需要访问 GPU 性能计数器。三个解法，任选其一：

```bash
# A. 临时用 root
sudo ncu --set full ./hello

# B. 给 ncu 加 capability（推荐，长期有效）
sudo setcap cap_sys_admin+ep $(which ncu)

# C. 驱动层放开限制（需要重启）
sudo sh -c 'echo "options nvidia NVreg_RestrictProfilingToAdminUsers=0" \
  > /etc/modprobe.d/nvidia-profiling.conf'
sudo reboot
```

顺带：`ncu` 在 WSL / 容器里需要 `--privileged` 或 `NVIDIA_DRIVER_CAPABILITIES=all`。

---

## 4. 远程服务器 + GUI 的正确姿势

VS Code Remote SSH 看不到 Nsight 的图形界面（除非配 X11 转发，很卡）。标准做法是**服务器端出报告、本地看报告**：

```bash
# 服务器上生成报告
nsys profile -t cuda,nvtx --stats=true -o trace ./hello   # → trace.nsys-rep
ncu --set full -o prof ./hello                            # → prof.ncu-rep
```

然后把 `.nsys-rep` / `.ncu-rep` 用 VS Code 的资源管理器**直接拖下来**，在本机装的 Nsight Systems / Nsight Compute 里 `File → Open` 打开。

也可以在本机用 `scp`：

```bash
scp server:/path/to/prof.ncu-rep .
```

> 注意：`.ncu-rep` 需要用**相同或更新版本**的 Nsight Compute 打开；版本不匹配会提示升级。

VS Code 里还有 NVIDIA 官方的 **Nsight Visual Studio Code Edition** 扩展，可以在编辑器内做断点调试和性能分析，但报告阅读体验仍不如原生 GUI。

---

## 5. 一套完整的诊断套路

以本仓库里的 `GEMM` / `transpose` 为例：

```bash
# ① 整体：先看哪个 kernel 最贵、有没有意外的同步或拷贝
nsys profile -t cuda,nvtx --stats=true -o /tmp/t1 ./build/bin/GEMMkernel
#    → 看 CUDA GPU Kernel Summary，记下 top kernel 名（如 gemm_naive）

# ② 深入：只针对热点 kernel 抓全量指标
ncu --kernel-name regex:gemm_naive --launch-skip 2 -c 1 \
    --set full -o /tmp/gemm_naive ./build/bin/GEMMkernel

# ③ 读报告顺序（GUI 里）
#   SpeedOfLight  → Memory 90%+ & Compute 20% ⇒ memory bound，优化 coalescing / shared memory tiling
#                 → Compute 90%+             ⇒ compute bound，考虑 FMA、指令级并行
#                 → 两者都低                ⇒ latency bound，看 Occupancy 和 Warp State
#   MemoryWorkload → Sector 利用率低 ⇒ 跨步/未对齐访存
#   Occupancy      → 看是寄存器还是共享内存卡住
#   Source         → 配合 -lineinfo 定位到具体行

# ④ 改代码后再跑同样命令，用 diff 对比
ncu --set full -o /tmp/gemm_tiled ./build/bin/GEMMkernel
```

GUI 里两个报告可以**并排对比**（`ncu` 的 baseline 功能：右键 Baseline → Compare），直接看到改 tiled 之后 L2 命中率和吞吐提升多少。

---

## 6. 小练习建议

`hello.cu` 只有一个 kernel 且几乎不耗时，`ncu` 抓到的指标意义不大（under-utilized 是正常的）。想练手可以直接上：

```bash
# reduction：对比 V1 / V2 / V3 的 D2H 拷贝次数和 kernel 耗时
nsys profile -t cuda --stats=true -o /tmp/rs1 ./build/bin/reducesum
nsys profile -t cuda --stats=true -o /tmp/rs3 ./build/bin/reducesumV3
```

`reducesumReadme.md` 里如果有讲优化点，用 `ncu` 验证"occupancy 提升了多少、bank conflict 少了多少"会非常有说服力。
