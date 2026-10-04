#include <cstdio>
#include <cuda_runtime.h>

__global__ void hello_kernel() {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    printf("Hello from thread %d (block %d, thread %d)\n",
           idx, blockIdx.x, threadIdx.x);
}

int main() {
    int dev_count = 0;
    cudaGetDeviceCount(&dev_count);
    printf("CUDA device count: %d\n", dev_count);

    for (int i = 0; i < dev_count; i++) {
        cudaDeviceProp prop;
        cudaGetDeviceProperties(&prop, i);
        printf("Device %d: %s, SM count: %d, global mem: %.2f GB\n",
               i, prop.name, prop.multiProcessorCount,
               prop.totalGlobalMem / (1024.0 * 1024.0 * 1024.0));
    }

    hello_kernel<<<2, 4>>>();
    cudaDeviceSynchronize();

    return 0;
}

/*
逐行说明（按上方源代码顺序）：
1. #include <cstdio>：引入 C 标准输入输出库，以便使用 printf。
2. #include <cuda_runtime.h>：引入 CUDA 运行时 API 和相关类型定义。
3. __global__ void hello_kernel() {：声明一个可从 CPU 调用、并在 GPU 上执行的 CUDA kernel。
4. int idx = blockIdx.x * blockDim.x + threadIdx.x;：根据 block 编号、每个 block 的线程数和线程编号计算全局线程编号。
5. printf("Hello from thread %d (block %d, thread %d)\n",：设置每个 GPU 线程输出的格式；%d 是整数占位符。
6. idx, blockIdx.x, threadIdx.x);：分别提供全局线程编号、block 编号和 block 内线程编号。
7. }：结束 hello_kernel 函数。
8. int main() {：定义程序入口，程序从这里开始执行。
9. int dev_count = 0;：声明并初始化变量，用来保存 CUDA 设备数量。
10. cudaGetDeviceCount(&dev_count);：查询可用的 CUDA 设备数量，并写入 dev_count。
11. printf("CUDA device count: %d\n", dev_count);：输出 CUDA 设备数量。
12. for (int i = 0; i < dev_count; i++) {：循环处理编号从 0 到 dev_count - 1 的设备。
13. cudaDeviceProp prop;：声明用于保存一张 CUDA 设备属性的结构体变量。
14. cudaGetDeviceProperties(&prop, i);：查询编号为 i 的设备属性并保存到 prop。
15. printf("Device %d: %s, SM count: %d, global mem: %.2f GB\n",：设置设备信息的输出格式；%.2f 表示小数点后显示两位。
16. i, prop.name, prop.multiProcessorCount,：传入设备编号、设备名称和 SM（流式多处理器）数量。
17. prop.totalGlobalMem / (1024.0 * 1024.0 * 1024.0));：将显存字节数换算为 GiB，并作为最后一个格式参数传入。
18. }：结束遍历设备的 for 循环。
19. hello_kernel<<<2, 4>>>();：启动 kernel，配置为 2 个 block、每个 block 4 个线程，共 8 个线程。
20. cudaDeviceSynchronize();：等待此前提交到当前设备的工作完成，再继续执行 CPU 端代码。
21. return 0;：从 main 返回 0，表示程序正常结束。
22. }：结束 main 函数。
*/