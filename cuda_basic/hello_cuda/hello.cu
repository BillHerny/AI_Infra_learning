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
CUDA 入门逐行讲解（对应上方源码）：

1. #include <cstdio>
   - 引入 C 标准输入输出库。
   - 本程序使用其中的 printf() 在终端显示设备信息和 GPU 线程输出。

2. #include <cuda_runtime.h>
   - 引入 CUDA Runtime API 的声明。
   - 设备查询函数、设备属性类型和 kernel 启动所需的定义都由 CUDA 运行时提供。

3. __global__ void hello_kernel() {
   - 定义一个 CUDA kernel，也就是由 GPU 执行的函数。
   - __global__ 表示 kernel 可从 CPU 端启动，并在 GPU 上运行。
   - 启动 kernel 后，GPU 会按照启动配置创建多个线程执行这个函数。

4. int idx = blockIdx.x * blockDim.x + threadIdx.x;
   - 为当前线程计算一个全局编号。
   - blockIdx.x 是当前线程所在 block 的编号。
   - blockDim.x 是每个 block 在 x 方向上的线程数。
   - threadIdx.x 是当前线程在所属 block 中的编号。
   - 将 block 编号乘以每个 block 的线程数，再加上线程在 block 内的编号，
     就得到当前线程在线性线程序列中的位置。

5. printf("Hello from thread %d (block %d, thread %d)\n",
   - 开始调用 printf()，并给出输出格式。
   - 每个 %d 都是一个整数占位符，将由后续参数依次替换。
   - \n 表示换行。设备端 printf 的输出顺序不保证与线程编号一致。

6. idx, blockIdx.x, threadIdx.x);
   - 依次提供要填入格式字符串的三个整数。
   - 它们分别表示线程的全局编号、所属 block 编号和 block 内线程编号。

7. }
   - 结束 hello_kernel() 函数定义。

8. int main() {
   - 定义 C++ 程序入口。
   - main() 中的语句由 CPU 执行；其中可以调用 CUDA Runtime API 来管理 GPU 工作。

9. int dev_count = 0;
   - 定义并初始化整数变量 dev_count。
   - 后续用它保存系统中可用的 CUDA 设备数量。

10. cudaGetDeviceCount(&dev_count);
    - 请求 CUDA 运行时查询可用的 CUDA 设备数量。
    - &dev_count 是变量的地址，函数通过这个地址写入查询结果。
    - 本示例没有检查函数返回的错误码；实际程序中应检查 CUDA API 调用是否成功。

11. printf("CUDA device count: %d\n", dev_count);
    - 在终端输出查询到的设备数量。
    - 格式字符串中的 %d 对应整数参数 dev_count。

12. for (int i = 0; i < dev_count; i++) {
    - 从设备编号 0 开始，逐个遍历所有检测到的 CUDA 设备。
    - 每次循环结束后 i 加 1；当 i 不再小于设备数量时，循环结束。

13. cudaDeviceProp prop;
    - 声明一个设备属性结构体变量 prop。
    - 该结构体可保存设备名称、SM 数量、显存容量等设备信息。

14. cudaGetDeviceProperties(&prop, i);
    - 查询编号为 i 的设备属性，并将结果写入 prop。
    - &prop 是接收属性信息的地址；i 指定要查询的设备。

15. printf("Device %d: %s, SM count: %d, global mem: %.2f GB\n",
    - 设置设备信息的输出格式。
    - %d 用于整数，%s 用于字符串，%.2f 用于保留两位小数的浮点数。
    - 后续参数必须按占位符出现的顺序提供。

16. i, prop.name, prop.multiProcessorCount,
    - 依次提供设备编号、设备名称和 SM 数量。
    - prop.name 是设备名称；multiProcessorCount 是设备的流式多处理器（SM）数量。

17. prop.totalGlobalMem / (1024.0 * 1024.0 * 1024.0));
    - 将设备的全局显存容量从字节换算为 GiB，并作为最后一个 printf() 参数。
    - totalGlobalMem 的单位是字节；除以 1024 的三次方后得到 GiB 数值。
    - 使用浮点数参与除法，避免整数除法丢弃小数部分。

18. }
    - 结束设备遍历循环的代码块。

19. hello_kernel<<<2, 4>>>();
    - 使用 CUDA 的执行配置语法启动 hello_kernel。
    - 第一个配置值 2 表示启动 2 个 block，第二个值 4 表示每个 block 启动 4 个线程。
    - 因此本次启动共创建 2 × 4 = 8 个线程，每个线程都会执行一次 kernel。

20. cudaDeviceSynchronize();
    - 等待当前设备上此前提交的 GPU 工作完成。
    - kernel 启动通常是异步的：CPU 发出启动请求后可以继续执行，而 GPU 可能仍在工作。
    - 同步后，CPU 才会继续执行下一条语句。
    - 本示例未检查同步调用的返回值；实际程序中应检查错误。

21. return 0;
    - 从 main() 返回 0，表示程序正常结束。

22. }
    - 结束 main() 函数，程序执行到此结束。
*/