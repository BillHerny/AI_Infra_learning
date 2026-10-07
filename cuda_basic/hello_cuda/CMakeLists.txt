add_executable(hello_cuda hello.cu)
set_target_properties(hello_cuda PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON
)

#[===[
========================================================================
本文件：hello_cuda 练习（子目录 / 目标级）
========================================================================

  这是 cuda_basic 工程下的一个练习子目录。按约定，这里只写"目标级"
  命令：全局设置（C++ 标准、CUDA 架构、输出目录等）都放在顶层
  cuda_basic/CMakeLists.txt 里，并通过 add_subdirectory(hello_cuda)
  自动向下传递，本文件无需重复。

  也就是说，本文件里"没有" cmake_minimum_required、project、
  set(CMAKE_CUDA_*) 这些工程级语句——它们属于顶层，不属于这里。

------------------------------------------------------------------------
【第 1 行】add_executable(hello_cuda hello.cu)
------------------------------------------------------------------------

  · 创建一个名为 hello_cuda 的可执行目标（build target）。
  · 源文件只有 hello.cu 一个。CMake 会依据扩展名 .cu 自动选用
    CUDA 编译流程（主机端 + 设备端），并自动完成两部分的链接。
  · 由于顶层工程已启用 CUDA 语言，CMake 会自动把 CUDA 运行时库
    （cudart）链接进该目标，无需手写 target_link_libraries(... cudart)。
  · 产物路径：因顶层设置了 CMAKE_RUNTIME_OUTPUT_DIRECTORY，最终
    可执行文件落在 build/hello_cuda（即 ./build/hello_cuda）。

------------------------------------------------------------------------
【第 2-4 行】set_target_properties(hello_cuda PROPERTIES
                  CUDA_SEPARABLE_COMPILATION ON)
------------------------------------------------------------------------

  · set_target_properties：为一个已存在的目标批量设置属性。
    第一个参数是目标名 hello_cuda，随后按 "属性名 值" 成对书写。
  · CUDA_SEPARABLE_COMPILATION ON（可分离编译，也称 RDC，
    Relocatable Device Code）：
      - OFF（默认）：nvcc 把主机端与设备端代码一次性编译并就地链接，
        跨翻译单元的 __device__ / __global__ 调用无法解析；
      - ON：将设备代码编译为"可重定位"的中间产物，先各自编译，
        再由设备链接器（nvcc -dlink）统一链接。
  · 对本例（单文件、单 kernel、无跨文件设备调用）其实并非必需，
    但作为可扩展的工程习惯保留：一旦将来拆分为多个 .cu 且互相调用
    设备函数，就必须开启，否则会报 "undefined reference" 之类的
    设备符号链接错误。
  · 代价：开启后会略微增加构建时间与产物体积。
  · 第 4 行的 ")" 用于闭合第 2 行开启的参数列表。

------------------------------------------------------------------------
【为什么这里不写 project()】
------------------------------------------------------------------------

  · project() 属于"工程级"，一个工程只在顶层写一次。子目录若再写
    project()，会在当前作用域里重置工程名等变量，属于多余且容易
    混淆的做法。
  · 同理，CMAKE_CUDA_STANDARD / CMAKE_CUDA_ARCHITECTURES 等全局设置
    也只在顶层写一次，子目录自动继承——这正是"工程级 / 目标级"
    分层的意义：公共设置集中一处，各练习只声明自己的目标。

  · 想新增练习时，照抄本文件的骨架即可（改成自己的目标名与源文件名）。

]===]
