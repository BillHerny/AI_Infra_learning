cmake_minimum_required(VERSION 3.20)
project(hello_cuda LANGUAGES CXX CUDA)

set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

add_executable(hello_cuda hello.cu)
set_target_properties(hello_cuda PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON
)

#[===[
========================================================================
逐行详细说明（按上方源代码顺序）
========================================================================

【第 1 行】cmake_minimum_required(VERSION 3.20)

  · 作用：声明本工程所需的 CMake 最低版本为 3.20。
  · 为什么是 3.20：版本号不仅决定"能否构建"，还会切换 CMake 的
    策略（Policy）默认行为。例如 CUDA 作为一等语言（LANGUAGES CUDA）、
    较新的 CUDA 架构检测逻辑等都依赖较新版本，写低了可能行为不一致。
  · 若本机 CMake 低于 3.20，配置阶段会直接报错并终止，而不是
    产生难以排查的构建错误——这是一种"尽早失败"的保护。

【第 2 行】project(hello_cuda LANGUAGES CXX CUDA)

  · hello_cuda：工程名，会写入 ${PROJECT_NAME} 等变量，
    并在默认情况下决定构建产物名（此处被 add_executable 显式覆盖）。
  · LANGUAGES CXX CUDA：启用两种源码语言——
      - CXX  → 由 C++ 编译器（g++/clang++）编译 .cpp 文件；
      - CUDA → 由 NVIDIA 的 nvcc 驱动编译 .cu 文件，
               程序的主机端代码交给底层 C++ 编译器，设备端代码
               编译成 GPU 可执行的 PTX/cubin。
  · 关键点：CMake ≥ 3.8 原生支持 CUDA 语言，因此无需再手写
    find_package(CUDA) 或自定义 nvcc 规则；CMake 会自动定位 nvcc
    并注入 cudart 运行时库。

【第 3 行】set(CMAKE_CUDA_STANDARD 17)

  · 指定编译 .cu 文件时所使用的 C++ 语言标准为 C++17。
  · 该变量是"目录级默认值"，影响本目录及子目录中之后创建的目标。
  · 与之对应的还有 CMAKE_CXX_STANDARD（用于 .cpp）。单独设置
    CMAKE_CUDA_STANDARD 是因为 CUDA 文件走的是 nvcc，
    CMake 会把它翻译成 nvcc 的 -std=c++17（具体旗标随 CUDA 版本而异）。
  · 为什么用 17：C++17 提供了结构化绑定、if constexpr、std::optional
    等特性，同时被主流 CUDA 工具链良好支持，兼顾现代性与兼容性。

【第 4 行】set(CMAKE_CUDA_STANDARD_REQUIRED ON)

  · 含义：强制要求编译器必须支持第 3 行指定的标准。
  · 若为 OFF（默认），当 nvcc 不支持 C++17 时，CMake 会"静默降级"
    到较低标准，导致你写的 C++17 代码编译失败且原因隐蔽。
  · 若为 ON，则在配置/编译阶段就能得到明确报错，便于定位环境问题。

【第 5 行】add_executable(hello_cuda hello.cu)

  · 创建一个名为 hello_cuda 的可执行目标（build target）。
  · 源文件只有 hello.cu 一个。CMake 会依据扩展名 .cu 自动选用
    CUDA 编译流程（主机端 + 设备端），并自动完成两部分的链接。
  · 由于工程启用了 CUDA 语言，CMake 还会自动把 CUDA 运行时库
    （cudart）链接进该目标，无需手写 target_link_libraries(... cudart)。

【第 6-8 行】set_target_properties(hello_cuda PROPERTIES
              CUDA_SEPARABLE_COMPILATION ON
          )

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
  · 第 8 行的 ")" 用于闭合第 6 行开启的参数列表。

========================================================================
补充：构建与运行
========================================================================

  cmake -S . -B build        # 配置：生成构建系统（本例为 Makefile）
  cmake --build build        # 构建：编译并链接出 ./build/hello_cuda
  ./build/hello_cuda         # 运行：打印 GPU 信息与 8 行 Hello

========================================================================
补充：关于本注释块语法
========================================================================

  · 本文件末尾使用的是 CMake 的"括号块注释"：以 #[[ 开始、以两个
    右方括号结束，中间内容原样保留、无需逐行加 "#"，适合放说明文档。
  · 关键陷阱：普通 #[[ ... ]] 注释在遇到第一个连续的两个右方括号时
    就会结束，且不支持嵌套；因此正文里不能直接书写这两个字符。
    否则注释会提前闭合，其后的文字会被当成 CMake 命令解析并报错：
    "Expected a command name, got quoted argument ..."。
  · 解决办法：改用"带等号"的定界符——在开头的两个方括号之间插入
    若干等号（本注释块用的是三个），结尾用相同数量的等号对称闭合。
    改为该形式后，正文即可安全地书写 ]] 这样的字符而不提前闭合。

]===]
