# GEMM/ 章节的"分组"CMakeLists.txt（工程级设置仍由顶层负责）
#
# 目录结构：
#   GEMM/transpose/   0 号过渡练习：共享内存分块转置
#   （后续 GEMM V1/V2/V3 等练习也会挂在这一层）
add_subdirectory(transpose)
add_subdirectory(GEMMkernel)
