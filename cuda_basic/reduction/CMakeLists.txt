add_executable(reduction reduction.cu)
set_target_properties(reduction PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON)