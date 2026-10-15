add_executable(transpose transpose.cu)
set_target_properties(transpose PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON)
