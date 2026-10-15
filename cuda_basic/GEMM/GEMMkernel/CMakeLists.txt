add_executable(GEMMkernel GEMMkernel.cu)
set_target_properties(GEMMkernel PROPERTIES
        CUDA_SEPARABLE_COMPILATION ON)