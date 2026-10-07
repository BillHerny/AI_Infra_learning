add_executable(vector_add vector_add.cu)
set_target_properties(vector_add PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON)

add_executable(my_vector_add my_vector_add.cu)
set_target_properties(my_vector_add PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON)