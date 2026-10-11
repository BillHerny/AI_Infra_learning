add_executable(reducesum reducesum.cu)
set_target_properties(reducesum PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON)

add_executable(reducesumV2 reducesumV2.cu)
set_target_properties(reducesumV2 PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON)

add_executable(reducesumV3 reducesumV3.cu)
set_target_properties(reducesumV3 PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON)

add_executable(reducemax reducemax.cu)
set_target_properties(reducemax PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON)

add_executable(argmax argmax.cu)
set_target_properties(argmax PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON)