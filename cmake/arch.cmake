if (NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
  set(CMAKE_CUDA_ARCHITECTURES native)
endif()

# # Build TORCH_CUDA_ARCH_LIST
# set(TORCH_CUDA_ARCH_LIST "")
# foreach(CUDA_ARCH IN LISTS CMAKE_CUDA_ARCHITECTURES)
#   if(CUDA_ARCH MATCHES "^([1-9][0-9]*)([0-9]a?)$")
#     set(TORCH_ARCH "${CMAKE_MATCH_1}.${CMAKE_MATCH_2}")
#   elseif(CUDA_ARCH STREQUAL "native")
#     set(TORCH_ARCH "Auto")
#   else()
#     message(FATAL_ERROR "${CUDA_ARCH} is not supported")
#   endif()
#   list(APPEND TORCH_CUDA_ARCH_LIST ${TORCH_ARCH})
# endforeach()

# message(STATUS "CMAKE_CUDA_ARCHITECTURES: ${CMAKE_CUDA_ARCHITECTURES}")
# message(STATUS "TORCH_CUDA_ARCH_LIST: ${TORCH_CUDA_ARCH_LIST}")
# message(STATUS "TORCH_CXX_FLAGS: ${TORCH_CXX_FLAGS}")

# add_compile_options(${TORCH_CXX_FLAGS})
# add_compile_definitions(TORCH_CUDA=1)

