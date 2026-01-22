include(cc_test)

cc_tests(
  g2s_test
  s2r_test
)

cc_test(
  NAME
    gemm_test
  SRCS
    gemm_test.cu
  DEPS
    CUDA::cublas
)
