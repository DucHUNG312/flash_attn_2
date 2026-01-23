#include <cublas_v2.h>
#include <gtest/gtest.h>
#include <cstdio>
#include <cstdlib>
#include "kernel/gemm_kernel.h"

namespace flash {

// Problem dimensions
static constexpr int kM = 1024;
static constexpr int kN = 1024;
static constexpr int kK = 2048;

// Block dimensions
static constexpr int BLK_M = 64;
static constexpr int BLK_N = 128;
static constexpr int BLK_K = 64;

// Warp layout (8 warps: 4 rows x 2 cols)
using WarpLayout = Layout_t<4, 2>;

using GProblemShape = Gemm_shape_t<kM, kN, kK>;
using SProblemShape = Gemm_shape_t<BLK_M, BLK_N, BLK_K>;

using GemmKeTraits = Gemm_kernel_traits_t<
    half, // Element
    float, // AccElement
    GProblemShape, // Global problem shape
    SProblemShape, // Shared memory tile shape
    WarpLayout, // Warp layout
    Mma_Atom_16x16x16,
    true, // kSwizzle
    128>; // kSwizzleBytes

using GemmKernel = Gemm_kernel_t<GemmKeTraits>;

// Reference GEMM using cuBLAS
void cublas_gemm_reference(
    const half* a,
    const half* b,
    float* c,
    int M,
    int N,
    int K) {
  cublasHandle_t handle;
  cublasCreate(&handle);

  // cuBLAS uses column-major, so we compute C = B^T * A^T to get row-major C
  // Our layout: A [M x K] row-major, B [N x K] row-major (B transposed in MMA)
  // C [M x N] row-major
  // cuBLAS: C_col = alpha * op(A_col) * op(B_col) + beta * C_col
  // For row-major A[M,K], treat as col-major A^T[K,M]
  // For row-major B[N,K], treat as col-major B^T[K,N]
  // We want C[M,N] = A[M,K] * B[N,K]^T = A[M,K] * B^T[K,N]
  // In col-major view: C^T[N,M] = B[N,K] * A[K,M]
  // So call cublas with: C = B * A^T, sizes (N, M, K), leading dims (K, K, N)

  float alpha = 1.0f;
  float beta = 0.0f;

  // Convert half inputs to float for reference computation
  float* a_f32;
  float* b_f32;
  cudaMalloc(&a_f32, M * K * sizeof(float));
  cudaMalloc(&b_f32, N * K * sizeof(float));

  // Copy to host and convert to float
  half* h_a = new half[M * K];
  half* h_b = new half[N * K];
  float* h_a_f32 = new float[M * K];
  float* h_b_f32 = new float[N * K];

  cudaMemcpy(h_a, a, M * K * sizeof(half), cudaMemcpyDeviceToHost);
  cudaMemcpy(h_b, b, N * K * sizeof(half), cudaMemcpyDeviceToHost);

  for (int i = 0; i < M * K; i++) {
    h_a_f32[i] = __half2float(h_a[i]);
  }
  for (int i = 0; i < N * K; i++) {
    h_b_f32[i] = __half2float(h_b[i]);
  }

  cudaMemcpy(a_f32, h_a_f32, M * K * sizeof(float), cudaMemcpyHostToDevice);
  cudaMemcpy(b_f32, h_b_f32, N * K * sizeof(float), cudaMemcpyHostToDevice);

  // C[M,N] = A[M,K] * B^T[K,N]
  // In column-major (cuBLAS): C_col[N,M] = B_col[N,K] * A_col[K,M]
  // cublasSgemm(handle, transB, transA, N, M, K, &alpha, B, ldb, A, lda, &beta,
  // C, ldc)
  cublasSgemm(
      handle,
      CUBLAS_OP_T, // B is [N,K] row-major, need transpose for col-major [K,N]
      CUBLAS_OP_N, // A is [M,K] row-major, treat as col-major [K,M]
      N, // rows of C (col-major)
      M, // cols of C (col-major)
      K,
      &alpha,
      b_f32,
      K, // ldb
      a_f32,
      K, // lda
      &beta,
      c,
      N); // ldc

  cudaFree(a_f32);
  cudaFree(b_f32);
  delete[] h_a;
  delete[] h_b;
  delete[] h_a_f32;
  delete[] h_b_f32;

  cublasDestroy(handle);
}

TEST(GemmTest, Gemm_1024x1024x2048_Blk256x128x64) {
  // Allocate host memory
  half* h_a = new half[kM * kK];
  half* h_b = new half[kN * kK];
  float* h_c = new float[kM * kN];
  float* h_c_ref = new float[kM * kN];

  // Initialize with random values
  srand(42);
  for (int i = 0; i < kM * kK; i++) {
    h_a[i] = __float2half((rand() % 10 - 5) * 0.1f);
  }
  for (int i = 0; i < kN * kK; i++) {
    h_b[i] = __float2half((rand() % 10 - 5) * 0.1f);
  }

  // Allocate device memory
  half *d_a, *d_b;
  float *d_c, *d_c_ref;
  cudaMalloc(&d_a, kM * kK * sizeof(half));
  cudaMalloc(&d_b, kN * kK * sizeof(half));
  cudaMalloc(&d_c, kM * kN * sizeof(float));
  cudaMalloc(&d_c_ref, kM * kN * sizeof(float));

  cudaMemcpy(d_a, h_a, kM * kK * sizeof(half), cudaMemcpyHostToDevice);
  cudaMemcpy(d_b, h_b, kN * kK * sizeof(half), cudaMemcpyHostToDevice);
  cudaMemset(d_c, 0, kM * kN * sizeof(float));
  cudaMemset(d_c_ref, 0, kM * kN * sizeof(float));

  // Compute reference using cuBLAS
  cublas_gemm_reference(d_a, d_b, d_c_ref, kM, kN, kK);

  // Launch our kernel
  dim3 gridDim(kM / BLK_M, kN / BLK_N);
  dim3 blockDim(WARP_SIZE * WarpLayout::kNumel);

  static constexpr int smem_size_inputs =
      BLK_K * (BLK_N + BLK_M) * sizeof(half);
  static constexpr int smem_size_accumulators = BLK_M * BLK_N * sizeof(float);
  static constexpr int smem_size = smem_size_inputs > smem_size_accumulators
      ? smem_size_inputs
      : smem_size_accumulators;

  printf(
      "[HOST] grid=(%d,%d), block=(%d), smem_size=%d bytes\n",
      gridDim.x,
      gridDim.y,
      blockDim.x,
      smem_size);
  printf(
      "[HOST] smem_size_inputs=%d, smem_size_accumulators=%d\n",
      smem_size_inputs,
      smem_size_accumulators);

  // Set max dynamic shared memory for the kernel
  cudaFuncSetAttribute(
      gemm_f16f16f32_kernel<GemmKernel>,
      cudaFuncAttributeMaxDynamicSharedMemorySize,
      smem_size);

  cudaError_t launch_err = cudaGetLastError();
  printf("[HOST] cudaFuncSetAttribute: %s\n", cudaGetErrorString(launch_err));

  gemm_f16f16f32_kernel<GemmKernel>
      <<<gridDim, blockDim, smem_size>>>(d_a, d_b, d_c);

  launch_err = cudaGetLastError();
  printf("[HOST] Kernel launch: %s\n", cudaGetErrorString(launch_err));

  cudaError_t err = cudaDeviceSynchronize();
  ASSERT_EQ(err, cudaSuccess) << "Kernel failed: " << cudaGetErrorString(err);

  // Copy results back
  cudaMemcpy(h_c, d_c, kM * kN * sizeof(float), cudaMemcpyDeviceToHost);
  cudaMemcpy(h_c_ref, d_c_ref, kM * kN * sizeof(float), cudaMemcpyDeviceToHost);

  // Compare results
  int mismatch_count = 0;
  float max_diff = 0.0f;
  for (int i = 0; i < kM; i++) {
    for (int j = 0; j < kN; j++) {
      int idx = i * kN + j;
      float diff = fabs(h_c[idx] - h_c_ref[idx]);
      float rel_diff =
          diff / (fabs(h_c_ref[idx]) > 1e-6f ? fabs(h_c_ref[idx]) : 1e-6f);

      if (diff > max_diff) {
        max_diff = diff;
      }

      // Allow some tolerance for FP16 accumulation errors
      if (rel_diff > 0.01f && diff > 0.1f) {
        if (mismatch_count < 10) {
          printf(
              "Mismatch at [%d,%d]: expected %.4f, got %.4f (diff=%.4f, "
              "rel=%.2f%%)\n",
              i,
              j,
              h_c_ref[idx],
              h_c[idx],
              diff,
              rel_diff * 100);
        }
        mismatch_count++;
      }
    }
  }

  printf("Max absolute difference: %.6f\n", max_diff);
  EXPECT_EQ(mismatch_count, 0) << "Found " << mismatch_count << " mismatches";

  // Cleanup
  cudaFree(d_a);
  cudaFree(d_b);
  cudaFree(d_c);
  cudaFree(d_c_ref);
  delete[] h_a;
  delete[] h_b;
  delete[] h_c;
  delete[] h_c_ref;
}

} // namespace flash