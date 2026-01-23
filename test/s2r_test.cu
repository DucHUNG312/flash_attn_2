#include <gtest/gtest.h>
#include "copy/g2s.h"
#include "copy/s2r.h"
#include "layout.h"
#include "ptx.h"
#include "swizzle.h"
#include "tensor.h"

namespace flash {

// =============================================================================
// S2R -> R2S roundtrip test kernel
// Tests: Global -> Shared -> Registers -> Shared -> Global
// =============================================================================

template <
    typename Element,
    typename WarpLayout,
    typename GLayout,
    typename SLayout,
    typename RLayout>
__global__ void s2r_r2s_roundtrip_kernel(Element* src, Element* dst) {
  using GSCopyTraits = GS_traits_t<Element, WarpLayout, GLayout, SLayout>;
  using SRCopyTraits = SR_traits_t<Element, WarpLayout, SLayout, RLayout>;
  using G2s = GSLoader<GSCopyTraits>;
  using S2g = GSStorer<GSCopyTraits>;
  using S2r = SRLoader<SRCopyTraits>;
  using R2s = SRStorer<SRCopyTraits>;
  using GTensor = Global_tensor_t<Element, GLayout>;
  using STensor = Shared_tensor_t<Element, SLayout>;
  using RTensor = Reg_tensor_t<Element, RLayout>;

  extern __shared__ char smem[];
  auto* s_data = reinterpret_cast<Element*>(smem);

  const int g_block_offset = blockIdx.y * SLayout::kRows * GLayout::kRowStride +
      blockIdx.x * SLayout::kCols;

  GTensor g_src_tensor(src + g_block_offset);
  GTensor g_dst_tensor(dst + g_block_offset);
  STensor s_tensor(s_data);
  RTensor reg{};

  // Step 1: Global -> Shared (g2s)
  G2s{}(g_src_tensor, s_tensor);

#ifdef FA_DEVICE_SUPPORTED
  cp_async_commit_and_wait_all();
#endif
  __syncthreads();
  // Step 2: Shared -> Registers (s2r)
  S2r{}(s_tensor, reg);

  __syncthreads();

  // Clear shared memory to ensure we're reading from registers
  if (threadIdx.x == 0) {
    for (int i = 0; i < SLayout::kNumel; i++) {
      s_data[i] = Element(0);
    }
  }
  __syncthreads();

  // Step 3: Registers -> Shared (r2s)
  R2s{}(reg, s_tensor);

  __syncthreads();

  // Step 4: Shared -> Global (s2g)
  S2g{}(s_tensor, g_dst_tensor);
}

// =============================================================================
// Test runner for S2R -> R2S roundtrip
// =============================================================================

template <
    typename Element,
    typename WarpLayout,
    typename GLayout,
    typename SLayout,
    typename RLayout>
void run_s2r_r2s_roundtrip_test() {
  constexpr int kGNumel = GLayout::kRows * GLayout::kCols;
  constexpr int kSNumel = SLayout::kRows * SLayout::kCols;
  const int smem_size = kSNumel * sizeof(Element);

  // Allocate and initialize host memory
  auto* h_src = new Element[kGNumel];
  for (int i = 0; i < kGNumel; ++i) {
    if constexpr (std::is_same_v<Element, half>) {
      // Use small values to stay in half precision range
      h_src[i] = __float2half(static_cast<float>(i % 256));
    } else {
      h_src[i] = static_cast<Element>(i % 256);
    }
  }

  // Allocate device memory
  Element *d_src, *d_dst;
  cudaMalloc(&d_src, kGNumel * sizeof(Element));
  cudaMalloc(&d_dst, kGNumel * sizeof(Element));
  cudaMemcpy(d_src, h_src, kGNumel * sizeof(Element), cudaMemcpyHostToDevice);
  cudaMemset(d_dst, 0, kGNumel * sizeof(Element));

  // Launch kernel
  dim3 gridDim(
      GLayout::kCols / SLayout::kCols, GLayout::kRows / SLayout::kRows, 1);
  dim3 blockDim(WARP_SIZE * WarpLayout::kNumel);

  s2r_r2s_roundtrip_kernel<Element, WarpLayout, GLayout, SLayout, RLayout>
      <<<gridDim, blockDim, smem_size>>>(d_src, d_dst);

  cudaError_t err = cudaDeviceSynchronize();
  ASSERT_EQ(err, cudaSuccess) << "Kernel failed: " << cudaGetErrorString(err);

  // Copy back results
  auto* h_dst = new Element[kGNumel];
  cudaMemcpy(h_dst, d_dst, kGNumel * sizeof(Element), cudaMemcpyDeviceToHost);

  // Verify results
  int mismatch_count = 0;
  for (int i = 0; i < GLayout::kRows; ++i) {
    for (int j = 0; j < GLayout::kCols; ++j) {
      int idx = i * GLayout::kRowStride + j * GLayout::kColStride;
      float expected = static_cast<float>(idx % 256);
      float actual;
      if constexpr (std::is_same_v<Element, half>) {
        actual = __half2float(h_dst[idx]);
      } else {
        actual = static_cast<float>(h_dst[idx]);
      }

      if (expected != actual) {
        if (mismatch_count < 10) {
          printf(
              "Mismatch at [%d,%d] (idx=%d): expected %.0f, got %.0f\n",
              i,
              j,
              idx,
              expected,
              actual);
        }
        mismatch_count++;
      }
    }
  }

  EXPECT_EQ(mismatch_count, 0) << "Found " << mismatch_count << " mismatches";

  // Cleanup
  cudaFree(d_src);
  cudaFree(d_dst);
  delete[] h_src;
  delete[] h_dst;
}

// =============================================================================
// Test Cases
// =============================================================================

// Test S2R -> R2S roundtrip with 16x64 shared memory tile, 1 warp
TEST(S2RTest, S2R_R2S_Roundtrip_16x64_Warp1x1_NoSwizzle) {
  using Element = half;
  using WarpLayout = Layout_t<1, 1>;
  using GLayout = Layout_t<16, 64>;
  using SLayout = Layout_t<16, 64>;
  using RLayout = Layout_t<1, 4>;

  run_s2r_r2s_roundtrip_test<Element, WarpLayout, GLayout, SLayout, RLayout>();
}

// Test S2R -> R2S roundtrip with 16x128 shared memory tile, 1 warp
TEST(S2RTest, S2R_R2S_Roundtrip_16x128_Warp1x1_NoSwizzle) {
  using Element = half;
  using WarpLayout = Layout_t<1, 1>;
  using GLayout = Layout_t<16, 128>;
  using SLayout = Layout_t<16, 128>;
  using RLayout = Layout_t<1, 8>;

  run_s2r_r2s_roundtrip_test<Element, WarpLayout, GLayout, SLayout, RLayout>();
}

// Test S2R -> R2S roundtrip with swizzle
TEST(S2RTest, S2R_R2S_Roundtrip_16x64_Warp1x1_Swizzle) {
  using Element = half;
  using WarpLayout = Layout_t<1, 1>;
  using GLayout = Layout_t<16, 64>;
  using SLayout = Layout_t<16, 64, 64, 1, true>;
  using RLayout = Layout_t<1, 4>;

  run_s2r_r2s_roundtrip_test<Element, WarpLayout, GLayout, SLayout, RLayout>();
}

// Test S2R -> R2S roundtrip with 4 warps
TEST(S2RTest, S2R_R2S_Roundtrip_64x64_Warp4x1_NoSwizzle) {
  using Element = half;
  using WarpLayout = Layout_t<4, 1>;
  using GLayout = Layout_t<64, 64>;
  using SLayout = Layout_t<64, 64>;
  using RLayout = Layout_t<1, 4>;

  run_s2r_r2s_roundtrip_test<Element, WarpLayout, GLayout, SLayout, RLayout>();
}

// Test S2R -> R2S roundtrip with 4 warps swizzle
TEST(S2RTest, S2R_R2S_Roundtrip_64x128_Warp4x1_Swizzle) {
  using Element = half;
  using WarpLayout = Layout_t<4, 1>;
  using GLayout = Layout_t<64, 128>;
  using SLayout = Layout_t<64, 128, 128, 1, true>;
  using RLayout = Layout_t<1, 8>;

  run_s2r_r2s_roundtrip_test<Element, WarpLayout, GLayout, SLayout, RLayout>();
}

} // namespace flash
