#include <gtest/gtest.h>
#include "copy/g2s.h"
#include "layout.h"
#include "ptx.h"
#include "swizzle.h"
#include "tensor.h"

namespace flash {

// =============================================================================
// Round-trip test kernel: G2S then S2G
// =============================================================================

template <
    typename Element,
    typename WarpLayout,
    typename GLayout,
    typename SLayout>
__global__ void roundtrip_kernel(Element* src, Element* dst) {
  using GSCopyTraits = GS_traits_t<Element, WarpLayout, GLayout, SLayout>;
  using G2s = GSLoader<GSCopyTraits>;
  using S2g = GSStorer<GSCopyTraits>;
  using GTensor = Tensor_t<Element, GLayout, g_tensor_tag>;
  using STensor = Tensor_t<Element, SLayout, s_tensor_tag>;

  extern __shared__ char smem[];
  auto* s_data = reinterpret_cast<Element*>(smem);

  const int g_block_offset = blockIdx.y * SLayout::kRows * GLayout::kRowStride +
      blockIdx.x * SLayout::kCols;

  GTensor g_src_tensor(src + g_block_offset);
  GTensor g_dst_tensor(dst + g_block_offset);
  STensor s_tensor(s_data);

  // copy to smem
  G2s{}(g_src_tensor, s_tensor);
#ifdef FA_DEVICE_SUPPORTED
  cp_async_commit_and_wait_all();
#endif
  // store back to gmem
  S2g{}(s_tensor, g_dst_tensor);
}

// =============================================================================
// Test Fixtures
// =============================================================================

template <
    typename Element,
    typename WarpLayout,
    typename GLayout,
    typename SLayout>
void run_load_g2s_test() {
  constexpr const int kGNumel = GLayout::kRows * GLayout::kCols;
  constexpr const int kSNumel = SLayout::kRows * SLayout::kCols;
  const int smem_size = kSNumel * sizeof(Element);

  // Allocate and initialize host memory
  // Use values up to 2048 to ensure exact representation in half precision
  auto* h_src = new Element[kGNumel];
  for (int i = 0; i < kGNumel; ++i) {
    if constexpr (std::is_same_v<Element, half>) {
      h_src[i] = __float2half(static_cast<float>(i % 2048));
    } else {
      h_src[i] = static_cast<Element>(i % 2048);
    }
  }

  // Allocate device memory and copy
  Element* d_src;
  Element* d_dst;
  cudaMalloc(&d_src, kGNumel * sizeof(Element));
  cudaMalloc(&d_dst, kGNumel * sizeof(Element));
  cudaMemcpy(d_src, h_src, kGNumel * sizeof(Element), cudaMemcpyHostToDevice);
  cudaMemset(d_dst, 0, kGNumel * sizeof(Element));

  // Launch round-trip kernel (G2S then S2G)
  dim3 gridDim;
  gridDim.y = GLayout::kRows / SLayout::kRows;
  gridDim.x = GLayout::kCols / SLayout::kCols;
  roundtrip_kernel<Element, WarpLayout, GLayout, SLayout>
      <<<gridDim, WARP_SIZE * WarpLayout::kNumel, smem_size>>>(d_src, d_dst);

  cudaError_t err = cudaDeviceSynchronize();
  ASSERT_EQ(err, cudaSuccess) << "Kernel failed: " << cudaGetErrorString(err);

  // Copy back and verify
  auto* h_dst = new Element[kGNumel];
  cudaMemcpy(h_dst, d_dst, kGNumel * sizeof(Element), cudaMemcpyDeviceToHost);

  // Verify that round-trip data matches original
  int mismatch_count = 0;
  for (int i = 0; i < GLayout::kRows; ++i) {
    for (int j = 0; j < GLayout::kCols; ++j) {
      int idx = i * GLayout::kRowStride + j * GLayout::kColStride;
      float expected = static_cast<float>(idx % 2048);
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

TEST(G2STest, LoadG2S_RowMajor_64x512_Warp4x1_NoSwizzle) {
  using Element = half;
  using WarpLayout = Layout_t<4, 1>;
  using GLayout = Layout_t<64, 512>;
  using SLayout = Layout_t<64, 128>;

  run_load_g2s_test<Element, WarpLayout, GLayout, SLayout>();
}

TEST(G2STest, LoadG2S_RowMajor_64x512_Warp4x1_Swizzle) {
  using Element = half;
  using WarpLayout = Layout_t<4, 1>;
  using GLayout = Layout_t<64, 512>;
  using SLayout = Layout_t<64, 128, 128, 1, true>;

  run_load_g2s_test<Element, WarpLayout, GLayout, SLayout>();
}

} // namespace flash