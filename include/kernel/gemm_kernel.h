#pragma once

#include <cassert>
#include <type_traits>
#include "copy/g2s.h"
#include "copy/s2r.h"
#include "def.h"
#include "gemm.h"
#include "layout.h"
#include "tensor.h"

namespace flash {

template <const int kM_, const int kN_, const int kK_>
struct Gemm_shape_t {
  static constexpr int kM = kM_;
  static constexpr int kN = kN_;
  static constexpr int kK = kK_;
};

template <typename Element, typename Layout, typename SubLayout, typename Tag>
struct Tiler_t {
  static constexpr int kRowIters = Layout::kRows / SubLayout::kRows;
  static constexpr int kColIters = Layout::kCols / SubLayout::kCols;

  using TiledLayout = Layout_t<
      SubLayout::kRows,
      SubLayout::kCols,
      Layout::kRowStride,
      Layout::kColStride>;
  using TiledTensor = Tensor_t<Element, TiledLayout, Tag>;

  FA_HOST_DEVICE Tiler_t(const Element* p) : base(const_cast<Element*>(p)) {}

  FA_HOST_DEVICE auto tensor(int tile) {
    assert(kRowIters == 1 || kColIters == 1);
    const int tile_i = kColIters == 1 ? 0 : tile;
    const int tile_j = kRowIters == 1 ? 0 : tile;
    auto* p = base + tile_i * SubLayout::kRows * Layout::kRowStride +
        tile_j * SubLayout::kCols * Layout::kColStride;
    return TiledTensor{p};
  }

  FA_HOST_DEVICE auto tensor(int tile_i, int tile_j) {
    auto* p = base + tile_i * SubLayout::kRows * Layout::kRowStride +
        tile_j * SubLayout::kCols * Layout::kColStride;
    return TiledTensor{p};
  }

  Element* base;
};

template <
    typename Element_,
    typename AccElement_,
    typename GProblemShape_,
    typename SProblemShape_,
    typename WarpLayout_,
    typename Mma_Atom_ = Mma_Atom_16x16x16,
    const bool kSwizzle_ = true,
    const int kSwizzleBytes = 128>
struct Gemm_kernel_traits_t {
  using Element = Element_;
  using AccElement = AccElement_;
  using WarpLayout = WarpLayout_;
  using Mma_Atom = Mma_Atom_;

  static constexpr int kM = GProblemShape_::kM;
  static constexpr int kN = GProblemShape_::kN;
  static constexpr int kK = GProblemShape_::kK;
  static constexpr int kTM = SProblemShape_::kM;
  static constexpr int kTN = SProblemShape_::kN;
  static constexpr int kTK = SProblemShape_::kK;

  static constexpr int kTKIters = kK / kTK;

  using A_GLayout = Layout_t<kTM, kK>;
  using B_GLayout = Layout_t<kTN, kK>;
  using C_GLayout = Layout_t<kTM, kTN>;

  using A_SLayout = std::conditional_t<
      kSwizzle_,
      Layout_t<kTM, kTK, kTK, 1, true, kSwizzleBytes>,
      Layout_t<kTM, kTK>>;
  using B_SLayout = std::conditional_t<
      kSwizzle_,
      Layout_t<kTN, kTK, kTK, 1, true, kSwizzleBytes>,
      Layout_t<kTN, kTK>>;
  using C_SLayout = std::conditional_t<
      kSwizzle_,
      Layout_t<kTM, kTN, kTN, 1, true, kSwizzleBytes>,
      Layout_t<kTM, kTN>>;

  using A_RLayout = Layout_t<
      kTM / WarpLayout::kRows / Mma_Atom::kRows,
      kTK / Mma_Atom::kCols>;
  using B_RLayout = Layout_t<
      kTN / WarpLayout::kCols / Mma_Atom::kCols,
      kTK / Mma_Atom::kRows>;
  using C_RLayout = Layout_t<
      kTM / WarpLayout::kRows / Mma_Atom::kRows,
      kTN / WarpLayout::kCols / Mma_Atom::kCols>;

  using A_GTiler = Tiler_t<Element, A_GLayout, A_SLayout, g_tensor_tag>;
  using B_GTiler = Tiler_t<Element, B_GLayout, B_SLayout, g_tensor_tag>;

  using A_GSCopyTraits = GS_traits_t<Element, WarpLayout, A_GLayout, A_SLayout>;
  using B_GSCopyTraits = GS_traits_t<Element, WarpLayout, B_GLayout, B_SLayout>;
  using C_GSCopyTraits =
      GS_traits_t<AccElement, WarpLayout, C_GLayout, C_SLayout>;

  using A_SRCopyTraits =
      SR_traits_t<Element, WarpLayout, A_SLayout, A_RLayout, Mma_Atom>;
  using B_SRCopyTraits =
      SR_traits_t<Element, WarpLayout, B_SLayout, B_RLayout, Mma_Atom>;
  using C_SRCopyTraits =
      SR_traits_t<AccElement, WarpLayout, C_SLayout, C_RLayout, Mma_Atom>;

  using A_G2s = GSLoader<A_GSCopyTraits>;
  using B_G2s = GSLoader<B_GSCopyTraits>;
  using C_S2g = GSStorer<C_GSCopyTraits>;

  using A_S2r = SRLoader<A_SRCopyTraits>;
  using B_S2r = SRLoader<B_SRCopyTraits>;
  using C_R2s = SRStorer<C_SRCopyTraits>;

  using A_GTensor = Global_tensor_t<Element, A_GLayout>;
  using B_GTensor = Global_tensor_t<Element, B_GLayout>;
  using C_GTensor = Global_tensor_t<AccElement, C_GLayout>;

  using A_STensor = Shared_tensor_t<Element, A_SLayout>;
  using B_STensor = Shared_tensor_t<Element, B_SLayout>;
  using C_STensor = Shared_tensor_t<AccElement, C_SLayout>;

  using A_RTensor = Reg_tensor_t<Element, A_RLayout>;
  using B_RTensor = Reg_tensor_t<Element, B_RLayout>;
  using C_RTensor = Reg_tensor_t<AccElement, C_RLayout>;

  using GemmTraits = Gemm_traits_t<A_RTensor, B_RTensor, C_RTensor, Mma_Atom>;
  using Gemm = Gemm_t<GemmTraits>;
};

template <typename GemmKeTraits_>
struct Gemm_kernel_t {
  using GemmKeTraits = GemmKeTraits_;

  using Element = typename GemmKeTraits::Element;
  using AccElement = typename GemmKeTraits::AccElement;
  using WarpLayout = typename GemmKeTraits::WarpLayout;

  static constexpr int kM = GemmKeTraits::kM;
  static constexpr int kN = GemmKeTraits::kN;
  static constexpr int kK = GemmKeTraits::kK;
  static constexpr int kTM = GemmKeTraits::kTM;
  static constexpr int kTN = GemmKeTraits::kTN;
  static constexpr int kTK = GemmKeTraits::kTK;

  using A_GTiler = typename GemmKeTraits::A_GTiler;
  using B_GTiler = typename GemmKeTraits::B_GTiler;

  using A_G2s = typename GemmKeTraits::A_G2s;
  using B_G2s = typename GemmKeTraits::B_G2s;
  using C_S2g = typename GemmKeTraits::C_S2g;

  using A_S2r = typename GemmKeTraits::A_S2r;
  using B_S2r = typename GemmKeTraits::B_S2r;
  using C_R2s = typename GemmKeTraits::C_R2s;

  using A_GTensor = typename GemmKeTraits::A_GTensor;
  using B_GTensor = typename GemmKeTraits::B_GTensor;
  using C_GTensor = typename GemmKeTraits::C_GTensor;

  using A_STensor = typename GemmKeTraits::A_STensor;
  using B_STensor = typename GemmKeTraits::B_STensor;
  using C_STensor = typename GemmKeTraits::C_STensor;

  using A_RTensor = typename GemmKeTraits::A_RTensor;
  using B_RTensor = typename GemmKeTraits::B_RTensor;
  using C_RTensor = typename GemmKeTraits::C_RTensor;

  using Gemm = typename GemmKeTraits::Gemm;

  static constexpr int kTKIters = GemmKeTraits::kTKIters;

  /*!
    \tparam a pointer to device data of A
    \tparam b pointer to device data of B
    \tparam c pointer to device data of C
  */
  FA_DEVICE void operator()(
      const Element* __restrict__ a,
      const Element* __restrict__ b,
      AccElement* __restrict__ c) {
    __shared__ Element a_smem[A_STensor::kNumel];
    __shared__ Element b_smem[B_STensor::kNumel];
    __shared__ AccElement c_smem[C_STensor::kNumel];

    const Element* a_base_gptr = a + blockIdx.x * kTM * kK;
    const Element* b_base_gptr = b + blockIdx.y * kTN * kK;
    AccElement* c_base_gptr = c + blockIdx.x * kTM * kN + blockIdx.y * kTN;

    A_GTiler a_gtiler{a_base_gptr};
    B_GTiler b_gtiler{b_base_gptr};
    C_GTensor c_gtensor{c_base_gptr};

    A_STensor a_stensor{a_smem};
    B_STensor b_stensor{b_smem};
    C_STensor c_stensor{c_smem};

    A_RTensor a_rtensor{};
    B_RTensor b_rtensor{};
    C_RTensor c_rtensor{};
    c_rtensor.fill(0);

    A_G2s g2s_a{};
    B_G2s g2s_b{};

    A_S2r s2r_a{};
    B_S2r s2r_b{};

    C_S2g s2g_c{};
    C_R2s r2s_c{};

    Gemm gemm{};

#pragma unroll
    for (int k1 = 0; k1 < kTKIters; k1++) {
      g2s_a(a_gtiler.tensor(k1), a_stensor);
      g2s_b(b_gtiler.tensor(k1), b_stensor);
      __syncthreads();
      s2r_a(a_stensor, a_rtensor);
      s2r_b(b_stensor, b_rtensor);
      gemm(a_rtensor, b_rtensor, c_rtensor);
    }
    __syncthreads();
    r2s_c(c_rtensor, c_stensor);
    s2g_c(c_stensor, c_gtensor);
  }
};

template <typename GemmKernel>
__global__ void gemm_f16f16f32_kernel(
    const half* __restrict__ a,
    const half* __restrict__ b,
    float* __restrict__ c) {
  GemmKernel{}(a, b, c);
}

} // namespace flash