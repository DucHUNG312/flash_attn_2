#pragma once

#include "gemm.h"
#include "layout.h"
#include "ptx.h"
#include "swizzle.h"
#include "tensor.h"
#include "warp.h"

namespace flash {

template <
    typename Element_,
    typename WarpLayout_,
    typename SLayout_,
    typename RLayout_,
    typename Mma_Atom_ = Mma_Atom_16x16x16>
struct SR_traits_t {
  using WarpLayout = WarpLayout_;
  using Element = Element_;
  using RLayout = RLayout_;
  using SLayout = SLayout_;
  using ThreadInfo = Thread_info_t<Element>;
  using Mma_Atom = Mma_Atom_;

  static constexpr int kTileRowStride = Mma_Atom::kRows;
  static constexpr int kTileColStride = Mma_Atom::kCols;
  static constexpr int kRowIters = RLayout::kRows;
  static constexpr int kColIters = RLayout::kCols;

  FA_DEVICE_CONSTEXPR static int shared_warp_base_offset() {
    const int row = get_warp_row<WarpLayout>() * Mma_Atom::kRows;
    const int col = get_warp_col<WarpLayout>() * Mma_Atom::kCols *
        ThreadInfo::kElePerThread;
    return (row * SLayout::kRowStride) + (col * SLayout::kColStride);
  }
  FA_DEVICE_CONSTEXPR static int s2r_lane_row_id() {
    return get_lane_id() % S2R_tile_shape_t::kThrRows;
  }
  FA_DEVICE_CONSTEXPR static int s2r_lane_col_id() {
    return (get_lane_id() / S2R_tile_shape_t::kThrRows) *
        ThreadInfo::kElePerThread;
  }
  FA_DEVICE_CONSTEXPR static int r2s_lane_row_id() {
    return get_lane_id() / R2S_tile_shape_t::kThrCols;
  }
  FA_DEVICE_CONSTEXPR static int r2s_lane_col_id() {
    return (get_lane_id() % R2S_tile_shape_t::kThrCols) *
        ThreadInfo::kElePerThread;
  }
};

template <typename SRTraits_>
struct SRLoader {
  using SRTraits = SRTraits_;
  using SLayout = typename SRTraits::SLayout;
  using ThreadInfo = typename SRTraits::ThreadInfo;
  static constexpr int kRowIters = SRTraits::kRowIters;
  static constexpr int kColIters = SRTraits::kColIters;

  template <typename STensor, typename RTensor>
  FA_DEVICE_CONSTEXPR void operator()(
      const STensor& s_tensor,
      RTensor& r_tensor) {
    const int s_warp_base_offset = SRTraits::shared_warp_base_offset();
#pragma unroll
    for (int i = 0; i < kRowIters; i++) {
#pragma unroll
      for (int j = 0; j < kColIters; j++) {
        const int s_lane_offset =
            s_warp_base_offset +
            SLayout{}(
                (i * SRTraits::kTileRowStride) + SRTraits::s2r_lane_row_id(),
                (j * SRTraits::kTileColStride) + SRTraits::s2r_lane_col_id());
        ldmatrix_x4<false>(
            s_tensor.data + s_lane_offset,
            reinterpret_cast<uint32_t&>(r_tensor(i, j)(0, 0)),
            reinterpret_cast<uint32_t&>(r_tensor(i, j)(1, 0)),
            reinterpret_cast<uint32_t&>(
                r_tensor(i, j)(0, ThreadInfo::kElePerReg)),
            reinterpret_cast<uint32_t&>(
                r_tensor(i, j)(1, ThreadInfo::kElePerReg)));
      }
    }
  }
};

template <typename SRTraits_>
struct SRStorer {
  using SRTraits = SRTraits_;
  using SLayout = typename SRTraits::SLayout;
  using ThreadInfo = typename SRTraits::ThreadInfo;
  static constexpr int kRowIters = SRTraits::kRowIters;
  static constexpr int kColIters = SRTraits::kColIters;

  template <typename RTensor, typename STensor>
  FA_DEVICE_CONSTEXPR void operator()(
      const RTensor& r_tensor,
      STensor& s_tensor) {
    const int s_warp_base_offset = SRTraits::shared_warp_base_offset();
#pragma unroll
    for (int i = 0; i < kRowIters; i++) {
#pragma unroll
      for (int j = 0; j < kColIters; j++) {
        const int s_lane_offset =
            s_warp_base_offset +
            SLayout{}(
                (i * R2S_tile_shape_t::kThrRows) + SRTraits::r2s_lane_row_id(),
                (j * SRTraits::kTileColStride) + SRTraits::r2s_lane_col_id());
        *reinterpret_cast<uint4*>(s_tensor.data + s_lane_offset) = make_uint4(
            reinterpret_cast<const uint32_t&>(r_tensor(i, j)(0, 0)),
            reinterpret_cast<const uint32_t&>(r_tensor(i, j)(1, 0)),
            reinterpret_cast<const uint32_t&>(
                r_tensor(i, j)(0, ThreadInfo::kElePerReg)),
            reinterpret_cast<const uint32_t&>(
                r_tensor(i, j)(1, ThreadInfo::kElePerReg)));
      }
    }
  }
};

} // namespace flash