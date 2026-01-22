#pragma once

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
    typename RLayout_>
struct SR_traits_t {
  using WarpLayout = WarpLayout_;
  using Element = Element_;
  using RLayout = RLayout_;
  using SLayout = SLayout_;
  using ThreadInfo = Thread_info_t<Element>;

  static constexpr int kTileRowStride = S2R_tile_shape_t::kThrRows;
  static constexpr int kTileColStride =
      S2R_tile_shape_t::kThrCols * ThreadInfo::kElePerThread;
  static constexpr int kRowIters = RLayout::kRows;
  static constexpr int kColIters = RLayout::kCols;

  FA_DEVICE_CONSTEXPR static int shared_warp_base_offset() {
    const int row = get_warp_row<WarpLayout>() * kTileRowStride;
    const int col = get_warp_col<WarpLayout>() * kTileColStride;
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
    using RFragment = typename RTensor::Element;
    const int s_warp_base_offset = SRTraits::shared_warp_base_offset();
#pragma unroll
    for (int i = 0; i < kRowIters; i++) {
#pragma unroll
      for (int j = 0; j < kColIters; j++) {
#pragma unroll
        for (int m = 0; m < RFragment::kRows; m++) {
          const int row = (i * SRTraits::kTileRowStride) +
              (m * FA_FRAGMENT_ROWS) + SRTraits::r2s_lane_row_id();
#pragma unroll
          for (int n = 0; n < RFragment::kCols / ThreadInfo::kElePerReg; n++) {
            const int col = (j * SRTraits::kTileColStride) +
                (n * FA_FRAGMENT_ROWS) + SRTraits::r2s_lane_col_id();
            const int s_lane_offset = s_warp_base_offset + SLayout{}(row, col);
            *reinterpret_cast<uint32_t*>(s_tensor.data + s_lane_offset) =
                reinterpret_cast<const uint32_t&>(
                    r_tensor(i, j)(m, n * ThreadInfo::kElePerReg));
          }
        }
      }
    }
  }
};

} // namespace flash