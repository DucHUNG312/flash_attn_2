#pragma once

#include "def.h"
#include "ptx.h"
#include "warp.h"

namespace flash {

template <
    typename Element_,
    typename WarpLayout_,
    typename GLayout_,
    typename SLayout_>
struct GS_traits_t {
  using WarpLayout = WarpLayout_;
  using Element = Element_;
  using GLayout = GLayout_;
  using SLayout = SLayout_;
  using ThreadInfo = Thread_info_t<Element>;

  static constexpr int kWarpRowStride =
      WarpLayout::kRows * G2S_tile_shape_t::kThrRows;
  static constexpr int kWarpColStride = WarpLayout::kCols *
      G2S_tile_shape_t::kThrCols * ThreadInfo::kElePerThread;
  static constexpr int kRowIters = SLayout::kRows / kWarpRowStride;
  static constexpr int kColIters = SLayout::kCols / kWarpColStride;

  FA_DEVICE_CONSTEXPR static int global_warp_base_offset() {
    const int row = get_warp_row<WarpLayout>() * G2S_tile_shape_t::kThrRows;
    const int col = get_warp_col<WarpLayout>() * G2S_tile_shape_t::kThrCols *
        ThreadInfo::kElePerThread;
    return (row * GLayout::kRowStride) + (col * GLayout::kColStride);
  }
  FA_DEVICE_CONSTEXPR static int shared_warp_base_offset() {
    const int row = get_warp_row<WarpLayout>() * G2S_tile_shape_t::kThrRows;
    const int col = get_warp_col<WarpLayout>() * G2S_tile_shape_t::kThrCols *
        ThreadInfo::kElePerThread;
    return (row * SLayout::kRowStride) + (col * SLayout::kColStride);
  }
  FA_DEVICE_CONSTEXPR static int lane_row_id() {
    return get_lane_id() / G2S_tile_shape_t::kThrCols;
  }
  FA_DEVICE_CONSTEXPR static int lane_col_id() {
    return (get_lane_id() % G2S_tile_shape_t::kThrCols) *
        ThreadInfo::kElePerThread;
  }
};

template <typename GSTraits_>
struct GSLoader {
  using GSTraits = GSTraits_;
  using GLayout = typename GSTraits::GLayout;
  using SLayout = typename GSTraits::SLayout;
  using ThreadInfo = typename GSTraits::ThreadInfo;
  static constexpr int kRowIters = GSTraits::kRowIters;
  static constexpr int kColIters = GSTraits::kColIters;

  template <typename GTensor, typename STensor>
  FA_DEVICE_CONSTEXPR void operator()(
      const GTensor& g_tensor,
      STensor& s_tensor) {
    const int g_warp_base_offset = GSTraits::global_warp_base_offset();
    const int s_warp_base_offset = GSTraits::shared_warp_base_offset();
#pragma unroll
    for (int i = 0; i < kRowIters; i++) {
#pragma unroll
      for (int j = 0; j < kColIters; j++) {
        const int g_lane_offset = g_warp_base_offset +
            GLayout{}((i * GSTraits::kWarpRowStride) + GSTraits::lane_row_id(),
                      (j * GSTraits::kWarpColStride) + GSTraits::lane_col_id());
        const int s_lane_offset = SLayout::swizzle_offset(
            s_warp_base_offset +
            (i * GSTraits::kWarpRowStride + GSTraits::lane_row_id()) *
                SLayout::kRowStride +
            (j * GSTraits::kWarpColStride + GSTraits::lane_col_id()) *
                SLayout::kColStride);
#ifdef FA_DEVICE_SUPPORTED
        cp_async<ThreadInfo::kMaxBytes>(
            g_tensor.data + g_lane_offset, s_tensor.data + s_lane_offset);
#else
        *reinterpret_cast<uint4*>(s_tensor.data + s_lane_offset) =
            *reinterpret_cast<uint4*>(g_tensor.data + g_lane_offset);
#endif
      }
    }
  }
};

template <typename GSTraits_>
struct GSStorer {
  using GSTraits = GSTraits_;
  using GLayout = typename GSTraits::GLayout;
  using SLayout = typename GSTraits::SLayout;
  using ThreadInfo = typename GSTraits::ThreadInfo;
  static constexpr int kRowIters = GSTraits::kRowIters;
  static constexpr int kColIters = GSTraits::kColIters;

  template <typename STensor, typename GTensor>
  FA_DEVICE_CONSTEXPR void operator()(
      const STensor& s_tensor,
      GTensor& g_tensor) {
    const int g_warp_base_offset = GSTraits::global_warp_base_offset();
    const int s_warp_base_offset = GSTraits::shared_warp_base_offset();
#pragma unroll
    for (int i = 0; i < kRowIters; i++) {
#pragma unroll
      for (int j = 0; j < kColIters; j++) {
        const int g_lane_offset = g_warp_base_offset +
            GLayout{}((i * GSTraits::kWarpRowStride) + GSTraits::lane_row_id(),
                      (j * GSTraits::kWarpColStride) + GSTraits::lane_col_id());
        const int s_lane_offset = SLayout::swizzle_offset(
            s_warp_base_offset +
            (i * GSTraits::kWarpRowStride + GSTraits::lane_row_id()) *
                SLayout::kRowStride +
            (j * GSTraits::kWarpColStride + GSTraits::lane_col_id()) *
                SLayout::kColStride);
        *reinterpret_cast<uint4*>(g_tensor.data + g_lane_offset) =
            *reinterpret_cast<uint4*>(s_tensor.data + s_lane_offset);
      }
    }
  }
};

} // namespace flash