#pragma once

#include "def.h"

namespace flash {

template <typename Element>
struct Thread_info_t {
  static constexpr int kMaxBytes = 16;
  static constexpr int kElePerThread = kMaxBytes / sizeof(Element);
  static constexpr int kElePerReg = sizeof(uint32_t) / sizeof(Element);
};

struct G2S_tile_shape_t {
  static constexpr int kThrRows = 4;
  static constexpr int kThrCols = 8;
};

struct S2R_tile_shape_t {
  static constexpr int kThrRows = 16;
  static constexpr int kThrCols = 2;
};

struct R2S_tile_shape_t {
  static constexpr int kThrRows = 8;
  static constexpr int kThrCols = 4;
};

FA_DEVICE int get_warp_id() {
  return static_cast<int>(threadIdx.x) / WARP_SIZE;
}
FA_DEVICE int get_lane_id() {
  return static_cast<int>(threadIdx.x) % WARP_SIZE;
}
template <typename WarpLayout>
FA_DEVICE int get_warp_row() {
  return get_warp_id() / WarpLayout::kCols;
}
template <typename WarpLayout>
FA_DEVICE int get_warp_col() {
  return get_warp_id() % WarpLayout::kCols;
}

} // namespace flash