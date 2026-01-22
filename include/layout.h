#pragma once

#include <type_traits>
#include "def.h"
#include "ptx.h"
#include "swizzle.h"

namespace flash {

template <
    const int kRows_,
    const int kCols_,
    const int kRowStride_ = kCols_,
    const int kColStride_ = 1,
    const bool kSwizzle_ = false,
    const int kSwizzleBytes_ = 128>
struct Layout_t {
  static constexpr int kRows = kRows_;
  static constexpr int kCols = kCols_;
  static constexpr int kRowStride = kRowStride_;
  static constexpr int kColStride = kColStride_;
  static constexpr bool kSwizzle = kSwizzle_;
  static constexpr int kSwizzleBytes = kSwizzleBytes_;
  static constexpr int kNumel = kRows * kCols;

  using SwizzleShape = Swizzle_tile_shape_t<kSwizzleBytes>;
  static constexpr int kSwizzleRows = SwizzleShape::kRows;
  static constexpr int kSwizzleCols = SwizzleShape::kCols;
  static constexpr int kBBits = SwizzleShape::kBBits;
  static constexpr int kMBase = SwizzleShape::kMBase;
  static constexpr int kSShift = SwizzleShape::kSShift;

  using Swizzle = std::
      conditional_t<kSwizzle, Swizzle_t<kBBits, kMBase, kSShift>, No_swizzle_t>;

  FA_DEVICE_CONSTEXPR int operator()(int i, int j) {
    return Swizzle::apply((i * kRowStride) + (j * kColStride_));
  }
};

} // namespace flash