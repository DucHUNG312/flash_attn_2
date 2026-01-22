#pragma once

#include "def.h"

namespace flash {

struct Swizzle_stride_t {
  int s0;
  int s1;
  int s2;

  FA_DEVICE_CONSTEXPR int offset(int idx) const noexcept {
    // get the 3rd bit of idx
    int i0 = (idx >> 2) & 1;
    // get the 2nd bit of idx
    int i1 = (idx >> 1) & 1;
    // get the 1st bit of idx
    int i2 = idx & 1;
    return (i0 * s0) + (i1 * s1) + (i2 * s2);
  }
};
template <int BBits, int MBase, int SShift = BBits>
struct Swizzle_t {
  static constexpr int mbase = MBase;
  static constexpr int mask_bits = BBits;
  static constexpr int mask_shift = SShift;

  static constexpr int bit_mask = (1 << mask_bits) - 1;
  // indicate group of 8 rows
  static constexpr int yy_mask = bit_mask << (mbase + mask_shift);
  // indicate did we cross into next group row
  static constexpr int yy_mask_lowest_bit = yy_mask & -yy_mask;

  FA_DEVICE_CONSTEXPR static auto apply(int offset) {
    const int row_shifted = (offset & yy_mask) >> mask_shift;
    return offset ^ row_shifted;
  }

  FA_DEVICE_CONSTEXPR static Swizzle_stride_t swizzle_stride(int base_offset) {
    const int base_offset_cmp = yy_mask_lowest_bit << 1;
    const int s1 =
        32 * binary_to_pm1((base_offset & (base_offset_cmp << 1)) == 0);
    const int s2 = 16 *
        binary_to_pm1(static_cast<int>((base_offset & base_offset_cmp) == 0));
    return {.s0 = 64, .s1 = s1, .s2 = s2};
  }

 private:
  FA_DEVICE_CONSTEXPR static int binary_to_pm1(int binary) {
    return (2 * binary) - 1;
  }
};
struct No_swizzle_t {
  FA_DEVICE_CONSTEXPR static auto apply(int offset) {
    return offset;
  }

  FA_DEVICE_CONSTEXPR static Swizzle_stride_t swizzle_stride(
      [[maybe_unused]] int base_offset) {
    return {.s0 = 64, .s1 = 32, .s2 = 16};
  }
};

template <const int kSwizzleBytes_>
struct Swizzle_tile_shape_t;

template <>
struct Swizzle_tile_shape_t<128> {
  static constexpr int kRows = 8;
  static constexpr int kCols = 64;

  static constexpr int kBBits = 3;
  static constexpr int kMBase = 3;
  static constexpr int kSShift = 4;
};

} // namespace flash