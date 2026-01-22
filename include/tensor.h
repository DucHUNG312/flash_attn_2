#pragma once

#include <type_traits>
#include "def.h"
#include "layout.h"

namespace flash {

template <typename Element_, typename Layout_, typename Tag_>
struct Tensor_t;

struct g_tensor_tag {};
struct s_tensor_tag {};
struct r_tensor_tag {};

template <typename T>
concept is_tensor_c = requires {
  typename T::is_tensor;
};
template <typename T>
static constexpr bool is_g_tensor_v =
    std::is_same_v<typename T::Tag, g_tensor_tag>;
template <typename T>
static constexpr bool is_s_tensor_v =
    std::is_same_v<typename T::Tag, s_tensor_tag>;
template <typename T>
static constexpr bool is_r_tensor_v =
    std::is_same_v<typename T::Tag, r_tensor_tag>;

template <typename T>
struct underlying_type {
  using type = T;
};
template <typename Element_, typename Layout_, typename Tag_>
struct underlying_type<Tensor_t<Element_, Layout_, Tag_>> {
  using type = typename underlying_type<
      typename Tensor_t<Element_, Layout_, Tag_>::Element>::type;
};
template <typename T>
using underlying_type_t = typename underlying_type<T>::type;

template <typename Element_, typename Layout_, typename Tag_>
struct Tensor_t {
  using Element = Element_;
  using Layout = Layout_;
  using Tag = Tag_;
  using is_tensor = bool;

  static constexpr int kRows = Layout::kRows;
  static constexpr int kCols = Layout::kCols;
  static constexpr int kRowStride = Layout::kRowStride;
  static constexpr int kColStride = Layout::kColStride;
  static constexpr int kNumel = Layout::kNumel;

  FA_HOST_DEVICE Tensor_t(Element* block_data) : data(block_data) {}

  FA_HOST_DEVICE static constexpr int size() noexcept {
    return kNumel * sizeof(Element);
  }

  FA_HOST_DEVICE static void dump_layout() {
    printf("(%d,%d):(%d,%d)", kRows, kCols, kRowStride, kColStride);
  }

  Element* data;
};

template <typename Element_, typename Layout_>
struct Tensor_t<Element_, Layout_, r_tensor_tag> {
  using Element = Element_;
  using Layout = Layout_;
  using Tag = r_tensor_tag;
  using is_tensor = bool;

  static constexpr int kRows = Layout::kRows;
  static constexpr int kCols = Layout::kCols;
  static constexpr int kRowStride = Layout::kRowStride;
  static constexpr int kColStride = Layout::kColStride;
  static constexpr int kNumel = Layout::kNumel;

  FA_HOST_DEVICE Tensor_t() = default;

  FA_HOST_DEVICE void fill(const underlying_type_t<Element>& value) {
    if constexpr (is_tensor_c<Element>) {
#pragma unroll
      for (int i = 0; i < kNumel; i++) {
        data[i].fill(value);
      }
    } else {
#pragma unroll
      for (int i = 0; i < kNumel; i++) {
        data[i] = value;
      }
    }
  }

  FA_HOST_DEVICE Element& operator()(int i, int j) {
    return data[(i * kRowStride) + (j * kColStride)];
  }
  FA_HOST_DEVICE const Element& operator()(int i, int j) const {
    return data[(i * kRowStride) + (j * kColStride)];
  }

  FA_HOST_DEVICE static void dump_layout() {
    printf("(%d,%d):(%d,%d)", kRows, kCols, kRowStride, kColStride);
    if constexpr (is_tensor_c<Element>) {
      printf(";");
      Element::dump_layout();
    }
  }

  Element data[kNumel];
};

template <typename Element, typename Layout>
using Global_tensor_t = Tensor_t<Element, Layout, g_tensor_tag>;
template <typename Element, typename Layout>
using Shared_tensor_t = Tensor_t<Element, Layout, s_tensor_tag>;
template <typename Element, typename Layout>
using Reg_tensor_t = std::conditional_t<
    std::is_same_v<Element, half> || std::is_same_v<Element, nv_bfloat16>,
    Tensor_t<
        Tensor_t<Element, Layout_t<2, 4>, r_tensor_tag>,
        Layout,
        r_tensor_tag>,
    Tensor_t<
        Tensor_t<Element, Layout_t<2, 2>, r_tensor_tag>,
        Layout,
        r_tensor_tag>>;

} // namespace flash