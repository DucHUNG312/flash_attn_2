#pragma once

#include "def.h"
#include "ptx.h"
#include "tensor.h"

namespace flash {

struct Mma_Atom_16x8x16 {
  static constexpr int kRows = 16;
  static constexpr int kCols = 8;
};
struct Mma_Atom_16x16x16 {
  static constexpr int kRows = 16;
  static constexpr int kCols = 16;
};

template <typename Element, typename AccElement, typename MmaAtom>
struct Mma_t;
template <>
struct Mma_t<half, float, Mma_Atom_16x8x16> {
  FA_DEVICE void operator()(
      const half* __restrict__ ra,
      const half* __restrict__ rb,
      float* __restrict__ rc) {
    const uint32_t* A = reinterpret_cast<const uint32_t*>(ra);
    const uint32_t* B = reinterpret_cast<const uint32_t*>(rb);
    float* C = static_cast<float*>(rc);
    mma_m16n8k16_f32_accum<half>(
        C[0],
        C[1],
        C[2],
        C[3],
        A[0],
        A[1],
        A[2],
        A[3],
        B[0],
        B[1],
        C[0],
        C[1],
        C[2],
        C[3]);
  }
};
template <>
struct Mma_t<nv_bfloat16, float, Mma_Atom_16x8x16> {
  FA_DEVICE void operator()(
      const nv_bfloat16* __restrict__ ra,
      const nv_bfloat16* __restrict__ rb,
      float* __restrict__ rc) {
    const uint32_t* A = reinterpret_cast<const uint32_t*>(ra);
    const uint32_t* B = reinterpret_cast<const uint32_t*>(rb);
    float* C = static_cast<float*>(rc);
    mma_m16n8k16_f32_accum<nv_bfloat16>(
        C[0],
        C[1],
        C[2],
        C[3],
        A[0],
        A[1],
        A[2],
        A[3],
        B[0],
        B[1],
        C[0],
        C[1],
        C[2],
        C[3]);
  }
};
template <>
struct Mma_t<half, float, Mma_Atom_16x16x16> {
  FA_DEVICE void operator()(
      const half* __restrict__ ra,
      const half* __restrict__ rb,
      float* __restrict__ rc) {
    const uint32_t* A = reinterpret_cast<const uint32_t*>(ra);
    const uint32_t* B = reinterpret_cast<const uint32_t*>(rb);
    float* C = static_cast<float*>(rc);
    mma_m16n8k16_f32_accum<half>(
        C[0],
        C[1],
        C[2],
        C[3],
        A[0],
        A[1],
        A[2],
        A[3],
        B[0],
        B[1],
        C[0],
        C[1],
        C[2],
        C[3]);
    mma_m16n8k16_f32_accum<half>(
        C[4],
        C[5],
        C[6],
        C[7],
        A[4],
        A[5],
        A[6],
        A[7],
        B[2],
        B[3],
        C[4],
        C[5],
        C[6],
        C[7]);
  }
};
template <>
struct Mma_t<nv_bfloat16, float, Mma_Atom_16x16x16> {
  FA_DEVICE void operator()(
      const nv_bfloat16* __restrict__ ra,
      const nv_bfloat16* __restrict__ rb,
      float* __restrict__ rc) {
    const uint32_t* A = reinterpret_cast<const uint32_t*>(ra);
    const uint32_t* B = reinterpret_cast<const uint32_t*>(rb);
    float* C = static_cast<float*>(rc);
    mma_m16n8k16_f32_accum<nv_bfloat16>(
        C[0],
        C[1],
        C[2],
        C[3],
        A[0],
        A[1],
        A[2],
        A[3],
        B[0],
        B[1],
        C[0],
        C[1],
        C[2],
        C[3]);
    mma_m16n8k16_f32_accum<nv_bfloat16>(
        C[4],
        C[5],
        C[6],
        C[7],
        A[4],
        A[5],
        A[6],
        A[7],
        B[2],
        B[3],
        C[4],
        C[5],
        C[6],
        C[7]);
  }
};

template <typename A_t_, typename B_t_, typename C_t_, typename MmaAtom_>
requires(is_r_tensor_v<A_t_>&& is_r_tensor_v<B_t_>&&
             is_r_tensor_v<C_t_>) struct Gemm_traits_t {
  using A_t = A_t_;
  using B_t = B_t_;
  using C_t = C_t_;
  using MmaAtom = MmaAtom_;
};

template <typename GemmTraits_>
struct Gemm_t {
  using GemmTraits = GemmTraits_;

  using A_t = typename GemmTraits::A_t;
  using B_t = typename GemmTraits::B_t;
  using C_t = typename GemmTraits::C_t;
  using MmaAtom = typename GemmTraits::MmaAtom;

  using A_Element = underlying_type_t<A_t>;
  using B_Element = underlying_type_t<B_t>;
  using C_Element = underlying_type_t<C_t>;

  static_assert(A_t::kRows == C_t::kRows);
  static_assert(B_t::kCols == C_t::kCols);
  static_assert(std::is_same_v<A_Element, B_Element>);

  using Mma = Mma_t<A_Element, C_Element, MmaAtom>;

  FA_DEVICE void operator()(const A_t& a, const B_t& b, C_t& c) {
#pragma unroll
    for (int i = 0; i < C_t::kRows; i++) {
#pragma unroll
      for (int j = 0; j < C_t::kCols; j++) {
#pragma unroll
        for (int k = 0; k < A_t::kCols; k++) {
          Mma{}(a(i, k).data, b(k, j).data, c(i, j).data);
        }
      }
    }
  }
};

} // namespace flash