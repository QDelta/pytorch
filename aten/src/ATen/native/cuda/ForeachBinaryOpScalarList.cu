#define TORCH_ASSERT_ONLY_METHOD_OPERATORS
#include <ATen/Dispatch.h>
#include <ATen/native/BinaryOps.h>
#include <ATen/native/ForeachUtils.h>
#include <ATen/native/cuda/ForeachFunctors.cuh>
#include <ATen/native/cuda/ForeachMinMaxFunctors.cuh>

#ifndef AT_PER_OPERATOR_HEADERS
#include <ATen/NativeFunctions.h>
#else
#include <ATen/ops/_foreach_add_native.h>
#include <ATen/ops/_foreach_clamp_max_native.h>
#include <ATen/ops/_foreach_clamp_min_native.h>
#include <ATen/ops/_foreach_div_native.h>
#include <ATen/ops/_foreach_mul_native.h>
#include <ATen/ops/_foreach_pow_native.h>
#include <ATen/ops/_foreach_sub_native.h>

#include <ATen/ops/empty_like_native.h>
#endif

namespace at::native {

template <typename T, template <class> class Op>
std::vector<Tensor> foreach_binary_op(
    TensorList tensors,
    at::ArrayRef<Scalar> scalars,
    bool has_empty_tensor) {
  std::vector<at::Tensor> vec_res;
  vec_res.reserve(tensors.size());
  for (const auto& t : tensors) {
    vec_res.emplace_back(at::native::empty_like(t));
  }

  std::vector<std::vector<at::Tensor>> tensor_lists;
  std::vector<Scalar> nonempty_scalars;
  std::pair<std::vector<std::vector<Tensor>>, std::vector<Scalar>> res;
  if (has_empty_tensor) {
    res = filter_out_empty_tensors({tensors, vec_res}, scalars);
    tensor_lists = res.first;
    nonempty_scalars = res.second;
  } else {
    tensor_lists.emplace_back(tensors.vec());
    tensor_lists.emplace_back(vec_res);
    nonempty_scalars = scalars.vec();
  }

  using opmath_t = at::opmath_type<T>;
  multi_tensor_apply<2, opmath_t>(
      tensor_lists,
      nonempty_scalars,
      BinaryOpScalarListFunctor<
          T,
          /* depth */ 2,
          /* r_args_depth */ 1,
          /* res_arg_index */ 1>(),

      Op<opmath_t>());
  return vec_res;
}

template <typename T, template <class> class Op>
void foreach_binary_op_(TensorList tensors, at::ArrayRef<Scalar> scalars) {
  std::vector<std::vector<at::Tensor>> tensor_lists;
  tensor_lists.emplace_back(tensors.vec());

  using opmath_t = at::opmath_type<T>;
  multi_tensor_apply<1, opmath_t>(
      tensor_lists,
      scalars,
      BinaryOpScalarListFunctor<
          T,
          /* depth */ 1,
          /* r_args_depth */ 1,
          /* res_arg_index */ 0>(),
      Op<opmath_t>());
  increment_version(tensors);
}

template <template <class> class Op>
std::vector<Tensor> all_types_complex_bool_half_bfloat16(
    TensorList tensors,
    at::ArrayRef<Scalar> scalars,
    bool has_empty_tensor) {
  return AT_DISPATCH_ALL_TYPES_AND_COMPLEX_AND3(
      kBool,
      kHalf,
      kBFloat16,
      tensors[0].scalar_type(),
      "foreach_binary_op_scalarlist_cuda",
      [&]() {
        return foreach_binary_op<scalar_t, Op>(
            tensors, scalars, has_empty_tensor);
      });
}

template <template <class> class Op>
void all_types_complex_bool_half_bfloat16_(
    TensorList tensors,
    at::ArrayRef<Scalar> scalars) {
  AT_DISPATCH_ALL_TYPES_AND_COMPLEX_AND3(
      kBool,
      kHalf,
      kBFloat16,
      tensors[0].scalar_type(),
      "foreach_binary_op_scalarlist_cuda_",
      [&]() { foreach_binary_op_<scalar_t, Op>(tensors, scalars); });
}

template <template <class> class Op>
std::vector<Tensor> all_types_half_bfloat16(
    TensorList tensors,
    at::ArrayRef<Scalar> scalars,
    bool has_empty_tensor) {
  return AT_DISPATCH_ALL_TYPES_AND2(
      kHalf,
      kBFloat16,
      tensors[0].scalar_type(),
      "foreach_binary_op_scalarlist_cuda",
      [&]() {
        return foreach_binary_op<scalar_t, Op>(
            tensors, scalars, has_empty_tensor);
      });
}

template <template <class> class Op>
void all_types_half_bfloat16_(
    TensorList tensors,
    at::ArrayRef<Scalar> scalars) {
  AT_DISPATCH_ALL_TYPES_AND2(
      kHalf,
      kBFloat16,
      tensors[0].scalar_type(),
      "foreach_binary_op_scalarlist_cuda_",
      [&]() { foreach_binary_op_<scalar_t, Op>(tensors, scalars); });
}

template <template <class> class Op>
std::vector<Tensor> all_types_complex_half_bfloat16(
    TensorList tensors,
    at::ArrayRef<Scalar> scalars,
    bool has_empty_tensor) {
  return AT_DISPATCH_ALL_TYPES_AND_COMPLEX_AND2(
      kHalf,
      kBFloat16,
      tensors[0].scalar_type(),
      "foreach_binary_op_scalarlist_cuda",
      [&]() {
        return foreach_binary_op<scalar_t, Op>(
            tensors, scalars, has_empty_tensor);
      });
}

template <template <class> class Op>
void all_types_complex_half_bfloat16_(
    TensorList tensors,
    at::ArrayRef<Scalar> scalars) {
  AT_DISPATCH_ALL_TYPES_AND_COMPLEX_AND2(
      kHalf,
      kBFloat16,
      tensors[0].scalar_type(),
      "foreach_binary_op_scalarlist_cuda_",
      [&]() { foreach_binary_op_<scalar_t, Op>(tensors, scalars); });
}

#define FOREACH_BINARY_OP_SCALARLIST(FUNCTION, NAME, OP, DIV_OP)            \
  void foreach_tensor_##NAME##_scalarlist_kernel_cuda_(                     \
      TensorList tensors, at::ArrayRef<Scalar> scalars) {                   \
    check_foreach_api_restrictions(tensors, scalars);                       \
    std::pair<bool, bool> p = can_use_fast_route(tensors, scalars, DIV_OP); \
    bool can_use_fast_route = p.first;                                      \
    bool has_empty_tensor = p.second;                                       \
    if (!can_use_fast_route) {                                              \
      return at::native::foreach_tensor_##NAME##_scalarlist_kernel_slow_(   \
          tensors, scalars);                                                \
    }                                                                       \
                                                                            \
    std::pair<std::vector<std::vector<Tensor>>, std::vector<Scalar>> res;   \
    if (has_empty_tensor) {                                                 \
      res = filter_out_empty_tensors({tensors}, scalars);                   \
      tensors = res.first[0];                                               \
      scalars = res.second;                                                 \
    }                                                                       \
                                                                            \
    FUNCTION##_<OP>(tensors, scalars);                                      \
  }                                                                         \
                                                                            \
  std::vector<Tensor> foreach_tensor_##NAME##_scalarlist_kernel_cuda(       \
      TensorList tensors, at::ArrayRef<Scalar> scalars) {                   \
    check_foreach_api_restrictions(tensors, scalars);                       \
    std::pair<bool, bool> p = can_use_fast_route(tensors, scalars, DIV_OP); \
    bool can_use_fast_route = p.first;                                      \
    bool has_empty_tensor = p.second;                                       \
    if (!can_use_fast_route) {                                              \
      return at::native::foreach_tensor_##NAME##_scalarlist_kernel_slow(    \
          tensors, scalars);                                                \
    }                                                                       \
                                                                            \
    return FUNCTION<OP>(tensors, scalars, has_empty_tensor);                \
  }

FOREACH_BINARY_OP_SCALARLIST(
    all_types_complex_bool_half_bfloat16,
    add,
    std::plus,
    /*div_op*/ false);
FOREACH_BINARY_OP_SCALARLIST(
    all_types_complex_bool_half_bfloat16,
    mul,
    std::multiplies,
    /*div_op*/ false);
FOREACH_BINARY_OP_SCALARLIST(
    all_types_complex_bool_half_bfloat16,
    div,
    std::divides,
    /*div_op*/ true);
// See [Why is foreach_pow's division_op=true?]
FOREACH_BINARY_OP_SCALARLIST(
    all_types_complex_half_bfloat16,
    pow,
    power_functor,
    /*div_op*/ true);

// This does not use FOREACH_BINARY_OP_SCALARLIST because
// In the case of subtraction, we dont allow scalar to be boolean following the
// torch.sub logic
void foreach_tensor_sub_scalarlist_kernel_cuda_(
    TensorList tensors,
    at::ArrayRef<Scalar> scalars) {
  check_foreach_api_restrictions(tensors, scalars);
  for (const auto i : c10::irange(tensors.size())) {
    sub_check(tensors[i], scalars[i]);
  }

  std::pair<bool, bool> p = can_use_fast_route({tensors}, scalars);
  bool can_use_fast_route = p.first;
  bool has_empty_tensor = p.second;
  if (!can_use_fast_route) {
    return at::native::foreach_tensor_sub_scalarlist_kernel_slow_(
        tensors, scalars);
  }

  std::pair<std::vector<std::vector<Tensor>>, std::vector<Scalar>> res;
  if (has_empty_tensor) {
    res = filter_out_empty_tensors({tensors}, scalars);
    tensors = res.first[0];
    scalars = res.second;
  }

  AT_DISPATCH_ALL_TYPES_AND_COMPLEX_AND3(
      kBool,
      kHalf,
      kBFloat16,
      tensors[0].scalar_type(),
      "foreach_binary_op_scalarlist_cuda_",
      [&]() { foreach_binary_op_<scalar_t, std::minus>(tensors, scalars); });
}

std::vector<Tensor> foreach_tensor_sub_scalarlist_kernel_cuda(
    TensorList tensors,
    at::ArrayRef<Scalar> scalars) {
  check_foreach_api_restrictions(tensors, scalars);
  for (const auto i : c10::irange(tensors.size())) {
    sub_check(tensors[i], scalars[i]);
  }

  std::pair<bool, bool> p = can_use_fast_route({tensors}, scalars);
  bool can_use_fast_route = p.first;
  bool has_empty_tensor = p.second;
  if (!can_use_fast_route) {
    return at::native::foreach_tensor_sub_scalarlist_kernel_slow(
        tensors, scalars);
  }

  return AT_DISPATCH_ALL_TYPES_AND_COMPLEX_AND3(
      kBool,
      kHalf,
      kBFloat16,
      tensors[0].scalar_type(),
      "foreach_binary_op_scalarlist_cuda_",
      [&]() {
        return foreach_binary_op<scalar_t, std::minus>(
            tensors, scalars, has_empty_tensor);
      });
}

FOREACH_BINARY_OP_SCALARLIST(
    all_types_half_bfloat16,
    clamp_max,
    minimum,
    false);
FOREACH_BINARY_OP_SCALARLIST(
    all_types_half_bfloat16,
    clamp_min,
    maximum,
    false);

} // namespace at::native
