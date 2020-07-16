// Functions that fill Tensors with constants. Implementations are in Fill.cpp.

#pragma once

#include <ATen/ATen.h>
#include <ATen/native/DispatchStub.h>

namespace at { namespace native {

using binary_fn_foreach_alpha = void(*)(TensorList, Scalar alpha);

DECLARE_DISPATCH(binary_fn_foreach_alpha, foreach_add_scalar_stub);
//DECLARE_DISPATCH(binary_fn_foreach_alpha, foreach_sub_scalar_stub);

}} // namespace at::native
