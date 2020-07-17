
//TODO: cleanup
#include <ATen/native/ForeachOps.h>
#include <type_traits>
#include <ATen/ATen.h>
#include <ATen/Dispatch.h>
#include <ATen/MemoryOverlap.h>
#include <ATen/NativeFunctions.h>
#include <ATen/native/DispatchStub.h>

namespace at {
namespace native {


std::vector<Tensor> foreach_add(TensorList tensors, Scalar scalar) {
  return foreach_tensor_add_scalar_stub(tensors[0].device().type(), tensors, scalar);
}
DEFINE_DISPATCH(foreach_tensor_add_scalar_stub);

}}