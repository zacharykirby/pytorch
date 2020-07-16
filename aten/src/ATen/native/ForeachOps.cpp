
//TODO: cleanup
#include <ATen/native/ForeachOps.h>
#include <type_traits>
#include <ATen/ATen.h>
#include <ATen/Dispatch.h>
#include <ATen/MemoryOverlap.h>
#include <ATen/NativeFunctions.h>
#include <torch/library.h>

namespace at {
namespace native {

DEFINE_DISPATCH(foreach_add_scalar_stub);
//DEFINE_DISPATCH(foreach_sub_scalar_stub);

std::vector<Tensor> foreach_add_scalar(TensorList tensors, Scalar scalar) {
  std::cout << "Hello from foreach" << std::endl;
  foreach_add_scalar_stub(DeviceType::CUDA, tensors, scalar);
  std::vector<Tensor> a;
  return a;
}

}}