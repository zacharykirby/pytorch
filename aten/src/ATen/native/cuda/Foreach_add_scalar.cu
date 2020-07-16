#include <ATen/Dispatch.h>
#include <ATen/native/DispatchStub.h>
#include <ATen/native/cuda/Loops.cuh>
#include <ATen/native/ForeachOps.h>


// NOTE: CUDA on Windows requires that the enclosing function
// of a __device__ lambda not have internal linkage.

namespace at { namespace native {

void foreach_add_scalar_kernel_cuda(TensorList tensors, Scalar scalar) {
  std::cout << "here we are again! cuda" << std::endl;
  //AT_DISPATCH_ALL_TYPES_AND3(kHalf, kBool, kBFloat16, tensors[0].scalar_type(), "foreach_add_scalar_cuda/foreach_sub_scalar_cuda", [&]() {
  //  
  //  //std::vector<Tensor> a;
  //  //return a;
  //});
}

//static void foreach_sub_scalar_kernel_cuda(TensorList& tensors, Scalar scalar) {
//    foreach_add_scalar_kernel_cuda(tensors, -scalar);
//}

REGISTER_DISPATCH(foreach_add_scalar_stub, &foreach_add_scalar_kernel_cuda);
//REGISTER_DISPATCH(foreach_sub_scalar_stub, &foreach_sub_scalar_kernel_cuda);

}} // namespace at::native
