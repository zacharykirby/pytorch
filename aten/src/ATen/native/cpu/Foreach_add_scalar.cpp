#include <ATen/Dispatch.h>
#include <ATen/native/ForeachOps.h>

namespace at { namespace native {

void foreach_add_scalar_kernel_cpu(TensorList tensors, Scalar scalar) {
  std::cout << "here we are again cpu!" << std::endl;
  //AT_DISPATCH_ALL_TYPES_AND3(kHalf, kBool, kBFloat16, tensors[0].scalar_type(), "foreach_add_scalar_cpu/foreach_sub_scalar_cpu", [&]() {
  //  
  //  //std::vector<Tensor> a;
  //  //return a;
  //});
}

REGISTER_DISPATCH(foreach_add_scalar_stub, &foreach_add_scalar_kernel_cpu);
//REGISTER_DISPATCH(foreach_sub_scalar_stub, &foreach_add_scalar_kernel_cpu);

} // namespace native
} // namespace at
