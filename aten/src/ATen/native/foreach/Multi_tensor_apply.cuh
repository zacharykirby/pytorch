
#include <iostream>
#include <math.h>
#include <torch/torch.h>
#include <torch/cuda.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <ATen/ATen.h>
#include <ATen/AccumulateType.h>
#include <ATen/cuda/CUDAContext.h>
#include <ATen/cuda/Exceptions.h>
#include <ATen/native/foreach/Utils.cuh>

#define BLOCK_SIZE 1
#define CHUNK_SIZE 1

namespace at { namespace native {

namespace {

//TODO: rename tl
//remove T
// 

template<typename T, typename U, typename... ArgTypes>
__global__ void 
__launch_bounds__(BLOCK_SIZE /*maxThreadsPerBlock*/,
                  1          /*minBlocksPerMultiprocessor*/)
multi_tensor_apply_kernel(
    int chunk_size,
    T tl,
    U callable,
    ArgTypes... args) {
  // Hand the chunk information to the user-supplied functor to process however it likes.
  callable(chunk_size, tl, args...); 
}

template<int depth, typename T, typename... ArgTypes>
void multi_tensor_apply(
    int block_size,
    int chunk_size, 
    std::vector<std::vector<at::Tensor>>& tensor_lists,
    T callable,
    ArgTypes... args) 
    {
        // TODO: 
        // 1. check sizes, dtypes, layouts, depth 
        // 2. 
        int n_tensors = tensor_lists[0].size();
        TensorListMetadata<depth> tl_meta;
    
        int loc_block_info = 0;
        int loc_tensor_info = 0;
        for(int t = 0; t < n_tensors; t++) 
        {   
            tl_meta.sizes[loc_tensor_info] = tensor_lists[0][t].numel();
            for (int d = 0; d < depth; d++) 
            {
                tl_meta.addresses[d][loc_tensor_info] = tensor_lists[d][t].data_ptr();
            }
            loc_tensor_info++;

            int chunks = (tensor_lists[0][t].numel() + chunk_size - 1)/chunk_size;
            for (int chunk = 0; chunk < chunks; chunk++) 
            {
                tl_meta.block_to_tensor[loc_block_info] = loc_tensor_info - 1;
                tl_meta.block_to_chunk[loc_block_info] = chunk;
                loc_block_info++;

                bool tensors_full = (loc_tensor_info == depth_to_max_tensors[depth-1] &&
                    chunk == chunks - 1);
                bool blocks_full = (loc_block_info == depth_to_max_blocks[depth-1]);
                bool last_chunk = (t == n_tensors - 1 && chunk == chunks - 1);
    
                if (tensors_full || blocks_full || last_chunk)
                {
                    multi_tensor_apply_kernel<<<loc_block_info, block_size, 0, at::cuda::getCurrentCUDAStream()>>>(
                        chunk_size,
                        tl_meta,
                        callable,
                        args...);
                  
                    AT_CUDA_CHECK(cudaGetLastError());
    
                    // Reset.
                    loc_block_info = 0;
                    if(chunk == chunks - 1)
                    {
                        loc_tensor_info = 0; 
                    }
                    else
                    {
                        tl_meta.sizes[0] = tl_meta.sizes[loc_tensor_info-1];
                        for(int d = 0; d < depth; d++)
                            tl_meta.addresses[d][0] = tl_meta.addresses[d][loc_tensor_info-1];
                        loc_tensor_info = 1;
                    }
                }
            }
        }
    }
} // namespace
}} // at::native