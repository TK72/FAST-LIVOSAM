//
// Created by liuminzhe on 25-11-9.
//
#pragma once
#ifndef CATKIN_FASTLIVOSAM_DYNAMIC_REMOVE_PREPROCESS_KERNELS_CUH
#define CATKIN_FASTLIVOSAM_DYNAMIC_REMOVE_PREPROCESS_KERNELS_CUH

#endif //CATKIN_FASTLIVOSAM_DYNAMIC_REMOVE_PREPROCESS_KERNELS_CUH

#include <cuda_runtime.h>
#include <cuda_fp16.h>

void launch_pack_to_nchw_kernel(  // from [V,P,C] to [1,C,V,P]
        const __half* vox_vpc, int V, int C, int P,
        __half* out_nchw, cudaStream_t stream);

// 把 [V,4] 的 voxel 索引（srcH×srcW 网格）缩放到网络 BEV 特征图（dstH×dstW）
// 注意：你可以在调用处传明确的 srcH = params_.getGridYSize(), srcW = params_.getGridXSize()
//       dstH = 128, dstW = 128 （从你的 binding 里读到）
void launch_pack_idx2_kernel(
        const unsigned int* vox_indices4, int V,
        int srcH, int srcW, int dstH, int dstW,
        int* out_yx, cudaStream_t stream);