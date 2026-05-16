/****************************************************************************************
 *
 * Copyright (c) 2024, Shenyang Institute of Automation, Chinese Academy of Sciences
 *
 * Authors: Yanpeng Jia
 * Contact: jiayanpeng@sia.cn
 *
 ****************************************************************************************/
/*
 * SPDX-FileCopyrightText: Copyright (c) 2023 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: MIT
 *
 * Permission is hereby granted, free of charge, to any person obtaining a
 * copy of this software and associated documentation files (the "Software"),
 * to deal in the Software without restriction, including without limitation
 * the rights to use, copy, modify, merge, publish, distribute, sublicense,
 * and/or sell copies of the Software, and to permit persons to whom the
 * Software is furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in
 * all copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL
 * THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
 * FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
 * DEALINGS IN THE SOFTWARE.
 */

#include "center_pointpillars/kernel.h"
#include "center_pointpillars/preprocess_kernels.cuh"
// src: [V, T, C]  (V<=MAX_V, T=20, C=10)
// dst: [1, C, MAX_V, T]  其余位置补 0
__global__ void pack_to_nchw_kernel(
        const __half* __restrict__ vpc, // [V,P,C]
        int V, int C, int P,
        __half* __restrict__ o          // [1,C,V,P]
){
    int c  = blockIdx.y * blockDim.y + threadIdx.y; // 0..C-1
    int vp = blockIdx.x * blockDim.x + threadIdx.x; // 0..V*P-1
    if (c >= C || vp >= V*P) return;
    int v = vp / P;
    int p = vp % P;

    // in:  ((v * P) + p) * C + c
    // out: ((c * V) + v) * P + p   (N=1 省略)
    o[((c*V + v)*P) + p] = vpc[((v*P + p)*C) + c];
}

void launch_pack_to_nchw_kernel(
        const __half* vox_vpc, int V, int C, int P,
        __half* out_nchw, cudaStream_t stream
){
    dim3 block(256, 1);
    dim3 grid((V*P + block.x - 1)/block.x, C);
    pack_to_nchw_kernel<<<grid, block, 0, stream>>>(vox_vpc, V, C, P, out_nchw);
}

static __global__ void pack_idx2_kernel(
        const unsigned int* __restrict__ idx4, // [V,4]，你之前分配的是 4 个 uint/voxel
        int V, int srcH, int srcW, int dstH, int dstW,
        int* __restrict__ out_yx               // [V,2]
){
    int v = blockIdx.x * blockDim.x + threadIdx.x;
    if (v >= V) return;

    // 你要确认 idx4 里 y/x 分别存在哪两个槽。
    // 典型自定义里常见：idx4[v*4 + 0]=z, +1=y, +2=x, +3=unused 或者别的排列。
    // 先按你之前的打印猜测：(y= idx[1], x= idx[2])，如果仍无响应，再交换试 (x= idx[1], y= idx[2])
    unsigned int iy_src = idx4[v*4 + 1];
    unsigned int ix_src = idx4[v*4 + 2];

    // 线性缩放到 0..dstH-1 / 0..dstW-1
    float fy = (float)iy_src * (float)dstH / (float)srcH;
    float fx = (float)ix_src * (float)dstW / (float)srcW;

    int y = min(max((int)floorf(fy), 0), dstH - 1);
    int x = min(max((int)floorf(fx), 0), dstW - 1);

    out_yx[v*2 + 0] = y;  // 如果没响应，尝试 out_yx[v*2+0]=x; out_yx[v*2+1]=y; A/B 一次即可定位
    out_yx[v*2 + 1] = x;
}

void launch_pack_idx2_kernel(
        const unsigned int* vox_indices4, int V,
        int srcH, int srcW, int dstH, int dstW,
        int* out_yx, cudaStream_t stream
){
    dim3 block(256);
    dim3 grid((V + block.x - 1)/block.x);
    pack_idx2_kernel<<<grid, block, 0, stream>>>(
            vox_indices4, V, srcH, srcW, dstH, dstW, out_yx
    );
}

__device__ inline uint64_t hash(uint64_t k) {
    k ^= k >> 16;
    k *= 0x85ebca6b;
    k ^= k >> 13;
    k *= 0xc2b2ae35;
    k ^= k >> 16;
    return k;
}

__device__ inline void insertHashTable(const uint32_t key, uint32_t *value,
                                       const uint32_t hash_size, uint32_t *hash_table) {
    uint64_t hash_value = hash(key);
    uint32_t slot = hash_value % (hash_size / 2)/*key, value*/;
    uint32_t empty_key = UINT32_MAX;
    while (true) {
        uint32_t pre_key = atomicCAS(hash_table + slot, empty_key, key);
        if (pre_key == empty_key) {
            hash_table[slot + hash_size / 2 /*offset*/] = atomicAdd(value, 1);
            break;
        } else if (pre_key == key) {
            break;
        }
        slot = (slot + 1) % (hash_size / 2);
    }
}

__device__ inline uint32_t lookupHashTable(const uint32_t key, const uint32_t hash_size, const uint32_t *hash_table) {
    uint64_t hash_value = hash(key);
    uint32_t slot = hash_value % (hash_size / 2)/*key, value*/;
    uint32_t empty_key = UINT32_MAX;
    int cnt = 0;
    while (cnt < 100 /* need to be adjusted according to data*/) {
        cnt++;
        if (hash_table[slot] == key) {
            return hash_table[slot + hash_size / 2];
        } else if (hash_table[slot] == empty_key) {
            return empty_key;
        } else {
            slot = (slot + 1) % (hash_size / 2);
        }
    }
    return empty_key;
}

__global__ void buildHashKernel(const float *points, size_t points_size,
                                float min_x_range, float max_x_range,
                                float min_y_range, float max_y_range,
                                float min_z_range, float max_z_range,
                                float voxel_x_size, float voxel_y_size, float voxel_z_size,
                                int grid_y_size, int grid_x_size, int feature_num,
                                unsigned int *hash_table, unsigned int *real_voxel_num) {
    int point_idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (point_idx >= points_size) {
        return;
    }

    float px = points[feature_num * point_idx];
    float py = points[feature_num * point_idx + 1];
    float pz = points[feature_num * point_idx + 2];

    if( px < min_x_range || px >= max_x_range || py < min_y_range || py >= max_y_range
        || pz < min_z_range || pz >= max_z_range) {
        return;
    }

    unsigned int voxel_idx = floorf((px - min_x_range) / voxel_x_size);
    unsigned int voxel_idy = floorf((py - min_y_range) / voxel_y_size);
    unsigned int voxel_idz = floorf((pz - min_z_range) / voxel_z_size);
    unsigned int voxel_offset = voxel_idz * grid_y_size * grid_x_size
                                + voxel_idy * grid_x_size
                                + voxel_idx;
    insertHashTable(voxel_offset, real_voxel_num, points_size * 2 * 2, hash_table);
}

__global__ void voxelizationKernel(const float *points, size_t points_size,
                                   float min_x_range, float max_x_range,
                                   float min_y_range, float max_y_range,
                                   float min_z_range, float max_z_range,
                                   float voxel_x_size, float voxel_y_size, float voxel_z_size,
                                   int grid_y_size, int grid_x_size, int feature_num, int max_voxels,
                                   int max_points_per_voxel,
                                   unsigned int *hash_table, unsigned int *num_points_per_voxel,
                                   float *voxels_temp, unsigned int *voxel_indices, unsigned int *real_voxel_num) {
    int point_idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (point_idx >= points_size) {
        return;
    }

    float px = points[feature_num * point_idx];
    float py = points[feature_num * point_idx + 1];
    float pz = points[feature_num * point_idx + 2];

    if( px < min_x_range || px >= max_x_range || py < min_y_range || py >= max_y_range
        || pz < min_z_range || pz >= max_z_range) {
        return;
    }

    unsigned int voxel_idx = floorf((px - min_x_range) / voxel_x_size);
    unsigned int voxel_idy = floorf((py - min_y_range) / voxel_y_size);
    unsigned int voxel_idz = floorf((pz - min_z_range) / voxel_z_size);
    unsigned int voxel_offset = voxel_idz * grid_y_size * grid_x_size
                                + voxel_idy * grid_x_size
                                + voxel_idx;

    // scatter to voxels
    unsigned int voxel_id = lookupHashTable(voxel_offset, points_size * 2 * 2, hash_table);
    if (voxel_id >= max_voxels) {
        return;
    }

    unsigned int current_num = atomicAdd(num_points_per_voxel + voxel_id, 1);
    if (current_num < max_points_per_voxel) {
        unsigned int dst_offset = voxel_id * (feature_num * max_points_per_voxel) + current_num * feature_num;
        unsigned int src_offset = point_idx * feature_num;
        for (int feature_idx = 0; feature_idx < feature_num; ++feature_idx) {
            voxels_temp[dst_offset + feature_idx] = points[src_offset + feature_idx];
        }

        // now only deal with batch_size = 1
        // since not sure what the input format will be if batch size > 1
        uint4 idx = {0, voxel_idz, voxel_idy, voxel_idx};
        ((uint4 *)voxel_indices)[voxel_id] = idx;

    }
}

__global__ void featureExtractionKernel(float *voxels_temp,
                                        unsigned int *num_points_per_voxel,
                                        int max_points_per_voxel, int feature_num, half *voxel_features) {
    int voxel_idx = blockIdx.x * blockDim.x + threadIdx.x;
    num_points_per_voxel[voxel_idx] = num_points_per_voxel[voxel_idx] > max_points_per_voxel ?
                                      max_points_per_voxel :  num_points_per_voxel[voxel_idx];
    int valid_points_num = num_points_per_voxel[voxel_idx];
    int offset = voxel_idx * max_points_per_voxel * feature_num;
    for (int feature_idx = 0; feature_idx< feature_num; ++feature_idx) {
        for (int point_idx = 0; point_idx < valid_points_num - 1; ++point_idx) {
            voxels_temp[offset + feature_idx] += voxels_temp[offset + (point_idx + 1) * feature_num + feature_idx];
        }
        voxels_temp[offset + feature_idx] /= valid_points_num;
    }

    // move to be continuous
    for (int feature_idx = 0; feature_idx < feature_num; ++feature_idx) {
        int dst_offset = voxel_idx * feature_num;
        int src_offset = voxel_idx * feature_num * max_points_per_voxel;
        voxel_features[dst_offset + feature_idx] = __float2half(voxels_temp[src_offset + feature_idx]);
    }
}

cudaError_t featureExtractionLaunch(float *voxels_temp, unsigned int *num_points_per_voxel,
                                    const unsigned int real_voxel_num, int max_points_per_voxel, int feature_num,
                                    half *voxel_features, cudaStream_t stream)
{
    int threadNum = THREADS_FOR_VOXEL;
    dim3 blocks((real_voxel_num + threadNum - 1) / threadNum);
    dim3 threads(threadNum);
    featureExtractionKernel<<<blocks, threads, 0, stream>>>
            (voxels_temp, num_points_per_voxel,
             max_points_per_voxel, feature_num, voxel_features);
    cudaError_t err = cudaGetLastError();
    return err;
}

cudaError_t voxelizationLaunch(const float *points, size_t points_size,
                               float min_x_range, float max_x_range,
                               float min_y_range, float max_y_range,
                               float min_z_range, float max_z_range,
                               float voxel_x_size, float voxel_y_size, float voxel_z_size,
                               int grid_y_size, int grid_x_size, int feature_num, int max_voxels,
                               int max_points_per_voxel,
                               unsigned int *hash_table, unsigned int *num_points_per_voxel,
                               float *voxel_features, unsigned int *voxel_indices,
                               unsigned int *real_voxel_num, cudaStream_t stream)
{
    int threadNum = THREADS_FOR_VOXEL;
    dim3 blocks((points_size+threadNum-1)/threadNum);
    dim3 threads(threadNum);
    buildHashKernel<<<blocks, threads, 0, stream>>>
            (points, points_size,
             min_x_range, max_x_range,
             min_y_range, max_y_range,
             min_z_range, max_z_range,
             voxel_x_size, voxel_y_size, voxel_z_size,
             grid_y_size, grid_x_size, feature_num, hash_table,
             real_voxel_num);
    voxelizationKernel<<<blocks, threads, 0, stream>>>
            (points, points_size,
             min_x_range, max_x_range,
             min_y_range, max_y_range,
             min_z_range, max_z_range,
             voxel_x_size, voxel_y_size, voxel_z_size,
             grid_y_size, grid_x_size, feature_num, max_voxels,
             max_points_per_voxel, hash_table,
             num_points_per_voxel, voxel_features, voxel_indices, real_voxel_num);
    cudaError_t err = cudaGetLastError();
    return err;
}
