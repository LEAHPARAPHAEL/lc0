/*
  This file is part of Leela Chess Zero.
  Copyright (C) 2018 The LCZero Authors

  Leela Chess is free software: you can redistribute it and/or modify
  it under the terms of the GNU General Public License as published by
  the Free Software Foundation, either version 3 of the License, or
  (at your option) any later version.

  Leela Chess is distributed in the hope that it will be useful,
  but WITHOUT ANY WARRANTY; without even the implied warranty of
  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
  GNU General Public License for more details.

  You should have received a copy of the GNU General Public License
  along with Leela Chess.  If not, see <http://www.gnu.org/licenses/>.

  Additional permission under GNU GPL version 3 section 7

  If you modify this Program, or any covered work, by linking or
  combining it with NVIDIA Corporation's libraries from the NVIDIA CUDA
  Toolkit and the NVIDIA CUDA Deep Neural Network library (or a
  modified version of those libraries), containing parts covered by the
  terms of the respective license agreement, the licensors of this
  Program grant you additional permission to convey the resulting work.
*/
#include <iostream>
#include <stdio.h>
#include <cuda_fp16.h>


#include "cuda_common.h"
#include "neural/tables/activation_function.h"



// Allow building on an old architecture.
#if __CUDA_ARCH__ < 530
#pragma message("Architecture < 530")
#define SKIP_FP16_BITS 1
#endif
#include "winograd_helper.inc"


// For fused DWPW
#define N_THREADS 384
#define TILE_W 4
#define TILE_H 4


#define THREAD_H 1
#define THREAD_W 1
#define PARALLEL_H 4
#define PARALLEL_W 4
#define PARALLEL_D 24
#define PW_THREAD_D 4
#define DW_THREAD_D 24


// For depthwise only 
// Each block covers a quarter of the channels and the upper half of the board : 8 blocks required for one position.
// 576 threads per block : each thread covers 1 square across 4 channels, which is 72 * 32 / 4 = 576 threads.
#define DW_N_THREADS 384
#define DW_BLOCK_W 8
#define DW_BLOCK_H 4
#define PARALLEL_BLOCKS 4
#define DW_PARALLEL_D 12





/*
#define THREAD_H 1
#define THREAD_W 1
#define PARALLEL_H 4
#define PARALLEL_W 4
#define PARALLEL_D 24
*/



namespace lczero {
namespace cudnn_backend {

/////////////////////////////////////////////////////////////////////////////
//          fp16-specific kernels used by certain layers                   //
/////////////////////////////////////////////////////////////////////////////




/*
Weights layout for chess masks (from top to bottom, then left to right): 
     [w0, w1, w2, w3, w4, w5, w6, w7, w8]

- Rook :
     [ 0, 0,w0, 0, 0,
       0, 0,w1, 0, 0,
      w2,w3,w4,w5,w6,
       0, 0,w7, 0, 0,
       0, 0,w8, 0, 0]
- Bishop :
     [w0, 0, 0, 0,w1,
       0,w2, 0,w3, 0,
       0, 0,w4, 0, 0,
       0,w5, 0,w6, 0,
      w7, 0, 0, 0,w8]
- Knight :
     [ 0,w0, 0,w1, 0,
      w2, 0, 0, 0, w3,
       0, 0,w4, 0, 0,
      w5, 0, 0, 0, w6,
       0,w7, 0,w8, 0]
*/

#if __CUDA_ARCH__ >= 800
inline __device__ float get_input_at(const half* ip, const int index_h,
    const int index_w, const int global_index) {
      if (index_h >= 0 && index_h < 8 && index_w >= 0
          && index_w < 8) {
        return __half2float(__ldg(&ip[global_index]));
      }
      return 0.0f;
  }

inline __device__ half2 get_input_half2_at(const half2* ip, const int index_h,
    const int index_w, const int global_index) {
      if (index_h >= 0 && index_h < 8 && index_w >= 0
          && index_w < 8) {
        return ip[global_index];
      }
      return make_half2(0.0f, 0.0f);
  }


__device__ inline float2 fma_half2(half2 a, half2 b, float2 acc) {
    acc.x = __fmaf_rn(__half2float(__low2half(a)),
                    __half2float(__low2half(b)),
                    acc.x);
    acc.y = __fmaf_rn(__half2float(__high2half(a)),
                    __half2float(__high2half(b)),
                    acc.y);
    return acc;
}

__device__ inline float2 operator+(float2 a, float2 b) {
    return make_float2(a.x + b.x, a.y + b.y);
}


__device__ inline float2 operator*(float2 a, float2 b) {
    return make_float2(a.x * b.x, a.y * b.y);
}
#endif


  __global__ void convert_float_to_half2_kernel(const float* __restrict__ input,
                                              half2* __restrict__ output,
                                              int C, int H, int W) {
    // Total number of half2 elements is (C / 2) * H * W
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int total_half2 = (C / 2) * H * W;

    if (tid >= total_half2) return;

    // Compute (c2, h, w) from flat index
    int w = tid % W;
    int h = (tid / W) % H;
    int c2 = (tid / (H * W)); // c2 = c / 2

    int c0 = c2 * 2;
    int c1 = c0 + 1;

    // Flat indices in (C, H, W) format
    int offset0 = (c0 * H + h) * W + w;
    int offset1 = (c1 * H + h) * W + w;

    // Load, convert, and pack
    float f0 = input[offset0];
    float f1 = input[offset1];

    half2 packed;
    packed.x = __float2half(f0);
    packed.y = __float2half(f1);

    output[tid] = packed;
}

void convert_float_to_half2(const float* input, half2* output, int C, int H, int W) {

  int num_half2 = (C / 2) * H * W;
  int threads = 256;
  int blocks = (num_half2 + threads - 1) / threads;

  convert_float_to_half2_kernel<<<blocks, threads>>>(input, output, C, H, W);
}


__global__ void convert_half_to_half2_kernel_nchw(const half* __restrict__ input,
                                                  half2* __restrict__ output,
                                                  int N, int C, int H, int W) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int total_half2 = N * (C / 2) * H * W;
    if (tid >= total_half2) return;

    // Decompose flat index
    int w = tid % W;
    int h = (tid / W) % H;
    int c2 = (tid / (H * W)) % (C / 2);
    int n = tid / ((C / 2) * H * W);

    int c0 = c2 * 2;
    int c1 = c0 + 1;

    // Compute flat indices in NCHW format
    int offset0 = ((n * C + c0) * H + h) * W + w;
    int offset1 = ((n * C + c1) * H + h) * W + w;

    // Load and pack
    half2 packed;
    packed.x = input[offset0];
    packed.y = input[offset1];

    output[tid] = packed;
}

void convert_half_to_half2_nchw(const half* input, half2* output,
                                int N, int C, int H, int W) {
    int total_half2 = N * (C / 2) * H * W;
    int threads = 256;
    int blocks = (total_half2 + threads - 1) / threads;

    convert_half_to_half2_kernel_nchw<<<blocks, threads>>>(input, output, N, C, H, W);
}


__global__ void DepthwiseKernel(int C_in, half* output, const half2* input,
                              const half2* weights, const half2* biases) {
#if __CUDA_ARCH__ >= 800   

    const int block_depth = C_in / (2 * PARALLEL_BLOCKS);
    const int thread_depth = block_depth / DW_PARALLEL_D;

    /*   

    8 blocks per chess position :
    - a block covers half of the board (the upper half or the lower half), and spans
      multiple channels (in the current design, a quarter of the total channels).
    - so, in total, for an input of shape (N,C,H,W), 8N blocks are required to compute
      the result of the depthwise convolution.
    - the number of blocks must be high enough to occupy all the streaming multiprocessors
      but at the same time not too high to avoid a long queue.
    - as this number depends on the size of the batch N, a compromise must be made, efficient for
      the most common batch sizes (between 5 and 40).

                -----------------------
               /          6          /|
              /---------------------/ |
          C  /          4          /| |
            /---------------------/ |6|
           /          2          /| | |
          /---------------------/ |4|/|
         /           0         /| | / |
        ----------------------- |2|/|7|
        |                     | | / | /
        |          0          |0|/|5|/
        |                     | / | /
      H |_____________________|/|3|/
        |                     | | / 
        |                     |1|/
        |          1          | /
        |                     |/
        -----------------------
                   W

    
    384 threads per block : 
    - each block computes one spatial position, but across several channels.
    - each channel is therefore covered by 32 threads, which is exactly the number of
      threads in a warp, allowing only 9 of them to load the 9 different weights and
      share them through registers.
    - as the values are in half2 format, two adjacent channels are computed simultaneously,
      so the position is treated as if it only contained 576 / 2 = 288 channels.
    - 12 threads in parallel cover the 72 channels of the block, so each thread is
      responsible for 6 channels (12 in reality, as each channel is doubled).
    
              
               ------------------------- 
              /_ /_ /_ /_ /_ /_ /_ /_ /|
             .  .  .  .  .  .  .  .  . |
            .  .  .  .  .  .  .  .  .  |
        D  /_ /_ /_ /_ /_ /_ /_ /_ /   |
          /_ /_ /_ /_ /_ /_ /_ /_ /|   /
         / 0/ 1/  /  /  /  /  /  /||  .
        ------------------------- || .
        | 0| 1|  |  |  |  |  |  | ||/
      H |  |  |  |  |  |  |  |  | |/
        |  |  |  |  |  |  |  |  | /
        |  |  |  |  |  |  |  |31|/
        -------------------------
                    W
                  

    thread_w : the horizontal position of the thread, goes from 0 to 8
    thread_h : the vertical position of the thread, goes from 0 to 4
    thread_d : the first channel covered by this thread, with 0 being the
    first channel of the block, not necessarily the first channel overall.
    */
    const int thread_w = threadIdx.x;
    const int thread_h = threadIdx.y;
    const int thread_d = threadIdx.z;

    const int block_n = blockIdx.x;
    const int block_d = blockIdx.y;
    const int block_h = blockIdx.z;

    // Absolute horizontal position of the thread
    const int abs_w = thread_w;

    // Absolute channel number of the beginning of the thread
    const int abs_d = block_depth * block_d + thread_depth * thread_d;

    // Absolute vertical position of the thread
    const int abs_h = block_h * DW_BLOCK_H + thread_h;

    // The nine depthwise weights and the bias to be loaded
    half2 w0, w1, w2, w3, w4, w5, w6, w7, w8, b;

    // Loops over the 6 channels covered by the thread executing the kernel.
    #pragma unroll
    for (int c = 0; c < thread_depth ; c+=1){

        // Current channel (beginning of the thread + offset)
        const int current_d = abs_d + c;
            
        // Used to share the 9 weights among the threads of the same warp
        half2 shared_weight;

        /*
          Each thread fetches at most one weight and shares it with the warp.
          For the sake of simplicity, the 9 first threads fetch the 9 weights and
          the 10th gets the bias. 
        */ 
        if (thread_h * DW_BLOCK_W + thread_w < 9){
            shared_weight = weights[current_d * 9 + thread_h * DW_BLOCK_W + thread_w];
        }
        if (thread_h * DW_BLOCK_W + thread_w == 9){
            shared_weight = biases[current_d];
        }


        /*
          The threads of the warp share the 9 weights and the bias using registers.
          For example, the first weight (w0) was fetched by the first thread of the warp
          (the thread at index 0), so the last argument of the __shfl_sync function is 0.
        */ 

        unsigned active_threads_mask = __activemask();

        w0 = __shfl_sync(active_threads_mask, shared_weight, 0);
        w1 = __shfl_sync(active_threads_mask, shared_weight, 1);
        w2 = __shfl_sync(active_threads_mask, shared_weight, 2);
        w3 = __shfl_sync(active_threads_mask, shared_weight, 3);
        w4 = __shfl_sync(active_threads_mask, shared_weight, 4);
        w5 = __shfl_sync(active_threads_mask, shared_weight, 5);
        w6 = __shfl_sync(active_threads_mask, shared_weight, 6);
        w7 = __shfl_sync(active_threads_mask, shared_weight, 7);
        w8 = __shfl_sync(active_threads_mask, shared_weight, 8);
        b = __shfl_sync(active_threads_mask, shared_weight, 9);
            

        /*
          Batch + channel index : 
          - misses only the height and width offsets to have
            the final index. 
          - counts with packed channels, hence the C_in / 2, so will have to be doubled when
            going back to half format for the output. 
        */ 

        const int offset_nc = (block_n * C_in / 2 + current_d) * 64;



        // Row in the 8 x 8 input of the channel : substract 2 for top padding
        const int abs_h_input = block_h * DW_BLOCK_H + thread_h - 2;

        // Column in the 8 x 8 input of the channel : substract 2 for left padding
        const int abs_w_input = thread_w - 2;

        const int index_input = offset_nc + abs_h_input * 8 + abs_w_input;

        // Accumulator
        half2 sum = make_half2(0.0f, 0.0f);

        /*
        Weights layout for chess masks (from top to bottom, then left to right): 
        [w0, w1, w2, w3, w4, w5, w6, w7, w8]

        - Rook (first third of the input channels):
          [0, 0,w0, 0, 0,
           0, 0,w1, 0, 0,
          w2,w3,w4,w5,w6,
           0, 0,w7, 0, 0,
           0, 0,w8, 0, 0]

        - Bishop (second third of the input channels):
          [w0, 0, 0, 0,w1,
            0,w2, 0,w3, 0,
            0, 0,w4, 0, 0,
            0,w5, 0,w6, 0,
           w7, 0, 0, 0,w8]

        - Knight (last third of the input channels):
          [0,w0, 0,w1, 0,
          w2, 0, 0, 0, w3,
           0, 0,w4, 0, 0,
          w5, 0, 0, 0, w6,
           0,w7, 0,w8, 0]



        A same padding is applied, which means that 2 rows and columns are added at each end of the board
        to keep the same size after the 5x5 convolution.
        For example, consider the first thread of the first channel of the first position of the batch :
        - block_n = block_d = block_h = thread_d = thread_w = thread_h = current_d = offset_nc = 0
        - first third of the channels, so it's a rook kernel.
        - we compute abs_h_input = -2, abs_w_input = -2, index_input = -18.
        - -18 marks the index of the X mark in the padded board.
        - as long as abs_h_input < 0 or abs_h_input > 7 or abs_w_input < 0 or abs_w_input > 7, the input
          will be 0 because out of bounds. 
        - so the first computation to be potentially non-zero is the one at the center of the kernel. We check
          that it corresponds to index_input + 18 = 0, which is indeed the first position of the board. 

       
        X  0  *  0  0  0  0  0  0  0  0  0                   
        0  0  *  0  0  0  0  0  0  0  0  0
        *  * [*][*][*][ ][ ][ ][ ][ ] 0  0
        0  0 [*][ ][ ][ ][ ][ ][ ][ ] 0  0
        0  0 [*][ ][ ][ ][ ][ ][ ][ ] 0  0
        0  0 [ ][ ][ ][ ][ ][ ][ ][ ] 0  0
        0  0 [ ][ ][ ][ ][ ][ ][ ][ ] 0  0
        0  0 [ ][ ][ ][ ][ ][ ][ ][ ] 0  0
        0  0 [ ][ ][ ][ ][ ][ ][ ][ ] 0  0
        0  0 [ ][ ][ ][ ][ ][ ][ ][ ] 0  0
        0  0  0  0  0  0  0  0  0  0  0  0
        0  0  0  0  0  0  0  0  0  0  0  0

      */
      

        // Rook filter
        if (2 * current_d < (C_in / 3)) {
            sum = __hfma2(w0, get_input_half2_at(input, abs_h_input, abs_w_input + 2, index_input + 2), sum);
            sum = __hfma2(w1, get_input_half2_at(input, abs_h_input + 1, abs_w_input + 2, index_input + 10), sum);
            sum = __hfma2(w2, get_input_half2_at(input, abs_h_input + 2, abs_w_input, index_input + 16), sum);
            sum = __hfma2(w3, get_input_half2_at(input, abs_h_input + 2 , abs_w_input + 1, index_input + 17), sum);
            sum = __hfma2(w4, get_input_half2_at(input, abs_h_input + 2, abs_w_input + 2, index_input + 18), sum);
            sum = __hfma2(w5, get_input_half2_at(input, abs_h_input + 2, abs_w_input + 3, index_input + 19), sum);
            sum = __hfma2(w6, get_input_half2_at(input, abs_h_input + 2, abs_w_input + 4, index_input + 20), sum);
            sum = __hfma2(w7, get_input_half2_at(input, abs_h_input + 3, abs_w_input + 2, index_input + 26), sum);
            sum = __hfma2(w8, get_input_half2_at(input, abs_h_input + 4, abs_w_input + 2, index_input + 34), sum);
        }

        // Bishop filter
        else if (2 * current_d < (2 * C_in / 3)) {
            sum = __hfma2(w0, get_input_half2_at(input, abs_h_input, abs_w_input, index_input), sum);
            sum = __hfma2(w1, get_input_half2_at(input, abs_h_input, abs_w_input + 4, index_input + 4), sum);
            sum = __hfma2(w2, get_input_half2_at(input, abs_h_input + 1, abs_w_input + 1, index_input + 9), sum);
            sum = __hfma2(w3, get_input_half2_at(input, abs_h_input + 1 , abs_w_input + 3, index_input + 11), sum);
            sum = __hfma2(w4, get_input_half2_at(input, abs_h_input + 2, abs_w_input + 2, index_input + 18), sum);
            sum = __hfma2(w5, get_input_half2_at(input, abs_h_input + 3, abs_w_input + 1, index_input + 25), sum);
            sum = __hfma2(w6, get_input_half2_at(input, abs_h_input + 3, abs_w_input + 3, index_input + 27), sum);
            sum = __hfma2(w7, get_input_half2_at(input, abs_h_input + 4, abs_w_input, index_input + 32), sum);
            sum = __hfma2(w8, get_input_half2_at(input, abs_h_input + 4, abs_w_input + 4, index_input + 36), sum);
        }

        // Knight filter
        else {
            sum = __hfma2(w0, get_input_half2_at(input, abs_h_input, abs_w_input + 1, index_input + 1), sum);
            sum = __hfma2(w1, get_input_half2_at(input, abs_h_input, abs_w_input + 3, index_input + 3), sum);
            sum = __hfma2(w2, get_input_half2_at(input, abs_h_input + 1, abs_w_input, index_input + 8), sum);
            sum = __hfma2(w3, get_input_half2_at(input, abs_h_input + 1 , abs_w_input + 4, index_input + 12), sum);
            sum = __hfma2(w4, get_input_half2_at(input, abs_h_input + 2, abs_w_input + 2, index_input + 18), sum);
            sum = __hfma2(w5, get_input_half2_at(input, abs_h_input + 3, abs_w_input, index_input + 24), sum);
            sum = __hfma2(w6, get_input_half2_at(input, abs_h_input + 3, abs_w_input + 4, index_input + 28), sum);
            sum = __hfma2(w7, get_input_half2_at(input, abs_h_input + 4, abs_w_input + 1, index_input + 33), sum);
            sum = __hfma2(w8, get_input_half2_at(input, abs_h_input + 4, abs_w_input + 3, index_input + 35), sum);
        }
                
        // Adds the bias to the sum
        sum = __hadd2(sum, b);

        // ReLU ! 
        sum = __hmax2(sum, __float2half2_rn(0.0f));

        
        /*
          Writes the result in the output, splitting the two halves of the sums, which correspond to the channels 
          at position 2*current_d and 2*current_d + 1, hence the 64 offset. 
        */ 
        output[2 * offset_nc + abs_h * 8 + abs_w] = __low2half(sum);
        output[2 * offset_nc + 64 + abs_h * 8 + abs_w] = __high2half(sum);
     
    }
            
#endif
}



void DepthwiseEval(int N, int C_in, half* output, const half* input, void* scratch,
                              const half2* w1, const half2* b1, cudaStream_t stream) {

    //std::cout << "Number of positions in the batch : " << N << std::endl;
    convert_half_to_half2_nchw(input, (half2*)scratch,
                         N, C_in, 8, 8); 

    dim3 threads(DW_BLOCK_W, DW_BLOCK_H, DW_PARALLEL_D);

    dim3 blocks(N, PARALLEL_BLOCKS, 8 / DW_BLOCK_H);
    DepthwiseKernel<<<blocks, threads>>>(C_in, output, (half2*)scratch, w1, b1);


    /*
    cudaDeviceSynchronize();
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
    std::cerr << "Kernel launch failed: " << cudaGetErrorString(err) << "\n";
    }
  

    std::exit(0);
    */
    

}




__global__ void FusedDWPWKernel(int C_in, int C, half* output, const half2* input,
                              const half2* w1, const half2* b1, const half2* w2) {
#if __CUDA_ARCH__ >= 800   

    //extern __shared__ half2 raw_mem[];
    //half2* intermediate = reinterpret_cast<half2*>(
    //        (reinterpret_cast<uintptr_t>(raw_mem) + 7) & ~static_cast<uintptr_t>(0x7));
    extern __shared__ half2 intermediate[];

    const int thread_w = threadIdx.x;
    const int thread_h = threadIdx.y;
    const int thread_d = threadIdx.z;

    const int block_n = blockIdx.x;
    const int block_w = blockIdx.y;
    const int block_h = blockIdx.z;

    // Column number of the beginning of the thread
    const int abs_w = block_w * TILE_W + thread_w * THREAD_W;

    // Channel number of the beginning of the thread
    const int abs_d = DW_THREAD_D * thread_d;

    // Row number of the beginning of the thread
    const int abs_h = block_h * TILE_H + thread_h * THREAD_H;

    // Specificaly for this configuration : must be generalized later
    // 16 threads before cycling to the same spatial position
    int warp_offset = thread_d % 2 == 0 ? 0 : 16;

    if (abs_w < 8) {
        half2 dweight0, dweight1, dweight2, dweight3, dweight4,
              dweight5, dweight6, dweight7, dweight8, dbias;

        for (int c = 0; c < DW_THREAD_D ; c+=2){

            const int current_d = abs_d + c;
            // Shared memory is in (C,H,W) layout. 
            // This computes the number of indices in all channels before this one
            // Plus the column offset
            // What remains is the offset linked to the row, which is done in the inner loop 
            const int offset_d_w = current_d / 2 * TILE_H * TILE_W + thread_w;
            
            half2 my_weight;

            //Each thread fetches at most one weight and shares it with the warp
            if (thread_h < 3 && thread_w < 3){
                my_weight = w1[current_d / 2 * 9 + thread_h * 3 + thread_w];
            }
            if (thread_h == 0 && thread_w == 3){
                my_weight = b1[current_d / 2];
            }
            unsigned active_threads_mask = __activemask();


            dweight0 = __shfl_sync(active_threads_mask, my_weight, warp_offset);
            dweight1 = __shfl_sync(active_threads_mask, my_weight, warp_offset + 1);
            dweight2 = __shfl_sync(active_threads_mask, my_weight, warp_offset + 2 );
            dweight3 = __shfl_sync(active_threads_mask, my_weight, warp_offset + PARALLEL_W );
            dweight4 = __shfl_sync(active_threads_mask, my_weight, warp_offset + PARALLEL_W + 1);
            dweight5 = __shfl_sync(active_threads_mask, my_weight, warp_offset + PARALLEL_W + 2);
            dweight6 = __shfl_sync(active_threads_mask, my_weight, warp_offset + 2 * PARALLEL_W );
            dweight7 = __shfl_sync(active_threads_mask, my_weight, warp_offset + 2 * PARALLEL_W  + 1);
            dweight8 = __shfl_sync(active_threads_mask, my_weight, warp_offset + 2 * PARALLEL_W + 2);
            dbias = __shfl_sync(active_threads_mask, my_weight, warp_offset + 3);
            

            const int offset_d = (block_n * C_in + current_d) / 2 * 8 * 8;

            for (int h = 0; h < THREAD_H; h++) {
                // Row in this tile
                const int row_in_tile = (thread_h * THREAD_H + h);

                // Row offset for shared memory
                const int offset_h = row_in_tile * TILE_W;

                // Row in the 8 x 8 input of the channel : substract 2 for top padding
                const int abs_h_input = block_h * TILE_H + row_in_tile - 2;

                // Column in the 8 x 8 input of the channel : substract 2 for left padding
                const int abs_w_input = block_w * TILE_W + thread_w * THREAD_W - 2;

                const int index_input = offset_d + abs_h_input * 8 + abs_w_input;

                // Accumulator
                half2 sum = make_half2(0.0f, 0.0f);

                /*
                Weights layout for chess masks (from top to bottom, then left to right): 
                    [w0, w1, w2, w3, w4, w5, w6, w7, w8]

                - Rook (first third of the input channels):
                    [ 0, 0,w0, 0, 0,
                      0, 0,w1, 0, 0,
                      w2,w3,w4,w5,w6,
                      0, 0,w7, 0, 0,
                      0, 0,w8, 0, 0]
                - Bishop (second third of the input channels):
                    [w0, 0, 0, 0,w1,
                      0,w2, 0,w3, 0,
                      0, 0,w4, 0, 0,
                      0,w5, 0,w6, 0,
                      w7, 0, 0, 0,w8]
                - Knight (last third of the input channels):
                    [ 0,w0, 0,w1, 0,
                      w2, 0, 0, 0, w3,
                      0, 0,w4, 0, 0,
                      w5, 0, 0, 0, w6,
                      0,w7, 0,w8, 0]
                */

                // Rook filter
                if (current_d < (C_in / 3)) {
                    sum = __hfma2(dweight0, get_input_half2_at(input, abs_h_input, abs_w_input + 2, index_input + 2), sum);
                    sum = __hfma2(dweight1, get_input_half2_at(input, abs_h_input + 1, abs_w_input + 2, index_input + 10), sum);
                    sum = __hfma2(dweight2, get_input_half2_at(input, abs_h_input + 2, abs_w_input, index_input + 16), sum);
                    sum = __hfma2(dweight3, get_input_half2_at(input, abs_h_input + 2 , abs_w_input + 1, index_input + 17), sum);
                    sum = __hfma2(dweight4, get_input_half2_at(input, abs_h_input + 2, abs_w_input + 2, index_input + 18), sum);
                    sum = __hfma2(dweight5, get_input_half2_at(input, abs_h_input + 2, abs_w_input + 3, index_input + 19), sum);
                    sum = __hfma2(dweight6, get_input_half2_at(input, abs_h_input + 2, abs_w_input + 4, index_input + 20), sum);
                    sum = __hfma2(dweight7, get_input_half2_at(input, abs_h_input + 3, abs_w_input + 2, index_input + 26), sum);
                    sum = __hfma2(dweight8, get_input_half2_at(input, abs_h_input + 4, abs_w_input + 2, index_input + 34), sum);

                    /*
                    sum = __hfma2(dummy_weight, get_input_half2_at(input, abs_h_input, abs_w_input, index_input), sum);
                    sum = __hfma2(dummy_weight, get_input_half2_at(input, abs_h_input, abs_w_input + 1, index_input + 1), sum);
                    sum = __hfma2(dummy_weight, get_input_half2_at(input, abs_h_input, abs_w_input + 3, index_input + 3), sum);
                    sum = __hfma2(dummy_weight, get_input_half2_at(input, abs_h_input, abs_w_input + 4, index_input + 4), sum);
                    sum = __hfma2(dummy_weight, get_input_half2_at(input, abs_h_input + 1, abs_w_input, index_input + 8), sum);
                    sum = __hfma2(dummy_weight, get_input_half2_at(input, abs_h_input + 1, abs_w_input + 1, index_input + 9), sum);
                    sum = __hfma2(dummy_weight, get_input_half2_at(input, abs_h_input + 1, abs_w_input + 3, index_input + 11), sum);
                    sum = __hfma2(dummy_weight, get_input_half2_at(input, abs_h_input + 1, abs_w_input + 4, index_input + 12), sum);
                    sum = __hfma2(dummy_weight, get_input_half2_at(input, abs_h_input + 3, abs_w_input, index_input + 24), sum);
                    sum = __hfma2(dummy_weight, get_input_half2_at(input, abs_h_input + 3, abs_w_input + 1, index_input + 25), sum);
                    sum = __hfma2(dummy_weight, get_input_half2_at(input, abs_h_input + 3, abs_w_input + 3, index_input + 27), sum);
                    sum = __hfma2(dummy_weight, get_input_half2_at(input, abs_h_input + 3, abs_w_input + 4, index_input + 28), sum);
                    sum = __hfma2(dummy_weight, get_input_half2_at(input, abs_h_input + 4, abs_w_input, index_input + 32), sum);
                    sum = __hfma2(dummy_weight, get_input_half2_at(input, abs_h_input + 4, abs_w_input + 1, index_input + 33), sum);
                    sum = __hfma2(dummy_weight, get_input_half2_at(input, abs_h_input + 4, abs_w_input + 3, index_input + 35), sum);
                    sum = __hfma2(dummy_weight, get_input_half2_at(input, abs_h_input + 4, abs_w_input + 4, index_input + 36), sum);
                    */

                }

                // Bishop filter
                else if (current_d < (2 * C_in / 3)) {
                    sum = __hfma2(dweight0, get_input_half2_at(input, abs_h_input, abs_w_input, index_input), sum);
                    sum = __hfma2(dweight1, get_input_half2_at(input, abs_h_input, abs_w_input + 4, index_input + 4), sum);
                    sum = __hfma2(dweight2, get_input_half2_at(input, abs_h_input + 1, abs_w_input + 1, index_input + 9), sum);
                    sum = __hfma2(dweight3, get_input_half2_at(input, abs_h_input + 1 , abs_w_input + 3, index_input + 11), sum);
                    sum = __hfma2(dweight4, get_input_half2_at(input, abs_h_input + 2, abs_w_input + 2, index_input + 18), sum);
                    sum = __hfma2(dweight5, get_input_half2_at(input, abs_h_input + 3, abs_w_input + 1, index_input + 25), sum);
                    sum = __hfma2(dweight6, get_input_half2_at(input, abs_h_input + 3, abs_w_input + 3, index_input + 27), sum);
                    sum = __hfma2(dweight7, get_input_half2_at(input, abs_h_input + 4, abs_w_input, index_input + 32), sum);
                    sum = __hfma2(dweight8, get_input_half2_at(input, abs_h_input + 4, abs_w_input + 4, index_input + 36), sum);

                    /*
                    sum = __hfma2(dummy_weight, get_input_half2_at(input, abs_h_input, abs_w_input + 1, index_input + 1), sum);
                    sum = __hfma2(dummy_weight, get_input_half2_at(input, abs_h_input, abs_w_input + 2, index_input + 2), sum);
                    sum = __hfma2(dummy_weight, get_input_half2_at(input, abs_h_input, abs_w_input + 3, index_input + 3), sum);
                    sum = __hfma2(dummy_weight, get_input_half2_at(input, abs_h_input + 1, abs_w_input, index_input + 8), sum);
                    sum = __hfma2(dummy_weight, get_input_half2_at(input, abs_h_input + 1, abs_w_input + 2, index_input + 10), sum);
                    sum = __hfma2(dummy_weight, get_input_half2_at(input, abs_h_input + 1, abs_w_input + 4, index_input + 12), sum);
                    sum = __hfma2(dummy_weight, get_input_half2_at(input, abs_h_input + 2, abs_w_input, index_input + 16), sum);
                    sum = __hfma2(dummy_weight, get_input_half2_at(input, abs_h_input + 2, abs_w_input + 1, index_input + 17), sum);
                    sum = __hfma2(dummy_weight, get_input_half2_at(input, abs_h_input + 2, abs_w_input + 3, index_input + 19), sum);
                    sum = __hfma2(dummy_weight, get_input_half2_at(input, abs_h_input + 2, abs_w_input + 4, index_input + 20), sum);
                    sum = __hfma2(dummy_weight, get_input_half2_at(input, abs_h_input + 3, abs_w_input, index_input + 24), sum);
                    sum = __hfma2(dummy_weight, get_input_half2_at(input, abs_h_input + 3, abs_w_input + 2, index_input + 26), sum);
                    sum = __hfma2(dummy_weight, get_input_half2_at(input, abs_h_input +3, abs_w_input + 4, index_input + 28), sum);
                    sum = __hfma2(dummy_weight, get_input_half2_at(input, abs_h_input + 4, abs_w_input + 1, index_input + 33), sum);
                    sum = __hfma2(dummy_weight, get_input_half2_at(input, abs_h_input + 4, abs_w_input + 2, index_input + 34), sum);
                    sum = __hfma2(dummy_weight, get_input_half2_at(input, abs_h_input + 4, abs_w_input + 3, index_input + 35), sum);
                    */
  
                }

                // Knight filter
                else {
                    sum = __hfma2(dweight0, get_input_half2_at(input, abs_h_input, abs_w_input + 1, index_input + 1), sum);
                    sum = __hfma2(dweight1, get_input_half2_at(input, abs_h_input, abs_w_input + 3, index_input + 3), sum);
                    sum = __hfma2(dweight2, get_input_half2_at(input, abs_h_input + 1, abs_w_input, index_input + 8), sum);
                    sum = __hfma2(dweight3, get_input_half2_at(input, abs_h_input + 1 , abs_w_input + 4, index_input + 12), sum);
                    sum = __hfma2(dweight4, get_input_half2_at(input, abs_h_input + 2, abs_w_input + 2, index_input + 18), sum);
                    sum = __hfma2(dweight5, get_input_half2_at(input, abs_h_input + 3, abs_w_input, index_input + 24), sum);
                    sum = __hfma2(dweight6, get_input_half2_at(input, abs_h_input + 3, abs_w_input + 4, index_input + 28), sum);
                    sum = __hfma2(dweight7, get_input_half2_at(input, abs_h_input + 4, abs_w_input + 1, index_input + 33), sum);
                    sum = __hfma2(dweight8, get_input_half2_at(input, abs_h_input + 4, abs_w_input + 3, index_input + 35), sum);

                    /*
                    sum = __hfma2(dummy_weight, get_input_half2_at(input, abs_h_input, abs_w_input, index_input), sum);
                    sum = __hfma2(dummy_weight, get_input_half2_at(input, abs_h_input, abs_w_input + 2, index_input + 2), sum);
                    sum = __hfma2(dummy_weight, get_input_half2_at(input, abs_h_input, abs_w_input + 4, index_input + 4), sum);
                    sum = __hfma2(dummy_weight, get_input_half2_at(input, abs_h_input + 1, abs_w_input + 1, index_input + 9), sum);
                    sum = __hfma2(dummy_weight, get_input_half2_at(input, abs_h_input + 1, abs_w_input + 2, index_input + 10), sum);
                    sum = __hfma2(dummy_weight, get_input_half2_at(input, abs_h_input + 1, abs_w_input + 3, index_input + 11), sum);
                    sum = __hfma2(dummy_weight, get_input_half2_at(input, abs_h_input + 2, abs_w_input, index_input + 16), sum);
                    sum = __hfma2(dummy_weight, get_input_half2_at(input, abs_h_input + 2, abs_w_input + 1, index_input + 17), sum);
                    sum = __hfma2(dummy_weight, get_input_half2_at(input, abs_h_input + 2, abs_w_input + 3, index_input + 19), sum);
                    sum = __hfma2(dummy_weight, get_input_half2_at(input, abs_h_input + 2, abs_w_input + 4, index_input + 20), sum);
                    sum = __hfma2(dummy_weight, get_input_half2_at(input, abs_h_input + 3, abs_w_input + 1, index_input + 25), sum);
                    sum = __hfma2(dummy_weight, get_input_half2_at(input, abs_h_input + 3, abs_w_input + 2, index_input + 26), sum);
                    sum = __hfma2(dummy_weight, get_input_half2_at(input, abs_h_input + 3, abs_w_input + 3, index_input + 27), sum);
                    sum = __hfma2(dummy_weight, get_input_half2_at(input, abs_h_input + 4, abs_w_input, index_input + 32), sum);
                    sum = __hfma2(dummy_weight, get_input_half2_at(input, abs_h_input + 4, abs_w_input + 2, index_input + 34), sum);
                    sum = __hfma2(dummy_weight, get_input_half2_at(input, abs_h_input + 4, abs_w_input + 4, index_input + 36), sum);
                    */
                }
                
               
                /*
                if (block_n == 0 && block_h == 0 && block_w == 0 && (offset_d_w + offset_h == 16 * 40 + 13)){
                  printf("thread_h : %d, thread_w : %d, thread_d : %d \n", thread_h, thread_w, thread_d);
                  printf("[%f,%f],\n", __half2float(dweight0.y), __half2float(get_input_half2_at(input, abs_h_input, abs_w_input + 2, index_input + 2).y));
                  printf("[%f,%f],\n", __half2float(dweight1.y), __half2float(get_input_half2_at(input, abs_h_input + 1, abs_w_input + 2, index_input + 10).y));
                  printf("[%f,%f],\n", __half2float(dweight2.y), __half2float(get_input_half2_at(input, abs_h_input + 2, abs_w_input, index_input + 16).y));
                  printf("[%f,%f],\n", __half2float(dweight3.y), __half2float(get_input_half2_at(input, abs_h_input + 2 , abs_w_input + 1, index_input + 17).y));
                  printf("[%f,%f],\n", __half2float(dweight4.y), __half2float(get_input_half2_at(input, abs_h_input + 2, abs_w_input + 2, index_input + 18).y));
                  printf("[%f,%f],\n", __half2float(dweight5.y), __half2float(get_input_half2_at(input, abs_h_input + 2, abs_w_input + 3, index_input + 19).y));
                  printf("[%f,%f],\n", __half2float(dweight6.y), __half2float(get_input_half2_at(input, abs_h_input + 2, abs_w_input + 4, index_input + 20).y));
                  printf("[%f,%f],\n", __half2float(dweight7.y), __half2float(get_input_half2_at(input, abs_h_input + 3, abs_w_input + 2, index_input + 26).y));
                  printf("[%f,%f],\n", __half2float(dweight8.y), __half2float(get_input_half2_at(input, abs_h_input + 4, abs_w_input + 2, index_input + 34).y));
                  printf("Bias : %f \n", __half2float(dbias.y));
                  printf("Dummy weight : %f \n", __half2float(dummy_weight.y));
              }
              */
                
              
                  

                  sum = __hadd2(sum, dbias);
                  sum = __hmax2(sum, __float2half2_rn(0.0f));
                  intermediate[offset_d_w + offset_h] = sum;

                /*
                if (block_n == 0 && block_h == 0 && block_w == 0 && thread_h == 3 && thread_w == 1){
                  printf("%f", __half2float(sum.x));
                  printf("%f", __half2float(sum.y));
                }
                */
                
                
            }
            
        }

      }

    __syncthreads();

    int pw_abs_d = thread_d * PW_THREAD_D;

    half2 ones = __float2half2_rn(1.0f);
    half one = __float2half_rn(1.0f);
    half2 zeros = __float2half2_rn(0.0f);
    half zero = __float2half_rn(0.0f);

    // Loops over the output channels of this thread
    for (int c_out = 0; c_out < PW_THREAD_D; c_out++) {

        const int current_d = pw_abs_d + c_out;

        const int offset_d = block_n * C * 8 * 8 + current_d * 8 * 8;

        for (int h = 0; h < THREAD_H; h++){

          const int output_index = offset_d + (abs_h + h) * 8 + abs_w;
               
          // Float accumulation to preserve numerical precision
          //float sum = 0.0f;
          half sum = __float2half_rn(0.0f);

          int offset_h_w = (thread_h * THREAD_H + h) * TILE_W + thread_w;

          for (int c_in = 0; c_in < C_in / 2; c_in++) {

            half2 pointwise_weight = w2[current_d * C_in / 2 + c_in];

            half2 input_val = intermediate[c_in * TILE_H * TILE_W + offset_h_w];

            
            input_val = __hmul2(pointwise_weight, input_val);
            //input_val = __hfma2(input_val, ones, zeros);
            
            sum = __hadd(input_val.x, sum);
            sum = __hadd(input_val.y, sum);
            
            //sum = sum + __half2float(input_val.x);
            //sum = sum + __half2float(input_val.y);

            /*
            if (output_index == 56*64 + 35 && c_in < 6) {
              printf("%d : [%f,%f],\n", 2*c_in, __half2float(pointwise_weight.x), __half2float(input_val.x));
              printf("%d : [%f,%f],\n", 2*c_in + 1, __half2float(pointwise_weight.y), __half2float(input_val.y));
            }
            */
            
            /*
            if (block_n == 0 && block_h == 0 && block_w == 0 && thread_h == 3 && thread_w == 1 && thread_d == 0 && c_out == 0)
            {
              printf("%f\n", __half2float(input_val.x));
              printf("%f\n", __half2float(input_val.y));
            }
            */

            
            

            //float2 result = __half22float2(__hmul2(pointwise_weight, input_val));


            //sum += result.x;
            //sum += result.y;

            //sum = __hfma(pointwise_weight.x, input_val.x, sum);
            //sum = __hfma(pointwise_weight.y, input_val.y, sum);
            //sum = __hadd(__hmul(pointwise_weight.x, input_val.x), sum);
            //sum = __hadd(__hmul(pointwise_weight.y, input_val.y), sum);
            //sum = sum + (__half22float2(pointwise_weight) * __half22float2(input_val));

            //sum = __fmaf_rn(__half2float(pointwise_weight.x), __half2float(input_val.x), sum);
            //sum = __fmaf_rn(__half2float(pointwise_weight.y), __half2float(input_val.y), sum);


            
            //sum += __half2float(pointwise_weight.x) *  __half2float(input_val.x);
            //sum += __half2float(pointwise_weight.y) *  __half2float(input_val.y);
          }

          //output[output_index] = __float2half_rn(sum);
          sum = __hfma(sum, one, zero);
          output[output_index] = sum;


          
          if (output_index >= 3000 && output_index < 3025){
              //printf("Output %d : %f\n", output_index, __half2float(__float2half_rn(sum)));
              printf("Output %d : %f\n", output_index, __half2float(sum));
          }
          
        }
    }
#endif
}


void FusedDWPWevalHalf2(int N, int C_in, int C, half* output, const half* input, void* scratch,
                              const half2* w1, const half2* b1, const half2* w2, cudaStream_t stream) {

    dim3 threads(PARALLEL_W, PARALLEL_H, PARALLEL_D);

    dim3 blocks(N, 8 / TILE_W, 8 / TILE_H);

    convert_half_to_half2_nchw(input, (half2*)scratch,
                         N, C_in, 8, 8); 


    FusedDWPWKernel<<<blocks, threads, C_in / 2 * TILE_W * TILE_H * sizeof(half2), stream>>>(C_in, C, output, (half2*)scratch, w1,
                      b1, w2);



    
    cudaDeviceSynchronize();
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
    std::cerr << "Kernel launch failed: " << cudaGetErrorString(err) << "\n";
    }
    
    

    std::exit(0);
    

}




__global__ void pack_weights_float_to_half2(
    const float* __restrict__ input,  
    half2* __restrict__ output,       
    int C_out, int C_in)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = C_out * (C_in / 2);

    if (idx < total) {
        int oc = idx / (C_in / 2);     
        int ic_pair = idx % (C_in / 2); 

        int in_idx0 = oc * C_in + 2 * ic_pair;
        int in_idx1 = in_idx0 + 1;

        float f0 = input[in_idx0];
        float f1 = input[in_idx1];

        half2 h2 = __halves2half2(__float2half(f0), __float2half(f1));
        output[idx] = h2;
    }
}

void pack_pointwise_weights(float* input, half2* output, const int C_in, const int C_out){

  int total_elements = C_out * (C_in / 2);
  int threads_per_block = 256;
  int blocks = (total_elements + threads_per_block - 1) / threads_per_block;

  pack_weights_float_to_half2<<<blocks, threads_per_block>>>(
      input, output, C_out, C_in);

}

































































// SE layer implementation using single fused kernel.

// N blocks.
// C threads per block.
// 'HWC' input data processed by thread block.
// Each thread processes 8x8 elements.
// K is the no. of outputs of first fully connected layer (same as no. of inputs
// for second fully connected layer).
// The kernel assumes K <= C.

template <int C, int K>
__global__ void SE_Layer_NHWC(half* output, const half* skip, const half* input,
                              const half* w1, const half* b1, const half* w2,
                              const half* b2, const half* bPrev,
                              ActivationFunction activation) {
#if __CUDA_ARCH__ >= 530
  const int elementsPerThread = 64;  // 8x8 board
  const int se_K = K;

  int n = blockIdx.x;
  int c = threadIdx.x;

  __shared__ half sharedData[C];

  half2 localData[elementsPerThread];

  half S = 0;

  half bias = 0;
  if (bPrev) bias = bPrev[c];

// 1. Global avg (1 avg per thread).
#pragma unroll
  for (int i = 0; i < elementsPerThread; i++) {
    int localIndex = i * C + c;
    int inputIndex = n * C * elementsPerThread + localIndex;
    localData[i].x = input[inputIndex] + bias;
    localData[i].y = skip[inputIndex];
    S += localData[i].x;
  }

  half avg = S / (half)elementsPerThread;
  sharedData[c] = avg;

  __syncthreads();

  // 2. First fully connected layer.
  if (c < K) {
    S = 0;

#pragma unroll
    for (int i = 0; i < C; i++) {
      S += sharedData[i] * readw1(i, c);
    }

    S += b1[c];

    S = activate(S, activation);

    sharedData[c] = S;
  }
  __syncthreads();

  // 3. Second fully connected layer.
  S = 0;
  half B = 0;
#pragma unroll
  for (int i = 0; i < K; i++) {
    half val = sharedData[i];
    S += val * readw2(i, c);
    B += val * readw2(i, c + C);
  }
  S += b2[c];
  B += b2[c + C];

  // Sigmoid (only on the scale part).
  S = (half)(1.0f / (1.0f + exp(-(float)(S))));

// 4. Scale, and add skip connection, perform relu, and write to output.
#pragma unroll
  for (int i = 0; i < elementsPerThread; i++) {
    int localIndex = i * C + c;
    int inputIndex = n * C * elementsPerThread + localIndex;
    half val = localData[i].y + localData[i].x * S + B;

    // Relu activation function.
    val = (half)activate((float)val, activation);

    output[inputIndex] = val;
  }
#endif
}

bool Se_Fp16_NHWC(int N, int C, int numFc1Out, half* output, const half* skip,
                  const half* input, const half* w1, const half* b1,
                  const half* w2, const half* b2, const half* bPrev,
                  ActivationFunction activation) {
  // TODO: Think of more elegant way to avoid this hardcoding :-/
  if (numFc1Out == 16) {
    if (C == 64) {
      SE_Layer_NHWC<64, 16>
          <<<N, C>>>(output, skip, input, w1, b1, w2, b2, bPrev, activation);
    } else {
      // TODO: support other channel counts.
      throw Exception("channel count unsupported by SE layer");
    }
  } else if (numFc1Out == 32) {
    if (C == 64) {
      SE_Layer_NHWC<64, 32>
          <<<N, C>>>(output, skip, input, w1, b1, w2, b2, bPrev, activation);
    } else if (C == 128) {
      SE_Layer_NHWC<128, 32>
          <<<N, C>>>(output, skip, input, w1, b1, w2, b2, bPrev, activation);
    } else if (C == 192) {
      SE_Layer_NHWC<192, 32>
          <<<N, C>>>(output, skip, input, w1, b1, w2, b2, bPrev, activation);
    } else if (C == 256) {
      SE_Layer_NHWC<256, 32>
          <<<N, C>>>(output, skip, input, w1, b1, w2, b2, bPrev, activation);
    } else if (C == 320) {
      SE_Layer_NHWC<320, 32>
          <<<N, C>>>(output, skip, input, w1, b1, w2, b2, bPrev, activation);
    } else if (C == 352) {
      SE_Layer_NHWC<352, 32>
          <<<N, C>>>(output, skip, input, w1, b1, w2, b2, bPrev, activation);
    } else if (C == 384) {
      SE_Layer_NHWC<384, 32>
          <<<N, C>>>(output, skip, input, w1, b1, w2, b2, bPrev, activation);
    } else {
      // TODO: support other channel counts.
      return false;
    }
  } else if (numFc1Out == 64) {
    if (C == 64) {
      SE_Layer_NHWC<64, 64>
          <<<N, C>>>(output, skip, input, w1, b1, w2, b2, bPrev, activation);
    } else if (C == 128) {
      SE_Layer_NHWC<128, 64>
          <<<N, C>>>(output, skip, input, w1, b1, w2, b2, bPrev, activation);
    } else if (C == 192) {
      SE_Layer_NHWC<192, 64>
          <<<N, C>>>(output, skip, input, w1, b1, w2, b2, bPrev, activation);
    } else if (C == 256) {
      SE_Layer_NHWC<256, 64>
          <<<N, C>>>(output, skip, input, w1, b1, w2, b2, bPrev, activation);
    } else if (C == 320) {
      SE_Layer_NHWC<320, 64>
          <<<N, C>>>(output, skip, input, w1, b1, w2, b2, bPrev, activation);
    } else if (C == 384) {
      SE_Layer_NHWC<384, 64>
          <<<N, C>>>(output, skip, input, w1, b1, w2, b2, bPrev, activation);
    } else {
      // TODO: support other channel counts.
      return false;
    }
  } else if (numFc1Out == 48) {
      if (C == 96) {
        SE_Layer_NHWC<96,48>
              <<<N, C>>>(output, skip, input, w1, b1, w2, b2, bPrev, activation);
      }

  } else {
    // TODO: support other sizes.
    return false;
  }
  ReportCUDAErrors(cudaGetLastError());
  return true;
}

// Get board for this thread from shared memory.
// We are just using shared memory to store local thread data in this kernel to
// help reduce some register pressure and spills to local memory.
#define BOARD(y, x) shboard[(y)*8 + (x)]

// input is in transformed space (HWNC layout) --- output of GEMM
// output is also in transformed space (HWNC layout) --- input to GEMM (for
// next layer)
// 'C' threads per block
// 'N' blocks
// Every thread generates an entire board/plane (8x8 elements).
template <ActivationFunction activation, bool use_bias, bool use_skip>
__global__ __launch_bounds__(
    kMaxResBlockFusingSeKFp16Ampere,
    1) void OutputInputTransformKernel_fp16_shmem_board(int N, int C, int se_K,
                                                        half* output,
                                                        const half* input,
                                                        half* skip,
                                                        const half* bias,
                                                        const half* w1,
                                                        const half* b1,
                                                        const half* w2,
                                                        const half* b2) {
#if __CUDA_ARCH__ >= 530
  int k = threadIdx.x;
  int n = blockIdx.x;

  extern __shared__ half _sboard[];
  half* shboard = &_sboard[k * 72];  // 72 instead of 64 to reduce shared
                                     // memory bank conflicts.
  half b = bias[k];

#pragma unroll
  for (int hStart = 0; hStart < 8; hStart += 4)
#pragma unroll
    for (int wStart = 0; wStart < 8; wStart += 4) {
      //  i) read to per thread registers (for doing output transform)
      int shln = n * 4 + (hStart / 4) * 2 + (wStart / 4);
      half outElTransformed[6][6];
#pragma unroll
      for (int y = 0; y < 6; y++)
#pragma unroll
        for (int x = 0; x < 6; x++)
          outElTransformed[y][x] = input[TEMP_INDEX_HWNC(y, x, shln, k)];

      // ii) transform it
      half outEl[4][4];
      OutputTransform4x4(&outEl[0][0], &outElTransformed[0][0]);

#pragma unroll
      for (int y = 0; y < 4; y++)
        copyAs<uint2>(&BOARD(hStart + y, wStart), &outEl[y][0]);
    }

  // Add bias, and compute the average for SE.
  float S = 0;
  float B = 0;

#pragma unroll
  for (int y = 0; y < 8; y++) {
    half boardRow[8];
    copyAs<uint4>(&boardRow, &BOARD(y, 0));
#pragma unroll
    for (int x = 0; x < 8; x++) {
      if (use_bias) boardRow[x] += b;
      S += (float)boardRow[x];
    }
    if (use_bias) copyAs<uint4>(&BOARD(y, 0), &boardRow);
  }

  __shared__ float shared_data[kMaxResBlockFusingSeKFp16Ampere];
  float avg = S / 64;
  shared_data[k] = avg;

  int lane = k & 0x1F;
  int warp = k >> 5;
  __syncthreads();

  // First fully-connected layer for SE

  // As se_K << C, we want to loop over se_K instead of C
  // even if it means taking the sum across threads

  __shared__ float shared_sums[kMaxResBlockFusingSeKFp16Ampere / 32]
                              [kMaxResBlockFusingSeK];  // per-warp sums

  for (int i = 0; i < se_K; i++) {
    float val = shared_data[k] * float(readw1(k, i));
    val = warpReduce(val);
    if (lane == 0) shared_sums[warp][i] = val;
  }
  __syncthreads();
  if (k < se_K) {
    S = 0;
    for (int i = 0; i < C / 32; i++) S += shared_sums[i][k];

    S += (float)b1[k];
    S = activate(S, activation);
    shared_data[k] = S;
  }

  __syncthreads();

  // Second fully-connected layer for SE
  S = 0;
  for (int i = 0; i < se_K; i++) {
    float val = shared_data[i];
    S += val * float(readw2(i, k));
    B += val * float(readw2(i, k + C));
  }
  S += (float)b2[k];
  B += (float)b2[k + C];

  // Sigmoid (only on the scale part).
  S = 1.0f / (1.0f + exp(-S));

  // Scale/bias, add skip connection, perform activation, and write to output.
  for (int h = 0; h < 8; h++) {
    half boardRow[8];
    copyAs<uint4>(&boardRow[0], &BOARD(h, 0));

#pragma unroll
    for (int w = 0; w < 8; w++) {
      boardRow[w] = (half)(float(boardRow[w]) * S + B);
    }

    // residual add
    if (use_skip) {
      half skipInp[8];
      copyAs<uint4>(&skipInp[0], &skip[INDEX_NHCW(n, k, h, 0)]);
#pragma unroll
      for (int w = 0; w < 8; w++) boardRow[w] += skipInp[w];
    }

    if (activation != ACTIVATION_NONE) {
#pragma unroll
      for (int w = 0; w < 8; w++)
        boardRow[w] = (half)activate((float)boardRow[w], activation);
    }

    // write un-transformed output to 'skip' if required
    if (use_skip) {
      copyAs<uint4>(&skip[INDEX_NHCW(n, k, h, 0)], &boardRow[0]);
    }

    copyAs<uint4>(&BOARD(h, 0), &boardRow);
  }

  // Perform input transform.

  int c = k;
  // top-left
  {
    half inEl[6][6] = {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                       0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0};

#pragma unroll
    for (int i = 0; i < 5; i++)
#pragma unroll
      for (int j = 0; j < 5; j++) inEl[i + 1][j + 1] = BOARD(i, j);

    InputTransform4x4(&inEl[0][0], &inEl[0][0]);

#pragma unroll
    for (int y = 0; y < 6; y++)
#pragma unroll
      for (int x = 0; x < 6; x++)
        output[TEMP_INDEX_HWNC(y, x, n * 4 + 0, c)] = inEl[y][x];
  }

  // top-right
  {
    half inEl[6][6] = {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                       0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0};

#pragma unroll
    for (int i = 0; i < 5; i++)
#pragma unroll
      for (int j = 0; j < 5; j++) inEl[i + 1][j] = BOARD(i, j + 3);

    InputTransform4x4(&inEl[0][0], &inEl[0][0]);

#pragma unroll
    for (int y = 0; y < 6; y++)
#pragma unroll
      for (int x = 0; x < 6; x++)
        output[TEMP_INDEX_HWNC(y, x, n * 4 + 1, c)] = inEl[y][x];
  }

  // bottom-left
  {
    half inEl[6][6] = {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                       0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0};

#pragma unroll
    for (int i = 0; i < 5; i++)
#pragma unroll
      for (int j = 0; j < 5; j++) inEl[i][j + 1] = BOARD(i + 3, j);

    InputTransform4x4(&inEl[0][0], &inEl[0][0]);

#pragma unroll
    for (int y = 0; y < 6; y++)
#pragma unroll
      for (int x = 0; x < 6; x++)
        output[TEMP_INDEX_HWNC(y, x, n * 4 + 2, c)] = inEl[y][x];
  }

  // bottom-right
  {
    half inEl[6][6] = {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                       0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0};

#pragma unroll
    for (int i = 0; i < 5; i++)
#pragma unroll
      for (int j = 0; j < 5; j++) inEl[i][j] = BOARD(i + 3, j + 3);

    InputTransform4x4(&inEl[0][0], &inEl[0][0]);

#pragma unroll
    for (int y = 0; y < 6; y++)
#pragma unroll
      for (int x = 0; x < 6; x++)
        output[TEMP_INDEX_HWNC(y, x, n * 4 + 3, c)] = inEl[y][x];
  }
#endif
}

template <typename T = half, bool use_se, ActivationFunction activation,
          bool use_bias, bool use_skip>
void OutputInputTransform(int N, int C, int se_K, T* output, const T* input,
                          const T* skip, const T* bias, const T* w1,
                          const T* b1, const T* w2, const T* b2,
                          cudaStream_t stream) {
  // Each thread processes entire chess board.
  if (use_se == false) {
    dim3 grid_dim(DivUp(C, kOpInpTransformBlockSize), N, 1);
    OutputTransform_relu_InputTransform_kernel<half, activation, use_bias,
                                               use_skip>
        <<<grid_dim, kOpInpTransformBlockSize, 0, stream>>>(N, C, output, input,
                                                            (half*)skip, bias);
  } else if (C > kMaxResBlockFusingChannels) {
    // Use special kernel with reduced register pressure - only works on Ampere,
    // and only for fp16.
    if (C <= kMaxResBlockFusingSeKFp16Ampere) {
      cudaFuncSetAttribute(
          OutputInputTransformKernel_fp16_shmem_board<activation, use_bias,
                                                      use_skip>,
          cudaFuncAttributeMaxDynamicSharedMemorySize, 72 * C * sizeof(half));
      OutputInputTransformKernel_fp16_shmem_board<activation, use_bias,
                                                  use_skip>
          <<<N, C, 72 * C * sizeof(half), stream>>>(
              N, C, se_K, (half*)output, (const half*)input, (half*)skip,
              (half*)bias, (half*)w1, (half*)b1, (half*)w2, (half*)b2);
    } else {
      throw Exception(
          "res block fusing opt not supported for the given data type and no "
          "of filters\n");
    }
  } else {
    OutputTransform_SE_relu_InputTransform_kernel<half, activation, use_bias,
                                                  use_skip>
        <<<N, C, 0, stream>>>(N, C, se_K, output, input, (half*)skip, bias, w1,
                              b1, w2, b2);
  }
  ReportCUDAErrors(cudaGetLastError());
}


template void FilterTransform<half>(int N, int C, half* transformedFilter,
                                    const half* filter);

template void InputTransform<half, true>(int N, int C, half* transformed_input,
                                         const half* input,
                                         cudaStream_t stream);
template void InputTransform<half, false>(int N, int C, half* transformed_input,
                                          const half* input,
                                          cudaStream_t stream);

template void OutputTransform<half, true, ACTIVATION_RELU, true, true, false,
                              false>(int N, int C, int se_K, half* output,
                                     const half* input, const half* skip,
                                     const half* bias, const half* w1,
                                     const half* b1, const half* w2,
                                     const half* b2, cudaStream_t stream);

template void OutputTransform<half, false, ACTIVATION_RELU, true, true, false,
                              false>(int N, int C, int se_K, half* output,
                                     const half* input, const half* skip,
                                     const half* bias, const half* w1,
                                     const half* b1, const half* w2,
                                     const half* b2, cudaStream_t stream);

template void OutputTransform<half, true, ACTIVATION_RELU, true, true, true,
                              false>(int N, int C, int se_K, half* output,
                                     const half* input, const half* skip,
                                     const half* bias, const half* w1,
                                     const half* b1, const half* w2,
                                     const half* b2, cudaStream_t stream);

template void OutputTransform<half, false, ACTIVATION_RELU, true, true, true,
                              false>(int N, int C, int se_K, half* output,
                                     const half* input, const half* skip,
                                     const half* bias, const half* w1,
                                     const half* b1, const half* w2,
                                     const half* b2, cudaStream_t stream);

template void OutputTransform<half, false, ACTIVATION_RELU, true, false, false,
                              false>(int N, int C, int se_K, half* output,
                                     const half* input, const half* skip,
                                     const half* bias, const half* w1,
                                     const half* b1, const half* w2,
                                     const half* b2, cudaStream_t stream);

template void OutputTransform<half, false, ACTIVATION_RELU, true, false, false,
                              true>(int N, int C, int se_K, half* output,
                                    const half* input, const half* skip,
                                    const half* bias, const half* w1,
                                    const half* b1, const half* w2,
                                    const half* b2, cudaStream_t stream);

template void OutputTransform<half, true, ACTIVATION_RELU, true, true, true,
                              true>(int N, int C, int se_K, half* output,
                                    const half* input, const half* skip,
                                    const half* bias, const half* w1,
                                    const half* b1, const half* w2,
                                    const half* b2, cudaStream_t stream);

template void OutputTransform<half, true, ACTIVATION_MISH, true, true, false,
                              false>(int N, int C, int se_K, half* output,
                                     const half* input, const half* skip,
                                     const half* bias, const half* w1,
                                     const half* b1, const half* w2,
                                     const half* b2, cudaStream_t stream);

template void OutputTransform<half, false, ACTIVATION_MISH, true, true, false,
                              false>(int N, int C, int se_K, half* output,
                                     const half* input, const half* skip,
                                     const half* bias, const half* w1,
                                     const half* b1, const half* w2,
                                     const half* b2, cudaStream_t stream);

template void OutputTransform<half, true, ACTIVATION_MISH, true, true, true,
                              false>(int N, int C, int se_K, half* output,
                                     const half* input, const half* skip,
                                     const half* bias, const half* w1,
                                     const half* b1, const half* w2,
                                     const half* b2, cudaStream_t stream);

template void OutputTransform<half, false, ACTIVATION_MISH, true, true, true,
                              false>(int N, int C, int se_K, half* output,
                                     const half* input, const half* skip,
                                     const half* bias, const half* w1,
                                     const half* b1, const half* w2,
                                     const half* b2, cudaStream_t stream);

template void OutputTransform<half, false, ACTIVATION_MISH, true, false, false,
                              false>(int N, int C, int se_K, half* output,
                                     const half* input, const half* skip,
                                     const half* bias, const half* w1,
                                     const half* b1, const half* w2,
                                     const half* b2, cudaStream_t stream);

template void OutputTransform<half, false, ACTIVATION_MISH, true, false, false,
                              true>(int N, int C, int se_K, half* output,
                                    const half* input, const half* skip,
                                    const half* bias, const half* w1,
                                    const half* b1, const half* w2,
                                    const half* b2, cudaStream_t stream);

template void OutputTransform<half, true, ACTIVATION_MISH, true, true, true,
                              true>(int N, int C, int se_K, half* output,
                                    const half* input, const half* skip,
                                    const half* bias, const half* w1,
                                    const half* b1, const half* w2,
                                    const half* b2, cudaStream_t stream);

template void OutputTransform<half, false, ACTIVATION_NONE, true, false, false,
                              false>(int N, int C, int se_K, half* output,
                                     const half* input, const half* skip,
                                     const half* bias, const half* w1,
                                     const half* b1, const half* w2,
                                     const half* b2, cudaStream_t stream);

template void OutputInputTransform<half, true, ACTIVATION_RELU, true, true>(
    int N, int C, int se_K, half* output, const half* input, const half* skip,
    const half* bias, const half* w1, const half* b1, const half* w2,
    const half* b2, cudaStream_t stream);

template void OutputInputTransform<half, false, ACTIVATION_RELU, true, true>(
    int N, int C, int se_K, half* output, const half* input, const half* skip,
    const half* bias, const half* w1, const half* b1, const half* w2,
    const half* b2, cudaStream_t stream);

template void OutputInputTransform<half, false, ACTIVATION_RELU, true, false>(
    int N, int C, int se_K, half* output, const half* input, const half* skip,
    const half* bias, const half* w1, const half* b1, const half* w2,
    const half* b2, cudaStream_t stream);

template void OutputInputTransform<half, true, ACTIVATION_MISH, true, true>(
    int N, int C, int se_K, half* output, const half* input, const half* skip,
    const half* bias, const half* w1, const half* b1, const half* w2,
    const half* b2, cudaStream_t stream);

template void OutputInputTransform<half, false, ACTIVATION_MISH, true, true>(
    int N, int C, int se_K, half* output, const half* input, const half* skip,
    const half* bias, const half* w1, const half* b1, const half* w2,
    const half* b2, cudaStream_t stream);

template void OutputInputTransform<half, false, ACTIVATION_MISH, true, false>(
    int N, int C, int se_K, half* output, const half* input, const half* skip,
    const half* bias, const half* w1, const half* b1, const half* w2,
    const half* b2, cudaStream_t stream);

}  // namespace cudnn_backend
}  // namespace lczero
