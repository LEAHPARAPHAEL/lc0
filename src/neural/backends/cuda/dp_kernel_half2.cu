
#include "cuda_common.h"
#include "neural/tables/activation_function.h"
#include "winograd_helper.inc"
#include "cuda_fp16.h"



namespace lczero {
namespace cudnn_backend {
/*
#define N_THREADS = 384

#define DW_TILE_W = 8
#define DW_TILE_H = 8
#define DW_THREAD_H = 1
#define DW_THREAD_W = 1
#define DW_THREADS_PER_TILE_H = DW_TILE_H / DW_THREADS_PER_TILE_H
#define DW_THREADS_PER_TILE_W = DW_TILE_W / DW_THREADS_PER_TILE_W

#define PW_TILE_W = 4
#define PW_TILE_H = 4
#define PW_THREAD_H = 1
#define PW_THREAD_W = 1
#define PW_THREADS_PER_TILE_H = PW_TILE_H / PW_THREADS_PER_TILE_H
#define PW_THREADS_PER_TILE_W = PW_TILE_W / PW_THREADS_PER_TILE_W







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

/*

//_syncthreads() waits for all threads of the block to have reached the _syncthreads()
//_shared_ declares a shared memory accessible for each thread in the same block
__global__ void FusedDepthwisePointwiseKernel(const int C_in, const int C, half2* output, const half2* input,
                              const half2* w1, const half2* b1, const half2* w2, int dw_thread_d,
                              int dw_threads_per_tile_d) {
    __shared__ half2 intermediate_output[C_in / 2 * PW_TILE_W * PW_TILE_H];

    const int thread_h = threadIdx.x;
    const int thread_w = threadIdx.y;
    const int thread_d = threadIdx.z;

    const int block_n = blockIdx.x;
    const int block_w = blockIdx.y;
    const int block_h = blockIdx.z;

    // Column number of the beginning of the thread
    const int abs_w = block_w * DW_THREADS_PER_TILE_W + thread_w;

    // Channel number of the beginning of the thread
    const int abs_d = dw_thread_d * thread_d;

    if (abs_w < 8) {
        half2 dweight0, dweight1, dweight2, dweight3, dweight4,
              dweight5, dweight6, dweight7, dweight8;
        for (int c = 0; c < thread_d ; c++){
            unsigned active_threads_mask = __activemask();

            // Shared memory is in (C,H,W) layout. 
            // This computes the number of indices in all channels before this one
            // Plus the column offset
            // What remains is the offset linked to the row, which is done in the inner loop
            const int offset_d_w = (abs_d + c) * PW_TILE_H * PW_TILE_W + thread_w;
            
            half2 my_weight;

            if (thread_h < 3 && thread_w < 3){
                my_weight = w1[(abs_d + c) * 3 * 3 + thread_h * 3 + thread_w];
            }

            dweight0 = __shfl_sync(active_threads_mask, my_weight, 0);
            dweight1 = __shfl_sync(active_threads_mask, my_weight, 1);
            dweight2 = __shfl_sync(active_threads_mask, my_weight, 2);
            dweight3 = __shfl_sync(active_threads_mask, my_weight, DW_THREADS_PER_TILE_W);
            dweight4 = __shfl_sync(active_threads_mask, my_weight, DW_THREADS_PER_TILE_W + 1);
            dweight5 = __shfl_sync(active_threads_mask, my_weight, DW_THREADS_PER_TILE_W + 2);
            dweight6 = __shfl_sync(active_threads_mask, my_weight, 2 * DW_THREADS_PER_TILE_W);
            dweight7 = __shfl_sync(active_threads_mask, my_weight, 2 * DW_THREADS_PER_TILE_W + 1);
            dweight8 = __shfl_sync(active_threads_mask, my_weight, 2 * DW_THREADS_PER_TILE_W + 2);

            // Threads can span multiple channels and rows, but only one column
            // This is why we only loop on the channels and the rows
            for (int h = 0; h < DW_THREAD_H; h ++) {
                // Row in this tile
                const int row_in_tile = (thread_h * DW_THREAD_H + h);

                // Row offset for shared memory
                const int offset_h = row_in_tile * PW_TILE_W;

                // Row in the 8 x 8 input of the channel : substract 2 for top padding
                int abs_h_input = block_h * PW_TILE_H + row_in_tile - 2;

                // Column in the 8 x 8 input of the channel : substract 2 for left padding
                // Note that as a thread spans only one column, DW_THREADS_PER_TILE = DW_TILE_W
                const int abs_w_input = block_w * PW_TILE_W + thread_w - 2;

                int global_index_in_input = block_n * C_in * 8 * 8 +
                                            (abs_d + c) * 8 * 8 +
                                            abs_h_input * 8 + 
                                            abs_w_input;

                // Accumulator
                float2 sum;
                int type_filter = C_in / (abs_d + c);
                
                // Rook filter
                if (type_filter == 0) {
                    sum = dweight0 * get_input_at(input, )
                }
            }
            
        }
    }








    

}

void FusedDepthwisePointwiseLayer(const int N, const int C_in, const int C, half2* output, const half2* input,
                              const half2* w1, const half2* b1, const half2* w2) {


    int dw_thread_d  = C_in * DW_TILE_W * DW_TILE_h / (N_THREADS * DW_THREAD_H * dw_thread_w);
    int dw_threads_per_tile_d = C_in / dw_thread_d;

    dim3 threads(DW_THREADS_PER_TILE_W, DW_THREADS_PER_TILE_H, threads_per_tile_d);

    // 1 block spans all output channels to maximize memory reuse of the depthwise convolution output. 
    dim3 blocks(N, 8 / DW_TILE_W, 8 / DW_TILE_h);

    FusedDepthwisePointwiseKernel<<<blocks, threads>>>(C_in, C, output, input, w1,
                        b1, w2, dw_thread_d, dw_threads_per_tile_d);
    

}


__global__ void pack_half_to_half2_kernel(const half* __restrict__ input,
                                          half2* __restrict__ output,
                                          int N, int C, int H, int W) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int total_elements = N * (C / 2) * H * W;

    if (tid >= total_elements) return;

    // Compute 4D index from flattened thread ID
    int w = tid % W;
    int h = (tid / W) % H;
    int c2 = (tid / (W * H)) % (C / 2);
    int n = tid / (W * H * (C / 2));

    int c0 = c2 * 2;
    int c1 = c0 + 1;

    // Compute offsets in the input tensor (NCHW layout)
    int offset0 = ((n * C + c0) * H + h) * W + w;
    int offset1 = ((n * C + c1) * H + h) * W + w;

    half2 val;
    val.x = input[offset0];
    val.y = input[offset1];

    // Compute output offset (NC/2HW)
    int out_offset = tid;
    output[out_offset] = val;
}

void pack_half_to_half2(const half* input, half2* output,
                        int N, int C, int H, int W,
                        cudaStream_t stream = 0) {
    int total_half2s = N * (C / 2) * H * W;
    int blockSize = 256;
    int gridSize = (total_half2s + blockSize - 1) / blockSize;

    pack_half_to_half2_kernel<<<gridSize, blockSize, 0, stream>>>(
        input, output, N, C, H, W);
}

__global__ void unpackHalf2ToHalf_kernel(half* op, const half2* ip, 
    int N, int C, int H, int W){

    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int total_elements = N * C * H * W;

    if (tid >= total_elements) return;

    // Compute coordinates in (N, C, H, W)
    int n = tid / (C * H * W);
    int rem = tid % (C * H * W);
    int c = rem / (H * W);
    int hw = rem % (H * W);
    int h = hw / W;
    int w = hw % W;

    // Load half2 element from input
    half2 val = ip[tid];

    // Extract two half values from half2
    half val_lo = __low2half(val);
    half val_hi = __high2half(val);

    // Compute output base index (N, C*2, H, W)
    int out_base = n * (2 * C) * H * W + c * 2 * H * W + h * W + w;

    // Store unpacked halves in consecutive channels
    op[out_base] = val_lo;
    op[out_base + H * W] = val_hi;
}

void unpack_half2_to_half(
    half* op, const half2* ip, int N, int C, int H, int W,
    cudaStream_t stream)
{
    int total_elements = N * C * H * W;
    int blockSize = 256;
    int blocks = (total_elements + blockSize - 1) / blockSize;
    unpackHalf2ToHalf_kernel<<<blocks, blockSize, 0, stream>>>(op, ip, N, C, H, W);
}

inline __device__ float2 get_input_at(const half2* ip, const int index_h,
    const int index_w, const int global_index) {
      if (index_h >= 0 && index_h < 8 && index_w >= 0
          && index_w < 8) {
        return __half22float2(ip[global_index]);
      }
      return make_float2(0.0f, 0.0f);
  }
*/
}
}