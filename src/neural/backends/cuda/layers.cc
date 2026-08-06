/*
  This file is part of Leela Chess Zero.
  Copyright (C) 2018-2019 The LCZero Authors

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
#include "layers.h"

#include <cassert>
#include <cstring>
#include <vector>

#include "cuda_common.h"
#include "kernels.h"
#include "neural/network.h"
#include "neural/tables/attention_policy_map.h"
#include "utils/fp16_utils.h"

namespace lczero {

#if 0
// debug code to dump allocation in GPU memory
template <typename T>
void dumpTensor(T* memory, int elements, const char* message, bool only_summary = false) {
    const bool fp16 = std::is_same<half, T>::value;
    printf("\n%s\n", message);
    int elementSize = (int) (fp16 ? sizeof(half) : sizeof(float));
    int bytes = elements * elementSize;
    void *temp = malloc(bytes);
    cudaMemcpy(temp, memory, bytes, cudaMemcpyDeviceToHost);
    float maxval = -std::numeric_limits<float>::max();
    float minval = std::numeric_limits<float>::max();
    int nans = 0;
    int nanss[10] {};

    for (int i = 0; i < elements; i++)
    {
        float val;
        if (fp16) 
        {
            half *arr = (half*)temp;
            val = (float)arr[i];
        }
        else
        {
            float *arr = (float *)temp;
            val = arr[i];
        }
        maxval = std::max(maxval, val);
        minval = std::min(minval, val);

        if (std::isnan(val)) {
          if (nans < 10) nanss[nans] = i;
          nans++;
        }

        if (!only_summary || i < 2 || i == elements - 1) {
          // printf("%8.4f ", val);
          // if ((i % 8) == 7) printf("\n");
          printf("%i;%.6f\n", i, val);
        }
    }
    free(temp);
    if (maxval == -std::numeric_limits<float>::max())
       maxval = std::numeric_limits<double>::quiet_NaN();
    if (minval == std::numeric_limits<float>::max())
       minval = std::numeric_limits<double>::quiet_NaN();

    printf("Max: %.6f, Min: %.6f, NaNs: %i of %i", maxval, minval, nans, elements);
    printf("\nNaN indices: ");
    for (int i=0; i<nans && i<10; i++) printf("%i ", nanss[i]);
    if (nans > 10) printf("......");
    printf("\n");
}
#endif

namespace cudnn_backend {

#define TRAP_CUDA(block_idx, phase_name) \
    do { \
        cudaError_t sync_err = cudaDeviceSynchronize(); \
        cudaError_t last_err = cudaGetLastError(); \
        cudaError_t err = (sync_err != cudaSuccess) ? sync_err : last_err; \
        if (err != cudaSuccess) { \
            printf("\n>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>\n"); \
            printf("FATAL GPU CRASH!\n"); \
            printf("Location: Block %d | Phase: %s\n", block_idx, phase_name); \
            printf("Error: %s\n", cudaGetErrorString(err)); \
            printf("<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<\n\n"); \
            fflush(stdout); \
            exit(1); \
        } \
    } while(0)


inline int GetPaddedBatchSize(int batch_size) {
  
  if (batch_size <= 32) {
      int padded = 1;
      while (padded < batch_size) {
          padded *= 2;
      }
      return padded;
  }
  constexpr int step = 32;
  return ((batch_size + step - 1) / step) * step;
  /*
  int padded = 1;
  while (padded < batch_size) {
    padded *= 2;
  }
  return padded;
  */
}

// Use Single kernel for entire SE operation.
// Right now supported only for fp16 with nhwc and it's quite a bit faster
// than using multiple passes. The flag can be set to false for debugging.
static constexpr bool kUseFusedSELayer = false;

template <typename DataType>
BaseLayer<DataType>::BaseLayer(int c, int h, int w, BaseLayer* ip, bool nhwc)
    : input_(ip), C(c), H(h), W(w), nhwc_(nhwc), use_gemm_ex_(false) {}

template <typename DataType>
BaseLayer<DataType>::BaseLayer(int c, int h, int w, BaseLayer* ip, bool nhwc,
                               bool gemm_ex)
    : input_(ip), C(c), H(h), W(w), nhwc_(nhwc), use_gemm_ex_(gemm_ex) {}

template <typename DataType>
BaseLayer<DataType>::BaseLayer(int c, int h, int w, BaseLayer* ip)
    : input_(ip),
      C(c),
      H(h),
      W(w),
      nhwc_(ip ? ip->nhwc_ : false),
      use_gemm_ex_(false) {}

#ifdef USE_CUDNN
template <typename DataType>
ConvLayer<DataType>::ConvLayer(
    BaseLayer<DataType>* prev, int C, int height, int width,
    int filter, int Cin, ActivationFunction act, bool bias, bool use_gemm_ex,
    int min_batch_size, int max_batch_size, cudnnHandle_t cudnn)
    : BaseLayer<DataType>(C, height, width, prev, true, use_gemm_ex),
      c_input_(Cin),
      filter_size_(filter),
      act_(act),
      use_bias_(bias) {

  for (int b = 1; b <= 32; b *= 2) {
        if (b >= min_batch_size) EnsureGraph(b, cudnn);
  }

  int start_batch = std::max(32, GetPaddedBatchSize(min_batch_size));
  int max_padded_batch = GetPaddedBatchSize(max_batch_size);

  for (int b = start_batch; b <= max_padded_batch; b += 32) {
      EnsureGraph(b, cudnn);
  }
}

template <typename DataType>
ConvLayer<DataType>::~ConvLayer() {
  if (weights_) ReportCUDAErrors(cudaFree(weights_));
  if (biases_) ReportCUDAErrors(cudaFree(biases_));
}

template <>
void ConvLayer<half>::LoadWeights(const std::vector<float>& weights,
                                 float* biases,
                                 void* scratch) {
  size_t num_weights = C * c_input_ * filter_size_ * filter_size_;
  size_t num_biases = C;

  ReportCUDAErrors(cudaMalloc(&weights_, num_weights * sizeof(half)));
  ReportCUDAErrors(cudaMalloc(&biases_, num_biases * sizeof(half)));

  // Copy raw host OIHW float weights to GPU staging scratchpad
  ReportCUDAErrors(cudaMemcpy(scratch, weights.data(), num_weights * sizeof(float), cudaMemcpyHostToDevice));
  
  // Slipped parameters map N=C (Output), C=c_input_ (Input). 
  // Performs layout transposition (OIHW -> OHWI) and Type Conversion (float -> half) simultaneously on GPU.
  convertNCHWtoNHWC((half*)weights_, (float*)scratch, C, c_input_, C, c_input_, filter_size_, filter_size_, 0);

  if (biases != nullptr) {
      ReportCUDAErrors(cudaMemcpy(scratch, biases, num_biases * sizeof(float), cudaMemcpyHostToDevice));
      copyTypeConverted((half*)biases_, (float*)scratch, num_biases, 0);
  } else {
      std::vector<float> zero_biases(num_biases, 0.0f);
      ReportCUDAErrors(cudaMemcpy(scratch, zero_biases.data(), num_biases * sizeof(float), cudaMemcpyHostToDevice));
      copyTypeConverted((half*)biases_, (float*)scratch, num_biases, 0);
  }
}

template <>
void ConvLayer<float>::LoadWeights(const std::vector<float>& weights,
                                  float* biases,
                                  void* scratch) {
  size_t num_weights = C * c_input_ * filter_size_ * filter_size_;
  size_t num_biases = C;

  ReportCUDAErrors(cudaMalloc(&weights_, num_weights * sizeof(float)));
  ReportCUDAErrors(cudaMalloc(&biases_, num_biases * sizeof(float)));

  // Copy raw host OIHW float weights to GPU staging scratchpad
  ReportCUDAErrors(cudaMemcpy(scratch, weights.data(), num_weights * sizeof(float), cudaMemcpyHostToDevice));

  // Performs pure layout transposition (OIHW -> OHWI) entirely on the GPU
  convertNCHWtoNHWC((float*)weights_, (float*)scratch, C, c_input_, C, c_input_, filter_size_, filter_size_, 0);

  if (biases != nullptr) {
      ReportCUDAErrors(cudaMemcpy(biases_, biases, num_biases * sizeof(float), cudaMemcpyHostToDevice));
  } else {
      ReportCUDAErrors(cudaMemset(biases_, 0, num_biases * sizeof(float)));
  }
}

template <typename DataType>
void ConvLayer<DataType>::EnsureGraph(int padded_batch, cudnnHandle_t cudnn) {
  if (plans_.count(padded_batch) > 0) return;

  namespace fe = cudnn_frontend;

  cudnn_frontend::DataType_t fe_dtype = std::is_same<DataType, half>::value ? 
      fe::DataType_t::HALF : fe::DataType_t::FLOAT;

  auto graph = std::make_shared<fe::graph::Graph>();
  graph->set_io_data_type(fe_dtype)
       .set_intermediate_data_type(fe::DataType_t::FLOAT)
       .set_compute_data_type(fe::DataType_t::FLOAT);

  // Input Activations (NHWC)
  auto X = graph->tensor(fe::graph::Tensor_attributes()
      .set_name("X")
      .set_dim({padded_batch, c_input_, this->H, this->W})
      .set_stride({this->H * this->W * c_input_, 1, this->W * c_input_, c_input_}));

  auto W = graph->tensor(fe::graph::Tensor_attributes()
      .set_name("W")
      .set_dim({this->C, c_input_, filter_size_, filter_size_})
      .set_stride({filter_size_ * filter_size_ * c_input_, 1, filter_size_ * c_input_, c_input_}));

  int padding = filter_size_ / 2;
  auto conv_options = fe::graph::Conv_fprop_attributes()
      .set_padding({padding, padding})
      .set_stride({1, 1})
      .set_dilation({1, 1});

  auto Y_conv = graph->conv_fprop(X, W, conv_options);

  // Output Activations (NHWC)
  Y_conv->set_output(true)
        .set_dim({padded_batch, this->C, this->H, this->W})
        .set_stride({this->H * this->W * this->C, 1, this->W * this->C, this->C});

  auto status = graph->build(cudnn, {fe::HeurMode_t::A});
  if (!status.is_good()) {
      throw Exception("cuDNN graph build failed for ConvLayer: " + status.get_message());
  }

  GraphPlan plan;
  plan.graph = graph;
  plan.X = X;
  plan.W = W;
  plan.Y = Y_conv; 
  plan.workspace_size = graph->get_workspace_size();

  plans_[padded_batch] = plan;
}

template <typename DataType>
void ConvLayer<DataType>::Eval(
    int batch_size, DataType* output, const DataType* input,
    const DataType* skip, void* scratch, size_t scratch_size,
    cudnnHandle_t cudnn, cublasHandle_t cublas, cudaStream_t stream,
    DataType***) {

  int padded_batch = GetPaddedBatchSize(batch_size);
  EnsureGraph(padded_batch, cudnn); 

  auto& plan = plans_[padded_batch];

  if (scratch_size < plan.workspace_size) {
      throw Exception("Scratch size is too small for cuDNN graph workspace in ConvLayer");
  }

  std::unordered_map<std::shared_ptr<cudnn_frontend::graph::Tensor_attributes>, void*> variant_pack = {
      {plan.X, (void*)input},
      {plan.W, (void*)weights_},
      {plan.Y, (void*)output}
  };

  auto status = plan.graph->execute(cudnn, variant_pack, scratch);
  if (!status.is_good()) {
      throw Exception("cuDNN graph execution failed in ConvLayer: " + status.get_message());
  }

  // Applies elements bias injection and activation loops on the contiguous channel data sequence outside the graph
  if (use_bias_) {
      addBiasBatched<DataType>(output, output, biases_, 1, batch_size * this->H * this->W, this->C, act_, stream);
  } else if (act_ != ACTIVATION_NONE) {
      addVectors(output, output, (DataType*)nullptr, batch_size * this->C * this->H * this->W, batch_size * this->C * this->H * this->W, 0, act_, stream);
  }
}


template <typename DataType>
DepthwiseConvLayer<DataType>::DepthwiseConvLayer(
    BaseLayer<DataType>* prev, int channels, int height, int width,
    ActivationFunction act, bool use_gemm_ex,  int min_batch_size, int max_batch_size, 
    int rook_channels, int bishop_channels, int knight_channels,
    cudnnHandle_t cudnn)
    // FIX 1: Toggle the 5th parameter from 'false' to 'true' to signal NHWC format
    : BaseLayer<DataType>(channels, height, width, prev, true, use_gemm_ex),
      channels_(channels),
      act_(act),
      rook_channels_(rook_channels),
      bishop_channels_(bishop_channels),
      knight_channels_(knight_channels) {

  for (int b = 1; b <= 32; b *= 2) {
        if (b >= min_batch_size) EnsureGraph(b, cudnn);
  }

  int start_batch = std::max(32, GetPaddedBatchSize(min_batch_size));
  int max_padded_batch = GetPaddedBatchSize(max_batch_size);

  for (int b = start_batch; b <= max_padded_batch; b += 32) {
      EnsureGraph(b, cudnn);
  }
}

template <typename DataType>
DepthwiseConvLayer<DataType>::~DepthwiseConvLayer() {
  if (weights_) ReportCUDAErrors(cudaFree(weights_));
  if (biases_) ReportCUDAErrors(cudaFree(biases_));
}

template <typename DataType>
void DepthwiseConvLayer<DataType>::LoadWeights(const std::vector<float>& weights,
                                               float* biases,
                                               void* scratch) {
  size_t num_weights_cudnn = channels_ * 1 * 5 * 5; 
  size_t num_biases = channels_;
  
  int weights_per_channel = weights.size() / channels_;
  assert(weights_per_channel == 25 || weights_per_channel == 9);

  ReportCUDAErrors(cudaMalloc(&weights_, num_weights_cudnn * sizeof(DataType)));
  ReportCUDAErrors(cudaMalloc(&biases_, num_biases * sizeof(DataType)));

  if (weights_per_channel == 25) {
      ReportCUDAErrors(cudaMemcpy(scratch, weights.data(), num_weights_cudnn * sizeof(float), cudaMemcpyHostToDevice));
      copyTypeConverted(weights_, (float*)scratch, num_weights_cudnn, 0);
  } 
  else {
      std::vector<float> dense_weights_host(num_weights_cudnn, 0.0f);
      const int rook_idx[9]   = {2, 7, 10, 11, 12, 13, 14, 17, 22};
      const int bishop_idx[9] = {0, 4,  6,  8, 12, 16, 18, 20, 24};
      const int knight_idx[9] = {1, 3,  5,  9, 12, 15, 19, 21, 23};

      for (int o = 0; o < channels_; o++) {
          const int* active_mask = nullptr;
          if (o < rook_channels_) {
              active_mask = rook_idx;
          } else if (o < (rook_channels_ + bishop_channels_)) {
              active_mask = bishop_idx;
          } else {
              active_mask = knight_idx;
          }

          for (int i = 0; i < 9; i++) {
              int src_compressed_idx = (o * 9) + i;
              int dst_dense_idx = (o * 25) + active_mask[i];
              dense_weights_host[dst_dense_idx] = weights[src_compressed_idx];
          }
      }

      ReportCUDAErrors(cudaMemcpy(scratch, dense_weights_host.data(), num_weights_cudnn * sizeof(float), cudaMemcpyHostToDevice));
      copyTypeConverted(weights_, (float*)scratch, num_weights_cudnn, 0);
  }

  if (biases != nullptr) {
      ReportCUDAErrors(cudaMemcpy(scratch, biases, num_biases * sizeof(float), cudaMemcpyHostToDevice));
      copyTypeConverted(biases_, (float*)scratch, num_biases, 0);
  } else {
      std::vector<float> zero_biases(num_biases, 0.0f);
      ReportCUDAErrors(cudaMemcpy(scratch, zero_biases.data(), num_biases * sizeof(float), cudaMemcpyHostToDevice));
      copyTypeConverted(biases_, (float*)scratch, num_biases, 0);
  }
}
template <typename DataType>
void DepthwiseConvLayer<DataType>::EnsureGraph(int padded_batch, cudnnHandle_t cudnn) {
  if (plans_.count(padded_batch) > 0) return;

  cudnn_frontend::DataType_t fe_dtype = std::is_same<DataType, half>::value ? 
      cudnn_frontend::DataType_t::HALF : cudnn_frontend::DataType_t::FLOAT;

  auto graph = std::make_shared<cudnn_frontend::graph::Graph>();
  graph->set_io_data_type(fe_dtype)
       .set_intermediate_data_type(cudnn_frontend::DataType_t::FLOAT)
       .set_compute_data_type(cudnn_frontend::DataType_t::FLOAT);

  // Tensor dimensions are strictly expected in logical NCHW format: {Batch, Channels, Height, Width}
  auto X = graph->tensor(cudnn_frontend::graph::Tensor_attributes()
      .set_name("X")
      .set_dim({padded_batch, channels_, 8, 8})
      .set_stride({64 * channels_, 1, 8 * channels_, channels_})); // NHWC Physical Strides

  // Filter dimensions: {Output_channels, Input_channels_per_group, Height, Width}
  // Setting Input_channels_per_group to 1 implicitly triggers the depthwise configuration!
  auto W = graph->tensor(cudnn_frontend::graph::Tensor_attributes()
      .set_name("W")
      .set_dim({channels_, 1, 5, 5})
      .set_stride({25, 25, 5, 1}));

  // FIX: Clear the non-existent .set_group_count call entirely
  auto conv_options = cudnn_frontend::graph::Conv_fprop_attributes()
      .set_padding({2, 2})
      .set_stride({1, 1})
      .set_dilation({1, 1});

  auto Y_conv = graph->conv_fprop(X, W, conv_options);

  Y_conv->set_output(true)
        .set_dim({padded_batch, channels_, 8, 8})
        .set_stride({64 * channels_, 1, 8 * channels_, channels_});

  auto status = graph->build(cudnn, {cudnn_frontend::HeurMode_t::A});
  if (!status.is_good()) {
      throw Exception("cuDNN graph build failed: " + status.get_message());
  }

  GraphPlan plan;
  plan.graph = graph;
  plan.X = X;
  plan.W = W;
  plan.Y = Y_conv; 
  plan.workspace_size = graph->get_workspace_size();

  plans_[padded_batch] = plan;
}

template <typename DataType>
void DepthwiseConvLayer<DataType>::Eval(
    int batch_size, DataType* output, const DataType* input,
    const DataType* skip, void* scratch, size_t scratch_size,
    cudnnHandle_t cudnn, cublasHandle_t cublas, cudaStream_t stream,
    DataType***) {

  int padded_batch = GetPaddedBatchSize(batch_size);
  
  EnsureGraph(padded_batch, cudnn); 

  auto& plan = plans_[padded_batch];

  if (scratch_size < plan.workspace_size) {
      throw Exception("Scratch size is too small for cuDNN graph workspace");
  }

  std::unordered_map<std::shared_ptr<cudnn_frontend::graph::Tensor_attributes>, void*> variant_pack = {
      {plan.X, (void*)input},
      {plan.W, (void*)weights_},
      {plan.Y, (void*)output}
  };

  auto status = plan.graph->execute(cudnn, variant_pack, scratch);
  if (!status.is_good()) {
      throw Exception("cuDNN graph execution failed: " + status.get_message());
  }

  addBiasBatched<DataType>(output, output, biases_, 1, batch_size * 64, channels_, act_, stream);
}
#endif


template<typename DataType>
DepthwiseCustom<DataType>::DepthwiseCustom(int C_in, int H, int W, ActivationFunction act, bool nhwc,
      int rook_channels,
      int bishop_channels, 
      int knight_channels)
    : BaseLayer<DataType>(C_in, H, W, nullptr, nhwc, false),
      c_input_(C_in),
      act_(act),
      rook_channels_(rook_channels),
      bishop_channels_(bishop_channels),
      knight_channels_(knight_channels) {
  // Allocate memory for weights (filter tensor) and biases.
  const size_t weight_size = sizeof(half) * c_input_ * 10;
  ReportCUDAErrors(cudaMalloc(&weights, weight_size));
  }

  
template <typename DataType>
void DepthwiseCustom<DataType>::LoadWeights(const std::vector<float>& pfilter, float* pBias, void* scratch) {
  assert(scratch != nullptr);

  std::vector<float> packed_weights;
  packed_weights.reserve(c_input_ * 10); 

  int weights_per_channel = pfilter.size() / c_input_;
  assert(weights_per_channel == 25 || weights_per_channel == 9);

  if (weights_per_channel == 25) {
      const int rook_idx[9]   = {2, 7, 10, 11, 12, 13, 14, 17, 22};
      const int bishop_idx[9] = {0, 4,  6,  8, 12, 16, 18, 20, 24};
      const int knight_idx[9] = {1, 3,  5,  9, 12, 15, 19, 21, 23};

      for (int o = 0; o < c_input_; o++) {
          const int* active_mask = nullptr;

          if (o < rook_channels_) active_mask = rook_idx;
          else if (o < (rook_channels_ + bishop_channels_)) active_mask = bishop_idx;
          else active_mask = knight_idx;

          for (int i = 0; i < 9; i++) {
              packed_weights.push_back(pfilter[o * 25 + active_mask[i]]);
          }
          packed_weights.push_back(pBias ? pBias[o] : 0.0f);
      }
  } 
  else {
      // COMPRESSED LAYOUT FORMAT (New style: load natively packed weights contiguously)
      for (int o = 0; o < c_input_; o++) {
          for (int i = 0; i < 9; i++) {
              packed_weights.push_back(pfilter[o * 9 + i]);
          }
          packed_weights.push_back(pBias ? pBias[o] : 0.0f);
      }
  }

  // 3. Convert and stage data into standard high-speed register layouts
  const size_t packed_size = sizeof(float) * c_input_ * 10;
  ReportCUDAErrors(
      cudaMemcpy(scratch, packed_weights.data(), packed_size, cudaMemcpyHostToDevice));
      
  if (nhwc_) {
    convert_float_to_half2_nhwc((float*)scratch, (half2*)weights, c_input_, 10, 1);
  }
  else {
    convert_float_to_half2_nchw((float*)scratch, (half2*)weights, c_input_, 10, 1);
  }
}
  
  

  template <>
  void DepthwiseCustom<half>::Eval(int N, half* output, const half* input,
            const half* input2, void* scratch, size_t scratch_size,
            cudnnHandle_t cudnn, cublasHandle_t cublas, cudaStream_t stream,
            half***) {
    
    try  {
      if (nhwc_) {
        DepthwiseEvalNHWC(N, c_input_, output, input, scratch, weights, act_, rook_channels_,
          bishop_channels_, knight_channels_, stream);
      }
        
      else {
        DepthwiseEvalNCHW(N, c_input_, output, input, scratch, weights, act_, rook_channels_,
          bishop_channels_, knight_channels_, stream);
      }
    }
    catch (const std::runtime_error& e) {
        std::cerr << "Runtime error: " << e.what() << std::endl;
    }
    catch (const std::exception& e) {
        std::cerr << "Standard exception: " << e.what() << std::endl;
    }
    catch (...) {
        std::cerr << "Unknown exception occurred!" << std::endl;
    }

            
  }

  template <>
  void DepthwiseCustom<float>::Eval(int N, float* output, const float* input,
            const float* input2, void* scratch, size_t scratch_size,
            cudnnHandle_t cudnn, cublasHandle_t cublas, cudaStream_t stream,
            float***) {
    return;      
  }

  template <typename DataType>
  DepthwiseCustom<DataType>::~DepthwiseCustom() {
    ReportCUDAErrors(cudaFree(weights));
  }

/*
template<typename DataType>
FusedDWPWLayer<DataType>::FusedDWPWLayer(
      int C_in, int C_out, int H, int W, 
      ActivationFunction dw_act, ActivationFunction pw_act, bool nhwc,
      int rook_channels, int bishop_channels, int knight_channels)
    : BaseLayer<DataType>(C_out, H, W, nullptr, nhwc, false),
      c_input_(C_in),
      c_output_(C_out),
      dw_act_(dw_act),
      pw_act_(pw_act),
      rook_channels_(rook_channels),
      bishop_channels_(bishop_channels),
      knight_channels_(knight_channels) {
          
  if (!nhwc) {
      throw std::runtime_error("FusedDWPWLayer requires NHWC layout!");
  }

  ReportCUDAErrors(cudaMalloc(&dw_weights_, sizeof(half) * c_input_ * 10));

  ReportCUDAErrors(cudaMalloc(&pw_weights_, sizeof(half) * c_output_ * c_input_));

  ReportCUDAErrors(cudaMalloc(&pw_biases_, sizeof(half) * c_output_));
}

template <typename DataType>
FusedDWPWLayer<DataType>::~FusedDWPWLayer() {
  if (dw_weights_) ReportCUDAErrors(cudaFree(dw_weights_));
  if (pw_weights_) ReportCUDAErrors(cudaFree(pw_weights_));
  if (pw_biases_) ReportCUDAErrors(cudaFree(pw_biases_));
}

template <typename DataType>
void FusedDWPWLayer<DataType>::LoadWeights(
        float* p_dw_filter, float* p_dw_bias, 
        float* p_pw_filter, float* p_pw_bias, void* scratch) {
            
  assert(p_dw_filter != nullptr);
  assert(p_pw_filter != nullptr);
  assert(scratch != nullptr);

  // ==========================================
  // 1. Pack Depthwise Weights
  // ==========================================
  std::vector<float> packed_dw;
  packed_dw.reserve(c_input_ * 10); 

  const int rook_idx[9]   = {2, 7, 10, 11, 12, 13, 14, 17, 22};
  const int bishop_idx[9] = {0, 4,  6,  8, 12, 16, 18, 20, 24};
  const int knight_idx[9] = {1, 3,  5,  9, 12, 15, 19, 21, 23};

  for (int o = 0; o < c_input_; o++) {
      const int* active_mask = nullptr;
      if (o < rook_channels_) active_mask = rook_idx;
      else if (o < (rook_channels_ + bishop_channels_)) active_mask = bishop_idx;
      else active_mask = knight_idx;

      for (int i = 0; i < 9; i++) {
          packed_dw.push_back(p_dw_filter[o * 25 + active_mask[i]]);
      }
      packed_dw.push_back(p_dw_bias ? p_dw_bias[o] : 0.0f);
  }

  // Upload and convert DW to half2
  ReportCUDAErrors(cudaMemcpy(scratch, packed_dw.data(), sizeof(float) * c_input_ * 10, cudaMemcpyHostToDevice));
  convert_float_to_half2_nhwc((float*)scratch, (half2*)dw_weights_, c_input_, 10, 1);

  // ==========================================
  // 2. Pack Pointwise Weights
  // ==========================================
  // Assuming p_pw_filter is exported as [C_out, C_in]
  size_t pw_w_bytes = sizeof(float) * c_output_ * c_input_;
  ReportCUDAErrors(cudaMemcpy(scratch, p_pw_filter, pw_w_bytes, cudaMemcpyHostToDevice));
  
  // Using NHWC conversion packs adjacent C_in elements into half2 pairs
  convert_float_to_half2_nhwc((float*)scratch, (half2*)pw_weights_, c_output_, c_input_, 1);

  // ==========================================
  // 3. Pack Pointwise Biases
  // ==========================================
  // We manually pad the biases array to ensure it's a multiple of 2 for half2 casting
  std::vector<float> packed_pw_b(c_output_, 0.0f);
  if (p_pw_bias) {
      for (int i = 0; i < c_output_; i++) packed_pw_b[i] = p_pw_bias[i];
  }
  
  size_t pw_b_bytes = sizeof(float) * c_output_;
  ReportCUDAErrors(cudaMemcpy(scratch, packed_pw_b.data(), pw_b_bytes, cudaMemcpyHostToDevice));
  convert_float_to_half2_nhwc((float*)scratch, (half2*)pw_biases_, 1, c_output_, 1);
}


template <>
void FusedDWPWLayer<half>::Eval(int N, half* output, const half* input,
          const half* input2, void* scratch, size_t scratch_size,
          cudnnHandle_t cudnn, cublasHandle_t cublas, cudaStream_t stream,
          half***) {
  try  {
      FusedDWPWEvalNHWC(N, c_input_, c_output_, output, input, 
                        (const half2*)dw_weights_, (const half2*)pw_weights_, (const half2*)pw_biases_,
                        dw_act_, pw_act_, 
                        rook_channels_, bishop_channels_, knight_channels_, stream);
  }
  catch (const std::exception& e) {
      std::cerr << "Standard exception in FusedDWPWLayer: " << e.what() << std::endl;
  }
}

// Fallback for FP32 (Not supported for this specific fused half2 kernel)
template <>
void FusedDWPWLayer<float>::Eval(int N, float* output, const float* input,
          const float* input2, void* scratch, size_t scratch_size,
          cudnnHandle_t cudnn, cublasHandle_t cublas, cudaStream_t stream,
          float***) {
  throw std::runtime_error("FusedDWPWLayer only supports FP16 (half) execution.");
}
*/


template <typename DataType>
SELayer<DataType>::SELayer(BaseLayer<DataType>* ip, int fc1Outputs,
                           bool addPrevLayerBias, ActivationFunction activation, bool activateOutput)
    : BaseLayer<DataType>(ip->GetC(), ip->GetH(), ip->GetW(), ip),
      numFc1Out_(fc1Outputs),
      addPrevLayerBias_(addPrevLayerBias),
      act_(activation),
      activateOutput_(activateOutput) {
  ReportCUDAErrors(cudaMalloc(&w1_, C * numFc1Out_ * sizeof(DataType)));
  ReportCUDAErrors(cudaMalloc(&w2_, 2 * C * numFc1Out_ * sizeof(DataType)));

  if (kUseFusedSELayer && nhwc_) {
    ReportCUDAErrors(cudaMalloc(&w1_t_, C * numFc1Out_ * sizeof(DataType)));
    ReportCUDAErrors(cudaMalloc(&w2_t_, 2 * C * numFc1Out_ * sizeof(DataType)));
  }

  ReportCUDAErrors(cudaMalloc(&b1_, numFc1Out_ * sizeof(DataType)));
  ReportCUDAErrors(cudaMalloc(&b2_, 2 * C * sizeof(DataType)));

  ReportCUDAErrors(cudaMalloc(&bPrev_, C * sizeof(DataType)));
}

template <typename DataType>
SELayer<DataType>::~SELayer() {
  ReportCUDAErrors(cudaFree(w1_));
  ReportCUDAErrors(cudaFree(w2_));
  ReportCUDAErrors(cudaFree(b1_));
  ReportCUDAErrors(cudaFree(b2_));
  ReportCUDAErrors(cudaFree(bPrev_));
}

template <>
void SELayer<float>::LoadWeights(float* w1, float* b1, float* w2, float* b2,
                                 float* prevLayerBias, void* /*scratch*/) {
  const size_t num_weights1 = C * numFc1Out_;
  const size_t weight_size1 = sizeof(float) * num_weights1;

  const size_t weight_size2 = 2 * weight_size1;

  // Weight for the first FC layer.
  ReportCUDAErrors(cudaMemcpy(w1_, w1, weight_size1, cudaMemcpyHostToDevice));

  // Weight for the second FC layer.
  ReportCUDAErrors(cudaMemcpy(w2_, w2, weight_size2, cudaMemcpyHostToDevice));

  // Bias for the first FC layer.
  ReportCUDAErrors(
      cudaMemcpy(b1_, b1, numFc1Out_ * sizeof(float), cudaMemcpyHostToDevice));

  // Bias for the second FC layer.
  ReportCUDAErrors(
      cudaMemcpy(b2_, b2, 2 * C * sizeof(float), cudaMemcpyHostToDevice));

  // Bias for previous layer (Convolution).
  if (prevLayerBias) {
    ReportCUDAErrors(cudaMemcpy(bPrev_, prevLayerBias, C * sizeof(float),
                                cudaMemcpyHostToDevice));
  }
}

void cpuTranspose(float* op, float* ip, int rows, int cols) {
  for (int i = 0; i < rows; i++)
    for (int j = 0; j < cols; j++) op[j * rows + i] = ip[i * cols + j];
}

template <>
void SELayer<half>::LoadWeights(float* w1, float* b1, float* w2, float* b2,
                                float* prevLayerBias, void* scratch) {
  const size_t num_weights1 = C * numFc1Out_;
  size_t weight_size1 = sizeof(float) * num_weights1;

  const size_t num_weights2 = 2 * num_weights1;
  size_t weight_size2 = 2 * weight_size1;

  // Transpose the weight matrices for the fused path.
  std::vector<float> temp(weight_size2);

  // Weight for the first FC layer.
  ReportCUDAErrors(
      cudaMemcpy(scratch, w1, weight_size1, cudaMemcpyHostToDevice));
  copyTypeConverted((half*)w1_, (float*)scratch, (int)num_weights1, 0);
  if (kUseFusedSELayer && nhwc_) {
    // transposed copy for fused SE kernel
    cpuTranspose(temp.data(), w1, numFc1Out_, C);
    ReportCUDAErrors(
        cudaMemcpy(scratch, temp.data(), weight_size1, cudaMemcpyHostToDevice));
    copyTypeConverted((half*)w1_t_, (float*)scratch, (int)num_weights1, 0);
  }

  // Weight for the second FC layer.
  ReportCUDAErrors(
      cudaMemcpy(scratch, w2, weight_size2, cudaMemcpyHostToDevice));
  copyTypeConverted((half*)w2_, (float*)scratch, (int)num_weights2, 0);
  if (kUseFusedSELayer && nhwc_) {
    cpuTranspose(temp.data(), w2, 2 * C, numFc1Out_);
    ReportCUDAErrors(
        cudaMemcpy(scratch, temp.data(), weight_size2, cudaMemcpyHostToDevice));
    copyTypeConverted((half*)w2_t_, (float*)scratch, (int)num_weights2, 0);
  }

  // Bias for the first FC layer.
  ReportCUDAErrors(cudaMemcpy(scratch, b1, numFc1Out_ * sizeof(float),
                              cudaMemcpyHostToDevice));
  copyTypeConverted((half*)b1_, (float*)scratch, numFc1Out_, 0);

  // Bias for the second FC layer.
  ReportCUDAErrors(
      cudaMemcpy(scratch, b2, 2 * C * sizeof(float), cudaMemcpyHostToDevice));
  copyTypeConverted((half*)b2_, (float*)scratch, 2 * C, 0);

  // Bias for previous layer (Convolution).
  if (prevLayerBias) {
    ReportCUDAErrors(cudaMemcpy(scratch, prevLayerBias, C * sizeof(float),
                                cudaMemcpyHostToDevice));
    copyTypeConverted((half*)bPrev_, (float*)scratch, C, 0);
  }
}

template <>
void SELayer<float>::Eval(int N, float* output, const float* input,
                          const float* /*input2*/, void* scratch,
                          size_t scratch_size, cudnnHandle_t /*cudnn*/,
                          cublasHandle_t cublas, cudaStream_t stream,
                          float***) {
  // Ping-pong between 'op1' and 'op2' (parts of scratch memory).
  float* op1 = (float*)scratch;
  float* op2 = (float*)scratch + scratch_size / sizeof(float) / 2;

  // 1. Global avg pooling (also adds previous layer bias before computing
  // averages).
  globalAvgPool(N, C, op2, input, bPrev_, false, stream);

  // 2. First fully connected layer.
  float alpha = 1.0f, beta = 0.0f;
  ReportCUBLASErrors(cublasSgemm(cublas, CUBLAS_OP_T, CUBLAS_OP_N, numFc1Out_,
                                 N, C, &alpha, w1_, C, op2, C, &beta, op1,
                                 numFc1Out_));
  addVectors(op1, b1_, op1, numFc1Out_ * N, numFc1Out_, numFc1Out_ * N, act_,
             stream);

  // 3. Second fully connected layer.
  ReportCUBLASErrors(cublasSgemm(cublas, CUBLAS_OP_T, CUBLAS_OP_N, 2 * C, N,
                                 numFc1Out_, &alpha, w2_, numFc1Out_, op1,
                                 numFc1Out_, &beta, op2, 2 * C));
  addVectors(op2, b2_, op2, 2 * C * N, 2 * C, 2 * C * N, ACTIVATION_NONE,
             stream);

  // 4. (Optional prev layer bias add), Global scale, residual add, relu and
  // bias.
  if (activateOutput_) {
    globalScale(N, C, output, input, op2, bPrev_, false, act_, stream);
  }
  else {
    globalScale(N, C, output, input, op2, bPrev_, false, ACTIVATION_NONE, stream);
  }

}

template <>
void SELayer<half>::Eval(int N, half* output, const half* input,
                         const half* input2, void* scratch, size_t scratch_size,
                         cudnnHandle_t /*cudnn*/, cublasHandle_t cublas,
                         cudaStream_t stream, half***) {
  bool se_done = false;
  if (kUseFusedSELayer && nhwc_) {
    se_done = Se_Fp16_NHWC(N, C, numFc1Out_, output, input2, input, w1_t_, b1_,
                           w2_t_, b2_, bPrev_, act_, stream);
  }
  if (!se_done) {
    assert(output == input2);
    // Ping-pong between 'op1' and 'op2' (parts of scratch memory).
    half* op1 = (half*)scratch;
    half* op2 = (half*)scratch + scratch_size / sizeof(half) / 2;

    // 1. Global avg pooling (also adds previous layer bias before computing
    // averages).
    globalAvgPool(N, C, op2, input, bPrev_, nhwc_, stream);

    // 2. First fully connected layer.
    __half_raw one_h{0x3C00};
    __half_raw zero_h{0};
    half alpha = one_h;
    half beta = zero_h;
    ReportCUBLASErrors(cublasHgemm(cublas, CUBLAS_OP_T, CUBLAS_OP_N, numFc1Out_,
                                   N, C, &alpha, w1_, C, op2, C, &beta, op1,
                                   numFc1Out_));
    addVectors(op1, b1_, op1, numFc1Out_ * N, numFc1Out_, numFc1Out_ * N, act_,
               stream);

    // 3. Second fully connected layer.
    ReportCUBLASErrors(cublasHgemm(cublas, CUBLAS_OP_T, CUBLAS_OP_N, 2 * C, N,
                                   numFc1Out_, &alpha, w2_, numFc1Out_, op1,
                                   numFc1Out_, &beta, op2, 2 * C));
    addVectors(op2, b2_, op2, 2 * C * N, 2 * C, 2 * C * N, ACTIVATION_NONE,
               stream);

    // 4. (Optional prev layer bias add), Global scale, residual add, relu and
    // bias.
    if (activateOutput_) {
      globalScale(N, C, output, input, op2, bPrev_, nhwc_, act_, stream);
    }
    else {
      globalScale(N, C, output, input, op2, bPrev_, nhwc_, ACTIVATION_NONE, stream);
    }

  }
}

template <typename DataType>
FCLayer<DataType>::FCLayer(BaseLayer<DataType>* ip, int C, int H, int W,
                           bool bias, ActivationFunction activation)
    : BaseLayer<DataType>(C, H, W, ip), use_bias_(bias), act_(activation) {
  const size_t weight_size =
      sizeof(DataType) * C * H * W * ip->GetC() * ip->GetH() * ip->GetW();
  const size_t bias_size = sizeof(DataType) * C * H * W;
  ReportCUDAErrors(cudaMalloc(&weights_, weight_size));
  if (use_bias_) {
    ReportCUDAErrors(cudaMalloc(&biases_, bias_size));
  } else {
    biases_ = nullptr;
  }
}

template <>
void FCLayer<half>::LoadWeights(float* cpuWeight, float* cpuBias,
                                void* scratch) {
  const size_t num_weights =
      C * H * W * input_->GetC() * input_->GetH() * input_->GetW();
  const size_t weight_size = sizeof(float) * num_weights;
  const size_t num_biases = C * H * W;
  const size_t bias_size = sizeof(float) * num_biases;

  // also need to convert from fp32 to fp16
  assert(scratch);
  ReportCUDAErrors(
      cudaMemcpy(scratch, cpuWeight, weight_size, cudaMemcpyHostToDevice));

  if (nhwc_) {
    convertNCHWtoNHWC((half*)weights_, (float*)scratch, (int)num_biases,
                      input_->GetC(), (int)num_biases, input_->GetC(),
                      input_->GetH(), input_->GetW(), 0);
  } else {
    copyTypeConverted((half*)weights_, (float*)scratch, (int)num_weights, 0);
  }

  if (cpuBias) {
    ReportCUDAErrors(
        cudaMemcpy(scratch, cpuBias, bias_size, cudaMemcpyHostToDevice));
    copyTypeConverted((half*)biases_, (float*)scratch, (int)num_biases, 0);
  }
}

template <>
void FCLayer<float>::LoadWeights(float* cpuWeight, float* cpuBias,
                                 void* /*scratch*/) {
  const size_t num_weights =
      C * H * W * input_->GetC() * input_->GetH() * input_->GetW();
  const size_t weight_size = sizeof(float) * num_weights;
  const size_t num_biases = C * H * W;
  const size_t bias_size = sizeof(float) * num_biases;

  ReportCUDAErrors(
      cudaMemcpy(weights_, cpuWeight, weight_size, cudaMemcpyHostToDevice));
  if (use_bias_) {
    ReportCUDAErrors(
        cudaMemcpy(biases_, cpuBias, bias_size, cudaMemcpyHostToDevice));
  }
}

template <>
void FCLayer<half>::Eval(int N, half* output_tensor, const half* input_tensor,
                         const half* /*input2*/, void* /*scratch*/,
                         size_t /*scratch_size*/, cudnnHandle_t /*cudnn*/,
                         cublasHandle_t cublas, cudaStream_t stream, half***) {
  const int num_outputs = C * H * W;
  const int num_inputs = input_->GetC() * input_->GetH() * input_->GetW();

  // half alpha = float2half_rn(1.0f), beta = float2half_rn(0.0f);
  const __half_raw one_h{0x3C00};
  const __half_raw zero_h{0};
  half alpha = one_h;
  half beta = zero_h;
  ReportCUBLASErrors(cublasHgemm(cublas, CUBLAS_OP_T, CUBLAS_OP_N, num_outputs,
                                 N, num_inputs, &alpha, weights_, num_inputs,
                                 input_tensor, num_inputs, &beta, output_tensor,
                                 num_outputs));

  if (use_bias_ || (act_ != ACTIVATION_NONE)) {
    addVectors(output_tensor, biases_, output_tensor, num_outputs * N,
               num_outputs, num_outputs * N, act_, stream);
  }
}

template <>
void FCLayer<float>::Eval(int N, float* output_tensor,
                          const float* input_tensor, const float* /*input2*/,
                          void* /*scratch*/, size_t /*scratch_size*/,
                          cudnnHandle_t /*cudnn*/, cublasHandle_t cublas,
                          cudaStream_t stream, float***) {
  const int num_outputs = C * H * W;
  const int num_inputs = input_->GetC() * input_->GetH() * input_->GetW();

  float alpha = 1.0f, beta = 0.0f;
  ReportCUBLASErrors(cublasSgemm(cublas, CUBLAS_OP_T, CUBLAS_OP_N, num_outputs,
                                 N, num_inputs, &alpha, weights_, num_inputs,
                                 input_tensor, num_inputs, &beta, output_tensor,
                                 num_outputs));

  if (use_bias_ || (act_ != ACTIVATION_NONE)) {
    addVectors(output_tensor, biases_, output_tensor, num_outputs * N,
               num_outputs, num_outputs * N, act_, stream);
  }
}

template <typename DataType>
FCLayer<DataType>::~FCLayer() {
  ReportCUDAErrors(cudaFree(weights_));
  ReportCUDAErrors(cudaFree(biases_));
}

template <typename DataType>
PolicyMapLayer<DataType>::PolicyMapLayer(BaseLayer<DataType>* ip, int C, int H,
                                         int W, int usedSize, bool attention)
    : BaseLayer<DataType>(C, H, W, ip),
      used_size_(usedSize),
      attention_map_(attention) {
  size_t weight_size = sizeof(short) * this->input_->GetC() * 64;
  if (attention) weight_size = sizeof(short) * usedSize;
  ReportCUDAErrors(cudaMalloc(&weights_, weight_size));
}

template <typename DataType>
void PolicyMapLayer<DataType>::LoadWeights(const short* cpuWeight,
                                           void* /*scratch*/) {
  size_t weight_size = sizeof(short) * used_size_;

  if (nhwc_ && !attention_map_) {
    // convert CHW to HWC
    int C = used_size_ / 64;
    int Cin = this->input_->GetC();

    // C is the no. of channels actually used (typically 73).
    // Cin the the no. of channels in previous layer (padded up to 80).
    // Weights of this layer is a mapping to select which output index of the
    // policy vector (1858 elements) maps to every element of input
    // tensor (assuming NCHW layout). Note that there are 73x64 valid inputs
    // (80x64 taking padding), and only 1858 outputs so the mapping isn't
    // one to one. Only few of the indices point to valid index in policy
    // vector. Invalid entries are set to -1.

    // In fp16 mode, the tensor layout is NHWC so the weights need to be
    // adjusted to make them work as intended.

    // This is how the original weights looks like (CHW layout):
    /*
               HW (64)
       ----|-------------|
           |             |
           |             |
    C (73) |             |
           |             |
           |             |
       ------------------|   Cin (80)
           |  padding    |
           |-------------|
    */
    // The padding is not part of the weights provided (used_size_ is 73 x 64).
    //
    // The weights converted to HWC looks like this
    /*
                 C (73)
            |-------------|---|
            |             | P |
            |             | a |
    HW (64) |             | d |
            |             |   |
            |             |   |
            |-----------------|
                     Cin (80)
    */
    // In HWC, because the padding is now part of each row
    // we need to increase the size of weights to account
    // for it.
    // The pad elements point to -1 (invalid output index) and the
    // same kernel works for both HWC and CHW layouts after used_size_
    // is updated to include padding (80x64).

    used_size_ = Cin * 64;
    std::vector<short> convertedWeights(used_size_);

    for (int hw = 0; hw < 64; hw++)
      for (int c = 0; c < Cin; c++) {
        if (c < C)
          convertedWeights[hw * Cin + c] = cpuWeight[c * 64 + hw];
        else
          convertedWeights[hw * Cin + c] = -1;
      }
    ReportCUDAErrors(cudaMemcpy(weights_, convertedWeights.data(),
                                used_size_ * sizeof(short),
                                cudaMemcpyHostToDevice));
  } else {
    ReportCUDAErrors(
        cudaMemcpy(weights_, cpuWeight, weight_size, cudaMemcpyHostToDevice));
  }
}

template <typename DataType>
void PolicyMapLayer<DataType>::Eval(int N, DataType* output_tensor,
                                    const DataType* input_tensor,
                                    const DataType* /*input2*/,
                                    void* /*scratch*/, size_t /*scratch_size*/,
                                    cudnnHandle_t /*cudnn*/,
                                    cublasHandle_t /*cublas*/,
                                    cudaStream_t stream, DataType***) {
  int inputSize =
      this->input_->GetC() * this->input_->GetH() * this->input_->GetW();
  if (attention_map_) inputSize = used_size_;
  int outputSize = this->C * this->H * this->W;
  PolicyMap(N, output_tensor, input_tensor, weights_, inputSize, used_size_,
            outputSize, stream);
}

template <typename DataType>
PolicyMapLayer<DataType>::~PolicyMapLayer() {
  ReportCUDAErrors(cudaFree(weights_));
}

template <typename DataType>
FusedWinogradConvSELayer<DataType>::FusedWinogradConvSELayer(
    BaseLayer<DataType>* ip, int C, int H, int W, int Cin,
    ActivationFunction activation, bool bias, bool skip_add, bool se, int se_k,
    bool use_gemm_ex, bool op_nhcw)
    : BaseLayer<DataType>(C, H, W, ip, true, use_gemm_ex),
      c_input_(Cin),
      act_(activation),
      use_bias_(bias),
      skip_add_(skip_add),
      has_se_(se),
      se_k_(se_k),
      op_nhcw_(op_nhcw) {
  if (act_ != ACTIVATION_RELU && act_ != ACTIVATION_MISH &&
      act_ != ACTIVATION_NONE) {
    throw Exception("Unsupported activation for fused winograd conv SE layer.");
  }
  // Allocate memory for weights (filter tensor) and biases.
  const size_t weight_size = sizeof(DataType) * c_input_ * C * 3 * 3;

  if (use_bias_) {
    const size_t bias_size = sizeof(DataType) * C;
    ReportCUDAErrors(cudaMalloc(&biases_, bias_size));
  }

  // 6x6 transformed filter size, for 3x3 convolution
  ReportCUDAErrors(cudaMalloc(&transformed_weights_, weight_size * 4));

  if (has_se_) {
    const size_t num_weights1 = C * se_k_;
    const size_t num_weights2 = num_weights1 * 2;
    const size_t num_biases1 = se_k_;
    const size_t num_biases2 = 2 * C;

    const size_t weight_size1 = sizeof(DataType) * num_weights1;
    const size_t weight_size2 = sizeof(DataType) * num_weights2;
    const size_t biases_size1 = sizeof(DataType) * num_biases1;
    const size_t biases_size2 = sizeof(DataType) * num_biases2;

    ReportCUDAErrors(cudaMalloc(&w1_, weight_size1));
    ReportCUDAErrors(cudaMalloc(&w2_, weight_size2));
    ReportCUDAErrors(cudaMalloc(&b1_, biases_size1));
    ReportCUDAErrors(cudaMalloc(&b2_, biases_size2));
  }
}

template <typename DataType>
void FusedWinogradConvSELayer<DataType>::LoadWeights(float* pfilter,
                                                     float* pBias,
                                                     void* scratch) {
  const size_t weight_size = sizeof(float) * c_input_ * C * 3 * 3;
  const size_t bias_size = sizeof(float) * C;

  // Store untransformed weights in scratch.
  const DataType* weights = (DataType*)scratch + weight_size + bias_size;

  // first copy from CPU memory to scratch space in GPU memory
  // and then do the type conversion using a kernel
  assert(scratch);
  ReportCUDAErrors(
      cudaMemcpy(scratch, pfilter, weight_size, cudaMemcpyHostToDevice));
  copyTypeConverted((DataType*)weights, (float*)scratch, C * c_input_ * 3 * 3,
                    0);

  if (pBias) {
    ReportCUDAErrors(
        cudaMemcpy(scratch, pBias, bias_size, cudaMemcpyHostToDevice));
    copyTypeConverted((DataType*)biases_, (float*)scratch, C, 0);
  }

  // run winograd transform kernel for the filter
  FilterTransform(C, c_input_, transformed_weights_, weights, 0);
}

// TODO: Do this on the GPU to improve network load time!
static inline void CpuTranspose(float* op, float* ip, size_t rows,
                                size_t cols) {
  for (size_t i = 0; i < rows; i++)
    for (size_t j = 0; j < cols; j++) op[j * rows + i] = ip[i * cols + j];
}

template <typename DataType>
void FusedWinogradConvSELayer<DataType>::LoadSEWeights(float* w1, float* b1,
                                                       float* w2, float* b2,
                                                       void* scratch) {
  const size_t num_weights1 = C * se_k_;
  const size_t num_weights2 = num_weights1 * 2;
  const size_t num_biases1 = se_k_;
  const size_t num_biases2 = 2 * C;

  // The shader uses transposed weight matrices.
  std::vector<float> temp_transposed(num_weights2);

  CpuTranspose(temp_transposed.data(), w1, se_k_, C);
  ReportCUDAErrors(cudaMemcpy(scratch, temp_transposed.data(),
                              num_weights1 * sizeof(float),
                              cudaMemcpyHostToDevice));
  copyTypeConverted((DataType*)w1_, (float*)scratch, (int)num_weights1, 0);

  CpuTranspose(temp_transposed.data(), w2, 2 * C, se_k_);
  ReportCUDAErrors(cudaMemcpy(scratch, temp_transposed.data(),
                              num_weights2 * sizeof(float),
                              cudaMemcpyHostToDevice));
  copyTypeConverted((DataType*)w2_, (float*)scratch, (int)num_weights2, 0);

  ReportCUDAErrors(cudaMemcpy(scratch, b1, num_biases1 * sizeof(float),
                              cudaMemcpyHostToDevice));
  copyTypeConverted((DataType*)b1_, (float*)scratch, (int)num_biases1, 0);

  ReportCUDAErrors(cudaMemcpy(scratch, b2, num_biases2 * sizeof(float),
                              cudaMemcpyHostToDevice));
  copyTypeConverted((DataType*)b2_, (float*)scratch, (int)num_biases2, 0);
}

template <>
void BaseLayer<half>::cublasRowMajorMatrixMul(const half* A, const half* B,
                                              half* Out, int M, int N, int K,
                                              int batchSize,
                                              cublasHandle_t cublas) {
  // Need to initialize 1.0 and 0.0 as hexadecimal for fp16 because typecasting
  // float to half type doesn't work before CUDA 10.0
  __half_raw one_h{0x3C00};
  __half_raw zero_h{0};
  half halfOne = one_h;
  half halfZero = zero_h;

  // dimensions of matrix A = M x K
  // dimensions of matrix B = K x N
  // dimensions of output   = M x N

  // cublas supports only col major output
  // to multiply row major matrices, use the trick below
  ReportCUBLASErrors(cublasGemmStridedBatchedEx(
      cublas, CUBLAS_OP_N, CUBLAS_OP_N, N, M, K, &halfOne, B, CUDA_R_16F, N,
      N * K, A, CUDA_R_16F, K, K * M, &halfZero, Out, CUDA_R_16F, N, N * M,
      batchSize, CUDA_R_16F, CUBLAS_GEMM_DEFAULT));
}

template <>
void BaseLayer<float>::cublasRowMajorMatrixMul(const float* A, const float* B,
                                               float* Out, int M, int N, int K,
                                               int batchSize,
                                               cublasHandle_t cublas) {
  float floatOne = 1.0f;
  float floatZero = 0.0f;
  if (use_gemm_ex_)
    ReportCUBLASErrors(cublasGemmStridedBatchedEx(
        cublas, CUBLAS_OP_N, CUBLAS_OP_N, N, M, K, &floatOne, B, CUDA_R_32F, N,
        N * K, A, CUDA_R_32F, K, K * M, &floatZero, Out, CUDA_R_32F, N, N * M,
        batchSize, CUDA_R_32F, CUBLAS_GEMM_DEFAULT));
  else
    // Much slower on RTX 2060.. why? Maybe a cublas bug :-/
    ReportCUBLASErrors(cublasSgemmStridedBatched(
        cublas, CUBLAS_OP_N, CUBLAS_OP_N, N, M, K, &floatOne, B, N, N * K, A, K,
        K * M, &floatZero, Out, N, N * M, batchSize));
}

template <typename DataType>
void FusedWinogradConvSELayer<DataType>::Eval(
    int N, DataType* output, const DataType* input, const DataType* input2,
    void* scratch, size_t scratch_size, cudnnHandle_t /*cudnn*/,
    cublasHandle_t cublas, cudaStream_t stream, DataType***) {
  // Split the scratch space into two parts - use first part for holding
  // transformed input and second part for transformed output.
  DataType* transformed_input = (DataType*)scratch;
  DataType* transformed_output =
      transformed_input + scratch_size / (2 * sizeof(DataType));

  InputTransform<DataType, false>(N, c_input_, transformed_input, input,
                                  stream);
  BaseLayer<DataType>::cublasRowMajorMatrixMul(
      transformed_input, transformed_weights_, transformed_output, N * 4, C,
      c_input_, 36, cublas);

  if (act_ == ACTIVATION_NONE) {
    if (!has_se_ && use_bias_ && !skip_add_)
      OutputTransform<DataType, false, ACTIVATION_NONE, true, false, false,
                      false>(N, C, 0, output, transformed_output, nullptr,
                             biases_, nullptr, nullptr, nullptr, nullptr,
                             stream);
    else
      throw Exception("unsupported network type!");
  } else if (act_ == ACTIVATION_RELU) {
    if (has_se_ && use_bias_ && skip_add_)
      OutputTransform<DataType, true, ACTIVATION_RELU, true, true, false,
                      false>(N, C, se_k_, output, transformed_output, input2,
                             biases_, w1_, b1_, w2_, b2_, stream);
    else if (!has_se_ && use_bias_ && !skip_add_) {
      if (op_nhcw_)
        OutputTransform<DataType, false, ACTIVATION_RELU, true, false, false,
                        true>(N, C, 0, output, transformed_output, nullptr,
                              biases_, nullptr, nullptr, nullptr, nullptr,
                              stream);
      else
        OutputTransform<DataType, false, ACTIVATION_RELU, true, false, false,
                        false>(N, C, 0, output, transformed_output, nullptr,
                               biases_, nullptr, nullptr, nullptr, nullptr,
                               stream);
    } else if (!has_se_ && use_bias_ && skip_add_)
      OutputTransform<DataType, false, ACTIVATION_RELU, true, true, false,
                      false>(N, C, 0, output, transformed_output, input2,
                             biases_, nullptr, nullptr, nullptr, nullptr,
                             stream);
    else
      throw Exception("unsupported network type!");
  } else if (act_ == ACTIVATION_MISH) {
    if (has_se_ && use_bias_ && skip_add_)
      OutputTransform<DataType, true, ACTIVATION_MISH, true, true, false,
                      false>(N, C, se_k_, output, transformed_output, input2,
                             biases_, w1_, b1_, w2_, b2_, stream);
    else if (!has_se_ && use_bias_ && !skip_add_) {
      if (op_nhcw_)
        OutputTransform<DataType, false, ACTIVATION_MISH, true, false, false,
                        true>(N, C, 0, output, transformed_output, nullptr,
                              biases_, nullptr, nullptr, nullptr, nullptr,
                              stream);
      else
        OutputTransform<DataType, false, ACTIVATION_MISH, true, false, false,
                        false>(N, C, 0, output, transformed_output, nullptr,
                               biases_, nullptr, nullptr, nullptr, nullptr,
                               stream);
    } else if (!has_se_ && use_bias_ && skip_add_)
      OutputTransform<DataType, false, ACTIVATION_MISH, true, true, false,
                      false>(N, C, 0, output, transformed_output, input2,
                             biases_, nullptr, nullptr, nullptr, nullptr,
                             stream);
    else
      throw Exception("unsupported network type!");
  } else
    throw Exception("unsupported network type!");
}

template <typename DataType>
FusedWinogradConvSELayer<DataType>::~FusedWinogradConvSELayer() {
  ReportCUDAErrors(cudaFree(transformed_weights_));
  if (use_bias_) ReportCUDAErrors(cudaFree(biases_));
  if (has_se_) {
    ReportCUDAErrors(cudaFree(w1_));
    ReportCUDAErrors(cudaFree(w2_));
    ReportCUDAErrors(cudaFree(b1_));
    ReportCUDAErrors(cudaFree(b2_));
  }
}

template <typename DataType>
ResidualBlock<DataType>::ResidualBlock(BaseLayer<DataType>* ip, int C, bool se,
                                       int se_k, bool use_gemm_ex, bool first,

                                       bool last, ActivationFunction activation,
                                       int shared_mem_size)
    : BaseLayer<DataType>(C, 8, 8, ip, ip->isNHWC(), use_gemm_ex),
      has_se_(se),
      se_k_(se_k),
      c_input_(C),
      first_block_(first),
      last_block_(last),
      shared_mem_size_(shared_mem_size),
      act_(activation) {
  if (act_ != ACTIVATION_RELU && act_ != ACTIVATION_MISH) {
    throw Exception("Unsupported activation for residual block.");
  }
  // Allocate memory for weights (filter tensor) and biases.
  const size_t weight_size = sizeof(DataType) * C * C * 3 * 3;

  const size_t bias_size = sizeof(DataType) * C;
  ReportCUDAErrors(cudaMalloc(&biases0_, bias_size));
  ReportCUDAErrors(cudaMalloc(&biases1_, bias_size));

  // 6x6 transformed filter size, for 3x3 convolution
  ReportCUDAErrors(cudaMalloc(&transformed_weights0_, weight_size * 4));
  ReportCUDAErrors(cudaMalloc(&transformed_weights1_, weight_size * 4));

  if (has_se_) {
    const size_t num_weights1 = C * se_k_;
    const size_t num_weights2 = num_weights1 * 2;
    const size_t num_biases1 = se_k_;
    const size_t num_biases2 = 2 * C;

    const size_t weight_size1 = sizeof(DataType) * num_weights1;
    const size_t weight_size2 = sizeof(DataType) * num_weights2;
    const size_t biases_size1 = sizeof(DataType) * num_biases1;
    const size_t biases_size2 = sizeof(DataType) * num_biases2;

    ReportCUDAErrors(cudaMalloc(&w1_, weight_size1));
    ReportCUDAErrors(cudaMalloc(&w2_, weight_size2));
    ReportCUDAErrors(cudaMalloc(&b1_, biases_size1));
    ReportCUDAErrors(cudaMalloc(&b2_, biases_size2));
  }
}

template <typename DataType>
void ResidualBlock<DataType>::LoadWeights0(float* pfilter, float* pBias,
                                           void* scratch) {
  const size_t weight_size = sizeof(float) * c_input_ * C * 3 * 3;
  const size_t bias_size = sizeof(float) * C;

  // Store untransformed weights in scratch.
  const DataType* weights = (DataType*)scratch + weight_size;

  // first copy from CPU memory to scratch space in GPU memory
  // and then do the type conversion using a kernel
  assert(scratch);
  ReportCUDAErrors(
      cudaMemcpy(scratch, pfilter, weight_size, cudaMemcpyHostToDevice));
  copyTypeConverted((DataType*)weights, (float*)scratch, C * c_input_ * 3 * 3,
                    0);

  if (pBias) {
    ReportCUDAErrors(
        cudaMemcpy(scratch, pBias, bias_size, cudaMemcpyHostToDevice));
    copyTypeConverted((DataType*)biases0_, (float*)scratch, C, 0);
  }

  // run winograd transform kernel for the filter
  FilterTransform(C, c_input_, transformed_weights0_, weights, 0);
}

template <typename DataType>
void ResidualBlock<DataType>::LoadWeights1(float* pfilter, float* pBias,
                                           void* scratch) {
  const size_t weight_size = sizeof(float) * C * C * 3 * 3;
  const size_t bias_size = sizeof(float) * C;

  // Store untransformed weights in scratch.
  const DataType* weights = (DataType*)scratch + weight_size;

  // first copy from CPU memory to scratch space in GPU memory
  // and then do the type conversion using a kernel
  assert(scratch);
  ReportCUDAErrors(
      cudaMemcpy(scratch, pfilter, weight_size, cudaMemcpyHostToDevice));
  copyTypeConverted((DataType*)weights, (float*)scratch, C * C * 3 * 3, 0);

  if (pBias) {
    ReportCUDAErrors(
        cudaMemcpy(scratch, pBias, bias_size, cudaMemcpyHostToDevice));
    copyTypeConverted((DataType*)biases1_, (float*)scratch, C, 0);
  }

  // run winograd transform kernel for the filter
  FilterTransform(C, C, transformed_weights1_, weights, 0);
}

template <typename DataType>
void ResidualBlock<DataType>::LoadSEWeights(float* w1, float* b1, float* w2,
                                            float* b2, void* scratch) {
  const size_t num_weights1 = C * se_k_;
  const size_t num_weights2 = num_weights1 * 2;
  const size_t num_biases1 = se_k_;
  const size_t num_biases2 = 2 * C;

  // The shader uses transposed weight matrices.
  std::vector<float> temp_transposed(num_weights2);

  CpuTranspose(temp_transposed.data(), w1, se_k_, C);
  ReportCUDAErrors(cudaMemcpy(scratch, temp_transposed.data(),
                              num_weights1 * sizeof(float),
                              cudaMemcpyHostToDevice));
  copyTypeConverted((DataType*)w1_, (float*)scratch, (int)num_weights1, 0);

  CpuTranspose(temp_transposed.data(), w2, 2 * C, se_k_);
  ReportCUDAErrors(cudaMemcpy(scratch, temp_transposed.data(),
                              num_weights2 * sizeof(float),
                              cudaMemcpyHostToDevice));
  copyTypeConverted((DataType*)w2_, (float*)scratch, (int)num_weights2, 0);

  ReportCUDAErrors(cudaMemcpy(scratch, b1, num_biases1 * sizeof(float),
                              cudaMemcpyHostToDevice));
  copyTypeConverted((DataType*)b1_, (float*)scratch, (int)num_biases1, 0);

  ReportCUDAErrors(cudaMemcpy(scratch, b2, num_biases2 * sizeof(float),
                              cudaMemcpyHostToDevice));
  copyTypeConverted((DataType*)b2_, (float*)scratch, (int)num_biases2, 0);
}

template <typename DataType>
void ResidualBlock<DataType>::Eval(int N, DataType* output,
                                   const DataType* input,
                                   const DataType* /*input2*/, void* scratch,
                                   size_t scratch_size, cudnnHandle_t /*cudnn*/,
                                   cublasHandle_t cublas, cudaStream_t stream,
                                   DataType***) {
  // normally:
  // - "output" initially contains the transformed input,
  //    and after this layer, it contains the transformed input for next layer
  // - "input" contains the original/untransformed input
  // special cases:
  //   - for first_block_, input is real input (untransformed)
  //   - for last_block_, output is the final output of this block
  //   (untransformed)

  // Split the scratch space into two parts - use first part for holding
  // transformed input and second part for transformed output.
  DataType* transformed_input;
  DataType* transformed_output;
  if (!scratch) {
    // Caller wants us to sub-allocate all memory we need from "output" tensor.
    transformed_input = output;  // This is true in normal cases too!
    transformed_output = transformed_input + (N * C * 8 * 8 * 36 / 16);
  } else {
    transformed_input = (DataType*)scratch;
    transformed_output =
        transformed_input + scratch_size / (2 * sizeof(DataType));
  }

  if (first_block_) {
    InputTransform<DataType, true>(N, c_input_, transformed_input, input,
                                   stream);
    BaseLayer<DataType>::cublasRowMajorMatrixMul(
        transformed_input, transformed_weights0_, transformed_output, N * 4, C,
        c_input_, 36, cublas);
  } else {
    BaseLayer<DataType>::cublasRowMajorMatrixMul(output, transformed_weights0_,
                                                 transformed_output, N * 4, C,
                                                 c_input_, 36, cublas);
  }

  if (act_ == ACTIVATION_RELU) {
    OutputInputTransform<DataType, false, ACTIVATION_RELU, true, false>(
        N, C, 0, transformed_input, transformed_output, nullptr, biases0_,
        nullptr, nullptr, nullptr, nullptr, stream);
  } else if (act_ == ACTIVATION_MISH) {
    OutputInputTransform<DataType, false, ACTIVATION_MISH, true, false>(
        N, C, 0, transformed_input, transformed_output, nullptr, biases0_,
        nullptr, nullptr, nullptr, nullptr, stream);
  }
  // "transformed_input" tensor now contains transformed input for the next
  // convolution

  BaseLayer<DataType>::cublasRowMajorMatrixMul(
      transformed_input, transformed_weights1_, transformed_output, N * 4, C, C,
      36, cublas);

  const bool fp16 = std::is_same<half, DataType>::value;
  bool allowFusing =
      (C <= kMaxResBlockFusingChannels) ||
      (fp16 && (shared_mem_size_ >= kMaxResBlockFusingSeFp16AmpereSmem) &&
       (C <= kMaxResBlockFusingSeKFp16Ampere));

  if (act_ == ACTIVATION_RELU) {
    if (last_block_) {
      if (has_se_)
        OutputTransform<DataType, true, ACTIVATION_RELU, true, true, true,
                        false>(N, C, se_k_, output, transformed_output, input,
                               biases1_, w1_, b1_, w2_, b2_, stream);
      else
        OutputTransform<DataType, false, ACTIVATION_RELU, true, true, true,
                        false>(N, C, se_k_, output, transformed_output, input,
                               biases1_, w1_, b1_, w2_, b2_, stream);
    } else {
      if (has_se_) {
        if (allowFusing) {
          OutputInputTransform<DataType, true, ACTIVATION_RELU, true, true>(
              N, C, se_k_, output, transformed_output, input, biases1_, w1_,
              b1_, w2_, b2_, stream);
        } else {
          OutputTransform<DataType, true, ACTIVATION_RELU, true, true, true,
                          true>(N, C, se_k_, (DataType*)input,
                                transformed_output, input, biases1_, w1_, b1_,
                                w2_, b2_, stream);
          InputTransform<DataType, true>(N, C, output, (DataType*)input,
                                         stream);
        }
      } else
        OutputInputTransform<DataType, false, ACTIVATION_RELU, true, true>(
            N, C, se_k_, output, transformed_output, input, biases1_, w1_, b1_,
            w2_, b2_, stream);
    }
  } else if (act_ == ACTIVATION_MISH) {
    if (last_block_) {
      if (has_se_)
        OutputTransform<DataType, true, ACTIVATION_MISH, true, true, true,
                        false>(N, C, se_k_, output, transformed_output, input,
                               biases1_, w1_, b1_, w2_, b2_, stream);
      else
        OutputTransform<DataType, false, ACTIVATION_MISH, true, true, true,
                        false>(N, C, se_k_, output, transformed_output, input,
                               biases1_, w1_, b1_, w2_, b2_, stream);
    } else {
      if (has_se_) {
        if (allowFusing) {
          OutputInputTransform<DataType, true, ACTIVATION_MISH, true, true>(
              N, C, se_k_, output, transformed_output, input, biases1_, w1_,
              b1_, w2_, b2_, stream);
        } else {
          OutputTransform<DataType, true, ACTIVATION_MISH, true, true, true,
                          true>(N, C, se_k_, (DataType*)input,
                                transformed_output, input, biases1_, w1_, b1_,
                                w2_, b2_, stream);
          InputTransform<DataType, true>(N, C, output, (DataType*)input,
                                         stream);
        }
      } else
        OutputInputTransform<DataType, false, ACTIVATION_MISH, true, true>(
            N, C, se_k_, output, transformed_output, input, biases1_, w1_, b1_,
            w2_, b2_, stream);
    }
  }
  // "output" tensor now contains transformed input for the next
  // convolution
}

template <typename DataType>
ResidualBlock<DataType>::~ResidualBlock() {
  ReportCUDAErrors(cudaFree(transformed_weights0_));
  ReportCUDAErrors(cudaFree(biases0_));
  ReportCUDAErrors(cudaFree(transformed_weights1_));
  ReportCUDAErrors(cudaFree(biases1_));
  if (has_se_) {
    ReportCUDAErrors(cudaFree(w1_));
    ReportCUDAErrors(cudaFree(w2_));
    ReportCUDAErrors(cudaFree(b1_));
    ReportCUDAErrors(cudaFree(b2_));
  }
}


template <typename DataType>
void allocAndUpload(DataType** gpu_dest, std::vector<float> cpu_src,
                    void* scratch) {
  size_t size = cpu_src.size() * sizeof(DataType);
  if (size == 0) {
    *gpu_dest = nullptr;
    return;
  }
  ReportCUDAErrors(cudaMalloc(gpu_dest, size));
  ReportCUDAErrors(cudaMemcpy(scratch, &cpu_src[0],
                              cpu_src.size() * sizeof(float),
                              cudaMemcpyHostToDevice));
  copyTypeConverted((DataType*)(*gpu_dest), (float*)scratch,
                    (int)cpu_src.size(), 0);
}

template <typename DataType>
AttentionPolicyHead<DataType>::AttentionPolicyHead(
    BaseLayer<DataType>* ip, const MultiHeadWeights::PolicyHead& weights,
    void* scratch, bool attention_body, ActivationFunction act,
    int max_batch_size, bool prenorm, bool use_gemm_ex, float epsilon)
    : BaseLayer<DataType>(64 * 64 + 24 * 8, 1, 1, ip),
      attention_body_(attention_body),
      // Old networks without attention body (e.g. T79) use hardcoded SELU
      // activations.
      act_(attention_body ? act : ACTIVATION_SELU),
      prenorm_(prenorm),
      default_epsilon_(epsilon) {
  embedding_op_size_ = weights.ip_pol_b.size();
  wq_op_size_ = weights.ip2_pol_b.size();
  wk_op_size_ = weights.ip3_pol_b.size();

  encoder_heads_ = weights.pol_encoder_head_count;
  policy_d_model_ = wq_op_size_;

  //default_epsilon_ = 1e-6;

  allocAndUpload<DataType>(&ip_pol_w_, weights.ip_pol_w, scratch);
  allocAndUpload<DataType>(&ip_pol_b_, weights.ip_pol_b, scratch);

  allocAndUpload<DataType>(&ip2_pol_w_, weights.ip2_pol_w, scratch);
  allocAndUpload<DataType>(&ip2_pol_b_, weights.ip2_pol_b, scratch);

  allocAndUpload<DataType>(&ip3_pol_w_, weights.ip3_pol_w, scratch);
  allocAndUpload<DataType>(&ip3_pol_b_, weights.ip3_pol_b, scratch);

  // big allocation to hold wq and wk weights one after the other
  {
    size_t elements = weights.ip2_pol_w.size();
    assert(elements == weights.ip3_pol_w.size());

    size_t size = elements * sizeof(DataType) * 2;
    ReportCUDAErrors(cudaMalloc(&wqk_w_, size));
    ReportCUDAErrors(
        cudaMemcpy(wqk_w_, ip2_pol_w_, size / 2, cudaMemcpyDeviceToDevice));
    ReportCUDAErrors(cudaMemcpy(wqk_w_ + elements, ip3_pol_w_, size / 2,
                                cudaMemcpyDeviceToDevice));

    elements = weights.ip2_pol_b.size();
    size = elements * sizeof(DataType) * 2;
    ReportCUDAErrors(cudaMalloc(&wqk_b_, size));
    ReportCUDAErrors(
        cudaMemcpy(wqk_b_, ip2_pol_b_, size / 2, cudaMemcpyDeviceToDevice));
    ReportCUDAErrors(cudaMemcpy(wqk_b_ + elements, ip3_pol_b_, size / 2,
                                cudaMemcpyDeviceToDevice));
  }

  allocAndUpload<DataType>(&ip4_pol_w_, weights.ip4_pol_w, scratch);

  for (const auto& enc : weights.pol_encoder) {
    EncoderBlock<DataType>* pW = new EncoderBlock<DataType>(
        enc, scratch, encoder_heads_, embedding_op_size_,
        1.0f,        // using alpha = 1 for now (TODO: may change?)
        nullptr, 0,  // smolgen weights not implemented in
                     // policy encoder heads yet.
        max_batch_size, ACTIVATION_SWISH, act_,
        default_epsilon_,          // attentionbody nets don't have policy encoders, so
        prenorm_,
        use_gemm_ex,   // using old epsilon for backward compatibility with T78.
        false);
    encoder_weights_.emplace_back(pW);
  }
}

template <typename DataType>
EncoderBlock<DataType>::EncoderBlock(
    const MultiHeadWeights::EncoderLayer& cpu_weights, void* scratch, int heads,
    int size, float alpha, DataType* smolgen_global_scratch,
    int smolgen_global_size, int max_batch_size, ActivationFunction smolgen_act,
    ActivationFunction ffn_act, float default_eps, bool prenorm, bool use_gemm_ex,
    bool fused_mha, const std::vector<float>& attention_mask)
    : embedding_op_size_(size),
      encoder_heads_(heads),
      alpha_(alpha),
      default_eps_(default_eps),
      has_smolgen_(cpu_weights.mha.has_smolgen),
      smolgen_activation_(smolgen_act),
      ffn_activation_(ffn_act),
      max_batch_size_(max_batch_size),
      use_fused_mha_(fused_mha),
      prenorm_(prenorm),
      use_gemm_ex_(use_gemm_ex) {

  mha_q_size_ = cpu_weights.mha.q_w.size() / size;
  mha_k_size_ = cpu_weights.mha.k_w.size() / size;
  mha_v_size_ = cpu_weights.mha.v_w.size() / size;

  mha_dense_size_ = cpu_weights.mha.dense_b.size();
  ffn_dense1_size_ = cpu_weights.ffn.dense1.biases.size();
  ffn_dense2_size_ = cpu_weights.ffn.dense2.biases.size();

  allocAndUpload<DataType>(&mha_q_w, cpu_weights.mha.q_w, scratch);
  allocAndUpload<DataType>(&mha_q_b, cpu_weights.mha.q_b, scratch);

  allocAndUpload<DataType>(&mha_k_w, cpu_weights.mha.k_w, scratch);
  allocAndUpload<DataType>(&mha_k_b, cpu_weights.mha.k_b, scratch);

  allocAndUpload<DataType>(&mha_v_w, cpu_weights.mha.v_w, scratch);
  allocAndUpload<DataType>(&mha_v_b, cpu_weights.mha.v_b, scratch);

  // big allocation to hold qkv weights one after the other
  {
    size_t elements = cpu_weights.mha.q_w.size();
    size_t size = elements * sizeof(DataType) * 3;
    ReportCUDAErrors(cudaMalloc(&mha_qkv_w, size));
    ReportCUDAErrors(
        cudaMemcpy(mha_qkv_w, mha_q_w, size / 3, cudaMemcpyDeviceToDevice));
    ReportCUDAErrors(cudaMemcpy(mha_qkv_w + elements, mha_k_w, size / 3,
                                cudaMemcpyDeviceToDevice));
    ReportCUDAErrors(cudaMemcpy(mha_qkv_w + elements * 2, mha_v_w, size / 3,
                                cudaMemcpyDeviceToDevice));

    elements = cpu_weights.mha.q_b.size();
    size = elements * sizeof(DataType) * 3;
    ReportCUDAErrors(cudaMalloc(&mha_qkv_b, size));
    ReportCUDAErrors(
        cudaMemcpy(mha_qkv_b, mha_q_b, size / 3, cudaMemcpyDeviceToDevice));
    ReportCUDAErrors(cudaMemcpy(mha_qkv_b + elements, mha_k_b, size / 3,
                                cudaMemcpyDeviceToDevice));
    ReportCUDAErrors(cudaMemcpy(mha_qkv_b + elements * 2, mha_v_b, size / 3,
                                cudaMemcpyDeviceToDevice));
  }

  allocAndUpload<DataType>(&mha_dense_w, cpu_weights.mha.dense_w, scratch);
  allocAndUpload<DataType>(&mha_dense_b, cpu_weights.mha.dense_b, scratch);

  allocAndUpload<DataType>(&ln1_gammas, cpu_weights.ln1_gammas, scratch);
  allocAndUpload<DataType>(&ln1_betas, cpu_weights.ln1_betas, scratch);

  allocAndUpload<DataType>(&ffn_dense1_w, cpu_weights.ffn.dense1.weights, scratch);
  allocAndUpload<DataType>(&ffn_dense1_b, cpu_weights.ffn.dense1.biases, scratch);

  allocAndUpload<DataType>(&ffn_dense2_w, cpu_weights.ffn.dense2.weights, scratch);
  allocAndUpload<DataType>(&ffn_dense2_b, cpu_weights.ffn.dense2.biases, scratch);

  allocAndUpload<DataType>(&ln2_gammas, cpu_weights.ln2_gammas, scratch);
  allocAndUpload<DataType>(&ln2_betas, cpu_weights.ln2_betas, scratch);

  if (cpu_weights.ffn.d_conv.biases.size() > 0) {

    d_conv = std::make_unique<DepthwiseCustom<DataType>>(ffn_dense1_size_, 8, 8, ffn_activation_, true,
          cpu_weights.ffn.d_conv.rook_channels,
          cpu_weights.ffn.d_conv.bishop_channels,
          cpu_weights.ffn.d_conv.knight_channels);
    d_conv->LoadWeights(cpu_weights.ffn.d_conv.weights,
                        const_cast<float*>(cpu_weights.ffn.d_conv.biases.data()),
                      scratch); 
  }


  // Smolgen weights.
  if (has_smolgen_) {
    smol_compress_size_ = cpu_weights.mha.smolgen.compress.size() / mha_q_size_;
    smol_dense_1_size_ = cpu_weights.mha.smolgen.dense1_b.size();
    smol_dense_2_size_ = cpu_weights.mha.smolgen.dense2_b.size();
    smol_global_size_ = smolgen_global_size;

    allocAndUpload<DataType>(&smol_compress, cpu_weights.mha.smolgen.compress,
                             scratch);
    allocAndUpload<DataType>(&smol_dense1_w, cpu_weights.mha.smolgen.dense1_w,
                             scratch);
    allocAndUpload<DataType>(&smol_dense1_b, cpu_weights.mha.smolgen.dense1_b,
                             scratch);
    allocAndUpload<DataType>(&smol_dense2_w, cpu_weights.mha.smolgen.dense2_w,
                             scratch);
    allocAndUpload<DataType>(&smol_dense2_b, cpu_weights.mha.smolgen.dense2_b,
                             scratch);

    allocAndUpload<DataType>(&smol_ln1_gammas,
                             cpu_weights.mha.smolgen.ln1_gammas, scratch);
    allocAndUpload<DataType>(&smol_ln1_betas, cpu_weights.mha.smolgen.ln1_betas,
                             scratch);
    allocAndUpload<DataType>(&smol_ln2_gammas,
                             cpu_weights.mha.smolgen.ln2_gammas, scratch);
    allocAndUpload<DataType>(&smol_ln2_betas, cpu_weights.mha.smolgen.ln2_betas,
                             scratch);

    // GPU memory already allocated in AttentionBody.
    smol_global = smolgen_global_scratch;
  }

  if (!attention_mask.empty()) {
    has_attention_mask_ = true;
    allocAndUpload<DataType>(&attention_mask_, attention_mask, scratch);
  }

}

template <typename DataType>
static void cublasXgemm(cublasHandle_t handle, cublasOperation_t transa,
                        cublasOperation_t transb, int m, int n, int k,
                        float alpha, const DataType* A, int lda,
                        const DataType* B, int ldb, float beta, DataType* C,
                        int ldc) {
  const bool fp16 = std::is_same<half, DataType>::value;
  if (fp16) {
    unsigned short alpha_h = FP32toFP16(alpha);
    unsigned short beta_h = FP32toFP16(beta);
    ReportCUBLASErrors(cublasHgemm(
        handle, transa, transb, m, n, k, (const half*)&alpha_h, (const half*)A,
        lda, (const half*)B, ldb, (const half*)&beta_h, (half*)C, ldc));
  } else {
    ReportCUBLASErrors(cublasSgemm(handle, transa, transb, m, n, k, &alpha,
                                   (const float*)A, lda, (const float*)B, ldb,
                                   &beta, (float*)C, ldc));
  }
}

template <typename DataType>
Conv1Layer<DataType>::Conv1Layer(BaseLayer<DataType>* ip, int C, int H, int W,
                                 int Cin, ActivationFunction activation,
                                 bool bias, bool use_gemm_ex, bool nhwc)
    : BaseLayer<DataType>(C, H, W, ip, nhwc, use_gemm_ex), 
      c_input_(Cin),
      act_(activation),
      use_bias_(bias) {
  // Allocate memory for weights (filter tensor) and biases.
  const size_t weight_size = sizeof(DataType) * c_input_ * C * 1 * 1;
  ReportCUDAErrors(cudaMalloc(&weights_, weight_size));

  if (use_bias_) {
    const size_t bias_size = sizeof(DataType) * C;
    ReportCUDAErrors(cudaMalloc(&biases_, bias_size));
  }
}

template <typename DataType>
void Conv1Layer<DataType>::LoadWeights(float* pfilter, float* pBias,
                                       void* scratch) {
  const size_t weight_size = sizeof(float) * c_input_ * C * 1 * 1;
  const size_t bias_size = sizeof(float) * C;

  assert(scratch);
  ReportCUDAErrors(
      cudaMemcpy(scratch, pfilter, weight_size, cudaMemcpyHostToDevice));
  copyTypeConverted((DataType*)weights_, (float*)scratch, C * c_input_ * 1 * 1,
                    0);

  if (pBias) {
    ReportCUDAErrors(
        cudaMemcpy(scratch, pBias, bias_size, cudaMemcpyHostToDevice));
    copyTypeConverted((DataType*)biases_, (float*)scratch, C, 0);
  }
}

template <>
void Conv1Layer<half>::cublasSpecialMatrixMul(const half* A, const half* B,
                                              half* Out, int M, int N, int K,
                                              int batchSize,
                                              cublasHandle_t cublas) {
  if (nhwc_) {
      int flat = batchSize * N;

      cublasXgemm<half>(
          cublas, CUBLAS_OP_T, CUBLAS_OP_N, M, flat, K, 1.0f, A, K, B, K,            
          0.0f, Out, M           
      );

  } else {
      __half_raw one_h{0x3C00};
      __half_raw zero_h{0};
      half halfOne = one_h;
      half halfZero = zero_h;
      
      ReportCUBLASErrors(cublasGemmStridedBatchedEx(
          cublas, CUBLAS_OP_N, CUBLAS_OP_N, N, M, K, &halfOne, B, CUDA_R_16F, N,
          N * K, A, CUDA_R_16F, K, 0, &halfZero, Out, CUDA_R_16F, N, N * M,
          batchSize, CUDA_R_16F, CUBLAS_GEMM_DEFAULT));
  }
}
template <>
void Conv1Layer<float>::cublasSpecialMatrixMul(const float* A, const float* B,
                                               float* Out, int M, int N, int K,
                                               int batchSize,
                                               cublasHandle_t cublas) {
  float floatOne = 1.0f;
  float floatZero = 0.0f;

  if (nhwc_) {
        int flat = batchSize * N;
        cublasXgemm<float>(
            cublas, CUBLAS_OP_T, CUBLAS_OP_N, M, flat, K, 1.0f, A, K, B, K,            
            0.0f, Out, M           
        );
  } else {
      if (use_gemm_ex_) {
          ReportCUBLASErrors(cublasGemmStridedBatchedEx(
              cublas, CUBLAS_OP_N, CUBLAS_OP_N, N, M, K, &floatOne, B, CUDA_R_32F, N,
              N * K, A, CUDA_R_32F, K, 0, &floatZero, Out, CUDA_R_32F, N, N * M,
              batchSize, CUDA_R_32F, CUBLAS_GEMM_DEFAULT));
      } else {
          ReportCUBLASErrors(cublasSgemmStridedBatched(
              cublas, CUBLAS_OP_N, CUBLAS_OP_N, N, M, K, &floatOne, B, N, N * K, A, K,
              0, &floatZero, Out, N, N * M, batchSize));
      }
  }
}

template <typename DataType>
void Conv1Layer<DataType>::Eval(int N, DataType* output, const DataType* input,
                                const DataType*, void*,
                                size_t,
                                cudnnHandle_t , cublasHandle_t cublas,
                                cudaStream_t stream, DataType***) {
  cublasSpecialMatrixMul(weights_, input, output, C, H * W, c_input_, N, cublas);

  if (use_bias_) {
      if (nhwc_) {
        addBiasBatched(output, output, biases_, 1, N * H * W, C, act_, stream);
      } else {
          addBias_NCHW(output, output, biases_, N, C, H, W, act_, stream);
      }
  } else if (act_ != ACTIVATION_NONE) {
      addVectors(output, output, (DataType*)nullptr, N * C * H * W, N * C * H * W, 0, act_, stream);
  }
}

template <typename DataType>
Conv1Layer<DataType>::~Conv1Layer() {
  ReportCUDAErrors(cudaFree(weights_));
  if (use_bias_) ReportCUDAErrors(cudaFree(biases_));
}

template <typename DataType>
static void cublasXGemmStridedBatched(
    cublasHandle_t handle, cublasOperation_t transa, cublasOperation_t transb,
    int m, int n, int k, float alpha, const void* A, int lda,
    long long int strideA, const void* B, int ldb, long long int strideB,
    float beta, void* C, int ldc, long long int strideC, int batchCount,
    bool use_gemm_ex) {
  const bool fp16 = std::is_same<half, DataType>::value;
  if (fp16) {
    unsigned short alpha_h = FP32toFP16(alpha);
    unsigned short beta_h = FP32toFP16(beta);
    ReportCUBLASErrors(cublasGemmStridedBatchedEx(
        handle, transa, transb, m, n, k, &alpha_h, A, CUDA_R_16F, lda, strideA,
        B, CUDA_R_16F, ldb, strideB, &beta_h, C, CUDA_R_16F, ldc, strideC,
        batchCount, CUDA_R_16F, CUBLAS_GEMM_DEFAULT));
  } else {
    if (use_gemm_ex) {
      ReportCUBLASErrors(cublasGemmStridedBatchedEx(
          handle, transa, transb, m, n, k, &alpha, A, CUDA_R_32F, lda, strideA,
          B, CUDA_R_32F, ldb, strideB, &beta, C, CUDA_R_32F, ldc, strideC,
          batchCount, CUDA_R_32F, CUBLAS_GEMM_DEFAULT));
    } else {
      ReportCUBLASErrors(cublasSgemmStridedBatched(
          handle, transa, transb, m, n, k, &alpha, (const float*)A, lda,
          strideA, (const float*)B, ldb, strideB, &beta, (float*)C, ldc,
          strideC, batchCount));
    }
  }
}

template <typename DataType>
static void cublasXGemmBatched(cublasHandle_t handle, cublasOperation_t transa,
                               cublasOperation_t transb, int m, int n, int k,
                               float alpha, DataType** A, int lda, DataType** B,
                               int ldb, float beta, DataType** C, int ldc,
                               int batchCount) {
  const bool fp16 = std::is_same<half, DataType>::value;
  if (fp16) {
    unsigned short alpha_h = FP32toFP16(alpha);
    unsigned short beta_h = FP32toFP16(beta);
    ReportCUBLASErrors(cublasHgemmBatched(
        handle, transa, transb, m, n, k, (const half*)&alpha_h, (half**)A, lda,
        (half**)B, ldb, (const half*)&beta_h, (half**)C, ldc, batchCount));
  } else {
    ReportCUBLASErrors(cublasSgemmBatched(
        handle, transa, transb, m, n, k, &alpha, (float**)A, lda, (float**)B,
        ldb, &beta, (float**)C, ldc, batchCount));
  }
}


template <typename DataType>
void EncoderBlock<DataType>::Eval(int N, DataType* in_out_tensor,
                                  DataType* scratch, DataType* buffer1,
                                  DataType* buffer2, cublasHandle_t cublas,
                                  cudaStream_t stream,
                                  DataType*** offset_pointers) const {
  const int d_model = mha_q_size_;
  const int depth = d_model / encoder_heads_;

  if (prenorm_) {
    LayerNorm<DataType>(N * 64, embedding_op_size_, buffer1, in_out_tensor,
                        (const DataType*)nullptr, (const DataType*)nullptr,
                        ln1_gammas, ln1_betas, default_eps_, 1.0f, ACTIVATION_NONE, stream);
  }

  DataType* mha_src_ptr = prenorm_ ? buffer1 : in_out_tensor;
  DataType* smol_tmp_buf = prenorm_ ? buffer2 : buffer1;

  if (has_smolgen_) {
    {
      const int num_inputs = d_model;
      const int num_outputs = smol_compress_size_;
      const int batch = N * 64;
      cublasXgemm<DataType>(
          cublas, CUBLAS_OP_T, CUBLAS_OP_N, num_outputs, batch, num_inputs,
          1.0f, (const DataType*)smol_compress, num_inputs, mha_src_ptr,
          num_inputs, 0.0f, scratch, num_outputs);
    }

    {
      const int num_inputs = 64 * smol_compress_size_;
      const int num_outputs = smol_dense_1_size_;
      const int batch = N;
      cublasXgemm<DataType>(cublas, CUBLAS_OP_T, CUBLAS_OP_N, num_outputs,
                            batch, num_inputs, 1.0f,
                            (const DataType*)smol_dense1_w, num_inputs, scratch,
                            num_inputs, 0.0f, smol_tmp_buf, num_outputs);

      LayerNorm<DataType>(batch, num_outputs, scratch, smol_tmp_buf, smol_dense1_b,
                          (DataType*)nullptr, smol_ln1_gammas, smol_ln1_betas,
                          default_eps_, 1.0, smolgen_activation_, stream);
    }

    {
      const int num_inputs = smol_dense_1_size_;
      const int num_outputs = smol_dense_2_size_;
      const int batch = N;
      cublasXgemm<DataType>(cublas, CUBLAS_OP_T, CUBLAS_OP_N, num_outputs,
                            batch, num_inputs, 1.0f,
                            (const DataType*)smol_dense2_w, num_inputs, scratch,
                            num_inputs, 0.0f, smol_tmp_buf, num_outputs);

      LayerNorm<DataType>(batch, num_outputs, scratch, smol_tmp_buf, smol_dense2_b,
                          (DataType*)nullptr, smol_ln2_gammas, smol_ln2_betas,
                          default_eps_, 1.0, smolgen_activation_, stream);
    }

    {
      const int num_inputs = smol_dense_2_size_ / encoder_heads_;
      const int num_outputs = smol_global_size_;
      const int batch = N * encoder_heads_;
      cublasXgemm<DataType>(cublas, CUBLAS_OP_T, CUBLAS_OP_N, num_outputs,
                            batch, num_inputs, 1.0f,
                            (const DataType*)smol_global, num_inputs, scratch,
                            num_inputs, 0.0f, buffer2, num_outputs);
    }
  }

  DataType* mha_q;
  DataType* mha_k;
  DataType* mha_v;

  {
    const int num_inputs = embedding_op_size_;
    const int num_outputs = d_model;
    const int batch = N * 64;
    const int max_batch = max_batch_size_ * 64;

    mha_q = scratch;
    mha_k = mha_q + num_outputs * max_batch;
    mha_v = mha_k + num_outputs * max_batch;

    cublasXGemmStridedBatched<DataType>(
        cublas, CUBLAS_OP_T, CUBLAS_OP_N, num_outputs, batch, num_inputs, 1.0f,
        mha_qkv_w, num_inputs, num_inputs * num_outputs, mha_src_ptr,
        num_inputs, 0, 0.0f, mha_q, num_outputs, num_outputs * max_batch, 3,
        use_gemm_ex_);
    if (mha_qkv_b != nullptr) {
      addBiasBatched<DataType>(mha_q, mha_q, mha_qkv_b, 3, batch, num_outputs,
                               max_batch, ACTIVATION_NONE, stream);
    }
  }

  float factor = 1.0f / sqrt((float)depth);

  bool has_bias = has_smolgen_ || has_attention_mask_;
  bool is_buffer_initialized = has_smolgen_;

  if (has_attention_mask_) {
      AddAttentionMask<DataType>(N, encoder_heads_, buffer2, attention_mask_, is_buffer_initialized, stream);
      is_buffer_initialized = true;
  }

#ifdef USE_CUTLASS
  if (use_fused_mha_) {
    fusedMHA(buffer2, mha_q, mha_k, mha_v, has_bias ? buffer2 : nullptr, N,
             encoder_heads_, depth, stream);
  } else
#endif
  {
    if (*offset_pointers == nullptr) {
#ifndef NDEBUG
      cudaStreamCaptureStatus capture;
      ReportCUDAErrors(cudaStreamIsCapturing(stream, &capture));
      assert(capture !=
                 cudaStreamCaptureStatus::cudaStreamCaptureStatusActive &&
             "Stream capture is active, cannot allocate memory for offset pointers");
#endif
      ReportCUDAErrors(
          cudaMalloc((void**)offset_pointers,
                     encoder_heads_ * max_batch_size_ * 5 * sizeof(DataType*)));
      genOffsetPointers((DataType**)*offset_pointers, encoder_heads_,
                        max_batch_size_, depth, d_model, mha_k, mha_q, buffer1,
                        mha_v, buffer2, stream);
    }

    cublasXGemmBatched<DataType>(
        cublas, CUBLAS_OP_T, CUBLAS_OP_N, 64, 64, depth, factor, 
        *offset_pointers, d_model,
        *offset_pointers + encoder_heads_ * max_batch_size_, d_model, 0.0f,
        *offset_pointers + encoder_heads_ * max_batch_size_ * 2, 64,
        N * encoder_heads_);

    if (has_bias) {
      Softmax(encoder_heads_ * N * 64, 64, buffer1, buffer1, buffer2, stream);
    } else {
      Softmax(encoder_heads_ * N * 64, 64, buffer1, buffer1,
              (const DataType*)nullptr, stream);
    }

    cublasXGemmBatched<DataType>(
        cublas, CUBLAS_OP_N, CUBLAS_OP_N, depth, 64, 64, 1.0f,
        *offset_pointers + encoder_heads_ * max_batch_size_ * 3, d_model,
        *offset_pointers + encoder_heads_ * max_batch_size_ * 2, 64, 0.0f,
        *offset_pointers + encoder_heads_ * max_batch_size_ * 4, d_model,
        N * encoder_heads_);
  }

  {
    const int num_inputs = d_model;
    const int num_outputs = embedding_op_size_;
    const int batch = N * 64;

    if (prenorm_) {
      cublasXgemm(cublas, CUBLAS_OP_T, CUBLAS_OP_N, num_outputs, batch,
                  num_inputs, 1.0f, (const DataType*)mha_dense_w, num_inputs,
                  buffer2, num_inputs, 1.0f, in_out_tensor, num_outputs);
      if (mha_dense_b != nullptr) {
        addBiasBatched<DataType>(in_out_tensor, in_out_tensor, mha_dense_b, 1, batch,
                                 num_outputs, ACTIVATION_NONE, stream);
      }
    } else {
      cublasXgemm(cublas, CUBLAS_OP_T, CUBLAS_OP_N, num_outputs, batch,
                  num_inputs, 1.0f, (const DataType*)mha_dense_w, num_inputs,
                  buffer2, num_inputs, 0.0f, buffer1, num_outputs);

      LayerNorm<DataType>(N * 64, embedding_op_size_, scratch, buffer1, mha_dense_b,
                          in_out_tensor, ln1_gammas, ln1_betas, default_eps_,
                          alpha_, ACTIVATION_NONE, stream);
    }
  }

  DataType* ffn_src_ptr = scratch; 
  DataType* ffn_inter_ptr = prenorm_ ? buffer2 : in_out_tensor;

  if (prenorm_) {
    LayerNorm<DataType>(N * 64, embedding_op_size_, ffn_src_ptr, in_out_tensor,
                        (const DataType*)nullptr, (const DataType*)nullptr,
                        ln2_gammas, ln2_betas, default_eps_, 1.0f, ACTIVATION_NONE, stream);
  }

  // #FFN dense 1
  if (d_conv == nullptr) {
    const int num_inputs = embedding_op_size_;
    const int num_outputs = ffn_dense1_size_;
    const int batch = N * 64;
    cublasXgemm(cublas, CUBLAS_OP_T, CUBLAS_OP_N, num_outputs, batch,
                num_inputs, 1.0f, (const DataType*)ffn_dense1_w, num_inputs,
                ffn_src_ptr, num_inputs, 0.0f, ffn_inter_ptr, num_outputs);
    addBiasBatched(ffn_inter_ptr, ffn_inter_ptr, ffn_dense1_b, 1, batch,
                   num_outputs, ffn_activation_, stream);
  } else {
    const int num_inputs = embedding_op_size_;
    const int num_outputs = ffn_dense1_size_;
    const int batch = N * 64;
    cublasXgemm(cublas, CUBLAS_OP_T, CUBLAS_OP_N, num_outputs, batch,
                num_inputs, 1.0f, (const DataType*)ffn_dense1_w, num_inputs,
                ffn_src_ptr, num_inputs, 0.0f, buffer1, num_outputs);
    addBiasBatched(buffer1, buffer1, ffn_dense1_b, 1, batch,
                   num_outputs, ffn_activation_, stream);

    d_conv->Eval(N, ffn_inter_ptr, buffer1, nullptr, scratch, 0, nullptr, cublas, stream, offset_pointers);
  }

  {
    const int num_inputs = ffn_dense1_size_;
    const int num_outputs = embedding_op_size_;
    const int batch = N * 64;

    if (prenorm_) {
      cublasXgemm(cublas, CUBLAS_OP_T, CUBLAS_OP_N, num_outputs, batch,
                  num_inputs, 1.0f, (const DataType*)ffn_dense2_w, num_inputs,
                  ffn_inter_ptr, num_inputs, 1.0f, in_out_tensor, num_outputs);
      if (ffn_dense2_b != nullptr) {
        addBiasBatched<DataType>(in_out_tensor, in_out_tensor, ffn_dense2_b, 1, batch,
                                 num_outputs, ACTIVATION_NONE, stream);
      }
    } else {
      cublasXgemm(cublas, CUBLAS_OP_T, CUBLAS_OP_N, num_outputs, batch,
                  num_inputs, 1.0f, (const DataType*)ffn_dense2_w, num_inputs,
                  ffn_inter_ptr, num_inputs, 0.0f, buffer1, num_outputs);

      LayerNorm<DataType>(N * 64, embedding_op_size_, in_out_tensor, buffer1,
                          ffn_dense2_b, scratch, ln2_gammas, ln2_betas,
                          default_eps_, alpha_, ACTIVATION_NONE, stream);
    }
  }
}


template <typename DataType>
void AttentionPolicyHead<DataType>::Eval(
    int N, DataType* output, const DataType* input, const DataType* input2,
    void* scratch, size_t scratch_size, cudnnHandle_t /*cudnn*/,
    cublasHandle_t cublas, cudaStream_t stream, DataType*** offset_pointers) {
  DataType* input2_tensor = (DataType*)input2;
  DataType* buffer1 = output + scratch_size / (2 * sizeof(DataType));
  DataType* buffer2 = input2_tensor + scratch_size / (2 * sizeof(DataType));

  int inputC = this->input_->GetC();
  bool input_nhwc = attention_body_ || this->input_->isNHWC();
  if (!input_nhwc)
    convertNCHWtoNHWC((DataType*)scratch, input, N, inputC, N, inputC, 8, 8,
                      stream);

  // 1. Policy embedding (fully connected layer)
  // Input data in NHWC layout N*(64)*C, output is N*(64)*embedding_op_size_
  DataType* pol_embedding = input2_tensor;
  {
    const int num_outputs = embedding_op_size_;
    const int num_inputs = inputC;
    const int batch = N * 64;
    cublasXgemm<DataType>(cublas, CUBLAS_OP_T, CUBLAS_OP_N, num_outputs, batch,
                          num_inputs, 1.0f, (const DataType*)ip_pol_w_,
                          num_inputs,
                          input_nhwc ? input : (DataType*)scratch,
                          num_inputs, 0.0f, pol_embedding, num_outputs);
    addBiasBatched(pol_embedding, pol_embedding, ip_pol_b_, 1, batch,
                   num_outputs, act_, stream);
  }

  // 2. Encoder layers
  for (const auto pEnc : encoder_weights_) {
    pEnc->Eval(N, input2_tensor, (DataType*)scratch, buffer1, buffer2, cublas,
               stream, offset_pointers);
  }  // End of encoder blocks

  DataType* wq;
  DataType* wk;
  {
    const int num_inputs = embedding_op_size_;
    const int num_outputs = policy_d_model_;
    const int batch = N * 64;
    wq = (DataType*)scratch;
    wk = wq + num_outputs * batch;

    cublasXGemmStridedBatched<DataType>(
        cublas, CUBLAS_OP_T, CUBLAS_OP_N, num_outputs, batch, num_inputs, 1.0f,
        wqk_w_, num_inputs, num_inputs * num_outputs, input2_tensor, num_inputs,
        0, 0.0f, wq, num_outputs, num_outputs * batch, 2, use_gemm_ex_);

    addBiasBatched<DataType>(wq, wq, wqk_b_, 2, batch, num_outputs,
                             ACTIVATION_NONE, stream);
  }

  // dk = tf.math.sqrt(tf.cast(tf.shape(keys)[-1], self.model_dtype))
  // policy matmul_qk = tf.matmul(queries, keys, transpose_b=True)
  // policy_attn_logits = matmul_qk / dk
  {
    // shape(keys)[-1] = policy_d_model_
    float factor = 1.0f / sqrt((float)policy_d_model_);

    // A/B, and M/N are swapped for row-major to col-major transform
    // leave 8*24 after each batch to interleave promotion_logits (computed
    // later below)
    cublasXGemmStridedBatched<DataType>(
        cublas, CUBLAS_OP_T, CUBLAS_OP_N, 64 /*M*/, 64 /*N*/,
        policy_d_model_ /*K*/,
        factor,  // to handle "/ tf.math.sqrt(dk)"
        wk /*A*/, policy_d_model_ /*LDA*/, 64 * policy_d_model_, /*strideA*/
        wq /*B*/, policy_d_model_ /*LDB*/, 64 * policy_d_model_, /*strideB*/
        0.0f, output /*C*/,  // output (policy_attn_logits)
        64 /*LDC*/, 64 * 64 + 8 * 24 /*strideC*/, N, use_gemm_ex_);
  }

  // Compute promotion_logits in a single kernel (and put the result just after
  // policy_attn_logits interleaved to get concat for free)
  DataType* promotion_logits = output + 64 * 64;

  ComputePromotionLogits<DataType>(N, policy_d_model_, promotion_logits, wk,
                                   ip4_pol_w_, output, stream);
}

template <typename DataType>
AttentionPolicyHead<DataType>::~AttentionPolicyHead() {
  ReportCUDAErrors(cudaFree(ip_pol_w_));
  ReportCUDAErrors(cudaFree(ip_pol_b_));
  ReportCUDAErrors(cudaFree(ip2_pol_w_));
  ReportCUDAErrors(cudaFree(ip2_pol_b_));
  ReportCUDAErrors(cudaFree(ip3_pol_w_));
  ReportCUDAErrors(cudaFree(ip3_pol_b_));
  ReportCUDAErrors(cudaFree(ip4_pol_w_));
  ReportCUDAErrors(cudaFree(wqk_w_));
  ReportCUDAErrors(cudaFree(wqk_b_));
  for (const auto pEnc : encoder_weights_) delete pEnc;
}

template <typename DataType>
EncoderBlock<DataType>::~EncoderBlock() {
  ReportCUDAErrors(cudaFree(mha_q_w));
  ReportCUDAErrors(cudaFree(mha_q_b));
  ReportCUDAErrors(cudaFree(mha_k_w));
  ReportCUDAErrors(cudaFree(mha_k_b));
  ReportCUDAErrors(cudaFree(mha_v_w));
  ReportCUDAErrors(cudaFree(mha_v_b));
  ReportCUDAErrors(cudaFree(mha_qkv_w));
  ReportCUDAErrors(cudaFree(mha_qkv_b));
  ReportCUDAErrors(cudaFree(mha_dense_w));
  ReportCUDAErrors(cudaFree(mha_dense_b));
  ReportCUDAErrors(cudaFree(ln1_gammas));
  ReportCUDAErrors(cudaFree(ln1_betas));
  ReportCUDAErrors(cudaFree(ln2_gammas));
  ReportCUDAErrors(cudaFree(ln2_betas));
  ReportCUDAErrors(cudaFree(ffn_dense1_w));
  ReportCUDAErrors(cudaFree(ffn_dense1_b));
  ReportCUDAErrors(cudaFree(ffn_dense2_w));
  ReportCUDAErrors(cudaFree(ffn_dense2_b));
  if (has_smolgen_) {
    ReportCUDAErrors(cudaFree(smol_compress));
    ReportCUDAErrors(cudaFree(smol_dense1_w));
    ReportCUDAErrors(cudaFree(smol_dense1_b));
    ReportCUDAErrors(cudaFree(smol_dense2_w));
    ReportCUDAErrors(cudaFree(smol_dense2_b));
    ReportCUDAErrors(cudaFree(smol_ln1_gammas));
    ReportCUDAErrors(cudaFree(smol_ln1_betas));
    ReportCUDAErrors(cudaFree(smol_ln2_gammas));
    ReportCUDAErrors(cudaFree(smol_ln2_betas));
  }
  if (has_attention_mask_) ReportCUDAErrors(cudaFree(attention_mask_));
}

template <typename DataType>
EmbeddingLayer<DataType>::EmbeddingLayer(BaseLayer<DataType>* ip,
                                         const std::vector<float>& weights,
                                         const std::vector<float>& biases,
                                         void* scratch, ActivationFunction act)
    : BaseLayer<DataType>(biases.size(), 8, 8, ip), act_(act) {
  allocAndUpload<DataType>(&weights_, weights, scratch);
  allocAndUpload<DataType>(&biases_, biases, scratch);
}

template <typename DataType>
EmbeddingLayer<DataType>::~EmbeddingLayer() {
  ReportCUDAErrors(cudaFree(weights_));
  ReportCUDAErrors(cudaFree(biases_));
}

template <typename DataType>
void EmbeddingLayer<DataType>::Eval(
    int N, DataType* output, const DataType* input, const DataType* /*input2*/,
    void* /*scratch*/, size_t /*scratch_size*/, cudnnHandle_t /*cudnn*/,
    cublasHandle_t cublas, cudaStream_t stream, DataType***) {
  const int num_outputs = this->GetC();
  const int num_inputs = this->input_->GetC();
  const int batch = N * 64;
  cublasXgemm<DataType>(cublas, CUBLAS_OP_T, CUBLAS_OP_N, num_outputs, batch,
                        num_inputs, 1.0f, weights_, num_inputs, input,
                        num_inputs, 0.0f, output, num_outputs);
  addBiasBatched(output, output, biases_, 1, batch, num_outputs, act_, stream);
}

template <typename DataType>
Backbone<DataType>::Backbone(const MultiHeadWeights& weights,
                            void* scratch, 
                            Activations activations,
                            int input_c,
                            bool prenorm,
                            int residual_blocks,
                            int mobilenet_blocks,
                            int convnext_blocks,
                            int encoder_blocks,
                            std::string first_block,
                            int min_batch_size,
                            int max_batch_size,
                            bool use_gemm_ex, 
                            bool fused_mha,
                            bool nhwc,
                            bool use_res_block_winograd_fuse_opt,
                            bool allow_cache_opt,
                            int l2_cache_size,
                            int shared_mem_per_block_optin,
                            bool use_custom_depthwise,
                            const std::vector<std::vector<float>>& layer_masks,
                            cudnnHandle_t cudnn)     
  : BaseLayer<DataType>(weights.ip_emb_b.size(), 8, 8, nullptr, nhwc, use_gemm_ex),
    embedding_op_size_(weights.ip_emb_b.size()),
    encoder_head_count_(weights.encoder_head_count),
    default_epsilon_(weights.epsilon),
    activations_(activations),
    act_(activations.default_activation),
    input_c_(input_c),
    prenorm_(prenorm),
    residual_blocks_(residual_blocks),
    mobilenet_blocks_(mobilenet_blocks),
    encoder_blocks_(encoder_blocks),
    convnext_blocks_(convnext_blocks),
    has_gating_(weights.ip_mult_gate.size() > 0 &&
                weights.ip_add_gate.size() > 0),
    has_smolgen_(weights.has_smolgen),
    use_fused_mha_(fused_mha),
    nhwc_(nhwc),
    use_res_block_winograd_fuse_opt_(use_res_block_winograd_fuse_opt),
    allow_cache_opt_(allow_cache_opt),
    l2_cache_size_(l2_cache_size),
    shared_mem_per_block_optin_(shared_mem_per_block_optin),
    use_custom_depthwise_(use_custom_depthwise),
    smolgen_global_(nullptr),
    ip_mult_gate_(nullptr),
    ip_add_gate_(nullptr),
    final_ln_gammas_(nullptr),
    final_ln_betas_(nullptr) {
    
  starts_with_encoder_ = (first_block == "T" || first_block == "B" || first_block == "D");
  starts_with_residual_ = first_block == "R";
  starts_with_mobilenet_ = first_block == "M";
  starts_with_convnext_ = first_block == "C";
  BaseLayer<DataType>* prev_layer = nullptr;
  
  int current_channels = input_c_;
  int encoder_count = 0;

  // Deep norm scaling for residual stream
  if (prenorm) {
    alpha_ = 1.0f;
  }
  else {
    alpha_ = (encoder_blocks_ + convnext_blocks_) > 0 ? (float)pow(2.0 * encoder_blocks_ + 1.0 * convnext_blocks_, -0.25) : 1.0f;
  }

  if (has_smolgen_) {
    allocAndUpload<DataType>(&smolgen_global_, weights.smolgen_w, scratch);
    smolgen_global_size_ = 64*64;
  }

  if (has_gating_) {
    allocAndUpload<DataType>(&ip_mult_gate_, weights.ip_mult_gate, scratch);
    allocAndUpload<DataType>(&ip_add_gate_, weights.ip_add_gate, scratch);
  }

  if (starts_with_encoder_) {
    allocAndUpload<DataType>(&ip_emb_w_, weights.ip_emb_w, scratch);
    allocAndUpload<DataType>(&ip_emb_b_, weights.ip_emb_b, scratch);
    allocAndUpload<DataType>(&ip_emb_pre_w_, weights.ip_emb_preproc_w, scratch);
    allocAndUpload<DataType>(&ip_emb_pre_b_, weights.ip_emb_preproc_b, scratch);

    allocAndUpload<DataType>(&ip_emb_ln_g_, weights.ip_emb_ln_gammas, scratch);
    allocAndUpload<DataType>(&ip_emb_ln_b_, weights.ip_emb_ln_betas, scratch);

    allocAndUpload<DataType>(&ip_emb_ffn_d1_w_, weights.ip_emb_ffn.dense1.weights,
                             scratch);
    allocAndUpload<DataType>(&ip_emb_ffn_d1_b_, weights.ip_emb_ffn.dense1.biases,
                             scratch);

    allocAndUpload<DataType>(&ip_emb_ffn_d2_w_, weights.ip_emb_ffn.dense2.weights,
                             scratch);
    allocAndUpload<DataType>(&ip_emb_ffn_d2_b_, weights.ip_emb_ffn.dense2.biases,
                             scratch);

    allocAndUpload<DataType>(&ip_emb_ffn_ln_g_, weights.ip_emb_ffn_ln_gammas,
                             scratch);
    allocAndUpload<DataType>(&ip_emb_ffn_ln_b_, weights.ip_emb_ffn_ln_betas,
                             scratch);

    embedding_dense_size_ = weights.ip_emb_preproc_b.size() / 64;
    embedding_ffn_size_ = weights.ip_emb_ffn.dense2.biases.size();
    embedding_ffn_dff_ = weights.ip_emb_ffn.dense1.biases.size();

    // Potential depthwise convolution between the two FFN dense layers of the embedding
    if (weights.ip_emb_ffn.d_conv.biases.size() > 0) {
        ip_emb_ffn_d_conv_ = std::make_unique<DepthwiseCustom<DataType>>(embedding_ffn_size_, 8, 8, 
            activations_.ffn_activation, true,
            weights.ip_emb_ffn.d_conv.rook_channels,
            weights.ip_emb_ffn.d_conv.bishop_channels,
            weights.ip_emb_ffn.d_conv.knight_channels);
        ip_emb_ffn_d_conv_->LoadWeights(weights.ip_emb_ffn.d_conv.weights,
                            const_cast<float*>(weights.ip_emb_ffn.d_conv.biases.data()),
                        scratch); 
    }
    prev_layer = this;
    current_channels = embedding_op_size_;
  }

  // CNN start for the network : 1x1 input conv
  else if (starts_with_mobilenet_ || starts_with_convnext_){
    auto input_conv = std::make_unique<Conv1Layer<DataType>>(
              prev_layer, weights.input.biases.size(), 8, 8, current_channels,
              act_, true, use_gemm_ex, nhwc_
          );
    
    input_conv->LoadWeights(const_cast<float*>(weights.input.weights.data()),
                                const_cast<float*>(weights.input.biases.data()),
                                scratch);
    
    prev_layer = input_conv.get();

    input_conv_ = std::move(input_conv);
    
    current_channels = weights.input.biases.size();
    input_conv_output_channels_ = current_channels;    
  }

  #ifdef USE_CUDNN
  else if (starts_with_residual_){
    current_channels = weights.input.biases.size();
    auto input_conv = std::make_unique<ConvLayer<DataType>>(
            prev_layer, current_channels, 8, 8, 3, input_c_, act_, true, use_gemm_ex,
            min_batch_size, max_batch_size, cudnn
    );

    input_conv->LoadWeights(weights.input.weights,
                              const_cast<float*>(weights.input.biases.data()),
                          scratch);

    prev_layer = input_conv.get();

    input_conv_ = std::move(input_conv);

    input_conv_output_channels_ = current_channels;   
  }
  #endif

  // Iterates over each block of the main backbone tower
  for (size_t i = 0; i < weights.tower.size(); ++i) {
      const auto& pb_block = weights.tower[i];
      TowerNode node;
      node.in_channels = current_channels;
      // CNN -> ENC transition between a CNN block and an encoder one.
      if (!pb_block.dense_w.empty()) {
          current_channels = pb_block.dense_b.size(); 
          allocAndUpload<DataType>(&node.dense_w, pb_block.dense_w, scratch);
          allocAndUpload<DataType>(&node.dense_b, pb_block.dense_b, scratch);
      }
      if (!pb_block.ln_betas.empty()) {
          allocAndUpload<DataType>(&node.ln_gammas, pb_block.ln_gammas, scratch);
          allocAndUpload<DataType>(&node.ln_betas, pb_block.ln_betas, scratch);
      }

      if (!pb_block.mult_gate.empty()){
        allocAndUpload<DataType>(&node.mult_gate, pb_block.mult_gate, scratch);
        allocAndUpload<DataType>(&node.add_gate, pb_block.add_gate, scratch);
      }
      node.out_channels = current_channels;
      embedding_op_size_ = current_channels;

      // ENC -> CNN
      if (!pb_block.enc_cnn.biases.empty()) {
          node.transition_cnn = std::make_unique<Conv1Layer<DataType>>(
              prev_layer, pb_block.enc_cnn.biases.size(), 8, 8, current_channels,
              act_, true, use_gemm_ex, nhwc_
          );
          node.transition_cnn->LoadWeights(const_cast<float*>(pb_block.enc_cnn.weights.data()),
                                           const_cast<float*>(pb_block.enc_cnn.biases.data()), 
                                           scratch);

          current_channels = pb_block.enc_cnn.biases.size();
          prev_layer = node.transition_cnn.get();
      }

      // CNN -> CNN
      else if (!pb_block.cnn_cnn.biases.empty()) {
          node.transition_cnn = std::make_unique<Conv1Layer<DataType>>(
              prev_layer, pb_block.cnn_cnn.biases.size(), 8, 8, current_channels,
              act_, true, use_gemm_ex, nhwc_
          );

          node.transition_cnn->LoadWeights(const_cast<float*>(pb_block.cnn_cnn.weights.data()),
                                  const_cast<float*>(pb_block.cnn_cnn.biases.data()), 
                                  scratch);
          
          current_channels = pb_block.cnn_cnn.biases.size();
          prev_layer = node.transition_cnn.get();
      }

      // Encoder block
      if (std::holds_alternative<BaseWeights::EncoderLayer>(pb_block.block)) {
          node.type = TowerNode::TRANSFORMER;
          const auto& enc_weights = std::get<BaseWeights::EncoderLayer>(pb_block.block);
          std::vector<float> mask = (encoder_count < layer_masks.size()) ? layer_masks[encoder_count] : std::vector<float>();
          node.encoder = std::make_unique<EncoderBlock<DataType>>(
              enc_weights, scratch, encoder_head_count_, embedding_op_size_, 
              alpha_,
              smolgen_global_, smolgen_global_size_, max_batch_size,
              activations_.smolgen_activation, activations_.ffn_activation,
              default_epsilon_, prenorm_, use_gemm_ex, use_fused_mha_, mask
          );
          encoder_count++;
      } 

      // Mobile Net block
      else if (std::holds_alternative<BaseWeights::MobileNet>(pb_block.block)) {
        node.type = TowerNode::MOBILENET;
        const auto& m_weights = std::get<BaseWeights::MobileNet>(pb_block.block);

        int se_k = m_weights.se.b1.size();
        int c_expand = m_weights.conv1.biases.size();

        auto conv1 = std::make_unique<Conv1Layer<DataType>>(prev_layer, 
          c_expand, 8, 8, current_channels, act_, true, use_gemm_ex, nhwc_);
        conv1->LoadWeights(const_cast<float*>(m_weights.conv1.weights.data()),
                          const_cast<float*>(m_weights.conv1.biases.data()),
                          scratch);
        prev_layer = conv1.get();
        node.cnn_layers.push_back(std::move(conv1));
        current_channels = c_expand;

        if (use_custom_depthwise_) {
          auto d_conv = std::make_unique<DepthwiseCustom<DataType>>(c_expand, 8, 8, act_, nhwc_,
              m_weights.d_conv.rook_channels,
              m_weights.d_conv.bishop_channels,
              m_weights.d_conv.knight_channels);
          d_conv->LoadWeights(m_weights.d_conv.weights,
                              const_cast<float*>(m_weights.d_conv.biases.data()),
                          scratch);
          prev_layer = d_conv.get();
          node.cnn_layers.push_back(std::move(d_conv));
        }
        #ifdef USE_CUDNN
        else {
          auto d_conv = std::make_unique<DepthwiseConvLayer<DataType>>(prev_layer, c_expand, 8, 8, act_, use_gemm_ex,
              min_batch_size, max_batch_size,
              m_weights.d_conv.rook_channels,
              m_weights.d_conv.bishop_channels,
              m_weights.d_conv.knight_channels,
              cudnn
            );
          d_conv->LoadWeights(m_weights.d_conv.weights,
                              const_cast<float*>(m_weights.d_conv.biases.data()),
                          scratch);
          prev_layer = d_conv.get();
          node.cnn_layers.push_back(std::move(d_conv));
        }
        #endif



        auto conv2 = std::make_unique<Conv1Layer<DataType>>(prev_layer, 
          m_weights.conv2.biases.size(), 8, 8, current_channels, ACTIVATION_NONE, false, use_gemm_ex, nhwc_);
        conv2->LoadWeights(const_cast<float*>(m_weights.conv2.weights.data()),
                          nullptr,
                          scratch);
        prev_layer = conv2.get();
        node.cnn_layers.push_back(std::move(conv2));
        current_channels = m_weights.conv2.biases.size();

        auto se = std::make_unique<SELayer<DataType>>(prev_layer,
        se_k, false, act_, false);
        se->LoadWeights(const_cast<float*>(m_weights.se.w1.data()),
                        const_cast<float*>(m_weights.se.b1.data()),
                        const_cast<float*>(m_weights.se.w2.data()),
                        const_cast<float*>(m_weights.se.b2.data()),
                        const_cast<float*>(m_weights.conv2.biases.data()),
                        scratch);

        prev_layer = se.get();
        node.cnn_layers.push_back(std::move(se));
      }

      #ifdef USE_CUDNN
      else if (std::holds_alternative<BaseWeights::Residual>(pb_block.block)) {
        node.type = TowerNode::RESIDUAL;
        const auto& r_weights = std::get<BaseWeights::Residual>(pb_block.block);
        int se_k = r_weights.se.b1.size();

        // 1. First Convolution Layer: Reads from prev_layer, writes to intermediate buf
        auto conv1 = std::make_unique<ConvLayer<DataType>>(
                prev_layer, current_channels, 8, 8, 3, current_channels, act_, true, use_gemm_ex,
                min_batch_size, max_batch_size, cudnn
        );

        conv1->LoadWeights(r_weights.conv1.weights,
                          const_cast<float*>(r_weights.conv1.biases.data()),
                          scratch);        
        
        prev_layer = conv1.get();
        node.cnn_layers.push_back(std::move(conv1));

        // 2. Second Convolution Layer: Performs the fused residual skip connection & optional SE
        auto conv2 = std::make_unique<ConvLayer<DataType>>(
                prev_layer, current_channels, 8, 8, 3, current_channels, ACTIVATION_NONE, false, use_gemm_ex,
                min_batch_size, max_batch_size, cudnn
        );

        conv2->LoadWeights(r_weights.conv2.weights,
                            nullptr,
                            scratch);           
        
        node.cnn_layers.push_back(std::move(conv2));

        auto se = std::make_unique<SELayer<DataType>>(prev_layer,
        se_k, true, act_, true);
        se->LoadWeights(const_cast<float*>(r_weights.se.w1.data()),
                        const_cast<float*>(r_weights.se.b1.data()),
                        const_cast<float*>(r_weights.se.w2.data()),
                        const_cast<float*>(r_weights.se.b2.data()),
                        const_cast<float*>(r_weights.conv2.biases.data()),
                        scratch);

        prev_layer = se.get();
        node.cnn_layers.push_back(std::move(se));
      }
      #endif

      // ConvNext block
      else if (std::holds_alternative<BaseWeights::ConvNext>(pb_block.block)) {
        node.type = TowerNode::CONVNEXT;
        const auto& c_weights = std::get<BaseWeights::ConvNext>(pb_block.block);
        allocAndUpload<DataType>(&node.convnext_ln1_betas, c_weights.ln1_betas, scratch);
        allocAndUpload<DataType>(&node.convnext_ln1_gammas, c_weights.ln1_gammas, scratch);
        allocAndUpload<DataType>(&node.convnext_ffn_dense1_w, c_weights.ffn.dense1.weights, scratch);
        allocAndUpload<DataType>(&node.convnext_ffn_dense1_b, c_weights.ffn.dense1.biases, scratch);
        allocAndUpload<DataType>(&node.convnext_ffn_dense2_w, c_weights.ffn.dense2.weights, scratch);
        allocAndUpload<DataType>(&node.convnext_ffn_dense2_b, c_weights.ffn.dense2.biases, scratch);
        allocAndUpload<DataType>(&node.convnext_ln2_betas, c_weights.ln2_betas, scratch);
        allocAndUpload<DataType>(&node.convnext_ln2_gammas, c_weights.ln2_gammas, scratch);
        node.dff_channels = c_weights.ffn.dense1.biases.size();

        auto d_conv = std::make_unique<DepthwiseCustom<DataType>>(current_channels, 8, 8, act_, nhwc_,
            c_weights.d_conv.rook_channels,
            c_weights.d_conv.bishop_channels,
            c_weights.d_conv.knight_channels);
        d_conv->LoadWeights(c_weights.d_conv.weights,
                            const_cast<float*>(c_weights.d_conv.biases.data()),
                        scratch);
        prev_layer = d_conv.get();
        node.cnn_layers.push_back(std::move(d_conv));
    

      }      
      node.out_channels = current_channels;
      tower_nodes_.push_back(std::move(node));

  }

  if (!weights.final_ln_gammas.empty()) {
    allocAndUpload<DataType>(&final_ln_gammas_, weights.final_ln_gammas, scratch);
    allocAndUpload<DataType>(&final_ln_betas_, weights.final_ln_betas, scratch);
  }

  prev_layer = this;
  this->C = current_channels;
}

template <typename DataType>
void Backbone<DataType>::Eval(int N, DataType* output,
                              const DataType* input,
                              const DataType* input2, void* scratch,
                              size_t scratch_size, cudnnHandle_t cudnn,
                              cublasHandle_t cublas, cudaStream_t stream,
                              DataType*** offset_pointers) {

  DataType* flow = (DataType*)output; 
  DataType* buf0 = (DataType*)input; 
  DataType* buf1 = (DataType*)input2;  
 
  DataType* buf2 = buf1 + scratch_size / (2 * sizeof(DataType));

  DataType* temp = (DataType*)scratch;

  int inputC = input_c_;
  int current_channels = inputC;

  if (starts_with_encoder_) {
      const int num_outputs = 64 * embedding_dense_size_;
      const int num_inputs = 64 * 12;
      const int batch = N;

      convertNCHWtoNHWC(temp, buf0, N, inputC, N, 12, 8, 8, stream);
      cublasXgemm<DataType>(
          cublas, CUBLAS_OP_T, CUBLAS_OP_N, num_outputs, batch, num_inputs,
          1.0f, (const DataType*)ip_emb_pre_w_, num_inputs,
          temp, num_inputs, 0.0f, buf1, num_outputs);

      const int size = num_outputs * N;
      addVectors(buf1, buf1, ip_emb_pre_b_, size, size, num_outputs, ACTIVATION_NONE, stream);
      inputPreprocessForAttentionBody(temp, buf0, buf1, N, kInputPlanes, embedding_dense_size_, true, stream);
      inputC += embedding_dense_size_;
      {
      const int num_outputs = embedding_op_size_;
      const int num_inputs = inputC;
      const int batch = N * 64;
      cublasXgemm<DataType>(cublas, CUBLAS_OP_T, CUBLAS_OP_N, num_outputs,
                            batch, num_inputs, 1.0f, (const DataType*)ip_emb_w_,
                            num_inputs, temp, num_inputs, 0.0f, flow, num_outputs);
      LayerNorm<DataType>(N * 64, embedding_op_size_, temp, flow,
                          ip_emb_b_, (DataType*)nullptr, ip_emb_ln_g_,
                          ip_emb_ln_b_, default_epsilon_, 1.0, act_, stream);
      }

      if (has_gating_) {
        applyInputGating<DataType>(temp, temp, ip_mult_gate_, ip_add_gate_, N, 64, embedding_op_size_, stream);
      }

      if (ip_emb_ffn_d_conv_ == nullptr) {
        {
        const int num_inputs = embedding_ffn_size_;
        const int num_outputs = embedding_ffn_dff_; 
        const int batch = N * 64;
        cublasXgemm(cublas, CUBLAS_OP_T, CUBLAS_OP_N, num_outputs, batch, num_inputs, 1.0f, 
                    (const DataType*)ip_emb_ffn_d1_w_, num_inputs, temp, num_inputs, 0.0f, buf1, num_outputs);
        addBiasBatched(buf1, buf1, ip_emb_ffn_d1_b_, 1, batch, num_outputs, activations_.ffn_activation, stream);
        }
      }
      else {
        {
        const int num_inputs = embedding_ffn_size_;
        const int num_outputs = embedding_ffn_dff_; 
        const int batch = N * 64;
        cublasXgemm(cublas, CUBLAS_OP_T, CUBLAS_OP_N, num_outputs, batch, num_inputs, 1.0f, 
                    (const DataType*)ip_emb_ffn_d1_w_, num_inputs, temp, num_inputs, 0.0f, buf0, num_outputs);
        addBiasBatched(buf0, buf0, ip_emb_ffn_d1_b_, 1, batch, num_outputs, activations_.ffn_activation, stream);

        ip_emb_ffn_d_conv_->Eval(N, buf1, buf0, nullptr, scratch, scratch_size, cudnn, cublas, stream, offset_pointers);
        }        
      }

      {
      const int num_inputs = embedding_ffn_dff_; 
      const int num_outputs = embedding_ffn_size_;
      const int batch = N * 64;
      
      cublasXgemm(cublas, CUBLAS_OP_T, CUBLAS_OP_N, num_outputs, batch, num_inputs, 1.0f, 
                  (const DataType*)ip_emb_ffn_d2_w_, num_inputs, buf1, num_inputs, 0.0f, buf2, num_outputs); 
      
      LayerNorm<DataType>(N * 64, embedding_ffn_size_, flow, buf2, ip_emb_ffn_d2_b_, temp, 
                          ip_emb_ffn_ln_g_, ip_emb_ffn_ln_b_, default_epsilon_, alpha_, ACTIVATION_NONE, stream);
      }

      current_channels = embedding_op_size_;
  }

  else if (starts_with_mobilenet_ || starts_with_convnext_ || starts_with_residual_) {
      input_conv_->Eval(N, flow, buf0, nullptr, (DataType*)scratch, scratch_size, cudnn, cublas, stream, offset_pointers);
      current_channels = input_conv_output_channels_;
  }

  //int b_idx = -1;
  for (auto& node : tower_nodes_) {
      //b_idx ++;
      DataType* temp_flow = flow;
      DataType* temp_spare = buf0;

      if (node.dense_w != nullptr) {
          const int batch = N * 64; 
          const int num_inputs = node.in_channels;
          const int num_outputs = node.out_channels;

          cublasXgemm<DataType>(
                cublas, CUBLAS_OP_T, CUBLAS_OP_N, num_outputs, batch, num_inputs, 1.0f, 
                (const DataType*)node.dense_w, num_inputs, temp_flow, num_inputs, 0.0f, temp_spare, num_outputs
          );
          std::swap(temp_flow, temp_spare);

          if (node.ln_betas == nullptr) {
              addBiasBatched(temp_flow, temp_flow, node.dense_b, 1, batch, num_outputs, ACTIVATION_NONE, stream);
          }
      }

      if (node.ln_betas != nullptr) {
          const int batch = N * 64; 
          const int num_inputs = node.in_channels;
          const int num_outputs = node.out_channels;
          LayerNorm<DataType>(batch, num_outputs, temp_spare, temp_flow, node.dense_b, (DataType*)nullptr, 
                              node.ln_gammas, node.ln_betas, default_epsilon_, 1.0, ACTIVATION_NONE, stream);
          std::swap(temp_flow, temp_spare);
      }

      if (node.mult_gate != nullptr) {
          const int num_outputs = node.out_channels;
          applyInputGating<DataType>(flow, temp_flow, node.mult_gate, node.add_gate, N, 64, num_outputs, stream);
      }

      if (node.transition_cnn != nullptr) {
          node.transition_cnn->Eval(N, temp_spare, temp_flow, nullptr, (DataType*)scratch, scratch_size, cudnn, cublas, stream, offset_pointers);
          std::swap(temp_flow, temp_spare);
      }

      current_channels = node.out_channels;



      if (temp_flow != flow) {
        cudaMemcpyAsync(flow, temp_flow, N * current_channels * 64 * sizeof(DataType), cudaMemcpyDeviceToDevice, stream);
        //copyTypeConverted((DataType*)flow, (DataType*)temp_flow, N * current_channels * 64, stream);
      }
      
      if (node.type == TowerNode::TRANSFORMER) {
        node.encoder->Eval(N, flow, (DataType*)scratch, buf1, buf2, cublas, stream, offset_pointers);
      } 
      
      else if (node.type == TowerNode::MOBILENET) { 
        node.cnn_layers[0]->Eval(N, buf0, flow, nullptr, temp, scratch_size, cudnn, cublas, stream, offset_pointers);
        node.cnn_layers[1]->Eval(N, buf1, buf0, nullptr, temp, scratch_size, cudnn, cublas, stream, offset_pointers);
        node.cnn_layers[2]->Eval(N, buf0, buf1, nullptr, temp, scratch_size, cudnn, cublas, stream, offset_pointers);
        node.cnn_layers[3]->Eval(N, flow, buf0, flow, temp, scratch_size, cudnn, cublas, stream, offset_pointers);
      }

      else if (node.type == TowerNode::RESIDUAL) {
        node.cnn_layers[0]->Eval(N, buf0, flow, nullptr, temp, scratch_size, cudnn, cublas, stream, offset_pointers);
        node.cnn_layers[1]->Eval(N, buf1, buf0, nullptr, temp, scratch_size, cudnn, cublas, stream, offset_pointers);
        node.cnn_layers[2]->Eval(N, flow, buf1, flow, temp, scratch_size, cudnn, cublas, stream, offset_pointers);
      }

      else if (node.type == TowerNode::CONVNEXT) {

        node.cnn_layers[0]->Eval(N, buf0, flow, nullptr, temp, scratch_size, cudnn, cublas, stream, offset_pointers);

        LayerNorm<DataType>(N * 64, current_channels, buf1, buf0, nullptr,
                      nullptr, node.convnext_ln1_gammas, node.convnext_ln1_betas, default_epsilon_,
                      alpha_, ACTIVATION_NONE, stream);

        // #FFN dense 1, scratch -> in_out_tensor
        {
        const int batch = N * 64;
        cublasXgemm(cublas, CUBLAS_OP_T, CUBLAS_OP_N, node.dff_channels, batch,
                    current_channels, 1.0f, (const DataType*)node.convnext_ffn_dense1_w, current_channels,
                    buf1, current_channels, 0.0f, buf0, node.dff_channels);
        addBiasBatched(buf0, buf0, node.convnext_ffn_dense1_b, 1, batch,
                        node.dff_channels, activations_.ffn_activation, stream);
        }
        {
        const int batch = N * 64;
        cublasXgemm(cublas, CUBLAS_OP_T, CUBLAS_OP_N, current_channels, batch,
                  node.dff_channels, 1.0f, (const DataType*)node.convnext_ffn_dense2_w, node.dff_channels,
                  buf0, node.dff_channels, 0.0f, buf1, current_channels);

        LayerNorm<DataType>(N * 64, current_channels, flow, buf1,
                            node.convnext_ffn_dense2_b, flow, node.convnext_ln2_gammas, 
                            node.convnext_ln2_betas, default_epsilon_, alpha_, ACTIVATION_NONE, stream);
        }
      }

      // Update channels for the next block
      current_channels = node.out_channels;
  }

  
  if (final_ln_gammas_ != nullptr) {
      const int batch = N * 64; 
      // Read from flow (output), write out-of-place to buf1 (input2)
      LayerNormInPlace<DataType>(batch, current_channels, flow, final_ln_gammas_, final_ln_betas_, 
        default_epsilon_, stream);
  }
  

  /*
  if (final_ln_gammas_ != nullptr) {
      const int batch = N * 64; 
      // Read from flow (output), write out-of-place to buf1 (input2)
      LayerNorm<DataType>(batch, current_channels, buf1, flow, nullptr, (DataType*)nullptr, 
                          final_ln_gammas_, final_ln_betas_, default_epsilon_, 1.0, ACTIVATION_NONE, stream);
      
      // Copy the final safely-normalized result back into the expected output tensor
      cudaMemcpyAsync(output, buf1, N * current_channels * 64 * sizeof(DataType), cudaMemcpyDeviceToDevice, stream);
  }*/
  

}

template <typename DataType>
Backbone<DataType>::~Backbone() {
  if (starts_with_encoder_) {
    if (ip_emb_w_) ReportCUDAErrors(cudaFree(ip_emb_w_));
    if (ip_emb_b_) ReportCUDAErrors(cudaFree(ip_emb_b_));

  
    if (ip_emb_pre_w_) ReportCUDAErrors(cudaFree(ip_emb_pre_w_));
    if (ip_emb_pre_b_) ReportCUDAErrors(cudaFree(ip_emb_pre_b_));
    if (ip_emb_ln_g_) ReportCUDAErrors(cudaFree(ip_emb_ln_g_));
    if (ip_emb_ln_b_) ReportCUDAErrors(cudaFree(ip_emb_ln_b_));
    
    if (ip_emb_ffn_d1_w_) ReportCUDAErrors(cudaFree(ip_emb_ffn_d1_w_));
    if (ip_emb_ffn_d1_b_) ReportCUDAErrors(cudaFree(ip_emb_ffn_d1_b_));
    if (ip_emb_ffn_d2_w_) ReportCUDAErrors(cudaFree(ip_emb_ffn_d2_w_));
    if (ip_emb_ffn_d2_b_) ReportCUDAErrors(cudaFree(ip_emb_ffn_d2_b_));
    
    if (ip_emb_ffn_ln_g_) ReportCUDAErrors(cudaFree(ip_emb_ffn_ln_g_));
    if (ip_emb_ffn_ln_b_) ReportCUDAErrors(cudaFree(ip_emb_ffn_ln_b_));


    if (has_gating_) {
      if (ip_mult_gate_) ReportCUDAErrors(cudaFree(ip_mult_gate_));
      if (ip_add_gate_) ReportCUDAErrors(cudaFree(ip_add_gate_));
    }
    
    if (has_smolgen_ && smolgen_global_) {
      ReportCUDAErrors(cudaFree(smolgen_global_));
    }

  }

  for (auto& node : tower_nodes_) {
      if (node.dense_w) ReportCUDAErrors(cudaFree(node.dense_w));
      if (node.dense_b) ReportCUDAErrors(cudaFree(node.dense_b));
      
      if (node.ln_gammas) {
        ReportCUDAErrors(cudaFree(node.ln_gammas));
        ReportCUDAErrors(cudaFree(node.ln_betas));
      }
      
      if (node.mult_gate) {
        ReportCUDAErrors(cudaFree(node.mult_gate));
        ReportCUDAErrors(cudaFree(node.add_gate));
      }

      if (node.type == TowerNode::CONVNEXT) {
        ReportCUDAErrors(cudaFree(node.convnext_ln1_betas));
        ReportCUDAErrors(cudaFree(node.convnext_ln1_gammas));
        ReportCUDAErrors(cudaFree(node.convnext_ffn_dense1_w));
        ReportCUDAErrors(cudaFree(node.convnext_ffn_dense1_b));
        ReportCUDAErrors(cudaFree(node.convnext_ffn_dense2_w));
        ReportCUDAErrors(cudaFree(node.convnext_ffn_dense2_b));
        ReportCUDAErrors(cudaFree(node.convnext_ln2_betas));
        ReportCUDAErrors(cudaFree(node.convnext_ln2_gammas));
      }
  }

  if (final_ln_gammas_ != nullptr) {
      ReportCUDAErrors(cudaFree(final_ln_gammas_));
      ReportCUDAErrors(cudaFree(final_ln_betas_));
  }

}


template <typename DataType>
AttentionBody<DataType>::AttentionBody(const MultiHeadWeights& weights,
                                       void* scratch, Activations activations,
                                       int num_res_blocks, int input_c,
                                       int max_batch_size,
                                       bool is_pe_dense_embedding,
                                       bool use_gemm_ex, bool fused_mha,
                                       bool nhwc)
    : BaseLayer<DataType>(weights.ip_emb_b.size(), 8, 8, nullptr, false,
                          use_gemm_ex),
      embedding_op_size_(weights.ip_emb_b.size()),
      encoder_head_count_(weights.encoder_head_count),
      activations_(activations),
      num_resi_blocks_(num_res_blocks),
      input_c_(input_c),
      has_gating_(weights.ip_mult_gate.size() > 0 &&
                  weights.ip_add_gate.size() > 0),
      has_smolgen_(weights.has_smolgen),
      is_pe_dense_embedding_(is_pe_dense_embedding),
      use_fused_mha_(fused_mha),
      nhwc_(nhwc) {
  allocAndUpload<DataType>(&ip_emb_w_, weights.ip_emb_w, scratch);
  allocAndUpload<DataType>(&ip_emb_b_, weights.ip_emb_b, scratch);

  if (is_pe_dense_embedding_) {
    allocAndUpload<DataType>(&ip_emb_pre_w_, weights.ip_emb_preproc_w, scratch);
    allocAndUpload<DataType>(&ip_emb_pre_b_, weights.ip_emb_preproc_b, scratch);

    allocAndUpload<DataType>(&ip_emb_ln_g_, weights.ip_emb_ln_gammas, scratch);
    allocAndUpload<DataType>(&ip_emb_ln_b_, weights.ip_emb_ln_betas, scratch);

    allocAndUpload<DataType>(&ip_emb_ffn_d1_w_, weights.ip_emb_ffn.dense1.weights,
                             scratch);
    allocAndUpload<DataType>(&ip_emb_ffn_d1_b_, weights.ip_emb_ffn.dense1.biases,
                             scratch);

    allocAndUpload<DataType>(&ip_emb_ffn_d2_w_, weights.ip_emb_ffn.dense2.weights,
                             scratch);
    allocAndUpload<DataType>(&ip_emb_ffn_d2_b_, weights.ip_emb_ffn.dense2.biases,
                             scratch);

    allocAndUpload<DataType>(&ip_emb_ffn_ln_g_, weights.ip_emb_ffn_ln_gammas,
                             scratch);
    allocAndUpload<DataType>(&ip_emb_ffn_ln_b_, weights.ip_emb_ffn_ln_betas,
                             scratch);

    // 12 is the number of input channels used for the input encoding.
    embedding_dense_size_ = weights.ip_emb_preproc_b.size() / 64;
    embedding_ffn_size_ = weights.ip_emb_ffn.dense2.biases.size();
    embedding_ffn_dff_ = weights.ip_emb_ffn.dense1.biases.size();
  } else {
    size_t size = 64 * kNumPosEncodingChannels * sizeof(float);
    ReportCUDAErrors(cudaMalloc(&pos_encoding_, size));
    ReportCUDAErrors(
        cudaMemcpy(scratch, kPosEncoding, size, cudaMemcpyHostToDevice));
    copyTypeConverted(pos_encoding_, (float*)scratch, size, 0);
  }

  if (has_gating_) {
    allocAndUpload<DataType>(&ip_mult_gate_, weights.ip_mult_gate, scratch);
    allocAndUpload<DataType>(&ip_add_gate_, weights.ip_add_gate, scratch);
  }

  if (has_smolgen_) {
    allocAndUpload<DataType>(&smolgen_global_, weights.smolgen_w, scratch);
    smolgen_global_size_ = 64 * 64;
  }

  int num_encoders = weights.encoder.size();
  float alpha = (float)pow(2.0 * num_encoders, -0.25);
  for (const auto& enc : weights.encoder) {
    EncoderBlock<DataType>* pW = new EncoderBlock<DataType>(
        enc, scratch, encoder_head_count_, embedding_op_size_, alpha,
        smolgen_global_, smolgen_global_size_, max_batch_size,
        activations_.smolgen_activation, activations_.ffn_activation,
        is_pe_dense_embedding_ ? 1e-3 : 1e-6, false, use_gemm_ex, use_fused_mha_);
    encoder_weights_.emplace_back(pW);
  }
}

template <typename DataType>
AttentionBody<DataType>::~AttentionBody() {
  ReportCUDAErrors(cudaFree(ip_emb_w_));
  ReportCUDAErrors(cudaFree(ip_emb_b_));
  if (is_pe_dense_embedding_) {
    ReportCUDAErrors(cudaFree(ip_emb_pre_w_));
    ReportCUDAErrors(cudaFree(ip_emb_pre_b_));
    ReportCUDAErrors(cudaFree(ip_emb_ln_g_));
    ReportCUDAErrors(cudaFree(ip_emb_ln_b_));
    ReportCUDAErrors(cudaFree(ip_emb_ffn_d1_w_));
    ReportCUDAErrors(cudaFree(ip_emb_ffn_d1_b_));
    ReportCUDAErrors(cudaFree(ip_emb_ffn_d2_w_));
    ReportCUDAErrors(cudaFree(ip_emb_ffn_d2_b_));
    ReportCUDAErrors(cudaFree(ip_emb_ffn_ln_g_));
    ReportCUDAErrors(cudaFree(ip_emb_ffn_ln_b_));
  } else {
    ReportCUDAErrors(cudaFree(pos_encoding_));
  }
  if (has_gating_) {
    ReportCUDAErrors(cudaFree(ip_mult_gate_));
    ReportCUDAErrors(cudaFree(ip_add_gate_));
  }
  if (has_smolgen_) {
    ReportCUDAErrors(cudaFree(smolgen_global_));
  }
  for (const auto pEnc : encoder_weights_) delete pEnc;
}

template <typename DataType>
void AttentionBody<DataType>::Eval(int N, DataType* output,
                                   const DataType* input,
                                   const DataType* input2, void* scratch,
                                   size_t scratch_size, cudnnHandle_t /*cudnn*/,
                                   cublasHandle_t cublas, cudaStream_t stream,
                                   DataType*** offset_pointers) {
  DataType* output_tensor = (DataType*)output;
  DataType* buffer1 = (DataType*)input2;
  DataType* buffer2 = buffer1 + scratch_size / (2 * sizeof(DataType));

  int inputC = input_c_;
  if (num_resi_blocks_ == 0) {
    assert(inputC == kInputPlanes);
    /*
      # if there are no residual blocks (pure transformer), do some input
      processing
    */
    if (is_pe_dense_embedding_) {
      // New encoding is made of dense layer fed with input from a 12-channel
      // slice of the input tensor.
      // pos_info = flow[..., :12]
      // pos_info_flat = tf.reshape(pos_info, [-1, 64 * 12])
      // pos_info_processed = tf.keras.layers.Dense(64*self.embedding_dense_sz,
      //                                            name=name+"embedding/preprocess")(pos_info_flat)
      const int num_outputs = 64 * embedding_dense_size_;
      const int num_inputs = 64 * 12;
      const int batch = N;

      convertNCHWtoNHWC((DataType*)scratch, input, N, inputC, N, 12, 8, 8,
                        stream);
      cublasXgemm<DataType>(
          cublas, CUBLAS_OP_T, CUBLAS_OP_N, num_outputs, batch, num_inputs,
          1.0f, (const DataType*)ip_emb_pre_w_, num_inputs,
          (const DataType*)scratch, num_inputs, 0.0f, buffer1, num_outputs);

      // addBiasBatched(buffer1, buffer1, ip_emb_pre_b_, batch, N, num_outputs,
      //               ACTIVATION_NONE, stream);
      const int size = num_outputs * N;
      // @todo addBiasBatched has a 4096 channel limit, needs refactoring.
      addVectors(buffer1, buffer1, ip_emb_pre_b_, size, size, num_outputs,
                 ACTIVATION_NONE, stream);
      inputPreprocessForAttentionBody((DataType*)scratch, input, buffer1, N,
                                      kInputPlanes, embedding_dense_size_, true,
                                      stream);
      inputC += embedding_dense_size_;
    } else {
      /*
      flow = tf.transpose(inputs, perm=[0, 2, 3, 1])
      flow = tf.reshape(flow, [-1, 64, tf.shape(inputs)[1]])
      # add positional encoding for each square to the input
      positional_encoding = tf.broadcast_to(tf.convert_to_tensor(self.POS_ENC,
      dtype=self.model_dtype), [tf.shape(flow)[0], 64,
      tf.shape(self.POS_ENC)[2]]) flow = tf.concat([flow, positional_encoding],
      axis=2)
      */
      inputPreprocessForAttentionBody((DataType*)scratch, input, pos_encoding_,
                                      N, kInputPlanes, kNumPosEncodingChannels,
                                      false, stream);
      inputC += kNumPosEncodingChannels;
    }
  } else if (!nhwc_) {
    // #redirect flow through encoder blocks
    // flow = tf.transpose(flow, perm = [ 0, 2, 3, 1 ])
    // flow = tf.reshape(flow, [ -1, 64, self.RESIDUAL_FILTERS ])
    convertNCHWtoNHWC((DataType*)scratch, input, N, inputC, N, inputC, 8, 8,
                      stream);
  }

  const DataType* attn_in = (num_resi_blocks_ > 0 && nhwc_) ? input : (const DataType*)scratch;

  if (is_pe_dense_embedding_) {
    // 1. square embedding (fully connected layer)
    // Input data in NHWC layout N*(64)*C, output is N*(64)*embedding_op_size_
    DataType* embedding = output_tensor;
    DataType* temp = (DataType*)scratch;
    {
      const int num_outputs = embedding_op_size_;
      const int num_inputs = inputC;
      const int batch = N * 64;
      cublasXgemm<DataType>(cublas, CUBLAS_OP_T, CUBLAS_OP_N, num_outputs,
                            batch, num_inputs, 1.0f, (const DataType*)ip_emb_w_,
                            num_inputs, attn_in, num_inputs, 0.0f, embedding,
                            num_outputs);
      // embedding layer norm with fused in bias add of previous gemm.
      LayerNorm<DataType>(N * 64, embedding_op_size_, temp, embedding,
                          ip_emb_b_, (DataType*)nullptr, ip_emb_ln_g_,
                          ip_emb_ln_b_, 1e-3, 1.0,
                          activations_.default_activation, stream);
    }

    // Input gating
    if (has_gating_) {
      applyInputGating<DataType>(temp, temp, ip_mult_gate_, ip_add_gate_, N, 64,
                                 embedding_op_size_, stream);
    }

    // embedding FFN dense 1
    {
      const int num_inputs = embedding_ffn_size_;
      const int num_outputs = embedding_ffn_dff_;  // encoder_dff
      const int batch = N * 64;
      cublasXgemm(cublas, CUBLAS_OP_T, CUBLAS_OP_N, num_outputs, batch,
                  num_inputs, 1.0f, (const DataType*)ip_emb_ffn_d1_w_,
                  num_inputs, temp, num_inputs, 0.0f, buffer1, num_outputs);
      addBiasBatched(buffer1, buffer1, ip_emb_ffn_d1_b_, 1, batch, num_outputs,
                     activations_.ffn_activation, stream);
    }

    // embedding FFN dense 2
    {
      const int num_inputs = embedding_ffn_dff_;  // encoder_dff
      const int num_outputs = embedding_ffn_size_;
      const int batch = N * 64;
      cublasXgemm(cublas, CUBLAS_OP_T, CUBLAS_OP_N, num_outputs, batch,
                  num_inputs, 1.0f, (const DataType*)ip_emb_ffn_d2_w_,
                  num_inputs, buffer1, num_inputs, 0.0f, buffer2, num_outputs);
      // Embedding LN: skip connection and layer normilization (also bias add of
      // prev gemm) buffer2 -> embedding
      float alpha = (float)pow(2. * encoder_weights_.size(), -0.25);
      LayerNorm<DataType>(N * 64, embedding_ffn_size_, embedding, buffer2,
                          ip_emb_ffn_d2_b_, temp, ip_emb_ffn_ln_g_,
                          ip_emb_ffn_ln_b_, 1e-3, alpha, ACTIVATION_NONE,
                          stream);
    }

  } else {
    // 1. square embedding (fully connected layer)
    // Input data in NHWC layout N*(64)*C, output is N*(64)*embedding_op_size_
    DataType* embedding = output_tensor;
    {
      const int num_outputs = embedding_op_size_;
      const int num_inputs = inputC;
      const int batch = N * 64;
      cublasXgemm<DataType>(cublas, CUBLAS_OP_T, CUBLAS_OP_N, num_outputs,
                            batch, num_inputs, 1.0f, (const DataType*)ip_emb_w_,
                            num_inputs, attn_in, num_inputs, 0.0f,
                            embedding, num_outputs);
      addBiasBatched(embedding, embedding, ip_emb_b_, 1, batch, num_outputs,
                     activations_.default_activation, stream);
    }
    // Input gating
    if (has_gating_) {
      applyInputGating<DataType>(embedding, embedding, ip_mult_gate_,
                                 ip_add_gate_, N, 64, embedding_op_size_,
                                 stream);
    }
  }

  // 2. Encoder blocks
  for (const auto pEnc : encoder_weights_) {
    pEnc->Eval(N, output_tensor, (DataType*)scratch, buffer1, buffer2, cublas,
               stream, offset_pointers);
  }  // End of encoder blocks
}

template <typename DataType>
ValueHead<DataType>::ValueHead(BaseLayer<DataType>* ip,
                               const MultiHeadWeights::ValueHead& weights,
                               void* scratch, bool attention_body, bool wdl,
                               ActivationFunction act, int /*max_batch_size*/,
                               bool use_gemm_ex)
    : BaseLayer<DataType>(weights.ip_val_b.size(), 8, 8, ip),
      embedding_size_(attention_body ? weights.ip_val_b.size()
                                     : weights.value.biases.size()),
      value_hidden_size_(weights.ip1_val_b.size()),
      wdl_(wdl),
      attention_body_(attention_body),
      act_(act) {
  if (attention_body_) {
    allocAndUpload<DataType>(&ip_val_w_, weights.ip_val_w, scratch);
    allocAndUpload<DataType>(&ip_val_b_, weights.ip_val_b, scratch);
  } else {
    conv_ = std::make_unique<Conv1Layer<DataType>>(
        ip, weights.value.biases.size(), 8, 8, ip->GetC(), act, true,
        use_gemm_ex);
    conv_->LoadWeights((float*)&weights.value.weights[0],
                       (float*)&weights.value.biases[0], scratch);
  }

  allocAndUpload<DataType>(&ip1_val_w_, weights.ip1_val_w, scratch);
  allocAndUpload<DataType>(&ip1_val_b_, weights.ip1_val_b, scratch);

  allocAndUpload<DataType>(&ip2_val_w_, weights.ip2_val_w, scratch);
  allocAndUpload<DataType>(&ip2_val_b_, weights.ip2_val_b, scratch);
}

template <typename DataType>
ValueHead<DataType>::~ValueHead() {
  if (attention_body_) {
    ReportCUDAErrors(cudaFree(ip_val_w_));
    ReportCUDAErrors(cudaFree(ip_val_b_));
  }
  ReportCUDAErrors(cudaFree(ip1_val_w_));
  ReportCUDAErrors(cudaFree(ip1_val_b_));
  ReportCUDAErrors(cudaFree(ip2_val_w_));
  ReportCUDAErrors(cudaFree(ip2_val_b_));
}

template <typename DataType>
void ValueHead<DataType>::Eval(int N, DataType* output, const DataType* input,
                               const DataType* input2, void* scratch,
                               size_t scratch_size, cudnnHandle_t /*cudnn*/,
                               cublasHandle_t cublas, cudaStream_t stream,
                               DataType***) {
  DataType* buffer = (DataType*)input2;
  {
    const int num_inputs = this->input_->GetC();
    const int num_outputs = embedding_size_;
    const int batch = N * 64;
    if (attention_body_) {
      cublasXgemm<DataType>(cublas, CUBLAS_OP_T, CUBLAS_OP_N, num_outputs,
                            batch, num_inputs, 1.0f, (const DataType*)ip_val_w_,
                            num_inputs, input, num_inputs, 0.0f, buffer,
                            num_outputs);
      addBiasBatched<DataType>(buffer, buffer, ip_val_b_, 1, batch, num_outputs,
                               act_, stream);

    } else {
      conv_->Eval(N, buffer, input, nullptr, scratch, scratch_size, nullptr,
                  cublas, stream);
    }
  }

  {
    // Value dense 1
    const int num_inputs = embedding_size_ * 64;
    const int num_outputs = value_hidden_size_;
    const int batch = N;
    DataType* layer_out = (DataType*)scratch;
    cublasXgemm<DataType>(cublas, CUBLAS_OP_T, CUBLAS_OP_N, num_outputs, batch,
                          num_inputs, 1.0f, (const DataType*)ip1_val_w_,
                          num_inputs, buffer, num_inputs, 0.0f, layer_out,
                          num_outputs);
    addBiasBatched<DataType>(layer_out, layer_out, ip1_val_b_, 1, batch,
                             num_outputs, act_, stream);
  }

  {
    // Value dense 2
    const int num_inputs = value_hidden_size_;
    const int num_outputs = wdl_ ? 3 : 1;
    const int batch = N;
    DataType* layer_out = (DataType*)output;
    cublasXgemm<DataType>(cublas, CUBLAS_OP_T, CUBLAS_OP_N, num_outputs, batch,
                          num_inputs, 1.0f, (const DataType*)ip2_val_w_,
                          num_inputs, (DataType*)scratch, num_inputs, 0.0f,
                          layer_out, num_outputs);
    addVectors(layer_out, layer_out, ip2_val_b_, num_outputs * batch,
               num_outputs * batch, num_outputs,
               wdl_ ? ACTIVATION_NONE : ACTIVATION_TANH, stream);
  }
}

// Template instantiation.
#ifdef USE_CUDNN
template class ConvLayer<half>;
template class ConvLayer<float>;

template class DepthwiseConvLayer<half>;
template class DepthwiseConvLayer<float>;

template class DepthwiseLegacy<half>;
template class DepthwiseLegacy<float>;
#endif

template class DepthwiseCustom<half>;
template class DepthwiseCustom<float>;


template class FCLayer<half>;
template class FCLayer<float>;

template class SELayer<half>;
template class SELayer<float>;

template class PolicyMapLayer<half>;
template class PolicyMapLayer<float>;

template class FusedWinogradConvSELayer<half>;
template class FusedWinogradConvSELayer<float>;

template class Conv1Layer<half>;
template class Conv1Layer<float>;

template class ResidualBlock<half>;
template class ResidualBlock<float>;

template class AttentionPolicyHead<half>;
template class AttentionPolicyHead<float>;

template class EncoderBlock<half>;
template class EncoderBlock<float>;

template class Backbone<half>;
template class Backbone<float>;

template class LayoutTransformLayer<half>;
template class LayoutTransformLayer<float>;


template class AttentionBody<half>;
template class AttentionBody<float>;

template class EmbeddingLayer<half>;
template class EmbeddingLayer<float>;

template class ValueHead<half>;
template class ValueHead<float>;

// Misc error handling stuff.
#ifdef USE_CUDNN
void CudnnError(cudnnStatus_t status, const char* file, const int& line) {
  if (status != CUDNN_STATUS_SUCCESS) {
    char message[128];
    sprintf(message, "CUDNN error: %s (%s:%d) ", cudnnGetErrorString(status),
            file, line);
    CERR << message;
    throw Exception(message);
  }
}
#endif

const char* CublasGetErrorString(cublasStatus_t status) {
  switch (status) {
    case CUBLAS_STATUS_SUCCESS:
      return "CUBLAS_STATUS_SUCCESS";
    case CUBLAS_STATUS_NOT_INITIALIZED:
      return "CUBLAS_STATUS_NOT_INITIALIZED";
    case CUBLAS_STATUS_ALLOC_FAILED:
      return "CUBLAS_STATUS_ALLOC_FAILED";
    case CUBLAS_STATUS_INVALID_VALUE:
      return "CUBLAS_STATUS_INVALID_VALUE";
    case CUBLAS_STATUS_ARCH_MISMATCH:
      return "CUBLAS_STATUS_ARCH_MISMATCH";
    case CUBLAS_STATUS_MAPPING_ERROR:
      return "CUBLAS_STATUS_MAPPING_ERROR";
    case CUBLAS_STATUS_EXECUTION_FAILED:
      return "CUBLAS_STATUS_EXECUTION_FAILED";
    case CUBLAS_STATUS_INTERNAL_ERROR:
      return "CUBLAS_STATUS_INTERNAL_ERROR";
    case CUBLAS_STATUS_NOT_SUPPORTED:
      return "CUBLAS_STATUS_NOT_SUPPORTED";
    case CUBLAS_STATUS_LICENSE_ERROR:
      return "CUBLAS_STATUS_LICENSE_ERROR";
  }
  return "unknown cublas error";
}

void CublasError(cublasStatus_t status, const char* file, const int& line) {
  if (status != CUBLAS_STATUS_SUCCESS) {
    char message[128];
    sprintf(message, "CUBLAS error: %s (%s:%d) ", CublasGetErrorString(status),
            file, line);
    CERR << message;
    throw Exception(message);
  }
}

void CudaError(cudaError_t status, const char* file, const int& line) {
  if (status != cudaSuccess) {
    char message[128];
    sprintf(message, "CUDA error: %s (%s:%d) ", cudaGetErrorString(status),
            file, line);
    CERR << message;
    throw Exception(message);
  }
}

}  // namespace cudnn_backend
}  // namespace lczero
