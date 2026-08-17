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
*/

#pragma once

#include <unordered_map>
#include <vector>

#include "proto/net.pb.h"
#include <variant>
#include <optional>

namespace lczero {

struct BaseWeights {
  explicit BaseWeights(const pblczero::Weights& weights);

  using Vec = std::vector<float>;
  struct ConvBlock {
    explicit ConvBlock(const pblczero::Weights::ConvBlock& block);

    Vec weights;
    Vec biases;
    Vec bn_gammas;
    Vec bn_betas;
    Vec bn_means;
    Vec bn_stddivs;
  };

  struct DepthwiseConvBlock {
    explicit DepthwiseConvBlock(const pblczero::Weights::DepthwiseConvBlock& block);

    Vec weights;
    Vec biases;
    Vec bn_gammas;
    Vec bn_betas;
    Vec bn_means;
    Vec bn_stddivs;
    int rook_channels;
    int bishop_channels;
    int knight_channels;
  };

  struct SEunit {
    explicit SEunit(const pblczero::Weights::SEunit& se);
    Vec w1;
    Vec b1;
    Vec w2;
    Vec b2;
  };

  struct Residual {
    explicit Residual(const pblczero::Weights::Residual& residual);
    ConvBlock conv1;
    ConvBlock conv2;
    SEunit se;
    bool has_se;
  };

  struct MobileNet {
    explicit MobileNet(const pblczero::Weights::MobileNet& block);
    ConvBlock conv1;
    DepthwiseConvBlock d_conv;
    ConvBlock conv2;
    bool has_se;
    SEunit se;
  };

  struct Smolgen {
    explicit Smolgen(const pblczero::Weights::Smolgen& smolgen);
    Vec compress;
    Vec dense1_w;
    Vec dense1_b;
    Vec ln1_gammas;
    Vec ln1_betas;
    Vec dense2_w;
    Vec dense2_b;
    Vec ln2_gammas;
    Vec ln2_betas;
  };

  struct MHA {
    explicit MHA(const pblczero::Weights::MHA& mha);
    Vec q_w;
    Vec q_b;
    Vec k_w;
    Vec k_b;
    Vec v_w;
    Vec v_b;
    Vec dense_w;
    Vec dense_b;
    Smolgen smolgen;
    bool has_smolgen;
    Vec static_b;
    bool has_static_b;
    Vec rpe_q;
    Vec rpe_k;
    bool has_rpe_q;
    bool has_rpe_k;
  };

  struct FFN {
    explicit FFN(const pblczero::Weights::FFN& ffn);
    ConvBlock dense1;
    ConvBlock dense2;
    DepthwiseConvBlock d_conv;
    //Vec ln_betas;
    //Vec ln_gammas;
  };

  struct EncoderLayer {
    explicit EncoderLayer(const pblczero::Weights::EncoderLayer& encoder);
    MHA mha;
    Vec ln1_gammas;
    Vec ln1_betas;
    FFN ffn;
    Vec ln2_gammas;
    Vec ln2_betas;
  };

  struct ConvNext {
    explicit ConvNext(const pblczero::Weights::ConvNext& block);
    DepthwiseConvBlock d_conv;
    Vec ln1_gammas;
    Vec ln1_betas;
    FFN ffn;
    Vec ln2_gammas;
    Vec ln2_betas;
  };


  struct TowerBlock {
    explicit TowerBlock(const pblczero::Weights::TowerBlock& pb_block);
    Vec dense_w;
    Vec dense_b;
    Vec mult_gate;
    Vec add_gate;
    Vec ln_gammas;
    Vec ln_betas;
    ConvBlock cnn_cnn;
    ConvBlock enc_cnn;

    std::variant<std::monostate, MobileNet, Residual, EncoderLayer, ConvNext> block;
    
  };

  // Input convnet.
  ConvBlock input;

  // Embedding preprocess layer.
  Vec ip_emb_preproc_w;
  Vec ip_emb_preproc_b;

  // Embedding layer
  Vec ip_emb_w;
  Vec ip_emb_b;

  // Embedding layernorm
  // @todo can this be folded into weights?
  Vec ip_emb_ln_gammas;
  Vec ip_emb_ln_betas;


  // Input gating
  Vec ip_mult_gate;
  Vec ip_add_gate;

  // Embedding feedforward network
  FFN ip_emb_ffn;
  Vec ip_emb_ffn_ln_gammas;
  Vec ip_emb_ffn_ln_betas;

  std::vector<TowerBlock> tower;
  // CNN -> Encoder
  Vec final_ln_gammas;
  Vec final_ln_betas;

  // Encoder stack.
  std::vector<EncoderLayer> encoder;
  int encoder_head_count;
  float epsilon;

  // Residual tower.
  std::vector<Residual> residual;

  // Moves left head
  ConvBlock moves_left;
  Vec ip_mov_w;
  Vec ip_mov_b;
  Vec ip1_mov_w;
  Vec ip1_mov_b;
  Vec ip2_mov_w;
  Vec ip2_mov_b;

  // Smolgen global weights
  Vec smolgen_w;
  bool has_smolgen;
};

struct LegacyWeights : public BaseWeights {
  explicit LegacyWeights(const pblczero::Weights& weights);

  // Policy head
  // Extra convolution for AZ-style policy head
  ConvBlock policy1;
  ConvBlock policy;
  Vec ip_pol_w;
  Vec ip_pol_b;
  // Extra params for attention policy head
  Vec ip2_pol_w;
  Vec ip2_pol_b;
  Vec ip3_pol_w;
  Vec ip3_pol_b;
  Vec ip4_pol_w;
  int pol_encoder_head_count;
  std::vector<EncoderLayer> pol_encoder;

  // Value head
  ConvBlock value;
  Vec ip_val_w;
  Vec ip_val_b;
  Vec ip1_val_w;
  Vec ip1_val_b;
  Vec ip2_val_w;
  Vec ip2_val_b;
};

struct MultiHeadWeights : public BaseWeights {
  explicit MultiHeadWeights(const pblczero::Weights& weights);

  struct PolicyHead {
    explicit PolicyHead(const pblczero::Weights::PolicyHead& policyhead, Vec& w,
                        Vec& b);
    // Policy head
   private:
    // Storage in case _ip_pol_w/b are not shared among heads.
    Vec _ip_pol_w;
    Vec _ip_pol_b;

   public:
    // Reference to possibly shared value (to avoid unnecessary copies).
    Vec& ip_pol_w;
    Vec& ip_pol_b;
    // Extra convolution for AZ-style policy head
    ConvBlock policy1;
    ConvBlock policy;
    // Extra params for attention policy head
    Vec ip2_pol_w;
    Vec ip2_pol_b;
    Vec ip3_pol_w;
    Vec ip3_pol_b;
    Vec ip4_pol_w;
    int pol_encoder_head_count;
    std::vector<EncoderLayer> pol_encoder;
  };

  struct ValueHead {
    explicit ValueHead(const pblczero::Weights::ValueHead& valuehead);
    // Value head
    ConvBlock value;
    Vec ip_val_w;
    Vec ip_val_b;
    Vec ip1_val_w;
    Vec ip1_val_b;
    Vec ip2_val_w;
    Vec ip2_val_b;
    Vec ip_val_err_w;
    Vec ip_val_err_b;
  };

 private:
  Vec ip_pol_w;
  Vec ip_pol_b;

 public:
  // Policy and value multiheads
  std::unordered_map<std::string, ValueHead> value_heads;
  std::unordered_map<std::string, PolicyHead> policy_heads;
};

enum InputEmbedding {
  INPUT_EMBEDDING_NONE = 0,
  INPUT_EMBEDDING_PE_MAP = 1,
  INPUT_EMBEDDING_PE_DENSE = 2,
};

}  // namespace lczero