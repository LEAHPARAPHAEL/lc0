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

#include "neural/network_legacy.h"

#include <algorithm>
#include <cmath>
#include <utility>

#include "utils/exception.h"
#include "utils/weights_adapter.h"

namespace lczero {
namespace {
static constexpr float kEpsilon = 1e-5f;
}  // namespace

BaseWeights::BaseWeights(const pblczero::Weights& weights)
    : input(weights.input()),
      ip_emb_preproc_w(LayerAdapter(weights.ip_emb_preproc_w()).as_vector()),
      ip_emb_preproc_b(LayerAdapter(weights.ip_emb_preproc_b()).as_vector()),
      ip_emb_w(LayerAdapter(weights.ip_emb_w()).as_vector()),
      ip_emb_b(LayerAdapter(weights.ip_emb_b()).as_vector()),
      ip_emb_ln_gammas(LayerAdapter(weights.ip_emb_ln_gammas()).as_vector()),
      ip_emb_ln_betas(LayerAdapter(weights.ip_emb_ln_betas()).as_vector()),
      ip_emb_expand(weights.ip_emb_expand()),
      mask_type(weights.mask_type()),
      ip_mult_gate(LayerAdapter(weights.ip_mult_gate()).as_vector()),
      ip_add_gate(LayerAdapter(weights.ip_add_gate()).as_vector()),
      ip_emb_ffn(weights.ip_emb_ffn()),
      ip_emb_ffn_ln_gammas(
          LayerAdapter(weights.ip_emb_ffn_ln_gammas()).as_vector()),
      ip_emb_ffn_ln_betas(
          LayerAdapter(weights.ip_emb_ffn_ln_betas()).as_vector()),
      moves_left(weights.moves_left()),
      ip_mov_w(LayerAdapter(weights.ip_mov_w()).as_vector()),
      ip_mov_b(LayerAdapter(weights.ip_mov_b()).as_vector()),
      ip1_mov_w(LayerAdapter(weights.ip1_mov_w()).as_vector()),
      ip1_mov_b(LayerAdapter(weights.ip1_mov_b()).as_vector()),
      ip2_mov_w(LayerAdapter(weights.ip2_mov_w()).as_vector()),
      ip2_mov_b(LayerAdapter(weights.ip2_mov_b()).as_vector()),
      smolgen_w(LayerAdapter(weights.smolgen_w()).as_vector()),
      has_smolgen(weights.has_smolgen_w()){
  for (const auto& res : weights.residual()) {
    residual.emplace_back(res);
  }
  encoder_head_count = weights.headcount();
  for (const auto& enc : weights.encoder()) {
    encoder.emplace_back(enc);
  }
  for (const auto& block : weights.ip_emb_mobilenet_tower()) {
    ip_emb_mobilenet_tower.emplace_back(block, this->mask_type);
  }
  for (const auto& block : weights.ip_emb_residual_tower()) {
    ip_emb_residual_tower.emplace_back(block);
  }
}

BaseWeights::SEunit::SEunit(const pblczero::Weights::SEunit& se)
    : w1(LayerAdapter(se.w1()).as_vector()),
      b1(LayerAdapter(se.b1()).as_vector()),
      w2(LayerAdapter(se.w2()).as_vector()),
      b2(LayerAdapter(se.b2()).as_vector()) {}

BaseWeights::Residual::Residual(const pblczero::Weights::Residual& residual)
    : conv1(residual.conv1()),
      conv2(residual.conv2()),
      se(residual.se()),
      has_se(residual.has_se()) {}

BaseWeights::MobileNet::MobileNet(const pblczero::Weights::MobileNet& block, 
                                  const std::string& mask_type)
    : conv1(block.conv1()),
      d_conv(block.d_conv(), mask_type),
      conv2(block.conv2()),
      has_se(block.has_se()),
      se(block.se()) {}

BaseWeights::ConvBlock::ConvBlock(const pblczero::Weights::ConvBlock& block)
    : weights(LayerAdapter(block.weights()).as_vector()),
      biases(LayerAdapter(block.biases()).as_vector()),
      bn_gammas(LayerAdapter(block.bn_gammas()).as_vector()),
      bn_betas(LayerAdapter(block.bn_betas()).as_vector()),
      bn_means(LayerAdapter(block.bn_means()).as_vector()),
      bn_stddivs(LayerAdapter(block.bn_stddivs()).as_vector()) {
  
  if (weights.size() == 0) return;

  if (bn_betas.size() == 0) {
    for (auto i = size_t{0}; i < bn_means.size(); i++) {
      bn_betas.emplace_back(0.0f);
      bn_gammas.emplace_back(1.0f);
    }
  }
  if (biases.size() == 0) {
    for (auto i = size_t{0}; i < bn_means.size(); i++) {
      biases.emplace_back(0.0f);
    }
  }

  if (bn_means.size() == 0) return;

  for (auto i = size_t{0}; i < bn_stddivs.size(); i++) {
    bn_gammas[i] *= 1.0f / std::sqrt(bn_stddivs[i] + kEpsilon);
    bn_means[i] -= biases[i];
  }

  auto outputs = biases.size();
  auto inputs = weights.size() / outputs;

  for (auto o = size_t{0}; o < outputs; o++) {
    for (auto c = size_t{0}; c < inputs; c++) {
      weights[o * inputs + c] *= bn_gammas[o];
    }
    biases[o] = -bn_gammas[o] * bn_means[o] + bn_betas[o];
  }

  bn_stddivs.clear();
  bn_means.clear();
  bn_betas.clear();
  bn_gammas.clear();
}


BaseWeights::DepthwiseConvBlock::DepthwiseConvBlock(const pblczero::Weights::DepthwiseConvBlock& block, 
                                  const std::string& mask_type)
    : weights(LayerAdapter(block.weights()).as_vector()),
      biases(LayerAdapter(block.biases()).as_vector()),
      bn_gammas(LayerAdapter(block.bn_gammas()).as_vector()),
      bn_betas(LayerAdapter(block.bn_betas()).as_vector()),
      bn_means(LayerAdapter(block.bn_means()).as_vector()),
      bn_stddivs(LayerAdapter(block.bn_stddivs()).as_vector()) {
  
  if (weights.size() == 0) return;

  if (bn_betas.size() == 0) {
    for (auto i = size_t{0}; i < bn_means.size(); i++) {
      bn_betas.emplace_back(0.0f);
      bn_gammas.emplace_back(1.0f);
    }
  }
  if (biases.size() == 0) {
    for (auto i = size_t{0}; i < bn_means.size(); i++) {
      biases.emplace_back(0.0f);
    }
  }

  if (bn_means.size() == 0) return;

  for (auto i = size_t{0}; i < bn_stddivs.size(); i++) {
    bn_gammas[i] *= 1.0f / std::sqrt(bn_stddivs[i] + kEpsilon);
    bn_means[i] -= biases[i];
  }

  auto outputs = biases.size();
  auto inputs = weights.size() / outputs;

  for (auto o = size_t{0}; o < outputs; o++) {
    for (auto c = size_t{0}; c < inputs; c++) {
      weights[o * inputs + c] *= bn_gammas[o];
    }
    biases[o] = -bn_gammas[o] * bn_means[o] + bn_betas[o];
  }

  /*
  if (mask_type == "rbk" || mask_type == "rb") {
    if (inputs != 25) {
      throw Exception("Mask type requires a 5x5 depthwise convolution kernel.");
    }

    std::vector<float> packed_weights;
    packed_weights.reserve(outputs * 10); 

    const int rook_idx[9]   = {2, 7, 10, 11, 12, 13, 14, 17, 22};
    const int bishop_idx[9] = {0, 4, 6, 8, 12, 16, 18, 20, 24};
    const int knight_idx[9] = {1, 3, 5, 9, 12, 15, 19, 21, 23};

    for (auto o = size_t{0}; o < outputs; o++) {
      const int* active_mask = nullptr;

      if (mask_type == "rbk") {
        if (o < outputs / 3) active_mask = rook_idx;
        else if (o < 2 * outputs / 3) active_mask = bishop_idx;
        else active_mask = knight_idx;
      } else { // "rb"
        if (o < outputs / 2) active_mask = rook_idx;
        else active_mask = bishop_idx;
      }

      for (int i = 0; i < 9; i++) {
        packed_weights.push_back(weights[o * inputs + active_mask[i]]);
      }
      
      packed_weights.push_back(biases[o]);
    }

    weights = std::move(packed_weights);
    
    biases.clear(); 
  }
  */

  bn_stddivs.clear();
  bn_means.clear();
  bn_betas.clear();
  bn_gammas.clear();
}

BaseWeights::MHA::MHA(const pblczero::Weights::MHA& mha)
    : q_w(LayerAdapter(mha.q_w()).as_vector()),
      q_b(LayerAdapter(mha.q_b()).as_vector()),
      k_w(LayerAdapter(mha.k_w()).as_vector()),
      k_b(LayerAdapter(mha.k_b()).as_vector()),
      v_w(LayerAdapter(mha.v_w()).as_vector()),
      v_b(LayerAdapter(mha.v_b()).as_vector()),
      dense_w(LayerAdapter(mha.dense_w()).as_vector()),
      dense_b(LayerAdapter(mha.dense_b()).as_vector()),
      smolgen(Smolgen(mha.smolgen())),
      has_smolgen(mha.has_smolgen()) {
      if (mha.has_rpe_q() || mha.has_rpe_k() || mha.has_rpe_v()) {
        throw Exception("RPE weights file not supported.");
      }
    }
      
    /*
    ,
      rpe_q(mha.has_rpe_q() ? LayerAdapter(mha.rpe_q()).as_vector() : Vec()),
      rpe_k(mha.has_rpe_k() ? LayerAdapter(mha.rpe_k()).as_vector() : Vec()),
      rpe_v(mha.has_rpe_v() ? LayerAdapter(mha.rpe_v()).as_vector() : Vec()),
      has_rpe_q(mha.has_rpe_q()),
      has_rpe_k(mha.has_rpe_k()),
      has_rpe_v(mha.has_rpe_v()) {}
    */





BaseWeights::FFN::FFN(const pblczero::Weights::FFN& ffn)
    : dense1_w(LayerAdapter(ffn.dense1_w()).as_vector()),
      dense1_b(LayerAdapter(ffn.dense1_b()).as_vector()),
      dense2_w(LayerAdapter(ffn.dense2_w()).as_vector()),
      dense2_b(LayerAdapter(ffn.dense2_b()).as_vector()) {}

BaseWeights::EncoderLayer::EncoderLayer(
    const pblczero::Weights::EncoderLayer& encoder)
    : mha(MHA(encoder.mha())),
      ln1_gammas(LayerAdapter(encoder.ln1_gammas()).as_vector()),
      ln1_betas(LayerAdapter(encoder.ln1_betas()).as_vector()),
      ffn(FFN(encoder.ffn())),
      ln2_gammas(LayerAdapter(encoder.ln2_gammas()).as_vector()),
      ln2_betas(LayerAdapter(encoder.ln2_betas()).as_vector()) {}

BaseWeights::Smolgen::Smolgen(const pblczero::Weights::Smolgen& smolgen)
    : compress(LayerAdapter(smolgen.compress()).as_vector()),
      dense1_w(LayerAdapter(smolgen.dense1_w()).as_vector()),
      dense1_b(LayerAdapter(smolgen.dense1_b()).as_vector()),
      ln1_gammas(LayerAdapter(smolgen.ln1_gammas()).as_vector()),
      ln1_betas(LayerAdapter(smolgen.ln1_betas()).as_vector()),
      dense2_w(LayerAdapter(smolgen.dense2_w()).as_vector()),
      dense2_b(LayerAdapter(smolgen.dense2_b()).as_vector()),
      ln2_gammas(LayerAdapter(smolgen.ln2_gammas()).as_vector()),
      ln2_betas(LayerAdapter(smolgen.ln2_betas()).as_vector()) {}

MultiHeadWeights::PolicyHead::PolicyHead(
    const pblczero::Weights::PolicyHead& policyhead, Vec& w, Vec& b)
    : _ip_pol_w(LayerAdapter(policyhead.ip_pol_w()).as_vector()),
      _ip_pol_b(LayerAdapter(policyhead.ip_pol_b()).as_vector()),
      ip_pol_w(_ip_pol_w.empty() ? w : _ip_pol_w),
      ip_pol_b(_ip_pol_b.empty() ? b : _ip_pol_b),
      policy1(policyhead.policy1()),
      policy(policyhead.policy()),
      ip2_pol_w(LayerAdapter(policyhead.ip2_pol_w()).as_vector()),
      ip2_pol_b(LayerAdapter(policyhead.ip2_pol_b()).as_vector()),
      ip3_pol_w(LayerAdapter(policyhead.ip3_pol_w()).as_vector()),
      ip3_pol_b(LayerAdapter(policyhead.ip3_pol_b()).as_vector()),
      ip4_pol_w(LayerAdapter(policyhead.ip4_pol_w()).as_vector()) {
  pol_encoder_head_count = policyhead.pol_headcount();
  for (const auto& enc : policyhead.pol_encoder()) {
    pol_encoder.emplace_back(enc);
  }
}

MultiHeadWeights::ValueHead::ValueHead(
    const pblczero::Weights::ValueHead& valuehead)
    : value(valuehead.value()),
      ip_val_w(LayerAdapter(valuehead.ip_val_w()).as_vector()),
      ip_val_b(LayerAdapter(valuehead.ip_val_b()).as_vector()),
      ip1_val_w(LayerAdapter(valuehead.ip1_val_w()).as_vector()),
      ip1_val_b(LayerAdapter(valuehead.ip1_val_b()).as_vector()),
      ip2_val_w(LayerAdapter(valuehead.ip2_val_w()).as_vector()),
      ip2_val_b(LayerAdapter(valuehead.ip2_val_b()).as_vector()),
      ip_val_err_w(LayerAdapter(valuehead.ip_val_err_w()).as_vector()),
      ip_val_err_b(LayerAdapter(valuehead.ip_val_err_b()).as_vector()) {}

LegacyWeights::LegacyWeights(const pblczero::Weights& weights)
    : BaseWeights(weights),
      policy1(weights.policy1()),
      policy(weights.policy()),
      ip_pol_w(LayerAdapter(weights.ip_pol_w()).as_vector()),
      ip_pol_b(LayerAdapter(weights.ip_pol_b()).as_vector()),
      ip2_pol_w(LayerAdapter(weights.ip2_pol_w()).as_vector()),
      ip2_pol_b(LayerAdapter(weights.ip2_pol_b()).as_vector()),
      ip3_pol_w(LayerAdapter(weights.ip3_pol_w()).as_vector()),
      ip3_pol_b(LayerAdapter(weights.ip3_pol_b()).as_vector()),
      ip4_pol_w(LayerAdapter(weights.ip4_pol_w()).as_vector()),
      value(weights.value()),
      ip_val_w(LayerAdapter(weights.ip_val_w()).as_vector()),
      ip_val_b(LayerAdapter(weights.ip_val_b()).as_vector()),
      ip1_val_w(LayerAdapter(weights.ip1_val_w()).as_vector()),
      ip1_val_b(LayerAdapter(weights.ip1_val_b()).as_vector()),
      ip2_val_w(LayerAdapter(weights.ip2_val_w()).as_vector()),
      ip2_val_b(LayerAdapter(weights.ip2_val_b()).as_vector()) {
  pol_encoder_head_count = weights.pol_headcount();
  for (const auto& enc : weights.pol_encoder()) {
    pol_encoder.emplace_back(enc);
  }
}

MultiHeadWeights::MultiHeadWeights(const pblczero::Weights& weights)
    : BaseWeights(weights),
      ip_pol_w(LayerAdapter(weights.policy_heads().has_ip_pol_w()
                                ? weights.policy_heads().ip_pol_w()
                                : weights.ip_pol_w())
                   .as_vector()),
      ip_pol_b(LayerAdapter(weights.policy_heads().has_ip_pol_b()
                                ? weights.policy_heads().ip_pol_b()
                                : weights.ip_pol_b())
                   .as_vector()) {
  policy_heads.emplace(std::piecewise_construct,
                       std::forward_as_tuple("vanilla"),
                       std::forward_as_tuple(weights.policy_heads().vanilla(),
                                             ip_pol_w, ip_pol_b));
  if (weights.has_policy_heads()) {
    if (weights.policy_heads().has_optimistic_st()) {
      policy_heads.emplace(
          std::piecewise_construct, std::forward_as_tuple("optimistic"),
          std::forward_as_tuple(weights.policy_heads().optimistic_st(),
                                ip_pol_w, ip_pol_b));
    }
    if (weights.policy_heads().has_soft()) {
      policy_heads.emplace(std::piecewise_construct,
                           std::forward_as_tuple("soft"),
                           std::forward_as_tuple(weights.policy_heads().soft(),
                                                 ip_pol_w, ip_pol_b));
    }
    if (weights.policy_heads().has_opponent()) {
      policy_heads.emplace(
          std::piecewise_construct, std::forward_as_tuple("opponent"),
          std::forward_as_tuple(weights.policy_heads().opponent(), ip_pol_w,
                                ip_pol_b));
    }
  } else {
    if (weights.has_policy() || weights.has_policy1() ||
        weights.has_ip_pol_w()) {
      auto& vanilla = policy_heads.at("vanilla");
      vanilla.policy1 = ConvBlock(weights.policy1());
      vanilla.policy = ConvBlock(weights.policy());
      vanilla.ip2_pol_w = LayerAdapter(weights.ip2_pol_w()).as_vector();
      vanilla.ip2_pol_b = LayerAdapter(weights.ip2_pol_b()).as_vector();
      vanilla.ip3_pol_w = LayerAdapter(weights.ip3_pol_w()).as_vector();
      vanilla.ip3_pol_b = LayerAdapter(weights.ip3_pol_b()).as_vector();
      vanilla.ip4_pol_w = LayerAdapter(weights.ip4_pol_w()).as_vector();
      vanilla.pol_encoder_head_count = weights.pol_headcount();
      for (const auto& enc : weights.pol_encoder()) {
        vanilla.pol_encoder.emplace_back(enc);
      }
    } else {
      throw Exception("Could not find valid policy head weights.");
    }
  }

  value_heads.emplace("winner", weights.value_heads().winner());
  if (weights.has_value_heads()) {
    if (weights.value_heads().has_q()) {
      value_heads.emplace("q", weights.value_heads().q());
    }
    if (weights.value_heads().has_st()) {
      value_heads.emplace("st", weights.value_heads().st());
    }
  } else {
    if (weights.has_value() || weights.has_ip_val_w()) {
      auto& winner = value_heads.at("winner");
      winner.value = ConvBlock(weights.value());
      winner.ip_val_w = LayerAdapter(weights.ip_val_w()).as_vector();
      winner.ip_val_b = LayerAdapter(weights.ip_val_b()).as_vector();
      winner.ip1_val_w = LayerAdapter(weights.ip1_val_w()).as_vector();
      winner.ip1_val_b = LayerAdapter(weights.ip1_val_b()).as_vector();
      winner.ip2_val_w = LayerAdapter(weights.ip2_val_w()).as_vector();
      winner.ip2_val_b = LayerAdapter(weights.ip2_val_b()).as_vector();
    } else {
      throw Exception("Could not find valid value head weights.");
    }
  }
}

}  // namespace lczero
