#pragma once

// Token embedding lookup: the first step of the forward pass.
//
// model.embed_tokens.weight is [vocab_size, hidden_size] in row-major
// order, so an embedding is a contiguous row and the "lookup" is a gather
// plus the BF16 widening. No arithmetic beyond that shift -- if this
// disagrees with the reference dump, the bug is in the loader or in the
// layout assumption, not in a kernel.

#include <cstdint>
#include <span>
#include <vector>

#include "config.h"
#include "safetensors.h"

namespace qllm {

// Returns [token_ids.size(), hidden_size] float32, row-major, laid out to
// match the reference dump. die()s on a shape that contradicts the config
// or on an out-of-range token id.
std::vector<float> embed(const SafeTensors &weights, const ModelConfig &cfg,
                         std::span<const std::int32_t> token_ids);

} // namespace qllm
