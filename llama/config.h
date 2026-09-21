#pragma once

// The architectural numbers the engine needs, read from a model
// directory's config.json. Mirrors extract_config_summary() in reference/dump_golden.py
// so the C++ side and the golden dump agree on where each value comes from
// -- nothing here is hardcoded to Llama 3.2.

#include <cstdint>
#include <filesystem>
#include <optional>
#include <string>

namespace qllm {

// Llama-3-style RoPE frequency rescaling. Absent for models that use plain
// RoPE, in which case rope_scaling on ModelConfig is nullopt.
struct RopeScaling {
    std::string rope_type;
    double factor = 1.0;
    double low_freq_factor = 1.0;
    double high_freq_factor = 1.0;
    std::int64_t original_max_position_embeddings = 0;
};

struct ModelConfig {
    std::string model_type;
    std::int64_t hidden_size = 0;
    std::int64_t num_hidden_layers = 0;
    std::int64_t num_attention_heads = 0;
    std::int64_t num_key_value_heads = 0;
    std::int64_t head_dim = 0; // defaults to hidden_size / num_attention_heads
    std::int64_t intermediate_size = 0;
    std::int64_t vocab_size = 0;
    double rms_norm_eps = 1e-5;
    // When true there is no lm_head.weight: the output projection reuses
    // model.embed_tokens.weight. True for Llama 3.2 1B.
    bool tie_word_embeddings = false;
    double rope_theta = 10000.0;
    std::optional<RopeScaling> rope_scaling;

    // Grouped-query attention: how many query heads share each KV head.
    std::int64_t heads_per_kv_group() const { return num_attention_heads / num_key_value_heads; }

    // Reads <dir>/config.json. die()s on missing or inconsistent fields.
    static ModelConfig load(const std::filesystem::path &model_dir);
};

} // namespace qllm
