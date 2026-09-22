// Step 1: read the model.
//
// Opens a Hugging Face model directory, parses config.json and the
// safetensors header, and reports what it found. Nothing is dequantized and
// no tensor data is copied -- the weights stay BF16 in the mapping, and a
// later step widens them in-register inside the kernels.
//
// Usage: llama [model-dir]
//   Falls back to $QLLM_MODEL_DIR when no argument is given.

#include <cstdio>
#include <cstdlib>
#include <filesystem>
#include <format>
#include <print>
#include <string>

#include "config.h"
#include "safetensors.h"

namespace {

std::string human_bytes(std::size_t n) {
    constexpr double kMiB = 1024.0 * 1024.0;
    if (n >= static_cast<std::size_t>(kMiB * 1024)) {
        return std::format("{:.2f} GiB", static_cast<double>(n) / (kMiB * 1024.0));
    }
    return std::format("{:.2f} MiB", static_cast<double>(n) / kMiB);
}

} // namespace

int main(int argc, char **argv) {
    std::string model_dir;
    if (argc > 1) {
        model_dir = argv[1];
    } else if (const char *env = std::getenv("QLLM_MODEL_DIR")) {
        model_dir = env;
    } else {
        std::println(stderr,
                     "usage: {} <model-dir>\n"
                     "  or set QLLM_MODEL_DIR to a directory containing\n"
                     "  config.json and model.safetensors",
                     argv[0]);
        return 2;
    }

    std::println("1. Reading Inputs");
    std::println("   model dir: {}", model_dir);

    const qllm::ModelConfig cfg = qllm::ModelConfig::load(model_dir);
    std::println();
    std::println("   config.json");
    std::println("     model_type          {}", cfg.model_type);
    std::println("     hidden_size         {}", cfg.hidden_size);
    std::println("     num_hidden_layers   {}", cfg.num_hidden_layers);
    std::println("     attention heads     {} q / {} kv  ({}:1 GQA)", cfg.num_attention_heads,
                 cfg.num_key_value_heads, cfg.heads_per_kv_group());
    std::println("     head_dim            {}", cfg.head_dim);
    std::println("     intermediate_size   {}", cfg.intermediate_size);
    std::println("     vocab_size          {}", cfg.vocab_size);
    std::println("     rms_norm_eps        {:g}", cfg.rms_norm_eps);
    std::println("     rope_theta          {:g}", cfg.rope_theta);
    std::println("     tie_word_embeddings {}", cfg.tie_word_embeddings);
    if (cfg.rope_scaling) {
        const auto &rs = *cfg.rope_scaling;
        std::println("     rope_scaling        type={} factor={:g} low={:g} high={:g} "
                     "orig_max_pos={}",
                     rs.rope_type, rs.factor, rs.low_freq_factor, rs.high_freq_factor,
                     rs.original_max_position_embeddings);
    }

    const qllm::SafeTensors weights =
        qllm::SafeTensors::open(std::filesystem::path(model_dir) / "model.safetensors");

    std::size_t total_bytes = 0;
    std::int64_t total_params = 0;
    for (const auto &t : weights.tensors()) {
        total_bytes += t.bytes.size();
        total_params += t.numel();
    }

    std::println();
    std::println("   model.safetensors");
    std::println("     file size           {}", human_bytes(weights.file_size()));
    std::println("     tensors             {}", weights.size());
    std::println("     parameters          {} ({} of weight data)", total_params,
                 human_bytes(total_bytes));
    for (const auto &[k, v] : weights.metadata()) {
        std::println("     metadata            {} = {}", k, v);
    }

    // Spot-check the tensors layer 0 needs, by the names Hugging Face's
    // Llama export uses. at() dies if the naming assumption is wrong,
    // which is the point -- better here than inside a kernel.
    std::println();
    std::println("   layer 0 tensors");
    for (const char *suffix :
         {"input_layernorm.weight", "self_attn.q_proj.weight", "self_attn.k_proj.weight",
          "self_attn.v_proj.weight", "self_attn.o_proj.weight", "post_attention_layernorm.weight",
          "mlp.gate_proj.weight", "mlp.up_proj.weight", "mlp.down_proj.weight"}) {
        const auto &t = weights.at(std::format("model.layers.0.{}", suffix));
        std::println("     {:<34} {:<5} {}", suffix, qllm::dtype_name(t.dtype), t.shape);
    }

    const auto &embed = weights.at("model.embed_tokens.weight");
    std::println();
    std::println("   model.embed_tokens.weight          {:<5} {}", qllm::dtype_name(embed.dtype),
                 embed.shape);
    std::println("   model.norm.weight                  {:<5} {}",
                 qllm::dtype_name(weights.at("model.norm.weight").dtype),
                 weights.at("model.norm.weight").shape);
    if (cfg.tie_word_embeddings) {
        std::println("   lm_head                            tied to embed_tokens (present in "
                     "file: {})",
                     weights.contains("lm_head.weight"));
    }

    // First four values of the embedding row for token 0, widened one at
    // a time -- proof the mapping and the BF16 shift are wired up.
    const auto rows = embed.bf16();
    std::println();
    std::println("   embed_tokens row 0, first 4: {:.6f} {:.6f} {:.6f} {:.6f}",
                 qllm::bf16_to_f32(rows[0]), qllm::bf16_to_f32(rows[1]), qllm::bf16_to_f32(rows[2]),
                 qllm::bf16_to_f32(rows[3]));

    return 0;
}
