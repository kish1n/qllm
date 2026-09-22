// Usage: llama [model-dir]
//   Falls back to $QLLM_MODEL_DIR when no argument is given.

#include <cstdlib>
#include <filesystem>
#include <format>
#include <print>
#include <string>
#include <string_view>
#include <utility>

#include "config.h"
#include "safetensors.h"
#include "summary.h"

// Owns the mmap: move-only, because SafeTensors is.
struct LoadedModel {
    qllm::ModelConfig config;
    qllm::SafeTensors weights;
};

LoadedModel load_model(std::string_view model_dir) {
    std::println("1. Reading Inputs");
    std::println("   model dir: {}", model_dir);

    qllm::ModelConfig cfg = qllm::ModelConfig::load(model_dir);
    qllm::print_config(cfg);
    qllm::SafeTensors weights =
        qllm::SafeTensors::open(std::filesystem::path(model_dir) / "model.safetensors");
    qllm::print_safetensors(weights, cfg);

    // Flat [vocab_size, hidden_size] in row-major order, so the first
    // hidden_size elements are token 0's embedding. Widening four of them
    // one at a time -- proof the mapping and the BF16 shift are wired up.
    // bf16() is a span into the mmap; nothing here copies the matrix.
    const auto token_embeddings = weights.at("model.embed_tokens.weight").bf16();
    std::println();
    std::println("   embed_tokens row 0, first 4: {:.6f} {:.6f} {:.6f} {:.6f}",
                 qllm::bf16_to_f32(token_embeddings[0]), qllm::bf16_to_f32(token_embeddings[1]),
                 qllm::bf16_to_f32(token_embeddings[2]), qllm::bf16_to_f32(token_embeddings[3]));
    return LoadedModel{std::move(cfg), std::move(weights)};
}

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

    LoadedModel model = load_model(model_dir);

    return 0;
}
