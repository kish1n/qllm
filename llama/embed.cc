#include "embed.h"

#include <cstddef>

#include "die.h"

namespace qllm {

std::vector<float> embed(const SafeTensors &weights, const ModelConfig &cfg,
                         std::span<const std::int32_t> token_ids) {
    const TensorView &table = weights.at("model.embed_tokens.weight");
    if (table.shape.size() != 2) {
        die("embed: model.embed_tokens.weight has rank {}, expected 2", table.shape.size());
    }
    if (table.shape[0] != cfg.vocab_size || table.shape[1] != cfg.hidden_size) {
        die("embed: model.embed_tokens.weight is [{}, {}] but config says [{}, {}]", table.shape[0],
            table.shape[1], cfg.vocab_size, cfg.hidden_size);
    }

    // Span into the mmap; the gather below is the only copy.
    const std::span<const std::uint16_t> src = table.bf16();
    const std::size_t hidden = static_cast<std::size_t>(cfg.hidden_size);

    std::vector<float> out(token_ids.size() * hidden);
    for (std::size_t t = 0; t < token_ids.size(); ++t) {
        const std::int32_t id = token_ids[t];
        if (id < 0 || id >= cfg.vocab_size) {
            die("embed: token id {} out of range [0, {})", id, cfg.vocab_size);
        }
        const std::uint16_t *row = src.data() + static_cast<std::size_t>(id) * hidden;
        float *dst = out.data() + t * hidden;
        for (std::size_t i = 0; i < hidden; ++i) {
            dst[i] = bf16_to_f32(row[i]);
        }
    }
    return out;
}

} // namespace qllm
