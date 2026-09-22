#include "config.h"

#include <fstream>
#include <sstream>

#include "die.h"
#include "json.h"

namespace qllm {
namespace {

[[noreturn]] void fail(const std::filesystem::path &path, std::string_view message) {
    die("config: {}: {}", path.string(), message);
}

const json::Value &require(const json::Value &root, std::string_view key,
                           const std::filesystem::path &path) {
    if (const json::Value *v = root.find(key)) {
        return *v;
    }
    fail(path, std::format("missing required field '{}'", key));
}

} // namespace

ModelConfig ModelConfig::load(const std::filesystem::path &model_dir) {
    const std::filesystem::path path = model_dir / "config.json";
    std::ifstream f(path, std::ios::binary);
    if (!f) {
        fail(path, "could not open");
    }
    std::ostringstream buf;
    buf << f.rdbuf();
    const std::string text = buf.str();

    const json::Value root = json::parse(text, std::format("config: {}", path.string()));
    if (!root.is_object()) {
        fail(path, "top level is not a JSON object");
    }

    ModelConfig cfg;
    {
        if (const json::Value *v = root.find("model_type")) {
            cfg.model_type = v->as_string("model_type");
        }
        cfg.hidden_size = require(root, "hidden_size", path).as_int64("hidden_size");
        cfg.num_hidden_layers =
            require(root, "num_hidden_layers", path).as_int64("num_hidden_layers");
        cfg.num_attention_heads =
            require(root, "num_attention_heads", path).as_int64("num_attention_heads");
        cfg.intermediate_size =
            require(root, "intermediate_size", path).as_int64("intermediate_size");
        cfg.vocab_size = require(root, "vocab_size", path).as_int64("vocab_size");

        cfg.num_key_value_heads = cfg.num_attention_heads;
        if (const json::Value *v = root.find("num_key_value_heads")) {
            cfg.num_key_value_heads = v->as_int64("num_key_value_heads");
        }

        // head_dim is explicit in newer configs and derived in older ones.
        // They agree for Llama 3.2 1B (2048 / 32 = 64) but need not in
        // general, so prefer the stored value whenever it is present.
        if (const json::Value *v = root.find("head_dim"); v != nullptr && !v->is_null()) {
            cfg.head_dim = v->as_int64("head_dim");
        } else {
            if (cfg.num_attention_heads == 0) {
                fail(path, "num_attention_heads is zero and head_dim is absent");
            }
            cfg.head_dim = cfg.hidden_size / cfg.num_attention_heads;
        }

        if (const json::Value *v = root.find("rms_norm_eps")) {
            cfg.rms_norm_eps = v->as_double("rms_norm_eps");
        }
        if (const json::Value *v = root.find("tie_word_embeddings")) {
            cfg.tie_word_embeddings = v->as_bool("tie_word_embeddings");
        }

        // Newer transformers folds rope_theta into rope_scaling/rope_parameters;
        // older ones keep it top-level. Check the nested spelling first, to
        // match dump_golden.py.
        const json::Value *scaling = root.find("rope_scaling");
        if (scaling == nullptr || scaling->is_null()) {
            scaling = root.find("rope_parameters");
        }
        bool have_theta = false;
        if (scaling != nullptr && scaling->is_object()) {
            if (const json::Value *v = scaling->find("rope_theta"); v != nullptr && !v->is_null()) {
                cfg.rope_theta = v->as_double("rope_scaling.rope_theta");
                have_theta = true;
            }
        }
        if (!have_theta) {
            if (const json::Value *v = root.find("rope_theta"); v != nullptr && !v->is_null()) {
                cfg.rope_theta = v->as_double("rope_theta");
            }
        }

        if (scaling != nullptr && scaling->is_object() && !scaling->object.empty()) {
            RopeScaling rs;
            if (const json::Value *v = scaling->find("rope_type")) {
                rs.rope_type = v->as_string("rope_scaling.rope_type");
            }
            if (const json::Value *v = scaling->find("factor")) {
                rs.factor = v->as_double("rope_scaling.factor");
            }
            if (const json::Value *v = scaling->find("low_freq_factor")) {
                rs.low_freq_factor = v->as_double("rope_scaling.low_freq_factor");
            }
            if (const json::Value *v = scaling->find("high_freq_factor")) {
                rs.high_freq_factor = v->as_double("rope_scaling.high_freq_factor");
            }
            if (const json::Value *v = scaling->find("original_max_position_embeddings")) {
                rs.original_max_position_embeddings =
                    v->as_int64("rope_scaling.original_max_position_embeddings");
            }
            cfg.rope_scaling = std::move(rs);
        }
    }

    if (cfg.num_key_value_heads == 0 || cfg.num_attention_heads % cfg.num_key_value_heads != 0) {
        fail(path,
             std::format("num_attention_heads ({}) is not a multiple of num_key_value_heads ({})",
                         cfg.num_attention_heads, cfg.num_key_value_heads));
    }

    return cfg;
}

} // namespace qllm
