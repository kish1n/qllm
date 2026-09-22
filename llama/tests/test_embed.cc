// Checks the token embedding gather against reference/golden/embed_output.bin.
//
// Expects a bit-exact match: BF16 -> F32 widening is lossless and the
// reference runs the same lookup in fp32, so any nonzero error means a
// wrong row, a wrong stride, or a bad mapping -- never rounding. Hence
// the zero tolerances below.
//
// Needs the weights, so it skips (exit 77) when QLLM_MODEL_DIR is unset.

#include <cstdint>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <print>
#include <sstream>
#include <string>
#include <vector>

#include "compare.h"
#include "config.h"
#include "die.h"
#include "embed.h"
#include "golden_dir_config.h"
#include "json.h"
#include "safetensors.h"

namespace {

constexpr int kSkip = 77; // matches SKIP_RETURN_CODE in CMakeLists.txt

// The prompt's token ids, read from the manifest rather than hardcoded so
// the test follows whatever prompt dump_golden.py was last run with.
std::vector<std::int32_t> golden_token_ids() {
    const std::filesystem::path path = std::filesystem::path(QLLM_GOLDEN_DIR) / "manifest.json";
    std::ifstream f(path, std::ios::binary);
    if (!f) {
        qllm::die("test_embed: could not open {}", path.string());
    }
    std::ostringstream buf;
    buf << f.rdbuf();

    const qllm::json::Value root = qllm::json::parse(buf.str(), path.string());
    const qllm::json::Value *ids = root.find("token_ids");
    if (ids == nullptr) {
        qllm::die("test_embed: {} has no token_ids", path.string());
    }

    std::vector<std::int32_t> out;
    for (const qllm::json::Value &v : ids->as_array("token_ids")) {
        out.push_back(static_cast<std::int32_t>(v.as_int64("token_ids[]")));
    }
    return out;
}

} // namespace

int main() {
    const char *model_dir = std::getenv("QLLM_MODEL_DIR");
    if (model_dir == nullptr) {
        std::println("[skip] test_embed: set QLLM_MODEL_DIR to a directory with "
                     "config.json and model.safetensors");
        return kSkip;
    }

    const std::vector<std::int32_t> token_ids = golden_token_ids();
    const qllm::ModelConfig cfg = qllm::ModelConfig::load(model_dir);
    const qllm::SafeTensors weights =
        qllm::SafeTensors::open(std::filesystem::path(model_dir) / "model.safetensors");

    const std::vector<float> out = qllm::embed(weights, cfg, token_ids);
    std::println("[info ] {} tokens x {} hidden = {} floats", token_ids.size(), cfg.hidden_size,
                 out.size());

    const auto result = qllm::testing::compare("embed_output", out.data(), out.size(),
                                               /*abs_tol=*/0.0, /*rel_tol=*/0.0, QLLM_GOLDEN_DIR);
    return result.ok ? 0 : 1;
}
