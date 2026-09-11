// Proves compare() actually discriminates matching vs. mismatching data:
// 1. Read a golden tensor back into memory and diff it against itself ->
//    must report zero error and OK.
// 2. Perturb one element far past tolerance and diff again -> must report
//    nonzero error and FAIL.
//
// Done-when check for Step 0: `ninja && ./llama/testing/compare_selftest`
// prints PASS once reference/golden/ has been produced by dump_golden.py.

#include <cstdio>
#include <fstream>
#include <vector>

#include "compare.h"
#include "golden_dir_config.h"

namespace {

std::vector<float> read_golden(const std::string& name, bool& ok) {
    const std::string path = std::string(QLLM_GOLDEN_DIR) + "/" + name + ".bin";
    std::ifstream f(path, std::ios::binary | std::ios::ate);
    if (!f) {
        std::fprintf(stderr,
                      "compare_selftest: could not open %s\n"
                      "  Run: reference/.venv/bin/python reference/dump_golden.py "
                      "--model <path-or-repo-id> --out-dir reference/golden\n",
                      path.c_str());
        ok = false;
        return {};
    }
    const std::streamsize bytes = f.tellg();
    f.seekg(0);
    std::vector<float> data(static_cast<std::size_t>(bytes) / sizeof(float));
    f.read(reinterpret_cast<char*>(data.data()), bytes);
    ok = static_cast<bool>(f);
    return data;
}

}  // namespace

int main() {
    using qllm::testing::compare;

    bool loaded = false;
    std::vector<float> data = read_golden("embed_output", loaded);
    if (!loaded || data.empty()) {
        std::printf("selftest: FAIL (could not load golden tensor)\n");
        return 1;
    }

    const auto identical = compare("embed_output", data.data(), data.size(), 1e-3, 1e-2,
                                    QLLM_GOLDEN_DIR);
    const bool identical_as_expected = identical.ok && identical.max_abs == 0.0 && identical.max_rel == 0.0;

    data[0] += 1.0f;  // well past both abs_tol and rel_tol
    const auto perturbed = compare("embed_output", data.data(), data.size(), 1e-3, 1e-2,
                                    QLLM_GOLDEN_DIR);
    const bool perturbed_as_expected = !perturbed.ok && perturbed.max_abs >= 1.0 - 1e-6;

    const bool pass = identical_as_expected && perturbed_as_expected;
    std::printf("selftest: %s\n", pass ? "PASS" : "FAIL");
    return pass ? 0 : 1;
}
