#pragma once

// Reusable golden-tensor diff helper. Every later step (RMSNorm, RoPE,
// attention, ...) reuses compare() to check a computed buffer against the
// float32 dump produced by reference/dump_golden.py.

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <string>
#include <vector>

namespace qllm::testing {

struct CompareResult {
    std::string name;
    std::size_t n = 0;
    double max_abs = 0.0;
    double max_rel = 0.0;
    bool ok = false;  // false also on a missing/short golden file
};

// Loads "<golden_dir>/<name>.bin" (n contiguous float32s, as written by
// dump_golden.py) and diffs it elementwise against ptr[0..n). Prints one
// report line and returns the measured error.
//
// golden_dir resolution when left empty: QLLM_GOLDEN_DIR env var, else
// "reference/golden" relative to the current working directory.
//
// Pass/fail uses numpy's allclose rule per element:
//   |a - b| <= abs_tol + rel_tol * |b|
inline CompareResult compare(const std::string& name, const float* ptr, std::size_t n,
                              double abs_tol = 1e-3, double rel_tol = 1e-2,
                              std::string golden_dir = "") {
    CompareResult result;
    result.name = name;
    result.n = n;

    if (golden_dir.empty()) {
        if (const char* env = std::getenv("QLLM_GOLDEN_DIR")) {
            golden_dir = env;
        } else {
            golden_dir = "reference/golden";
        }
    }

    const std::string path = golden_dir + "/" + name + ".bin";
    std::ifstream f(path, std::ios::binary | std::ios::ate);
    if (!f) {
        std::printf("[compare] %-28s FAIL  could not open %s\n", name.c_str(), path.c_str());
        return result;
    }

    const std::streamsize bytes = f.tellg();
    const std::streamsize expected = static_cast<std::streamsize>(n * sizeof(float));
    if (bytes != expected) {
        std::printf("[compare] %-28s FAIL  size mismatch: golden has %lld bytes, expected %lld\n",
                    name.c_str(), static_cast<long long>(bytes), static_cast<long long>(expected));
        return result;
    }
    f.seekg(0);
    std::vector<float> golden(n);
    f.read(reinterpret_cast<char*>(golden.data()), bytes);
    if (!f) {
        std::printf("[compare] %-28s FAIL  short read from %s\n", name.c_str(), path.c_str());
        return result;
    }

    double max_abs = 0.0;
    double max_rel = 0.0;
    bool all_close = true;
    constexpr double kRelEps = 1e-12;  // avoids inflating max_rel near-zero golden values
    for (std::size_t i = 0; i < n; ++i) {
        const double a = static_cast<double>(ptr[i]);
        const double b = static_cast<double>(golden[i]);
        const double abs_diff = std::fabs(a - b);
        const double rel_diff = abs_diff / (std::fabs(b) + kRelEps);
        max_abs = std::max(max_abs, abs_diff);
        max_rel = std::max(max_rel, rel_diff);
        if (abs_diff > abs_tol + rel_tol * std::fabs(b)) {
            all_close = false;
        }
    }

    result.max_abs = max_abs;
    result.max_rel = max_rel;
    result.ok = all_close;

    std::printf("[compare] %-28s n=%-8zu max_abs=%.3e max_rel=%.3e  %s\n", name.c_str(), n, max_abs,
                max_rel, result.ok ? "OK" : "FAIL");
    return result;
}

}  // namespace qllm::testing
