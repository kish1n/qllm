#pragma once

// Hard-fail. The engine does not use exceptions: anything that would be an
// error is unrecoverable here (a malformed header, a missing tensor, a
// config that contradicts the weights), so we print one line naming the
// problem and exit rather than unwinding to a handler that could only do
// the same thing.
//
// die() does not return and does not run local destructors -- std::exit
// skips them. That is fine for the resources this code holds: the kernel
// reclaims mappings and descriptors at process teardown. Don't rely on a
// destructor for anything observable outside the process.

#include <format>
#include <string_view>
#include <utility>

namespace qllm {

[[noreturn]] void die_message(std::string_view message);

template <typename... Args> [[noreturn]] void die(std::format_string<Args...> fmt, Args &&...args) {
    die_message(std::format(fmt, std::forward<Args>(args)...));
}

} // namespace qllm
