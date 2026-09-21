#include "die.h"

#include <cstdio>
#include <cstdlib>
#include <print>

namespace qllm {

void die_message(std::string_view message) {
    std::fflush(stdout); // keep partial progress output ahead of the error
    std::println(stderr, "qllm: {}", message);
    std::exit(EXIT_FAILURE);
}

} // namespace qllm
