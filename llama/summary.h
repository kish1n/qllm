#pragma once

// Human-readable dump of what was just loaded: the parsed config and the
// safetensors header. Kept out of config.cc and safetensors.cc so those
// stay free of I/O, and out of main.cc so the startup path reads as a
// list of steps rather than a wall of format strings.

#include "config.h"
#include "safetensors.h"

namespace qllm {

// Both write to stdout, indented to sit under the "1. Reading Inputs"
// heading.

void print_config(const ModelConfig &cfg);

// File size, tensor count, parameter count, __metadata__, and the
// per-tensor dtype/shape for the entries layer 0 needs. Takes cfg only to
// decide whether to mention the tied lm_head.
void print_safetensors(const SafeTensors &weights, const ModelConfig &cfg);

} // namespace qllm
