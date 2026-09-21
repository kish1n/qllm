# No test targets right now. compare.h (the golden-tensor diff helper) and
# the generated golden_dir_config.h stay here so a later step can diff a
# computed buffer against reference/golden without re-deriving paths.
set(QLLM_GOLDEN_DIR "${CMAKE_SOURCE_DIR}/reference/golden")
configure_file(golden_dir_config.h.in golden_dir_config.h @ONLY)
