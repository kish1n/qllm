set(QLLM_GOLDEN_DIR "${CMAKE_SOURCE_DIR}/reference/golden")
configure_file(golden_dir_config.h.in golden_dir_config.h @ONLY)

add_executable(compare_selftest compare_selftest.cxx)
target_include_directories(compare_selftest PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}
    ${CMAKE_CURRENT_BINARY_DIR}
)

add_test(NAME compare_selftest COMMAND compare_selftest)
