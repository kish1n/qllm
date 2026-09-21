# Loader + (later) kernels, built as a library so the tests can link the
# same code the binary runs.
add_library(qllm_core STATIC
    die.cxx
    json.cxx
    safetensors.cxx
    config.cxx
)
target_include_directories(qllm_core PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})

add_executable(llama main.cxx)
target_link_libraries(llama PRIVATE qllm_core)

add_subdirectory(tests)
