#include "safetensors.h"

#include <bit>
#include <fcntl.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>

#include <cerrno>
#include <cstring>
#include <system_error>
#include <utility>

#include "die.h"
#include "json.h"

static_assert(std::endian::native == std::endian::little,
              "safetensors is little-endian; a byte-swapping path would be needed here");

namespace qllm {
namespace {

// A header larger than this is corruption, not a real weights file
// (Llama 3.2 1B's is 16 KB). Bounds the allocation before we trust the
// length.
constexpr std::uint64_t kMaxHeaderBytes = 256ull << 20;

[[noreturn]] void fail(const std::filesystem::path &path, std::string_view message) {
    die("safetensors: {}: {}", path.string(), message);
}

DType parse_dtype(std::string_view name, const std::filesystem::path &path,
                  std::string_view tensor) {
    if (name == "BF16")
        return DType::BF16;
    if (name == "F16")
        return DType::F16;
    if (name == "F32")
        return DType::F32;
    if (name == "F64")
        return DType::F64;
    if (name == "I8")
        return DType::I8;
    if (name == "U8")
        return DType::U8;
    if (name == "I16")
        return DType::I16;
    if (name == "I32")
        return DType::I32;
    if (name == "I64")
        return DType::I64;
    if (name == "BOOL")
        return DType::Bool;
    fail(path, std::format("tensor '{}' has unsupported dtype '{}'", tensor, name));
}

} // namespace

std::size_t dtype_size(DType dtype) {
    switch (dtype) {
    case DType::BF16:
    case DType::F16:
    case DType::I16:
        return 2;
    case DType::F32:
    case DType::I32:
        return 4;
    case DType::F64:
    case DType::I64:
        return 8;
    case DType::I8:
    case DType::U8:
    case DType::Bool:
        return 1;
    }
    return 0;
}

std::string_view dtype_name(DType dtype) {
    switch (dtype) {
    case DType::BF16:
        return "BF16";
    case DType::F16:
        return "F16";
    case DType::F32:
        return "F32";
    case DType::F64:
        return "F64";
    case DType::I8:
        return "I8";
    case DType::U8:
        return "U8";
    case DType::I16:
        return "I16";
    case DType::I32:
        return "I32";
    case DType::I64:
        return "I64";
    case DType::Bool:
        return "BOOL";
    }
    return "?";
}

std::int64_t TensorView::numel() const {
    std::int64_t n = 1;
    for (const std::int64_t d : shape) {
        n *= d;
    }
    return n;
}

std::span<const std::uint16_t> TensorView::bf16() const {
    if (dtype != DType::BF16) {
        die("safetensors: tensor '{}' is {}, not BF16", name, dtype_name(dtype));
    }
    return {reinterpret_cast<const std::uint16_t *>(bytes.data()), bytes.size() / 2};
}

std::vector<float> TensorView::to_f32() const {
    const std::size_t n = static_cast<std::size_t>(numel());
    std::vector<float> out(n);
    switch (dtype) {
    case DType::BF16: {
        const auto src = bf16();
        for (std::size_t i = 0; i < n; ++i) {
            out[i] = bf16_to_f32(src[i]);
        }
        return out;
    }
    case DType::F32:
        std::memcpy(out.data(), bytes.data(), n * sizeof(float));
        return out;
    default:
        die("safetensors: to_f32() does not handle {} (tensor '{}')", dtype_name(dtype), name);
    }
}

SafeTensors SafeTensors::open(const std::filesystem::path &path) {
    const int fd = ::open(path.c_str(), O_RDONLY | O_CLOEXEC);
    if (fd < 0) {
        fail(path, std::format("open failed: {}",
                               std::error_code(errno, std::generic_category()).message()));
    }

    struct stat st{};
    if (::fstat(fd, &st) != 0) {
        const int saved = errno;
        ::close(fd);
        fail(path, std::format("fstat failed: {}",
                               std::error_code(saved, std::generic_category()).message()));
    }
    const auto file_size = static_cast<std::size_t>(st.st_size);
    if (file_size < 8) {
        ::close(fd);
        fail(path, "file is shorter than the 8-byte header length");
    }

    void *mapping = ::mmap(nullptr, file_size, PROT_READ, MAP_PRIVATE, fd, 0);
    const int mapping_error = errno;
    // The mapping keeps the pages alive on its own; the descriptor is not
    // needed past this point.
    ::close(fd);
    if (mapping == MAP_FAILED) {
        fail(path, std::format("mmap failed: {}",
                               std::error_code(mapping_error, std::generic_category()).message()));
    }

    // No unmap-on-failure path below: fail() exits the process, and the
    // kernel drops the mapping then.
    const auto *base = static_cast<const std::byte *>(mapping);

    std::uint64_t header_len = 0;
    std::memcpy(&header_len, base, sizeof header_len);
    if (header_len > kMaxHeaderBytes) {
        fail(path, std::format("header length {} is implausibly large", header_len));
    }
    if (header_len > file_size - 8) {
        fail(path, std::format("header length {} overruns the file", header_len));
    }

    const std::byte *data_base = base + 8 + header_len;
    const std::size_t data_size = file_size - 8 - static_cast<std::size_t>(header_len);

    const json::Value header = json::parse(
        {reinterpret_cast<const char *>(base + 8), static_cast<std::size_t>(header_len)},
        std::format("safetensors: {}: header", path.string()));
    if (!header.is_object()) {
        fail(path, "header is not a JSON object");
    }

    SafeTensors out;
    for (const json::Member &member : header.object) {
        const std::string &name = member.key;
        const json::Value &entry = member.value;
        if (name == "__metadata__") {
            if (entry.is_object()) {
                for (const json::Member &kv : entry.object) {
                    out.metadata_.emplace_back(kv.key, kv.value.is_string() ? kv.value.string
                                                                            : std::string{});
                }
            }
            continue;
        }

        TensorView view;
        view.name = name;
        {
            const json::Value *dtype = entry.find("dtype");
            const json::Value *shape = entry.find("shape");
            const json::Value *offsets = entry.find("data_offsets");
            if (dtype == nullptr || shape == nullptr || offsets == nullptr) {
                fail(path, std::format("tensor '{}' is missing dtype/shape/data_offsets", name));
            }

            view.dtype = parse_dtype(dtype->as_string(name + ".dtype"), path, name);

            for (const json::Value &dim : shape->as_array(name + ".shape")) {
                const std::int64_t d = dim.as_int64(name + ".shape[]");
                if (d < 0) {
                    fail(path, std::format("tensor '{}' has a negative dimension", name));
                }
                view.shape.push_back(d);
            }

            const auto &pair = offsets->as_array(name + ".data_offsets");
            if (pair.size() != 2) {
                fail(path,
                     std::format("tensor '{}' has {} data_offsets, expected 2", name, pair.size()));
            }
            const std::int64_t begin = pair[0].as_int64(name + ".data_offsets[0]");
            const std::int64_t end = pair[1].as_int64(name + ".data_offsets[1]");
            if (begin < 0 || end < begin || static_cast<std::size_t>(end) > data_size) {
                fail(path,
                     std::format("tensor '{}' offsets [{}, {}) do not fit the {}-byte data section",
                                 name, begin, end, data_size));
            }

            const auto span_bytes = static_cast<std::size_t>(end - begin);
            const auto expected = static_cast<std::size_t>(view.numel()) * dtype_size(view.dtype);
            if (span_bytes != expected) {
                fail(path, std::format("tensor '{}' spans {} bytes but its shape and dtype need {}",
                                       name, span_bytes, expected));
            }
            view.bytes = {data_base + begin, span_bytes};
        }

        const auto [it, inserted] = out.index_.emplace(view.name, out.tensors_.size());
        if (!inserted) {
            fail(path, std::format("duplicate tensor name '{}'", view.name));
        }
        out.tensors_.push_back(std::move(view));
    }

    out.path_ = path;
    out.base_ = base;
    out.size_ = file_size;
    return out;
}

void SafeTensors::reset() noexcept {
    if (base_ != nullptr) {
        ::munmap(const_cast<std::byte *>(base_), size_);
    }
    base_ = nullptr;
    size_ = 0;
    tensors_.clear();
    index_.clear();
    metadata_.clear();
}

SafeTensors::~SafeTensors() { reset(); }

SafeTensors::SafeTensors(SafeTensors &&other) noexcept
    : path_(std::move(other.path_)), base_(std::exchange(other.base_, nullptr)),
      size_(std::exchange(other.size_, 0)), tensors_(std::move(other.tensors_)),
      index_(std::move(other.index_)), metadata_(std::move(other.metadata_)) {}

SafeTensors &SafeTensors::operator=(SafeTensors &&other) noexcept {
    if (this != &other) {
        reset();
        path_ = std::move(other.path_);
        base_ = std::exchange(other.base_, nullptr);
        size_ = std::exchange(other.size_, 0);
        tensors_ = std::move(other.tensors_);
        index_ = std::move(other.index_);
        metadata_ = std::move(other.metadata_);
    }
    return *this;
}

const TensorView *SafeTensors::find(std::string_view name) const {
    const auto it = index_.find(name);
    return it == index_.end() ? nullptr : &tensors_[it->second];
}

const TensorView &SafeTensors::at(std::string_view name) const {
    if (const TensorView *v = find(name)) {
        return *v;
    }
    die("safetensors: no tensor named '{}' in {}", name, path_.string());
}

} // namespace qllm
