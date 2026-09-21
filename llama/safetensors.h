#pragma once

// Zero-copy reader for a .safetensors file.
//
// The file is mmap'd read-only and every TensorView is a span into that
// mapping -- opening a 2.3 GiB weights file costs a header parse and nothing
// else, and pages fault in only as kernels touch them. Weights stay in
// their stored dtype (BF16 for Llama 3.2); widening to float happens at the
// point of use, not at load.
//
// Layout:
//   [0, 8)              uint64 LE header length N
//   [8, 8 + N)          UTF-8 JSON header
//   [8 + N, file_end)   tensor data; data_offsets in the header are
//                       relative to this base
//
// Every accessor validates, so a truncated or hand-edited file dies with a
// named error rather than faulting deep inside a matmul.

#include <bit>
#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <span>
#include <string>
#include <string_view>
#include <unordered_map>
#include <utility>
#include <vector>

namespace qllm {

enum class DType { BF16, F16, F32, F64, I8, U8, I16, I32, I64, Bool };

// Bytes per element. BF16/F16 are 2, as stored.
std::size_t dtype_size(DType dtype);
std::string_view dtype_name(DType dtype);

// BF16 is the upper 16 bits of an IEEE float32, so widening is a shift --
// no table, no branch, exact for finite values and preserves NaN/Inf.
constexpr float bf16_to_f32(std::uint16_t v) noexcept {
    const std::uint32_t bits = static_cast<std::uint32_t>(v) << 16;
    return std::bit_cast<float>(bits);
}

struct TensorView {
    std::string name;
    DType dtype = DType::BF16;
    std::vector<std::int64_t> shape;
    std::span<const std::byte> bytes; // into the mmap; valid while SafeTensors lives

    std::int64_t numel() const;

    // Raw BF16 elements. die()s unless dtype is BF16.
    std::span<const std::uint16_t> bf16() const;

    // Materializes the whole tensor as float32.
    //
    // This is a debugging scaffold, not the inference path: it copies, and
    // for embed_tokens that is a 1 GB allocation. Kernels should widen in
    // registers from bf16() instead. Useful for diffing a single tensor
    // against reference/golden while bringing a step up.
    std::vector<float> to_f32() const;
};

class SafeTensors {
  public:
    // Opens and validates the file. die()s on a bad header, an unknown
    // dtype, or offsets that don't fit the file.
    static SafeTensors open(const std::filesystem::path &path);

    SafeTensors() = default;
    ~SafeTensors();
    SafeTensors(SafeTensors &&) noexcept;
    SafeTensors &operator=(SafeTensors &&) noexcept;
    SafeTensors(const SafeTensors &) = delete;
    SafeTensors &operator=(const SafeTensors &) = delete;

    // Lookup by tensor name. at() dies naming the missing key; find()
    // returns nullptr.
    const TensorView &at(std::string_view name) const;
    const TensorView *find(std::string_view name) const;
    bool contains(std::string_view name) const { return find(name) != nullptr; }

    // Tensors in header order.
    const std::vector<TensorView> &tensors() const { return tensors_; }
    std::size_t size() const { return tensors_.size(); }

    // The optional __metadata__ object, as raw key/value strings.
    const std::vector<std::pair<std::string, std::string>> &metadata() const { return metadata_; }

    std::size_t file_size() const { return size_; }
    const std::filesystem::path &path() const { return path_; }

  private:
    void reset() noexcept;

    std::filesystem::path path_;
    const std::byte *base_ = nullptr; // start of the mapping
    std::size_t size_ = 0;
    std::vector<TensorView> tensors_;
    // Heterogeneous so find(string_view) doesn't allocate. Keys are owned
    // copies rather than views into tensors_, which would dangle on a move
    // of a short (SSO) name.
    struct TransparentHash {
        using is_transparent = void;
        std::size_t operator()(std::string_view s) const {
            return std::hash<std::string_view>{}(s);
        }
    };
    std::unordered_map<std::string, std::size_t, TransparentHash, std::equal_to<>> index_;
    std::vector<std::pair<std::string, std::string>> metadata_;
};

} // namespace qllm
