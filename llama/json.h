#pragma once

// Minimal recursive-descent JSON parser.
//
// Exists so the engine can read model.safetensors' header and config.json
// without pulling in a dependency. Deliberately small: no comments, no
// trailing commas, no streaming -- the inputs are a 16 KB tensor header and
// a 1 KB config, both machine-generated and well-formed.
//
// Value is a plain struct rather than a std::variant to sidestep the
// incomplete-type rules a self-referential variant runs into. The extra
// per-node bytes are irrelevant at these sizes. Object members go through a
// forward-declared Member for the same reason: std::vector is the one
// container the standard lets you instantiate with an incomplete element
// type, and std::pair<std::string, Value> would not qualify.

#include <cstdint>
#include <string>
#include <string_view>
#include <vector>

namespace qllm::json {

struct Member;

struct Value {
    enum class Kind { Null, Bool, Number, String, Array, Object };

    Kind kind = Kind::Null;
    bool boolean = false;
    double number = 0.0;
    std::string string;
    std::vector<Value> array;
    std::vector<Member> object; // insertion-ordered

    bool is_null() const { return kind == Kind::Null; }
    bool is_bool() const { return kind == Kind::Bool; }
    bool is_number() const { return kind == Kind::Number; }
    bool is_string() const { return kind == Kind::String; }
    bool is_array() const { return kind == Kind::Array; }
    bool is_object() const { return kind == Kind::Object; }

    // Object lookup. Returns nullptr when this isn't an object or the key is
    // absent, so callers can treat "missing" and "wrong shape" uniformly.
    const Value *find(std::string_view key) const;

    // Typed accessors. Each die()s naming `context` on a type mismatch;
    // as_int64 additionally rejects non-integral and out-of-range numbers.
    bool as_bool(std::string_view context) const;
    double as_double(std::string_view context) const;
    std::int64_t as_int64(std::string_view context) const;
    const std::string &as_string(std::string_view context) const;
    const std::vector<Value> &as_array(std::string_view context) const;
    const std::vector<Member> &as_object(std::string_view context) const;
};

struct Member {
    std::string key;
    Value value;
};

// Parses `text` as a single JSON document. die()s on malformed input, on
// nesting deeper than 64, or on trailing non-whitespace. `origin` names the
// source in that message.
Value parse(std::string_view text, std::string_view origin);

} // namespace qllm::json
