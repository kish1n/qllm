#include "json.h"

#include <charconv>
#include <cmath>
#include <system_error>

#include "die.h"

namespace qllm::json {
namespace {

constexpr int kMaxDepth = 64;

class Parser {
  public:
    Parser(std::string_view text, std::string_view origin) : text_(text), origin_(origin) {}

    Value parse_document() {
        skip_whitespace();
        Value v = parse_value(0);
        skip_whitespace();
        if (pos_ != text_.size()) {
            fail("trailing content after top-level value");
        }
        return v;
    }

  private:
    std::string_view text_;
    std::string_view origin_;
    std::size_t pos_ = 0;

    [[noreturn]] void fail(std::string_view message) const {
        die("{}: {} at offset {}", origin_, message, pos_);
    }

    bool eof() const { return pos_ >= text_.size(); }
    char peek() const {
        if (eof()) {
            fail("unexpected end of input");
        }
        return text_[pos_];
    }

    void skip_whitespace() {
        while (pos_ < text_.size()) {
            const char c = text_[pos_];
            if (c == ' ' || c == '\t' || c == '\n' || c == '\r') {
                ++pos_;
            } else {
                break;
            }
        }
    }

    void expect(char c) {
        if (eof() || text_[pos_] != c) {
            fail(std::format("expected '{}'", c));
        }
        ++pos_;
    }

    bool consume_literal(std::string_view literal) {
        if (text_.substr(pos_).starts_with(literal)) {
            pos_ += literal.size();
            return true;
        }
        return false;
    }

    Value parse_value(int depth) {
        if (depth > kMaxDepth) {
            fail("nesting too deep");
        }
        switch (peek()) {
        case '{':
            return parse_object(depth);
        case '[':
            return parse_array(depth);
        case '"':
            return parse_string_value();
        case 't':
        case 'f':
            return parse_bool();
        case 'n':
            return parse_null();
        default:
            return parse_number();
        }
    }

    Value parse_object(int depth) {
        Value v;
        v.kind = Value::Kind::Object;
        expect('{');
        skip_whitespace();
        if (peek() == '}') {
            ++pos_;
            return v;
        }
        for (;;) {
            skip_whitespace();
            if (peek() != '"') {
                fail("expected object key");
            }
            std::string key = parse_raw_string();
            skip_whitespace();
            expect(':');
            skip_whitespace();
            v.object.push_back(Member{std::move(key), parse_value(depth + 1)});
            skip_whitespace();
            if (peek() == ',') {
                ++pos_;
                continue;
            }
            expect('}');
            return v;
        }
    }

    Value parse_array(int depth) {
        Value v;
        v.kind = Value::Kind::Array;
        expect('[');
        skip_whitespace();
        if (peek() == ']') {
            ++pos_;
            return v;
        }
        for (;;) {
            skip_whitespace();
            v.array.push_back(parse_value(depth + 1));
            skip_whitespace();
            if (peek() == ',') {
                ++pos_;
                continue;
            }
            expect(']');
            return v;
        }
    }

    Value parse_string_value() {
        Value v;
        v.kind = Value::Kind::String;
        v.string = parse_raw_string();
        return v;
    }

    Value parse_bool() {
        Value v;
        v.kind = Value::Kind::Bool;
        if (consume_literal("true")) {
            v.boolean = true;
        } else if (consume_literal("false")) {
            v.boolean = false;
        } else {
            fail("invalid literal");
        }
        return v;
    }

    Value parse_null() {
        if (!consume_literal("null")) {
            fail("invalid literal");
        }
        return Value{};
    }

    Value parse_number() {
        const std::size_t start = pos_;
        if (!eof() && (text_[pos_] == '-' || text_[pos_] == '+')) {
            ++pos_;
        }
        while (!eof()) {
            const char c = text_[pos_];
            const bool numeric =
                (c >= '0' && c <= '9') || c == '.' || c == 'e' || c == 'E' || c == '+' || c == '-';
            if (!numeric) {
                break;
            }
            ++pos_;
        }
        if (pos_ == start) {
            fail("expected a value");
        }
        const std::string_view literal = text_.substr(start, pos_ - start);
        // from_chars rather than stod: it doesn't throw, doesn't allocate, and
        // doesn't consult the locale. It also rejects a leading '+' and hex
        // forms, which JSON disallows anyway.
        double parsed = 0.0;
        const auto [stop, ec] =
            std::from_chars(literal.data(), literal.data() + literal.size(), parsed);
        if (ec != std::errc{} || stop != literal.data() + literal.size()) {
            pos_ = start;
            fail(std::format("malformed number '{}'", literal));
        }
        Value v;
        v.kind = Value::Kind::Number;
        v.number = parsed;
        return v;
    }

    // Decodes a JSON string including \uXXXX escapes (encoded as UTF-8, with
    // surrogate pairs combined). Tensor names are ASCII, but config.json and
    // tokenizer metadata are not guaranteed to be.
    std::string parse_raw_string() {
        expect('"');
        std::string out;
        for (;;) {
            if (eof()) {
                fail("unterminated string");
            }
            const char c = text_[pos_++];
            if (c == '"') {
                return out;
            }
            if (c != '\\') {
                out.push_back(c);
                continue;
            }
            if (eof()) {
                fail("unterminated escape");
            }
            switch (const char esc = text_[pos_++]) {
            case '"':
            case '\\':
            case '/':
                out.push_back(esc);
                break;
            case 'b':
                out.push_back('\b');
                break;
            case 'f':
                out.push_back('\f');
                break;
            case 'n':
                out.push_back('\n');
                break;
            case 'r':
                out.push_back('\r');
                break;
            case 't':
                out.push_back('\t');
                break;
            case 'u': {
                std::uint32_t cp = parse_hex4();
                if (cp >= 0xD800 && cp <= 0xDBFF && text_.substr(pos_).starts_with("\\u")) {
                    const std::size_t saved = pos_;
                    pos_ += 2;
                    const std::uint32_t low = parse_hex4();
                    if (low >= 0xDC00 && low <= 0xDFFF) {
                        cp = 0x10000 + ((cp - 0xD800) << 10) + (low - 0xDC00);
                    } else {
                        pos_ = saved;
                    }
                }
                append_utf8(out, cp);
                break;
            }
            default:
                fail("invalid escape");
            }
        }
    }

    std::uint32_t parse_hex4() {
        if (pos_ + 4 > text_.size()) {
            fail("truncated \\u escape");
        }
        std::uint32_t cp = 0;
        for (int i = 0; i < 4; ++i) {
            const char c = text_[pos_++];
            cp <<= 4;
            if (c >= '0' && c <= '9') {
                cp |= static_cast<std::uint32_t>(c - '0');
            } else if (c >= 'a' && c <= 'f') {
                cp |= static_cast<std::uint32_t>(c - 'a' + 10);
            } else if (c >= 'A' && c <= 'F') {
                cp |= static_cast<std::uint32_t>(c - 'A' + 10);
            } else {
                fail("invalid hex digit in \\u escape");
            }
        }
        return cp;
    }

    static void append_utf8(std::string &out, std::uint32_t cp) {
        if (cp < 0x80) {
            out.push_back(static_cast<char>(cp));
        } else if (cp < 0x800) {
            out.push_back(static_cast<char>(0xC0 | (cp >> 6)));
            out.push_back(static_cast<char>(0x80 | (cp & 0x3F)));
        } else if (cp < 0x10000) {
            out.push_back(static_cast<char>(0xE0 | (cp >> 12)));
            out.push_back(static_cast<char>(0x80 | ((cp >> 6) & 0x3F)));
            out.push_back(static_cast<char>(0x80 | (cp & 0x3F)));
        } else {
            out.push_back(static_cast<char>(0xF0 | (cp >> 18)));
            out.push_back(static_cast<char>(0x80 | ((cp >> 12) & 0x3F)));
            out.push_back(static_cast<char>(0x80 | ((cp >> 6) & 0x3F)));
            out.push_back(static_cast<char>(0x80 | (cp & 0x3F)));
        }
    }
};

[[noreturn]] void type_error(std::string_view context, std::string_view wanted) {
    die("json: {} is not {}", context, wanted);
}

} // namespace

const Value *Value::find(std::string_view key) const {
    if (kind != Kind::Object) {
        return nullptr;
    }
    for (const Member &m : object) {
        if (m.key == key) {
            return &m.value;
        }
    }
    return nullptr;
}

bool Value::as_bool(std::string_view context) const {
    if (kind != Kind::Bool) {
        type_error(context, "a bool");
    }
    return boolean;
}

double Value::as_double(std::string_view context) const {
    if (kind != Kind::Number) {
        type_error(context, "a number");
    }
    return number;
}

std::int64_t Value::as_int64(std::string_view context) const {
    const double d = as_double(context);
    if (!std::isfinite(d) || d != std::floor(d)) {
        type_error(context, "an integer");
    }
    // 2^53 is where doubles stop representing consecutive integers; every
    // safetensors offset and shape is far below it.
    constexpr double kMaxExact = 9007199254740992.0;
    if (d < -kMaxExact || d > kMaxExact) {
        type_error(context, "an integer in range");
    }
    return static_cast<std::int64_t>(d);
}

const std::string &Value::as_string(std::string_view context) const {
    if (kind != Kind::String) {
        type_error(context, "a string");
    }
    return string;
}

const std::vector<Value> &Value::as_array(std::string_view context) const {
    if (kind != Kind::Array) {
        type_error(context, "an array");
    }
    return array;
}

const std::vector<Member> &Value::as_object(std::string_view context) const {
    if (kind != Kind::Object) {
        type_error(context, "an object");
    }
    return object;
}

Value parse(std::string_view text, std::string_view origin) {
    return Parser(text, origin).parse_document();
}

} // namespace qllm::json
