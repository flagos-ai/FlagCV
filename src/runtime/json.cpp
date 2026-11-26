/* Copyright (c) 2025-2026 BAAI. SPDX-License-Identifier: Apache-2.0 */

#include "src/runtime/json.hpp"

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <stdexcept>

namespace flagcv {
namespace json {

namespace {

[[noreturn]] void type_error(const char* expected) {
  throw std::runtime_error(std::string("json: value is not ") + expected);
}

void append_utf8(std::string& out, unsigned code) {
  if (code < 0x80) {
    out += static_cast<char>(code);
  } else if (code < 0x800) {
    out += static_cast<char>(0xC0 | (code >> 6));
    out += static_cast<char>(0x80 | (code & 0x3F));
  } else if (code < 0x10000) {
    out += static_cast<char>(0xE0 | (code >> 12));
    out += static_cast<char>(0x80 | ((code >> 6) & 0x3F));
    out += static_cast<char>(0x80 | (code & 0x3F));
  } else {
    out += static_cast<char>(0xF0 | (code >> 18));
    out += static_cast<char>(0x80 | ((code >> 12) & 0x3F));
    out += static_cast<char>(0x80 | ((code >> 6) & 0x3F));
    out += static_cast<char>(0x80 | (code & 0x3F));
  }
}

}  // namespace

class Parser {
 public:
  explicit Parser(const std::string& text) : text_(text) {}

  Value document() {
    Value value = parse_value();
    skip_space();
    if (pos_ != text_.size()) fail("trailing characters");
    return value;
  }

 private:
  [[noreturn]] void fail(const std::string& what) const {
    throw std::runtime_error("json: " + what + " at offset " +
                             std::to_string(pos_));
  }

  void skip_space() {
    while (pos_ < text_.size() &&
           (text_[pos_] == ' ' || text_[pos_] == '\t' || text_[pos_] == '\n' ||
            text_[pos_] == '\r')) {
      ++pos_;
    }
  }

  char peek() {
    skip_space();
    if (pos_ >= text_.size()) fail("unexpected end of input");
    return text_[pos_];
  }

  void expect(const char* literal) {
    for (const char* c = literal; *c; ++c) {
      if (pos_ >= text_.size() || text_[pos_] != *c)
        fail(std::string("expected '") + literal + "'");
      ++pos_;
    }
  }

  Value parse_value() {
    Value value;
    switch (peek()) {
      case '{':
        value.type_ = Value::Type::kObject;
        ++pos_;
        if (peek() == '}') {
          ++pos_;
          return value;
        }
        for (;;) {
          if (peek() != '"') fail("expected object key");
          std::string key = parse_string();
          if (peek() != ':') fail("expected ':'");
          ++pos_;
          value.object_[key] = parse_value();
          char c = peek();
          ++pos_;
          if (c == '}') return value;
          if (c != ',') fail("expected ',' or '}'");
        }
      case '[':
        value.type_ = Value::Type::kArray;
        ++pos_;
        if (peek() == ']') {
          ++pos_;
          return value;
        }
        for (;;) {
          value.array_.push_back(parse_value());
          char c = peek();
          ++pos_;
          if (c == ']') return value;
          if (c != ',') fail("expected ',' or ']'");
        }
      case '"':
        value.type_ = Value::Type::kString;
        value.string_ = parse_string();
        return value;
      case 't':
        expect("true");
        value.type_ = Value::Type::kBool;
        value.bool_ = true;
        return value;
      case 'f':
        expect("false");
        value.type_ = Value::Type::kBool;
        return value;
      case 'n':
        expect("null");
        return value;
      default:
        value.type_ = Value::Type::kNumber;
        value.number_ = parse_number();
        return value;
    }
  }

  double parse_number() {
    const char* begin = text_.c_str() + pos_;
    char* end = nullptr;
    double number = std::strtod(begin, &end);
    if (end == begin) fail("invalid value");
    pos_ += static_cast<std::size_t>(end - begin);
    return number;
  }

  unsigned parse_hex4() {
    if (pos_ + 4 > text_.size()) fail("truncated \\u escape");
    unsigned code = 0;
    for (int i = 0; i < 4; ++i) {
      char c = text_[pos_++];
      code <<= 4;
      if (c >= '0' && c <= '9')
        code |= static_cast<unsigned>(c - '0');
      else if (c >= 'a' && c <= 'f')
        code |= static_cast<unsigned>(c - 'a' + 10);
      else if (c >= 'A' && c <= 'F')
        code |= static_cast<unsigned>(c - 'A' + 10);
      else
        fail("invalid \\u escape");
    }
    return code;
  }

  std::string parse_string() {
    ++pos_;  // opening quote
    std::string out;
    for (;;) {
      if (pos_ >= text_.size()) fail("unterminated string");
      char c = text_[pos_++];
      if (c == '"') return out;
      if (c != '\\') {
        out += c;
        continue;
      }
      if (pos_ >= text_.size()) fail("unterminated escape");
      char e = text_[pos_++];
      switch (e) {
        case '"':
          out += '"';
          break;
        case '\\':
          out += '\\';
          break;
        case '/':
          out += '/';
          break;
        case 'b':
          out += '\b';
          break;
        case 'f':
          out += '\f';
          break;
        case 'n':
          out += '\n';
          break;
        case 'r':
          out += '\r';
          break;
        case 't':
          out += '\t';
          break;
        case 'u': {
          unsigned code = parse_hex4();
          if (code >= 0xD800 && code < 0xDC00 && pos_ + 1 < text_.size() &&
              text_[pos_] == '\\' && text_[pos_ + 1] == 'u') {
            pos_ += 2;
            unsigned low = parse_hex4();
            code = 0x10000 + ((code - 0xD800) << 10) + (low - 0xDC00);
          }
          append_utf8(out, code);
          break;
        }
        default:
          fail("invalid escape");
      }
    }
  }

  const std::string& text_;
  std::size_t pos_ = 0;
};

bool Value::as_bool() const {
  if (type_ != Type::kBool) type_error("a boolean");
  return bool_;
}

double Value::as_number() const {
  if (type_ != Type::kNumber) type_error("a number");
  return number_;
}

std::int64_t Value::as_int() const {
  double number = as_number();
  if (std::floor(number) != number) type_error("an integer");
  return static_cast<std::int64_t>(number);
}

const std::string& Value::as_string() const {
  if (type_ != Type::kString) type_error("a string");
  return string_;
}

const std::vector<Value>& Value::as_array() const {
  if (type_ != Type::kArray) type_error("an array");
  return array_;
}

const Value& Value::at(const std::string& key) const {
  if (type_ != Type::kObject) type_error("an object");
  auto it = object_.find(key);
  if (it == object_.end())
    throw std::runtime_error("json: missing key '" + key + "'");
  return it->second;
}

bool Value::has(const std::string& key) const {
  return type_ == Type::kObject && object_.count(key) != 0;
}

Value Value::parse(const std::string& text) { return Parser(text).document(); }

std::string quote(const std::string& text) {
  std::string out = "\"";
  for (unsigned char c : text) {
    switch (c) {
      case '"':
        out += "\\\"";
        break;
      case '\\':
        out += "\\\\";
        break;
      case '\n':
        out += "\\n";
        break;
      case '\r':
        out += "\\r";
        break;
      case '\t':
        out += "\\t";
        break;
      default:
        if (c < 0x20) {
          char buffer[8];
          std::snprintf(buffer, sizeof(buffer), "\\u%04x", c);
          out += buffer;
        } else {
          out += static_cast<char>(c);
        }
    }
  }
  return out + "\"";
}

}  // namespace json
}  // namespace flagcv
