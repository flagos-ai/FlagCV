/* Copyright (c) 2025-2026 BAAI. SPDX-License-Identifier: Apache-2.0 */

#ifndef FLAGCV_SRC_RUNTIME_JSON_HPP_
#define FLAGCV_SRC_RUNTIME_JSON_HPP_

#include <cstdint>
#include <map>
#include <string>
#include <vector>

namespace flagcv {
namespace json {

/*
 * Minimal JSON value and parser for the compiler protocol and kernel
 * registries. The core library takes no third-party dependency for this;
 * the documents involved are small, machine-written, and UTF-8.
 */
class Value {
 public:
  enum class Type { kNull, kBool, kNumber, kString, kArray, kObject };

  Type type() const { return type_; }
  bool is_null() const { return type_ == Type::kNull; }
  bool is_number() const { return type_ == Type::kNumber; }
  bool is_string() const { return type_ == Type::kString; }
  bool is_array() const { return type_ == Type::kArray; }
  bool is_object() const { return type_ == Type::kObject; }

  bool as_bool() const;
  double as_number() const;
  std::int64_t as_int() const;
  const std::string& as_string() const;
  const std::vector<Value>& as_array() const;

  /* Object member; throws when absent. */
  const Value& at(const std::string& key) const;
  bool has(const std::string& key) const;

  static Value parse(const std::string& text);

 private:
  friend class Parser;
  Type type_ = Type::kNull;
  bool bool_ = false;
  double number_ = 0;
  std::string string_;
  std::vector<Value> array_;
  std::map<std::string, Value> object_;
};

/* Quoted, escaped JSON string literal. */
std::string quote(const std::string& text);

}  // namespace json
}  // namespace flagcv

#endif  // FLAGCV_SRC_RUNTIME_JSON_HPP_
