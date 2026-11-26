/* Copyright (c) 2025-2026 BAAI. SPDX-License-Identifier: Apache-2.0 */

#include "src/runtime/compiler_client.hpp"

#include <spawn.h>
#include <sys/socket.h>
#include <sys/wait.h>
#include <unistd.h>

#include <cerrno>
#include <cstdlib>
#include <cstring>
#include <flagcv/handle.hpp>
#include <vector>

#include "src/runtime/paths.hpp"

extern char** environ;

namespace flagcv {

CompilerClient& CompilerClient::instance() {
  static CompilerClient client;
  return client;
}

CompilerClient::~CompilerClient() { stop(); }

void CompilerClient::start() {
  int sockets[2];
  if (::socketpair(AF_UNIX, SOCK_STREAM | SOCK_CLOEXEC, 0, sockets) != 0) {
    throw Error(std::string("flagcv: socketpair failed: ") +
                std::strerror(errno));
  }

  const std::string data = data_dir();
  const char* python_env = std::getenv("FLAGCV_PYTHON");
  const std::string python =
      python_env != nullptr && *python_env ? python_env : "python3";

  // The child sees the compiler package on PYTHONPATH and the data directory
  // in FLAGCV_DATA_DIR; the rest of the environment passes through untouched.
  std::vector<std::string> env_storage;
  const char* old_path = std::getenv("PYTHONPATH");
  env_storage.push_back("PYTHONPATH=" + data + "/compiler" +
                        (old_path != nullptr && *old_path
                             ? std::string(":") + old_path
                             : std::string()));
  env_storage.push_back("FLAGCV_DATA_DIR=" + data);
  for (char** e = environ; *e != nullptr; ++e) {
    if (std::strncmp(*e, "PYTHONPATH=", 11) != 0 &&
        std::strncmp(*e, "FLAGCV_DATA_DIR=", 16) != 0) {
      env_storage.emplace_back(*e);
    }
  }
  std::vector<char*> envp;
  for (auto& entry : env_storage) envp.push_back(entry.data());
  envp.push_back(nullptr);

  std::string arg_m = "-m", arg_module = "flagcv_codegen.main";
  std::string arg_python = python;
  char* argv[] = {arg_python.data(), arg_m.data(), arg_module.data(), nullptr};

  // dup2 onto 0/1 clears close-on-exec for those copies only; both original
  // socket ends close in the child at exec.
  posix_spawn_file_actions_t actions;
  posix_spawn_file_actions_init(&actions);
  posix_spawn_file_actions_adddup2(&actions, sockets[1], STDIN_FILENO);
  posix_spawn_file_actions_adddup2(&actions, sockets[1], STDOUT_FILENO);

  pid_t pid = -1;
  int rc = ::posix_spawnp(&pid, python.c_str(), &actions, nullptr, argv,
                          envp.data());
  posix_spawn_file_actions_destroy(&actions);
  ::close(sockets[1]);
  if (rc != 0) {
    ::close(sockets[0]);
    throw Error("flagcv: cannot start the compiler with '" + python +
                "': " + std::strerror(rc) +
                " (set FLAGCV_PYTHON to an interpreter with Triton installed)");
  }
  pid_ = pid;
  fd_ = sockets[0];
  pending_.clear();
}

void CompilerClient::stop() {
  if (fd_ >= 0) {
    ::close(fd_);  // EOF on stdin ends the child's request loop
    fd_ = -1;
  }
  if (pid_ > 0) {
    int status = 0;
    while (::waitpid(pid_, &status, 0) < 0 && errno == EINTR) {
    }
    pid_ = -1;
  }
}

void CompilerClient::send_line(const std::string& line) {
  std::size_t sent = 0;
  while (sent < line.size()) {
    // MSG_NOSIGNAL: a dead child must surface as an error, not SIGPIPE.
    ssize_t n =
        ::send(fd_, line.data() + sent, line.size() - sent, MSG_NOSIGNAL);
    if (n < 0 && errno == EINTR) continue;
    if (n <= 0)
      throw Error(std::string("flagcv: compiler process write failed: ") +
                  std::strerror(errno));
    sent += static_cast<std::size_t>(n);
  }
}

std::string CompilerClient::receive_line() {
  for (;;) {
    auto newline = pending_.find('\n');
    if (newline != std::string::npos) {
      std::string line = pending_.substr(0, newline);
      pending_.erase(0, newline + 1);
      return line;
    }
    char buffer[65536];
    ssize_t n = ::recv(fd_, buffer, sizeof(buffer), 0);
    if (n < 0 && errno == EINTR) continue;
    if (n <= 0)
      throw Error(
          "flagcv: compiler process exited unexpectedly (see its stderr "
          "above)");
    pending_.append(buffer, static_cast<std::size_t>(n));
  }
}

json::Value CompilerClient::request(const std::string& body) {
  std::lock_guard<std::mutex> lock(mutex_);
  if (fd_ < 0) start();

  const std::int64_t id = next_id_++;
  json::Value reply;
  try {
    send_line("{\"id\":" + std::to_string(id) + "," + body + "}\n");
    reply = json::Value::parse(receive_line());
  } catch (...) {
    stop();  // the stream state is unknown; restart on the next request
    throw;
  }
  if (!reply.has("id") || reply.at("id").is_null() ||
      reply.at("id").as_int() != id) {
    stop();
    throw Error("flagcv: compiler reply does not match request " +
                std::to_string(id));
  }
  if (!reply.at("ok").as_bool()) {
    throw Error("flagcv: compile failed: " + reply.at("error").as_string());
  }
  return reply;
}

}  // namespace flagcv
