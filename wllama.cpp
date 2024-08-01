#include <iostream>
#include <vector>
#include <string>
#include <sstream>
#include <stdio.h>

#include <stdlib.h>
#include <unistd.h>
#include <malloc.h>
#include <emscripten/emscripten.h>

#include "llama.h"
#include "json.hpp"
#include "common.h"
#include "actions.hpp"

#define WLLAMA_ACTION(name)              \
  if (action == #name)                   \
  {                                      \
    res = action_##name(app, body_json); \
  }

static void llama_log_callback_logTee(ggml_log_level level, const char *text, void *user_data)
{
  (void)user_data;
  const char *lvl = "@@DEBUG";
  size_t len = strlen(text);
  if (len == 0 || text[len - 1] != '\n') {
    // do not print if the line does not terminate with \n
    return;
  }
  if (level == GGML_LOG_LEVEL_ERROR) {
    lvl = "@@ERROR";
  } else if (level == GGML_LOG_LEVEL_WARN) {
    lvl = "@@WARN";
  } else if (level == GGML_LOG_LEVEL_INFO) {
    lvl = "@@INFO";
  }
  fprintf(stderr, "%s@@%s", lvl, text);
}

static std::string result;
static app_t app;

extern "C" const char *wllama_start()
{
  try
  {
    log_disable();
    llama_backend_init();
    // std::cerr << llama_print_system_info() << "\n";
    llama_log_set(llama_log_callback_logTee, nullptr);
    return "{\"success\":true}";
  }
  catch (std::exception &e)
  {
    json ex{{"__exception", std::string(e.what())}};
    result = std::string(ex.dump());
    return result.c_str();
  }
}

extern "C" const char *wllama_action(const char *name, const char *body)
{
  try
  {
    json res;
    std::string body_str(body);
    json body_json = json::parse(body_str);
    std::string action(name);
    WLLAMA_ACTION(load);
    WLLAMA_ACTION(set_options);
    WLLAMA_ACTION(sampling_init);
    WLLAMA_ACTION(sampling_sample);
    WLLAMA_ACTION(sampling_accept);
    WLLAMA_ACTION(get_vocab);
    WLLAMA_ACTION(lookup_token);
    WLLAMA_ACTION(tokenize);
    WLLAMA_ACTION(detokenize);
    WLLAMA_ACTION(decode);
    WLLAMA_ACTION(encode);
    WLLAMA_ACTION(get_logits);
    WLLAMA_ACTION(embeddings);
    WLLAMA_ACTION(kv_remove);
    WLLAMA_ACTION(kv_clear);
    WLLAMA_ACTION(current_status);
    WLLAMA_ACTION(session_save);
    WLLAMA_ACTION(session_load);
    result = std::string(res.dump());
    return result.c_str();
  }
  catch (std::exception &e)
  {
    json ex{{"__exception", std::string(e.what())}};
    result = std::string(ex.dump());
    return result.c_str();
  }
}

extern "C" const char *wllama_exit()
{
  try
  {
    free_all(app);
    llama_backend_free();
    return "{\"success\":true}";
  }
  catch (std::exception &e)
  {
    json ex{{"__exception", std::string(e.what())}};
    result = std::string(ex.dump());
    return result.c_str();
  }
}

extern "C" const char *wllama_debug()
{
  auto get_mem_total = [&]()
  {
    return EM_ASM_INT(return HEAP8.length);
  };
  auto get_mem_free = [&]()
  {
    auto i = mallinfo();
    unsigned int total_mem = get_mem_total();
    unsigned int dynamic_top = (unsigned int)sbrk(0);
    return total_mem - dynamic_top + i.fordblks;
  };
  json res = json{
      {"mem_total_MB", get_mem_total() / 1024 / 1024},
      {"mem_free_MB", get_mem_free() / 1024 / 1024},
      {"mem_used_MB", (get_mem_total() - get_mem_free()) / 1024 / 1024},
  };
  result = std::string(res.dump());
  return result.c_str();
}

int main()
{
  std::cerr << "Unused\n";
  return 0;
}
