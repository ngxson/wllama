#include <iostream>
#include <vector>
#include <string>
#include <sstream>
#include <stdio.h>

#include <stdlib.h>
#include <unistd.h>

#ifdef __EMSCRIPTEN__
#include <malloc.h>
#include <emscripten/emscripten.h>
#endif

// #define GLUE_DEBUG(...) fprintf(stderr, "@@ERROR@@" __VA_ARGS__)

#include "llama.h"
#include "helpers/wcommon.h"
#include "actions.hpp"

#define MAX(a, b) ((a) > (b) ? (a) : (b))

#define WLLAMA_ACTION(name)                 \
  else if (action == #name)                 \
  {                                         \
    auto res = action_##name(app, req_raw); \
    res.handler.serialize(output_buffer);   \
  }

static void llama_log_callback_logTee(ggml_log_level level, const char *text, void *user_data)
{
  (void)user_data;
  const char *lvl = "@@DEBUG";
  size_t len = strlen(text);
  if (len == 0 || text[len - 1] != '\n')
  {
    // do not print if the line does not terminate with \n
    return;
  }
  if (level == GGML_LOG_LEVEL_ERROR)
  {
    lvl = "@@ERROR";
  }
  else if (level == GGML_LOG_LEVEL_WARN)
  {
    lvl = "@@WARN";
  }
  else if (level == GGML_LOG_LEVEL_INFO)
  {
    lvl = "@@INFO";
  }
  fprintf(stderr, "%s@@%s", lvl, text);
}

static void printStr(ggml_log_level level, const char *text)
{
  std::string str = std::string(text) + "\n";
  llama_log_callback_logTee(level, str.c_str(), nullptr);
}

static glue_outbuf output_buffer;
static app_t app;

static std::vector<char> input_buffer;
// second argument is dummy
extern "C" const char *wllama_malloc(size_t size, uint32_t)
{
  if (input_buffer.size() < size)
  {
    input_buffer.resize(size);
  }
  return input_buffer.data();
}

extern "C" const char *wllama_start()
{
  try
  {
    llama_backend_init();
    // std::cerr << llama_print_system_info() << "\n";
    llama_log_set(llama_log_callback_logTee, nullptr);
    wllama_malloc(1024, 0);
    return "{\"success\":true}";
  }
  catch (std::exception &e)
  {
    printStr(GGML_LOG_LEVEL_ERROR, e.what());
    return "{\"error\":true}";
  }
}

extern "C" const char *wllama_action(const char *name, const char *req_raw)
{
  try
  {
    std::string action(name);

    if (action.empty())
    {
      printStr(GGML_LOG_LEVEL_ERROR, "Empty action");
      abort();
    }

    WLLAMA_ACTION(load)
    WLLAMA_ACTION(set_options)
    WLLAMA_ACTION(sampling_init)
    WLLAMA_ACTION(sampling_sample)
    WLLAMA_ACTION(sampling_accept)
    WLLAMA_ACTION(get_vocab)
    WLLAMA_ACTION(lookup_token)
    WLLAMA_ACTION(tokenize)
    WLLAMA_ACTION(detokenize)
    WLLAMA_ACTION(decode)
    WLLAMA_ACTION(encode)
    WLLAMA_ACTION(get_logits)
    WLLAMA_ACTION(embeddings)
    WLLAMA_ACTION(chat_format)
    WLLAMA_ACTION(kv_remove)
    WLLAMA_ACTION(kv_clear)
    WLLAMA_ACTION(current_status)
    // WLLAMA_ACTION(session_save)
    // WLLAMA_ACTION(session_load)
    WLLAMA_ACTION(test_benchmark)
    WLLAMA_ACTION(test_perplexity)

    else
    {
      printStr(GGML_LOG_LEVEL_ERROR, (std::string("Unknown action: ") + name).c_str());
      abort();
    }

    // length of response is written inside input_buffer
    uint32_t *output_len = (uint32_t *)req_raw;
    output_len[0] = output_buffer.data.size();
    return output_buffer.data.data();
  }
  catch (std::exception &e)
  {
    printStr(GGML_LOG_LEVEL_ERROR, e.what());
    return nullptr;
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
    printStr(GGML_LOG_LEVEL_ERROR, e.what());
    return "{\"error\":true}";
  }
}

extern "C" const char *wllama_debug()
{
  auto get_mem_total = [&]()
  {
#ifdef __EMSCRIPTEN__
    return EM_ASM_INT(return HEAP8.length);
#else
    return 0;
#endif
  };
  auto get_mem_free = [&]()
  {
#ifdef __EMSCRIPTEN__
    auto i = mallinfo();
    unsigned int total_mem = get_mem_total();
    unsigned int dynamic_top = (unsigned int)(size_t)sbrk(0);
    return total_mem - dynamic_top + i.fordblks;
#else
    return 0;
#endif
  };
  /*json res = json{
      {"mem_total_MB", get_mem_total() / 1024 / 1024},
      {"mem_free_MB", get_mem_free() / 1024 / 1024},
      {"mem_used_MB", (get_mem_total() - get_mem_free()) / 1024 / 1024},
  };
  result = std::string(res.dump());
  return result.c_str();*/
  return nullptr;
}

int main()
{
  std::cerr << "Unused\n";
  return 0;
}
