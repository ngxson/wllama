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
#include "wllama-context.h"
#include "wllama-fs.h"
#include "wllama.h"

#define MAX(a, b) ((a) > (b) ? (a) : (b))

#define WLLAMA_ACTION(name)                \
  else if (action == #name)                \
  {                                        \
    auto res = app.action_##name(req_raw); \
    res.handler.serialize(output_buffer);  \
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
static wllama_context app;

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

    wllama_fs::make_sure_ready();
    if (wllama_fs::use_async)
    {
      printStr(GGML_LOG_LEVEL_INFO, "Using async file read");
    }

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
    WLLAMA_ACTION(completion)
    WLLAMA_ACTION(embedding)
    WLLAMA_ACTION(get_result)

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
    // app.unload();
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
    size_t total_mem = get_mem_total();
    size_t dynamic_top = (size_t)sbrk(0);
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
