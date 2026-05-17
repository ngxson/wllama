#pragma once

#ifdef __EMSCRIPTEN__
#include <emscripten/emscripten.h>
#endif

#include <algorithm>
#include <map>
#include <vector>
#include <cstring>

static std::map<FILE *, std::string> s_file_path_map;

namespace wllama_fs
{
  bool ready = false;
  bool use_async = false;

  static const size_t CACHE_SIZE = 1024 * 1024; // 1 MB read-ahead

  std::vector<uint8_t> cache_data;
  size_t cache_start = 0;
  FILE *cache_file = nullptr;

  void make_sure_ready()
  {
    if (ready)
      return;
    use_async = getenv("USE_ASYNC_FILE") != nullptr;
    ready = true;
  }

  size_t try_cache(FILE *f, char *ptr, size_t req_bytes, size_t fpos)
  {
    if (f != cache_file || cache_data.empty())
      return 0;
    if (fpos >= cache_start && fpos + req_bytes <= cache_start + cache_data.size())
    {
      memcpy(ptr, cache_data.data() + (fpos - cache_start), req_bytes);
      return req_bytes;
    }
    return 0;
  }
}

// Thin stub — real implementation lives in llama-cpp.js to avoid
// C++ formatter mangling the JS syntax inside EM_ASYNC_JS macros.

EM_ASYNC_JS(size_t, js_file_read, (const char *path_ptr, size_t offset, size_t req_size, void *out_ptr), {
  return await _wllama_js_file_read(UTF8ToString(Number(path_ptr)), Number(offset), Number(req_size), Number(out_ptr));
});

extern "C"
{
  FILE *__real_fopen(const char *path, const char *mode);
  int __real_fclose(FILE *f);
  size_t __real_fread(void *ptr, size_t size, size_t nmemb, FILE *f);
  int __real_fseek(FILE *f, long offset, int whence);
  long __real_ftell(FILE *f);

  FILE *__wrap_fopen(const char *path, const char *mode)
  {
    wllama_fs::make_sure_ready();
    FILE *f = __real_fopen(path, mode);
    if (f)
    {
      s_file_path_map[f] = path;
    }
    return f;
  }

  int __wrap_fclose(FILE *f)
  {
    if (wllama_fs::cache_file == f)
    {
      wllama_fs::cache_file = nullptr;
      wllama_fs::cache_data.clear();
    }
    s_file_path_map.erase(f);
    return __real_fclose(f);
  }

  int __wrap_fseek(FILE *f, long offset, int whence)
  {
    return __real_fseek(f, offset, whence);
  }

  long __wrap_ftell(FILE *f)
  {
    return __real_ftell(f);
  }

  size_t __wrap_fread(void *ptr, size_t size, size_t nmemb, FILE *f)
  {
    wllama_fs::make_sure_ready();
    if (!wllama_fs::use_async)
      return __real_fread(ptr, size, nmemb, f);

    auto nit = s_file_path_map.find(f);
    if (nit == s_file_path_map.end())
      return __real_fread(ptr, size, nmemb, f);

    size_t req_bytes = size * nmemb;
    if (req_bytes == 0)
      return 0;

    size_t fpos = (size_t)__real_ftell(f);

    // Large reads (>= 1 MB): write directly into ptr, skip cache entirely.
    if (req_bytes >= wllama_fs::CACHE_SIZE)
    {
      size_t actual = (size_t)js_file_read(
          nit->second.c_str(), fpos, req_bytes, ptr);
      if (actual == 0)
        return 0;
      size_t copy_bytes = std::min(req_bytes, actual);
      __real_fseek(f, fpos + copy_bytes, SEEK_SET);
      return copy_bytes / size;
    }

    // Small reads: try cache first.
    size_t cached = wllama_fs::try_cache(f, (char *)ptr, req_bytes, fpos);
    if (cached == req_bytes)
    {
      __real_fseek(f, fpos + req_bytes, SEEK_SET);
      return nmemb;
    }

    // Cache miss: fetch a full CACHE_SIZE block from main thread.
    wllama_fs::cache_data.resize(wllama_fs::CACHE_SIZE);
    size_t actual = (size_t)js_file_read(
        nit->second.c_str(), fpos, wllama_fs::CACHE_SIZE,
        wllama_fs::cache_data.data());

    wllama_fs::cache_data.resize(actual);
    wllama_fs::cache_file  = f;
    wllama_fs::cache_start = fpos;

    if (actual == 0)
      return 0;

    size_t copy_bytes = std::min(req_bytes, actual);
    memcpy(ptr, wllama_fs::cache_data.data(), copy_bytes);
    __real_fseek(f, fpos + copy_bytes, SEEK_SET);

    return copy_bytes / size;
  }
}
