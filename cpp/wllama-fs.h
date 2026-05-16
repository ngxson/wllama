#pragma once

#ifdef __EMSCRIPTEN__
#include <emscripten/emscripten.h>
#endif

#include <map>

// ---------------------------------------------------------------------------
// fopen / fclose / fread intercept layer
//
// llama_file::impl uses fopen + fseek + std::fread (buffered I/O path that
// is always taken on Emscripten when use_mmap=false).  Wrapping at this
// level avoids the wasm64 type-mismatch that arises from wrapping the raw
// open/read syscalls (which are varargs in musl and therefore have wasm
// type (i64,i32)->i32 rather than the fixed-arg type we would declare).
// ---------------------------------------------------------------------------

static std::map<FILE *, std::string> s_file_path_map;

namespace wllama_fs {
  bool ready = false;
  bool use_async = false;

  // only support one cached file
  std::vector<char> cache_data;
  size_t cache_offset = 0;
  FILE * cache_file = nullptr;

  void make_sure_ready() {
    if (ready) return;
    use_async = getenv("USE_ASYNC_FILE") != nullptr;
    if (use_async) {
      cache_data.reserve(1024 * 1024); // pre-allocate 1MB for async read
    }
    ready = true;
  }

  // return false if the read request is not handled (no changes to ptr is made)
  bool read_cache(FILE *f, char *ptr, size_t size, size_t nmemb) {
  }
}

extern "C" {
  FILE   *__real_fopen(const char *path, const char *mode);
  int     __real_fclose(FILE *f);
  size_t  __real_fread(void *ptr, size_t size, size_t nmemb, FILE *f);

  FILE *__wrap_fopen(const char *path, const char *mode) {
    wllama_fs::make_sure_ready();
    // TODO: if use_async, we call EM_ASYNC_JS to read the file
    FILE *f = __real_fopen(path, mode);
    if (f) s_file_path_map[f] = path;
    fprintf(stderr, "__wrap_fopen(%s) = %p\n", path, (void *)f);
    return f;
  }

  int __wrap_fclose(FILE *f) {
    auto it = s_file_path_map.find(f);
    fprintf(stderr, "__wrap_fclose(%p, path=%s)\n", (void *)f,
            it != s_file_path_map.end() ? it->second.c_str() : "?");
    s_file_path_map.erase(f);
    return __real_fclose(f);
  }

  size_t __wrap_fread(void *ptr, size_t size, size_t nmemb, FILE *f) {
    auto it = s_file_path_map.find(f);
    const char *path = it != s_file_path_map.end() ? it->second.c_str() : "?";
    fprintf(stderr, "__wrap_fread(%p, size=%zu, nmemb=%zu, path=%s)\n",
            (void *)f, size, nmemb, path);
    return __real_fread(ptr, size, nmemb, f);
  }
}
