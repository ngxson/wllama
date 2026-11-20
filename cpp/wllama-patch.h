#pragma once

// state of the patches in wllama-patch.cpp, used by the rest of the glue

struct mtmd_context;

// mtmd context of the server, recorded by __wrap_mtmd_init_from_file
extern mtmd_context *shared_mctx;

// if true, __wrap_ggml_graph_plan runs graphs on 1 thread
extern bool force_single_thread;

// true if model files are read from JS (async file read)
bool wllama_fs_use_async();
