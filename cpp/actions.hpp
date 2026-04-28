#pragma once

#include <iostream>
#include <vector>
#include <string>
#include <sstream>
#include <stdio.h>
#include <cmath>
#include <fstream>

#include "llama.h"
#include "common.h"
#include "sampling.h"
#include "chat.h"
#include "wllama.h"

#include "server-context.h"

#include "glue.hpp"

#define PARSE_REQ(msg_typename) \
  msg_typename req;             \
  glue_inbuf inbuf(req_raw);    \
  req.handler.deserialize(inbuf);

inline std::vector<char> convert_string_to_buf(std::string &input)
{
  std::vector<char> output;
  output.reserve(input.size());
  output.insert(output.end(), input.begin(), input.end());
  return output;
}

inline static ggml_type kv_cache_type_from_str(const std::string &s)
{
  if (s == "f32")
    return GGML_TYPE_F32;
  if (s == "f16")
    return GGML_TYPE_F16;
  if (s == "q8_0")
    return GGML_TYPE_Q8_0;
  if (s == "q4_0")
    return GGML_TYPE_Q4_0;
  if (s == "q4_1")
    return GGML_TYPE_Q4_1;
  if (s == "q5_0")
    return GGML_TYPE_Q5_0;
  if (s == "q5_1")
    return GGML_TYPE_Q5_1;
  throw std::runtime_error("Invalid cache type: " + s);
}

inline static enum llama_pooling_type pooling_type_from_str(const std::string &s)
{
  if (s == "LLAMA_POOLING_TYPE_UNSPECIFIED")
    return LLAMA_POOLING_TYPE_UNSPECIFIED;
  if (s == "LLAMA_POOLING_TYPE_NONE")
    return LLAMA_POOLING_TYPE_NONE;
  if (s == "LLAMA_POOLING_TYPE_MEAN")
    return LLAMA_POOLING_TYPE_MEAN;
  if (s == "LLAMA_POOLING_TYPE_CLS")
    return LLAMA_POOLING_TYPE_CLS;
  throw std::runtime_error("Invalid pooling type: " + s);
}

inline static llama_rope_scaling_type rope_scaling_type_from_str(const std::string &s)
{
  if (s == "LLAMA_ROPE_SCALING_TYPE_UNSPECIFIED")
    return LLAMA_ROPE_SCALING_TYPE_UNSPECIFIED;
  if (s == "LLAMA_ROPE_SCALING_TYPE_NONE")
    return LLAMA_ROPE_SCALING_TYPE_NONE;
  if (s == "LLAMA_ROPE_SCALING_TYPE_LINEAR")
    return LLAMA_ROPE_SCALING_TYPE_LINEAR;
  if (s == "LLAMA_ROPE_SCALING_TYPE_YARN")
    return LLAMA_ROPE_SCALING_TYPE_YARN;
  throw std::runtime_error("Invalid RoPE scaling type: " + s);
}

class app_exception : public std::exception
{
public:
  app_exception(const std::string &msg) throw() : message(msg) {}
  virtual ~app_exception() throw() {}
  const char *what() const throw() { return message.c_str(); }

private:
  std::string message;
};

struct kv_dump
{
  std::vector<std::string> keys;
  std::vector<std::string> vals;
};

//////////////////////////////////////////
//////////////////////////////////////////
//////////////////////////////////////////

enum display_type {
  DISPLAY_TYPE_RESET = 0,
  DISPLAY_TYPE_INFO,
  DISPLAY_TYPE_PROMPT,
  DISPLAY_TYPE_REASONING,
  DISPLAY_TYPE_USER_INPUT,
  DISPLAY_TYPE_ERROR
};

struct wllama_context
{
  server_context ctx_server;
  llama_context *ctx = nullptr;
  llama_model *model = nullptr;
  llama_vocab *vocab = nullptr;
  json messages = json::array();
  std::vector<raw_buffer> input_files;
  task_params defaults;

  std::function<bool()> should_stop = []() { return false; };
  std::string last_error;

  struct dummy_atomic {
    bool value = false;
    void store(bool v) { value = v; }
    operator bool() const { return value; }
  } g_is_interrupted;

  struct console
  {
    struct spinner
    {
      static void start() {}
      static void stop() {}
    } spinner;
    static void set_display(display_type display) {}
    static void flush() {}
  } console;

  explicit wllama_context() {};
  explicit wllama_context(const common_params &params)
  {
    defaults.sampling = params.sampling;
    defaults.speculative = params.speculative;
    defaults.n_keep = params.n_keep;
    defaults.n_predict = params.n_predict;
    defaults.antiprompt = params.antiprompt;
    defaults.stream = true;
  }

  std::string generate_completion(result_timings &out_timings)
  {
    server_response_reader rd = ctx_server.get_response_reader();
    auto chat_params = format_chat();
    {
      // TODO: reduce some copies here in the future
      server_task task = server_task(SERVER_TASK_TYPE_COMPLETION);
      task.id = rd.get_new_id();
      task.index = 0;
      task.params = defaults;               // copy
      task.cli_prompt = chat_params.prompt; // copy
      task.cli_files = input_files;         // copy
      task.cli = true;

      // chat template settings
      task.params.chat_parser_params = common_chat_parser_params(chat_params);
      task.params.chat_parser_params.reasoning_format = COMMON_REASONING_FORMAT_DEEPSEEK;
      if (!chat_params.parser.empty())
      {
        task.params.chat_parser_params.parser.load(chat_params.parser);
      }

      // reasoning budget sampler
      if (!chat_params.thinking_end_tag.empty())
      {
        const llama_vocab *vocab = llama_model_get_vocab(
            llama_get_model(ctx_server.get_llama_context()));

        task.params.sampling.reasoning_budget_tokens = defaults.sampling.reasoning_budget_tokens;
        task.params.sampling.generation_prompt = chat_params.generation_prompt;

        if (!chat_params.thinking_start_tag.empty())
        {
          task.params.sampling.reasoning_budget_start =
              common_tokenize(vocab, chat_params.thinking_start_tag, false, true);
        }
        task.params.sampling.reasoning_budget_end =
            common_tokenize(vocab, chat_params.thinking_end_tag, false, true);
        task.params.sampling.reasoning_budget_forced =
            common_tokenize(vocab, defaults.sampling.reasoning_budget_message + chat_params.thinking_end_tag, false, true);
      }

      rd.post_task({std::move(task)});
    }

    // wait for first result
    console::spinner::start();
    server_task_result_ptr result = rd.next(should_stop);

    console::spinner::stop();
    std::string curr_content;
    bool is_thinking = false;

    while (result)
    {
      if (should_stop())
      {
        break;
      }
      if (result->is_error())
      {
        json err_data = result->to_json();
        if (err_data.contains("message"))
        {
          last_error = err_data["message"].get<std::string>();
          // console::error("Error: %s\n", last_error.c_str());
        }
        else
        {
          last_error = err_data.dump();
          // console::error("Error: %s\n", last_error.c_str());
        }
        return curr_content;
      }
      auto res_partial = dynamic_cast<server_task_result_cmpl_partial *>(result.get());
      if (res_partial)
      {
        out_timings = std::move(res_partial->timings);
        for (const auto &diff : res_partial->oaicompat_msg_diffs)
        {
          if (!diff.content_delta.empty())
          {
            if (is_thinking)
            {
              // console::log("\n[End thinking]\n\n");
              console::set_display(DISPLAY_TYPE_RESET);
              is_thinking = false;
            }
            curr_content += diff.content_delta;
            // console::log("%s", diff.content_delta.c_str());
            console::flush();
          }
          if (!diff.reasoning_content_delta.empty())
          {
            console::set_display(DISPLAY_TYPE_REASONING);
            if (!is_thinking)
            {
              // console::log("[Start thinking]\n");
            }
            is_thinking = true;
            // console::log("%s", diff.reasoning_content_delta.c_str());
            console::flush();
          }
        }
      }
      auto res_final = dynamic_cast<server_task_result_cmpl_final *>(result.get());
      if (res_final)
      {
        out_timings = std::move(res_final->timings);
        break;
      }
      result = rd.next(should_stop);
    }
    g_is_interrupted.store(false);
    // server_response_reader automatically cancels pending tasks upon destruction
    return curr_content;
  }

  // TODO: support remote files in the future (http, https, etc)
  std::string load_input_file(const std::string &fname, bool is_media)
  {
    std::ifstream file(fname, std::ios::binary);
    if (!file)
    {
      return "";
    }
    if (is_media)
    {
      raw_buffer buf;
      buf.assign((std::istreambuf_iterator<char>(file)), std::istreambuf_iterator<char>());
      input_files.push_back(std::move(buf));
      return get_media_marker();
    }
    else
    {
      std::string content((std::istreambuf_iterator<char>(file)), std::istreambuf_iterator<char>());
      return content;
    }
  }

  common_chat_params format_chat()
  {
    auto meta = ctx_server.get_meta();
    auto &chat_params = meta.chat_params;

    auto caps = common_chat_templates_get_caps(chat_params.tmpls.get());

    common_chat_templates_inputs inputs;
    inputs.messages = common_chat_msgs_parse_oaicompat(messages);
    inputs.tools = {}; // TODO
    inputs.tool_choice = COMMON_CHAT_TOOL_CHOICE_NONE;
    inputs.json_schema = ""; // TODO
    inputs.grammar = "";     // TODO
    inputs.use_jinja = chat_params.use_jinja;
    inputs.parallel_tool_calls = caps["supports_parallel_tool_calls"];
    inputs.add_generation_prompt = true;
    inputs.reasoning_format = COMMON_REASONING_FORMAT_DEEPSEEK;
    inputs.force_pure_content = chat_params.force_pure_content;
    inputs.enable_thinking = chat_params.enable_thinking ? common_chat_templates_support_enable_thinking(chat_params.tmpls.get()) : false;

    // Apply chat template to the list of messages
    return common_chat_templates_apply(chat_params.tmpls.get(), inputs);
  }

  kv_dump dump_metadata()
  {
    kv_dump output;
    int count = llama_model_meta_count(model);
    std::string key;
    std::string val;
    std::vector<char> buf(1024);
    int res = 0;
    for (int i = 0; i < count; i++)
    {
      res = llama_model_meta_val_str_by_index(model, i, buf.data(), buf.size());
      if (res < 0)
        continue;
      if (res > buf.size())
      {
        buf.resize(res + 1);
        res = llama_model_meta_val_str_by_index(model, i, buf.data(), buf.size());
      }
      val = std::string(buf.data(), res);
      res = llama_model_meta_key_by_index(model, i, buf.data(), buf.size());
      if (res < 0)
        continue;
      if (res > buf.size())
      {
        buf.resize(res + 1);
        res = llama_model_meta_key_by_index(model, i, buf.data(), buf.size());
      }
      key = std::string(buf.data(), res);
      output.keys.push_back(std::move(key));
      output.vals.push_back(std::move(val));
    }
    return output;
  }

  //////////////////////////////////////////
  //////////////////////////////////////////
  //////////////////////////////////////////

  glue_msg_load_res action_load(const char *req_raw)
  {
    PARSE_REQ(glue_msg_load_req);
    assert(ctx == nullptr);
    std::vector<std::string> &model_paths = req.model_paths.arr;
    bool n_ctx_auto = req.n_ctx_auto.value;

    common_params params;

    // model params
    if (req.use_mmap.not_null())
      params.use_mmap = req.use_mmap.value;
    if (req.use_mlock.not_null())
      params.use_mlock = req.use_mlock.value;
    if (req.n_gpu_layers.not_null())
      params.n_gpu_layers = req.n_gpu_layers.value;

    params.sampling.seed = req.seed.value;
    params.n_ctx = req.n_ctx.value;
    params.cpuparams.n_threads = req.n_threads.value;
    params.cpuparams_batch.n_threads = req.n_threads.value;
    if (req.embeddings.not_null())
      params.embedding = req.embeddings.value;
    // if (req.offload_kqv.not_null())
    //   params.no_kv_offload = !req.offload_kqv.value;
    if (req.n_batch.not_null())
      params.n_batch = req.n_batch.value;
    if (req.n_seq_max.not_null())
      params.n_parallel = req.n_seq_max.value;
    if (req.pooling_type.not_null())
      params.pooling_type = pooling_type_from_str(req.pooling_type.value);
    // context extending: https://github.com/ggerganov/llama.cpp/pull/2054
    if (req.rope_scaling_type.not_null())
      params.rope_scaling_type = rope_scaling_type_from_str(req.rope_scaling_type.value);
    if (req.rope_freq_base.not_null())
      params.rope_freq_base = req.rope_freq_base.value;
    if (req.rope_freq_scale.not_null())
      params.rope_freq_scale = req.rope_freq_scale.value;
    if (req.yarn_ext_factor.not_null())
      params.yarn_ext_factor = req.yarn_ext_factor.value;
    if (req.yarn_attn_factor.not_null())
      params.yarn_attn_factor = req.yarn_attn_factor.value;
    if (req.yarn_beta_fast.not_null())
      params.yarn_beta_fast = req.yarn_beta_fast.value;
    if (req.yarn_beta_slow.not_null())
      params.yarn_beta_slow = req.yarn_beta_slow.value;
    if (req.yarn_orig_ctx.not_null())
      params.yarn_orig_ctx = req.yarn_orig_ctx.value;
    // optimizations
    if (req.cache_type_k.not_null())
      params.cache_type_k = kv_cache_type_from_str(req.cache_type_k.value);
    if (req.cache_type_v.not_null())
      params.cache_type_v = kv_cache_type_from_str(req.cache_type_v.value);
    if (req.swa_full.not_null())
      params.swa_full = req.swa_full.value;
    if (req.flash_attn.not_null())
      params.flash_attn_type = req.flash_attn.value ? LLAMA_FLASH_ATTN_TYPE_AUTO : LLAMA_FLASH_ATTN_TYPE_DISABLED;

    // init threadpool
    ggml_threadpool_params_default(params.cpuparams.n_threads);

    // load model
    llama_backend_init();
    llama_numa_init(params.numa);
    if (!ctx_server.load_model(params))
    {
      glue_msg_load_res res;
      res.success.value = false;
      return res;
    }

    // prepare model paths
    std::vector<const char *> model_paths_ptrs;
    for (auto &path : model_paths)
    {
      model_paths_ptrs.push_back(path.c_str());
    }
    auto metadata = dump_metadata();

    // get EOG tokens
    std::vector<llama_token> list_tokens_eog;
    auto n_vocab = llama_vocab_n_tokens(vocab);
    {
      for (int i = 0; i < n_vocab; i++)
      {
        if (llama_vocab_is_eog(vocab, i))
        {
          list_tokens_eog.push_back(i);
        }
      }
    }

    glue_msg_load_res res;
    res.success.value = true;
    res.n_ctx.value = params.n_ctx;
    res.n_batch.value = llama_n_batch(ctx);
    res.n_ubatch.value = llama_n_ubatch(ctx);
    res.n_vocab.value = n_vocab;
    res.n_ctx_train.value = llama_model_n_ctx_train(model);
    res.n_embd.value = llama_model_n_embd(model);
    res.n_layer.value = llama_model_n_layer(model);
    res.metadata_key.arr = metadata.keys;
    res.metadata_val.arr = metadata.vals;
    res.token_bos.value = llama_vocab_bos(vocab);
    res.token_eos.value = llama_vocab_eos(vocab);
    res.token_eot.value = llama_vocab_eot(vocab);
    res.list_tokens_eog.arr = std::move(list_tokens_eog);
    res.add_bos_token.value = llama_vocab_get_add_bos(vocab) == 1;
    res.add_eos_token.value = llama_vocab_get_add_eos(vocab) == 1;
    res.has_encoder.value = llama_model_has_encoder(model);
    res.token_decoder_start.value = llama_model_decoder_start_token(model);
    return res;
  }
};
