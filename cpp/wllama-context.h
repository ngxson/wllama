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
#include "fit.h"
#include "log.h"
#include "download.h"
#include "wllama.h"

#include "server-context.h"
#include "server-queue.h"

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

enum display_type
{
  DISPLAY_TYPE_RESET = 0,
  DISPLAY_TYPE_INFO,
  DISPLAY_TYPE_PROMPT,
  DISPLAY_TYPE_REASONING,
  DISPLAY_TYPE_USER_INPUT,
  DISPLAY_TYPE_ERROR
};

static bool has_more_tasks = false;

struct wllama_context
{
  server_context ctx_server;
  llama_context *ctx = nullptr;
  const llama_model *model = nullptr;
  const llama_vocab *vocab = nullptr;
  common_params params;

  std::function<bool()> should_stop = []()
  { return false; };
  std::string last_error;
  // using unique_ptr to allow late initialization
  std::unique_ptr<server_response_reader> rd;
  std::unique_ptr<const server_context_meta> meta;

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

  void create_completion_task(std::string &req_raw, std::vector<raw_buffer> &files, bool is_chat)
  {
    json body = json::parse(req_raw);
    task_response_type res_type = TASK_RESPONSE_TYPE_OAI_CMPL;

    if (is_chat)
    {
      std::vector<raw_buffer> dummy_files; // unused
      json body_parsed = oaicompat_chat_params_parse(
          body,
          meta->chat_params,
          dummy_files);
      body = std::move(body_parsed);
      res_type = TASK_RESPONSE_TYPE_OAI_CHAT;
    }

    {
      const auto &prompt = body.at("prompt");

      // TODO: reduce some copies here in the future
      server_task task = server_task(SERVER_TASK_TYPE_COMPLETION);
      task.id = rd->get_new_id();
      task.index = 0;
      task.params = server_task::params_from_json_cmpl(
          vocab,
          params,
          meta->slot_n_ctx,
          meta->logit_bias_eog,
          body);
      task.params.res_type = res_type;
      task.cli_prompt = prompt;
      task.cli_files = files;
      task.cli = true;

      rd->post_task({std::move(task)});
    }
  }

  std::pair<std::string, bool> get_next_result()
  {
    server_task_result_ptr result = rd->next(should_stop);
    if (result)
      return {result->to_json().dump(), result->is_error()};
    else
      return {"", false};
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

  // returns true if there are more tasks in the queue after this one
  int run_loop()
  {
    ctx_server.start_loop(); // only run one iteration of the generation loop (i.e. generating one token)
    return has_more_tasks;
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

    assert(model_paths.size() > 0);
    params.model.path = model_paths[0];

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
    if (req.chat_template.not_null())
      params.chat_template = req.chat_template.value;
    if (req.jinja.not_null())
      params.use_jinja = req.jinja.value;

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

    LOG_INF("%s", "Model loaded successfully\n");

    ctx = ctx_server.get_llama_context();
    model = llama_get_model(ctx);
    vocab = llama_model_get_vocab(model);
    meta = std::make_unique<server_context_meta>(ctx_server.get_meta());
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

  glue_msg_completion_res action_completion(const char *req_raw)
  {
    PARSE_REQ(glue_msg_completion_req);
    glue_msg_completion_res res;

    // prepare
    rd = std::make_unique<server_response_reader>(ctx_server.get_response_reader());
    last_error = "";
    std::vector<raw_buffer> input_files;
    for (const auto &file : req.files.arr)
    {
      input_files.push_back(file);
    }

    // create completion task and post to the queue
    create_completion_task(req.data_json.value, input_files, req.is_chat.value);

    res.success.value = true;
    return res;
  }

  glue_msg_get_result_res action_get_result(const char *req_raw)
  {
    PARSE_REQ(glue_msg_get_result_req);
    glue_msg_get_result_res res;

    bool has_more = run_loop();
    auto [data_json, is_error] = get_next_result();

    res.success.value = true;
    res.has_more.value = has_more;
    res.data_json.value = data_json;
    res.is_error.value = is_error;
    return res;
  }
};

////////////////////////////
// server_queue

int server_queue::get_new_id()
{
  return id++;
}

int server_queue::post(server_task &&task, bool front)
{
  GGML_ASSERT(task.id != -1);
  // if this is cancel task make sure to clean up pending tasks
  if (task.type == SERVER_TASK_TYPE_CANCEL)
  {
    cleanup_pending_task(task.id_target);
  }
  const int task_id = task.id;
  if (front)
  {
    queue_tasks.push_front(std::move(task));
  }
  else
  {
    queue_tasks.push_back(std::move(task));
  }
  time_last_task = ggml_time_ms();
  return task_id;
}

int server_queue::post(std::vector<server_task> &&tasks, bool front)
{
  for (auto &task : tasks)
  {
    if (task.id == -1)
    {
      task.id = id++;
    }
    // if this is cancel task make sure to clean up pending tasks
    if (task.type == SERVER_TASK_TYPE_CANCEL)
    {
      cleanup_pending_task(task.id_target);
    }
    if (front)
    {
      queue_tasks.push_front(std::move(task));
    }
    else
    {
      queue_tasks.push_back(std::move(task));
    }
  }
  time_last_task = ggml_time_ms();
  return 0;
}

void server_queue::cleanup_pending_task(int id_target)
{
  auto rm_func = [id_target](const server_task &task)
  {
    return task.id == id_target;
  };
  queue_tasks.erase(
      std::remove_if(queue_tasks.begin(), queue_tasks.end(), rm_func),
      queue_tasks.end());
  queue_tasks_deferred.erase(
      std::remove_if(queue_tasks_deferred.begin(), queue_tasks_deferred.end(), rm_func),
      queue_tasks_deferred.end());
}

void server_queue::defer(server_task &&task)
{
  assert(false && "should not be called in wllama");
}

void server_queue::pop_deferred_task(int id_slot)
{
  // no deferred task in wllama, so this is a no-op
}

void server_response::send(server_task_result_ptr &&result)
{
  LOG_DBG("%s\n", __func__);
  queue_results.push_back(std::move(result));
}

void server_response::add_waiting_task_id(int id)
{
  // no-op
}

void server_response::remove_waiting_task_ids(const std::unordered_set<int> &id_tasks)
{
  for (const auto &id_task : id_tasks)
  {
    waiting_task_ids.erase(id_task);
  }
}

server_task_result_ptr server_response::recv(const std::unordered_set<int> &)
{
  for (size_t i = 0; i < queue_results.size(); i++)
  {
    server_task_result_ptr res = std::move(queue_results[i]);
    queue_results.erase(queue_results.begin() + i);
    return res;
  }
  return nullptr;
}

void server_queue::start_loop(int64_t idle_sleep_ms)
{
  while (true)
  {
    if (queue_tasks.empty())
    {
      break;
    }
    server_task task = std::move(queue_tasks.front());
    queue_tasks.pop_front();

    LOG_DBG("processing task, id = %d\n", task.id);
    callback_new_task(std::move(task));
  }
  // all tasks in the current loop is processed, slots data is now ready
  LOG_DBG("%s", "update slots\n");

  // this will run the main inference process for all slots
  callback_update_slots();

  has_more_tasks = !queue_tasks.empty();
}

const char *llama_build_info()
{
  return "wllama";
}

////////////////////////////
// server_response_reader

void server_response_reader::post_task(server_task &&task, bool front)
{
  LOG_DBG("%s\n", __func__);
  GGML_ASSERT(id_tasks.empty() && "post_task() can only be called once per reader");
  GGML_ASSERT(!task.is_parent() && "not supported, use post_tasks() instead");
  task.index = 0;
  id_tasks.insert(task.id);
  states.push_back(task.create_state());
  queue_results.add_waiting_task_id(task.id);
  queue_tasks.post(std::move(task), front);
}

void server_response_reader::post_tasks(std::vector<server_task> &&tasks, bool front)
{
  LOG_DBG("%s\n", __func__);
  GGML_ASSERT(id_tasks.empty() && "post_tasks() can only be called once per reader");
  id_tasks = server_task::get_list_id(tasks);
  states.reserve(tasks.size());
  size_t index = 0;
  for (auto &task : tasks)
  {
    task.index = index++;
    states.push_back(task.create_state());
    // for child tasks
    for (auto &child_task : task.child_tasks)
    {
      child_task.index = index++;
      states.push_back(child_task.create_state());
    }
  }
  GGML_ASSERT(states.size() == id_tasks.size());
  queue_results.add_waiting_task_ids(id_tasks);
  queue_tasks.post(std::move(tasks), front);
}

bool server_response_reader::has_next() const
{
  return !cancelled && received_count < id_tasks.size();
}

// return nullptr if should_stop() is true before receiving a result
// note: if one error is received, it will stop further processing and return error result
server_task_result_ptr server_response_reader::next(const std::function<bool()> &should_stop)
{
  LOG_DBG("%s\n", __func__);
  auto result = queue_results.recv(id_tasks);
  if (result && !states.empty())
  {
    // update the generation state if needed
    LOG_DBG("%s: update result\n", __func__);
    const size_t idx = result->index;
    GGML_ASSERT(idx < states.size());
    result->update(states[idx]);
  }
  if (result && result->is_error())
  {
    LOG_DBG("%s: received error result, stop further processing\n", __func__);
    stop();
  }
  return result;
}

void server_response_reader::stop()
{
  queue_results.remove_waiting_task_ids(id_tasks);
  cancelled = true;
  std::vector<server_task> cancel_tasks;
  cancel_tasks.reserve(id_tasks.size());
  for (const auto &id_task : id_tasks)
  {
    LOG_DBG("cancel task, id_task = %d\n", id_task);
    server_task task(SERVER_TASK_TYPE_CANCEL);
    task.id_target = id_task;
    cancel_tasks.push_back(std::move(task));
  }
  // push to beginning of the queue, so it has highest priority
  queue_tasks.post(std::move(cancel_tasks), true);
}

////////////////////////////
// common_log

int common_log_verbosity_thold = LOG_LEVEL_INFO;

int common_log_get_verbosity_thold(void)
{
  return common_log_verbosity_thold;
}

void common_log_set_verbosity_thold(int verbosity)
{
  common_log_verbosity_thold = verbosity;
}

struct common_log
{
  void add(enum ggml_log_level level, const char *fmt, va_list args)
  {
    static std::vector<char> msg;

    const size_t n = vsnprintf(msg.data(), msg.size(), fmt, args);
    if (n >= msg.size())
    {
      msg.resize(n + 1);
      // cannot use args twice, so make a copy in case we need to expand the buffer
      va_list args_copy;
      va_copy(args_copy, args);
      vsnprintf(msg.data(), msg.size(), fmt, args_copy);
    }

    printf("%s", msg.data());
  }
};

struct common_log *common_log_init()
{
  return new common_log;
}

struct common_log *common_log_main()
{
  static struct common_log log;
  return &log;
}

void common_log_add(struct common_log *log, enum ggml_log_level level, const char *fmt, ...)
{
  va_list args;
  va_start(args, fmt);
  log->add(level, fmt, args);
  va_end(args);
}

static int common_get_verbosity(enum ggml_log_level level)
{
  switch (level)
  {
  case GGML_LOG_LEVEL_DEBUG:
    return LOG_LEVEL_DEBUG;
  case GGML_LOG_LEVEL_INFO:
    return LOG_LEVEL_INFO;
  case GGML_LOG_LEVEL_WARN:
    return LOG_LEVEL_WARN;
  case GGML_LOG_LEVEL_ERROR:
    return LOG_LEVEL_ERROR;
  case GGML_LOG_LEVEL_CONT:
    return LOG_LEVEL_INFO; // same as INFO
  case GGML_LOG_LEVEL_NONE:
  default:
    return LOG_LEVEL_OUTPUT;
  }
}

void common_log_default_callback(enum ggml_log_level level, const char *text, void * /*user_data*/)
{
  auto verbosity = common_get_verbosity(level);
  if (verbosity <= common_log_verbosity_thold)
  {
    common_log_add(common_log_main(), level, "%s", text);
  }
}

enum common_params_fit_status common_fit_params(
    const char *path_model,
    llama_model_params *mparams,
    llama_context_params *cparams,
    float *tensor_split,
    llama_model_tensor_buft_override *tensor_buft_overrides,
    size_t *margins,
    uint32_t n_ctx_min,
    ggml_log_level log_level)
{
  return COMMON_PARAMS_FIT_STATUS_FAILURE;
}

std::pair<long, std::vector<char>> common_remote_get_content(const std::string &url,
                                                             const common_remote_params &params)
{
  throw std::runtime_error("common_remote_get_content is not implemented in wllama");
}
