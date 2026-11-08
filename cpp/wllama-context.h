#pragma once

#include <iostream>
#include <vector>
#include <string>
#include <sstream>
#include <stdio.h>
#include <cmath>
#include <fstream>
#include <unordered_map>
#include <algorithm>

#include "llama.h"
#include "common.h"
#include "sampling.h"
#include "chat.h"
#include "fit.h"
#include "log.h"
#include "download.h"
#include "wllama.h"

#include "server-context.h"
#include "server-decision.h"
#include "server-queue.h"
#include "server-schema.h"
// realtime_conversation is private to this file and its run() needs a thread, so wllama includes it and calls the steps itself
#include "server-realtime.cpp"

#include "ggml-cpu.h"
#include "ggml-backend.h"

#include "glue.hpp"

#ifdef WLLAMA_TEST_BACKEND
int main_test_backend_ops(int argc, char **argv);
#else
int main_test_backend_ops(int, char **)
{
  fprintf(stderr, "@@ERROR@@test-backend-ops is not enabled, please refer to README-dev.md for how to build it\n");
  return -1000;
}
#endif

#define PARSE_REQ(msg_typename) \
  msg_typename req;             \
  glue_inbuf inbuf(req_raw);    \
  req.handler.deserialize(inbuf);

// for debugging
enum TEST_STACK_TRACE
{
  TEST_STACK_TRACE_NONE = 0,
  TEST_STACK_TRACE_ABORT = 1,
  TEST_STACK_TRACE_OOB = 2,
};
static TEST_STACK_TRACE test_stack_trace = TEST_STACK_TRACE_NONE;
extern "C" void __real_abort(void);
extern "C" void __wrap_abort(void)
{
  char buf[4096];
  emscripten_get_callstack(EM_LOG_JS_STACK | EM_LOG_NO_PATHS, buf, sizeof(buf));
  for (size_t i = 0; i < sizeof(buf); i++)
  {
    if (buf[i] == '\n')
      buf[i] = '|';
  }
  fprintf(stderr, "@@STACK@@%s\n", buf);
  fflush(stderr);
  __real_abort();
}

// graphs with no pool (clip, draft model) use this pool. A second pool needs more workers, and a new worker cannot start while the main thread waits at a barrier
static struct ggml_threadpool *shared_threadpool = nullptr;

extern "C" struct ggml_threadpool *__real_ggml_threadpool_new(struct ggml_threadpool_params *params);
extern "C" struct ggml_threadpool *__wrap_ggml_threadpool_new(struct ggml_threadpool_params *params)
{
  shared_threadpool = __real_ggml_threadpool_new(params);
  return shared_threadpool;
}

extern "C" void __real_ggml_threadpool_free(struct ggml_threadpool *threadpool);
extern "C" void __wrap_ggml_threadpool_free(struct ggml_threadpool *threadpool)
{
  if (threadpool == shared_threadpool)
    shared_threadpool = nullptr;
  __real_ggml_threadpool_free(threadpool);
}

// the mtmd context of the server is private, TTS and realtime need it
static mtmd_context *shared_mctx = nullptr;

extern "C" mtmd_context *__real_mtmd_init_from_file(const char *mmproj_fname, const struct llama_model *text_model, const struct mtmd_context_params ctx_params);
extern "C" mtmd_context *__wrap_mtmd_init_from_file(const char *mmproj_fname, const struct llama_model *text_model, const struct mtmd_context_params ctx_params)
{
  shared_mctx = __real_mtmd_init_from_file(mmproj_fname, text_model, ctx_params);
  return shared_mctx;
}

static bool force_single_thread = false;
extern "C" struct ggml_cplan __real_ggml_graph_plan(
    const struct ggml_cgraph *cgraph,
    int n_threads,
    struct ggml_threadpool *threadpool);

extern "C" struct ggml_cplan __wrap_ggml_graph_plan(
    const struct ggml_cgraph *cgraph,
    int n_threads,
    struct ggml_threadpool *threadpool)
{
  return __real_ggml_graph_plan(cgraph, force_single_thread ? 1 : n_threads, threadpool ? threadpool : shared_threadpool);
}

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
  // legacy values
  if (s == "LLAMA_POOLING_TYPE_UNSPECIFIED")
    return LLAMA_POOLING_TYPE_UNSPECIFIED;
  if (s == "LLAMA_POOLING_TYPE_NONE")
    return LLAMA_POOLING_TYPE_NONE;
  if (s == "LLAMA_POOLING_TYPE_MEAN")
    return LLAMA_POOLING_TYPE_MEAN;
  if (s == "LLAMA_POOLING_TYPE_CLS")
    return LLAMA_POOLING_TYPE_CLS;
  // new values
  if (s == "unspecified")
    return LLAMA_POOLING_TYPE_UNSPECIFIED;
  if (s == "none")
    return LLAMA_POOLING_TYPE_NONE;
  if (s == "mean")
    return LLAMA_POOLING_TYPE_MEAN;
  if (s == "cls")
    return LLAMA_POOLING_TYPE_CLS;
  if (s == "last")
    return LLAMA_POOLING_TYPE_LAST;
  if (s == "rank")
    return LLAMA_POOLING_TYPE_RANK;
  // for internal wllama testing
  if (s == "test_stack_trace_abort")
  {
    test_stack_trace = TEST_STACK_TRACE_ABORT;
    return LLAMA_POOLING_TYPE_NONE;
  }
  if (s == "test_stack_trace_oob")
  {
    test_stack_trace = TEST_STACK_TRACE_OOB;
    return LLAMA_POOLING_TYPE_NONE;
  }
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

inline static common_reasoning_format reasoning_format_from_str(const std::string &s)
{
  if (s == "none")
    return COMMON_REASONING_FORMAT_NONE;
  if (s == "deepseek-legacy")
    return COMMON_REASONING_FORMAT_DEEPSEEK_LEGACY;
  if (s == "deepseek")
    return COMMON_REASONING_FORMAT_DEEPSEEK;
  throw std::runtime_error("Invalid reasoning format: " + s);
}

inline static llama_model_kv_override parse_kv_override(const std::string &key, const std::string &val_str)
{
  llama_model_kv_override kvo;
  strncpy(kvo.key, key.c_str(), sizeof(kvo.key) - 1);
  kvo.key[sizeof(kvo.key) - 1] = '\0';
  auto colon = val_str.find(':');
  if (colon == std::string::npos)
    throw std::runtime_error("Invalid kv_override value, expected TYPE:value: " + val_str);
  const std::string type_str = val_str.substr(0, colon);
  const std::string value_str = val_str.substr(colon + 1);
  if (type_str == "int")
  {
    kvo.tag = LLAMA_KV_OVERRIDE_TYPE_INT;
    kvo.val_i64 = std::stoll(value_str);
  }
  else if (type_str == "float")
  {
    kvo.tag = LLAMA_KV_OVERRIDE_TYPE_FLOAT;
    kvo.val_f64 = std::stod(value_str);
  }
  else if (type_str == "bool")
  {
    kvo.tag = LLAMA_KV_OVERRIDE_TYPE_BOOL;
    if (value_str == "true" || value_str == "1")
    {
      kvo.val_bool = true;
    }
    else if (value_str == "false" || value_str == "0")
    {
      kvo.val_bool = false;
    }
    else
    {
      throw std::runtime_error("Invalid bool value for kv_override: " + value_str);
    }
  }
  else if (type_str == "str")
  {
    kvo.tag = LLAMA_KV_OVERRIDE_TYPE_STR;
    strncpy(kvo.val_str, value_str.c_str(), sizeof(kvo.val_str) - 1);
    kvo.val_str[sizeof(kvo.val_str) - 1] = '\0';
  }
  else
  {
    throw std::runtime_error("Invalid kv_override type: " + type_str);
  }
  return kvo;
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
static ggml_log_level log_level = GGML_LOG_LEVEL_INFO;

// realtime code waits for its result inside next(); no server thread here, so next() must run the queue itself
static bool run_queue_in_next = false;
struct run_queue_scope
{
  run_queue_scope() { run_queue_in_next = true; }
  ~run_queue_scope() { run_queue_in_next = false; }
};

// no socket here: client events come from action_realtime_send, server events wait in the list until the next poll
struct realtime_conn : server_http_context::ws_conn
{
  std::vector<std::string> events;
  bool open = true;

  int read(std::string &) override { return 0; }
  bool send(const std::string &msg) override
  {
    events.push_back(msg);
    return true;
  }
  bool send_binary(const std::string &) override { return false; }
  bool is_open() const override { return open; }
  void close() override { open = false; }
};

struct realtime_session
{
  realtime_conn conn;

  virtual ~realtime_session() = default;
  virtual const std::string &get_id() const = 0;
  virtual json session_json() const = 0;
  virtual void handle(const std::string &msg) = 0;
  virtual void on_result(server_task_result &result) = 0;

  void send_event(json ev)
  {
    ev["event_id"] = "event_" + random_string();
    conn.send(safe_json_to_str(ev));
  }

  void send_error(const std::string &message, const std::string &type)
  {
    send_event({
        {"type", "error"},
        {"error", {{"type", type}, {"code", nullptr}, {"message", message}, {"param", nullptr}}},
    });
  }

  // as a json array, empty string if there is no event
  std::string take_events()
  {
    if (conn.events.empty())
    {
      return "";
    }
    std::string out = "[";
    for (size_t i = 0; i < conn.events.size(); i++)
    {
      out += (i > 0 ? "," : "") + conn.events[i];
    }
    conn.events.clear();
    return out + "]";
  }
};

// models that answer after the audio, see server_realtime_conversation()
struct realtime_session_conversation : realtime_session
{
  common_params_sampling sampling;
  mtmd_helper_cap cap;
  server_realtime_deps deps;
  realtime_conversation conversation;

  realtime_session_conversation(server_response_reader &rd, const server_context_meta &meta, const common_params_sampling &sampling, const mtmd_helper_cap &cap, const llama_vocab *vocab)
      : sampling(sampling), cap(cap), deps{rd.queue_tasks, rd.queue_results, meta, this->sampling, shared_mctx, this->cap, vocab}, conversation(deps, conn)
  {
    conversation.input_rate = mtmd_get_audio_sample_rate(shared_mctx);
    if (cap.audio_gen)
    {
      conversation.output_rate = mtmd_gen_audio_get_info(shared_mctx).sample_rate;
    }
  }

  const std::string &get_id() const override { return conversation.id; }
  json session_json() const override { return conversation.session_json(); }
  void handle(const std::string &msg) override { conversation.handle(msg); }
  void on_result(server_task_result &result) override
  {
    if (auto *res = dynamic_cast<server_task_result_realtime *>(&result))
    {
      conversation.on_result(*res);
    }
  }
};

// models that write the text while they listen, same events as the /v1/realtime route of llama-server
struct realtime_session_live : realtime_session
{
  server_queue &queue_tasks;
  server_response &queue_results;
  std::string id = "sess_" + random_string();
  std::string model_name;
  int sample_rate;

  // one item per committed audio buffer, deltas go to the current item
  std::string item_id = "item_" + random_string();
  std::string item_prev;
  std::string item_text;
  bool input_ended = false;

  realtime_session_live(server_response_reader &rd, const std::string &model_name, int sample_rate)
      : queue_tasks(rd.queue_tasks), queue_results(rd.queue_results), model_name(model_name), sample_rate(sample_rate) {}

  const std::string &get_id() const override { return id; }

  json session_json() const override
  {
    return {
        {"id", id},
        {"object", "realtime.transcription_session"},
        {"type", "transcription"},
        {"audio", {{"input", {
                                 {"format", {{"type", "audio/pcm"}, {"rate", sample_rate}}},
                                 {"transcription", {{"model", model_name}}},
                                 {"turn_detection", nullptr},
                                 {"noise_reduction", nullptr},
                             }}}},
    };
  }

  bool control(server_control_action action, std::vector<float> &&audio = {})
  {
    server_response_reader ctl(queue_tasks, queue_results, 1);
    server_task task(SERVER_TASK_TYPE_CONTROL);
    task.id = ctl.get_new_id();
    task.params.control_action = action;
    task.params.control_cmpl_id = id;
    task.params.control_audio = std::move(audio);
    ctl.post_task(std::move(task));
    auto result = ctl.next([]()
                           { return false; });
    auto *res = result ? dynamic_cast<server_task_result_control *>(result.get()) : nullptr;
    return res && res->success;
  }

  void send_completed()
  {
    send_event({
        {"type", "conversation.item.input_audio_transcription.completed"},
        {"item_id", item_id},
        {"content_index", 0},
        {"transcript", item_text},
    });
    item_text.clear();
  }

  void handle(const std::string &msg) override
  {
    json ev;
    try
    {
      ev = json::parse(msg);
    }
    catch (const std::exception &)
    {
      send_error("invalid JSON", "invalid_request_error");
      return;
    }
    const std::string type = json_value(ev, "type", std::string());
    if (type == "session.update" || type == "transcription_session.update")
    {
      send_event({{"type", "session.updated"}, {"session", session_json()}});
    }
    else if (type == "input_audio_buffer.append")
    {
      std::string pcm;
      try
      {
        pcm = base64::decode(json_value(ev, "audio", std::string()));
      }
      catch (const std::exception &)
      {
        send_error("invalid base64 audio", "invalid_request_error");
        return;
      }
      std::vector<float> samples(pcm.size() / 2);
      for (size_t i = 0; i < samples.size(); i++)
      {
        int16_t v;
        memcpy(&v, pcm.data() + 2 * i, sizeof(v));
        samples[i] = v / 32768.0f;
      }
      if (!control(SERVER_CONTROL_ACTION_AUDIO_APPEND, std::move(samples)))
      {
        send_error("failed to append audio", "server_error");
      }
    }
    else if (type == "input_audio_buffer.commit")
    {
      // "final": true ends the input, the last item is completed when the stream drains
      if (json_value(ev, "final", false))
      {
        input_ended = true;
        control(SERVER_CONTROL_ACTION_AUDIO_END);
        return;
      }
      if (!control(SERVER_CONTROL_ACTION_AUDIO_COMMIT))
      {
        send_error("failed to commit audio", "server_error");
        return;
      }
      send_event({
          {"type", "input_audio_buffer.committed"},
          {"previous_item_id", item_prev.empty() ? json(nullptr) : json(item_prev)},
          {"item_id", item_id},
      });
    }
    else if (type == "input_audio_buffer.clear")
    {
      send_event({{"type", "input_audio_buffer.cleared"}});
    }
    else
    {
      send_error("unknown event type: " + type, "invalid_request_error");
    }
  }

  void on_result(server_task_result &result) override
  {
    if (auto *partial = dynamic_cast<server_task_result_cmpl_partial *>(&result))
    {
      if (partial->audio_commit_drained)
      {
        // complete the committed item and open the next one
        send_completed();
        item_prev = item_id;
        item_id = "item_" + random_string();
      }
      else if (!partial->content.text.empty())
      {
        item_text += partial->content.text;
        send_event({
            {"type", "conversation.item.input_audio_transcription.delta"},
            {"item_id", item_id},
            {"content_index", 0},
            {"delta", partial->content.text},
        });
      }
    }
    else if (result.is_stop())
    {
      send_completed();
    }
  }
};

struct wllama_context
{
  server_context ctx_server;
  llama_context *ctx = nullptr;
  const llama_model *model = nullptr;
  const llama_vocab *vocab = nullptr;
  common_params params;

  std::function<bool()> should_stop = []()
  { return false; };
  // one reader per in-flight request, keyed by req_id; erasing an entry cancels its unfinished tasks
  std::unordered_map<int, std::unique_ptr<server_response_reader>> readers;
  int next_req_id = 1;
  std::unique_ptr<const server_context_meta> meta;

  // the one inside ctx_server is private, so we keep our own copy
  server_decision_context decision;
  struct decision_request
  {
    std::vector<server_decision_question> questions;
    std::vector<std::vector<float>> scores; // indexed by task index
    int32_t n_tokens = 0;
  };
  std::unordered_map<int, decision_request> decision_reqs; // keyed by req_id
  std::unordered_map<int, std::vector<json>> embedding_reqs; // keyed by req_id
  std::unordered_map<int, std::unique_ptr<realtime_session>> realtime_sessions; // keyed by req_id

  mtmd_helper_cap mtmd_cap{};
  llama_tokens asr_prompt; // prompt of the "asr" chat template, for live ASR

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

  void create_completion_task(server_response_reader &rd, std::string &req_raw, std::vector<raw_buffer> &files, bool is_chat, bool is_asr)
  {
    json body = json::parse(req_raw);
    task_response_type res_type = TASK_RESPONSE_TYPE_OAI_CMPL;
    common_chat_session chat_session;

    if (is_asr)
    {
      if (files.size() != 1)
      {
        throw app_exception("Transcription needs exactly one audio file");
      }
      std::map<std::string, uploaded_file> in_files;
      in_files["file"].data = std::move(files[0]);
      body = convert_transcriptions_to_chatcmpl(body, meta->chat_params.tmpls.get(), in_files, files);
    }

    if (is_chat || is_asr)
    {
      std::vector<raw_buffer> dummy_files; // unused
      json body_parsed = oaicompat_chat_params_parse(
          vocab,
          body,
          meta->chat_params,
          dummy_files,
          chat_session);
      body = std::move(body_parsed);
      res_type = is_asr ? TASK_RESPONSE_TYPE_OAI_ASR : TASK_RESPONSE_TYPE_OAI_CHAT;
    }

    {
      const auto &prompt = body.at("prompt");

      // TODO: reduce some copies here in the future
      server_task task = server_task(SERVER_TASK_TYPE_COMPLETION);
      task.id = rd.get_new_id();
      task.index = 0;
      task.params = server_schema::eval_llama_cmpl_schema(
          vocab,
          params,
          meta->logit_bias_eog,
          body);
      task.params.res_type = res_type;
      task.cli_prompt = prompt;
      task.cli_files = files;
      task.cli = true;
      task.apply_chat_session(chat_session);

      std::vector<server_task> tasks;
      tasks.push_back(std::move(task));
      rd.post_tasks(std::move(tasks), chat_session);
    }
  }

  int register_reader(std::unique_ptr<server_response_reader> rd)
  {
    const int req_id = next_req_id++;
    readers.emplace(req_id, std::move(rd));
    return req_id;
  }

  std::pair<server_task_result_ptr, bool> get_next_result(server_response_reader &rd)
  {
    server_task_result_ptr result = rd.next(should_stop);
    if (result)
    {
      const bool is_error = result->is_error();
      return {std::move(result), is_error};
    }
    else
    {
      return {nullptr, false};
    }
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

    llama_log_set(common_log_default_callback, nullptr);

    assert(ctx == nullptr);
    std::vector<std::string> &model_paths = req.model_paths.arr;
    bool n_ctx_auto = req.n_ctx_auto.value;

    assert(model_paths.size() > 0);
    params.model.path = model_paths[0];

    if (req.log_level.not_null())
      log_level = static_cast<ggml_log_level>(req.log_level.value);

    // mmproj params
    if (req.mmproj_path.not_null())
      params.mmproj.path = req.mmproj_path.value;
    if (req.image_min_tokens.not_null())
      params.image_min_tokens = req.image_min_tokens.value;
    if (req.image_max_tokens.not_null())
      params.image_max_tokens = req.image_max_tokens.value;

    // model params
    // load_mode replaces the old use_mmap/use_mlock flags, keep it on auto if neither is given
    if (req.use_mmap.not_null() || req.use_mlock.not_null())
    {
      const bool use_mmap = req.use_mmap.not_null() ? req.use_mmap.value : true;
      const bool use_mlock = req.use_mlock.not_null() ? req.use_mlock.value : false;
      params.load_mode = use_mmap
                             ? (use_mlock ? LLAMA_LOAD_MODE_MMAP_MLOCK : LLAMA_LOAD_MODE_MMAP)
                             : (use_mlock ? LLAMA_LOAD_MODE_MLOCK : LLAMA_LOAD_MODE_NONE);
    }
    if (req.n_gpu_layers.not_null())
      params.n_gpu_layers = req.n_gpu_layers.value;
    if (req.model_alias.not_null())
      params.model_alias.insert(req.model_alias.value);

    params.n_ctx = req.n_ctx.value;
    params.cpuparams.n_threads = req.n_threads.value;
    params.cpuparams_batch.n_threads = req.n_threads.value;
    if (req.embeddings.not_null())
      params.embedding = req.embeddings.value;
    if (req.n_batch.not_null())
      params.n_batch = req.n_batch.value;
    if (req.n_ubatch.not_null())
      params.n_ubatch = req.n_ubatch.value;
    if (req.n_parallel.not_null())
      params.n_parallel = req.n_parallel.value;
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
    if (req.warmup.not_null())
      params.warmup = req.warmup.value;

    // optimizations
    if (req.cache_type_k.not_null())
      params.cache_type_k = kv_cache_type_from_str(req.cache_type_k.value);
    if (req.cache_type_v.not_null())
      params.cache_type_v = kv_cache_type_from_str(req.cache_type_v.value);
    if (req.kv_unified.not_null())
      params.kv_unified = req.kv_unified.value;
    if (req.flash_attn.not_null())
      params.flash_attn_type = req.flash_attn.value ? LLAMA_FLASH_ATTN_TYPE_AUTO : LLAMA_FLASH_ATTN_TYPE_DISABLED;
    if (req.swa_full.not_null())
      params.swa_full = req.swa_full.value;
    if (req.n_ctx_checkpoints.not_null())
      params.n_ctx_checkpoints = req.n_ctx_checkpoints.value;
    if (req.checkpoint_min_step.not_null())
      params.checkpoint_min_step = req.checkpoint_min_step.value;

    // template params
    if (req.chat_template.not_null())
      params.chat_template = req.chat_template.value;
    if (req.jinja.not_null())
      params.use_jinja = req.jinja.value;
    if (req.reasoning.not_null())
    {
      if (req.reasoning.value)
      {
        params.enable_reasoning = 1;
        params.default_template_kwargs["enable_thinking"] = "true";
      }
      else
      {
        params.enable_reasoning = 0;
        params.default_template_kwargs["enable_thinking"] = "false";
      }
    }
    if (req.default_template_kwargs_keys.not_null() && req.default_template_kwargs_vals.not_null())
    {
      auto &keys = req.default_template_kwargs_keys.arr;
      auto &vals = req.default_template_kwargs_vals.arr;
      if (keys.size() != vals.size())
      {
        throw app_exception("default_template_kwargs_keys and default_template_kwargs_vals must have the same length");
      }
      for (size_t i = 0; i < keys.size(); i++)
      {
        params.default_template_kwargs[keys[i]] = vals[i];
      }
    }

    // GPU
    if (req.no_kv_offload.not_null())
      params.no_kv_offload = req.no_kv_offload.value;
    if (req.mmproj_offload.not_null())
      params.mmproj_use_gpu = req.mmproj_offload.value;

    // batch / context scheduling
    if (req.cont_batching.not_null())
      params.cont_batching = req.cont_batching.value;
    if (req.n_keep.not_null())
      params.n_keep = req.n_keep.value;
    if (req.ctx_shift.not_null())
      params.ctx_shift = req.ctx_shift.value;
    if (req.cache_idle_slots.not_null())
      params.cache_idle_slots = req.cache_idle_slots.value;
    if (req.n_cache_reuse.not_null())
      params.n_cache_reuse = req.n_cache_reuse.value;

    // lora
    if (req.lora_paths.not_null())
    {
      const auto &paths = req.lora_paths.arr;
      const auto &scales = req.lora_scales.arr;
      if (!scales.empty() && scales.size() != paths.size())
        throw app_exception("lora_paths and lora_scales must have the same length");
      for (size_t i = 0; i < paths.size(); i++)
      {
        common_adapter_lora_info info;
        info.path = paths[i];
        info.scale = scales.empty() ? 1.0f : scales[i];
        params.lora_adapters.push_back(std::move(info));
      }
    }
    if (req.lora_init_without_apply.not_null())
      params.lora_init_without_apply = req.lora_init_without_apply.value;

    // speculative decoding
    if (req.spec_draft_model.not_null())
      params.speculative.draft.mparams.path = req.spec_draft_model.value;
    if (req.spec_draft_ngl.not_null())
      params.speculative.draft.n_gpu_layers = req.spec_draft_ngl.value;
    if (req.spec_draft_n_max.not_null())
      params.speculative.draft.n_max = req.spec_draft_n_max.value;
    if (req.spec_draft_n_min.not_null())
      params.speculative.draft.n_min = req.spec_draft_n_min.value;
    if (req.spec_draft_p_min.not_null())
      params.speculative.draft.p_min = req.spec_draft_p_min.value;
    if (req.spec_draft_threads.not_null())
      params.speculative.draft.cpuparams.n_threads = req.spec_draft_threads.value;
    if (req.spec_draft_threads_batch.not_null())
      params.speculative.draft.cpuparams_batch.n_threads = req.spec_draft_threads_batch.value;

    // kv overrides
    if (req.kv_overrides_keys.not_null() && req.kv_overrides_vals.not_null())
    {
      const auto &keys = req.kv_overrides_keys.arr;
      const auto &vals = req.kv_overrides_vals.arr;
      if (keys.size() != vals.size())
        throw app_exception("kv_overrides_keys and kv_overrides_vals must have the same length");
      for (size_t i = 0; i < keys.size(); i++)
        params.kv_overrides.push_back(parse_kv_override(keys[i], vals[i]));
    }

    // reasoning
    if (req.reasoning_budget_tokens.not_null())
      params.sampling.reasoning_budget_tokens = req.reasoning_budget_tokens.value;
    if (req.reasoning_budget_message.not_null())
      params.sampling.reasoning_budget_message = req.reasoning_budget_message.value;
    if (req.reasoning_format.not_null())
      params.reasoning_format = reasoning_format_from_str(req.reasoning_format.value);

    // other
    if (req.skip_chat_parsing.not_null())
      params.force_pure_content_parser = req.skip_chat_parsing.value;
    if (req.prefill_assistant.not_null())
      params.prefill_assistant = req.prefill_assistant.value;

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
    decision.init(model);
    mtmd_cap = mtmd_helper_get_capabilities(model, shared_mctx);
    asr_prompt = server_chat_get_asr_prompt(model);
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

    // multimodal
    auto server_meta = ctx_server.get_meta();

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
    res.media_marker.value = get_media_marker();
    res.has_image_input.value = server_meta.chat_params.allow_image;
    res.has_audio_input.value = server_meta.chat_params.allow_audio;
    res.has_audio_output.value = server_meta.has_cap_tts;
    res.has_audio_stream.value = server_meta.has_audio_stream;
    res.has_live_asr.value = server_meta.has_audio_stream && (!mtmd_cap.chat || !asr_prompt.empty());
    res.audio_input_rate.value = server_meta.has_inp_audio ? mtmd_get_audio_sample_rate(shared_mctx) : 0;
    res.audio_output_rate.value = server_meta.has_cap_tts ? mtmd_gen_audio_get_info(shared_mctx).sample_rate : 0;
    res.audio_voices.arr = server_meta.tts_voices;
    return res;
  }

  glue_msg_completion_res action_completion(const char *req_raw)
  {
    PARSE_REQ(glue_msg_completion_req);
    glue_msg_completion_res res;

    // prepare
    auto rd = std::make_unique<server_response_reader>(ctx_server.get_response_reader());
    std::vector<raw_buffer> input_files;
    for (const auto &file : req.files.arr)
    {
      input_files.push_back(file);
    }

    // create completion task and post to the queue
    create_completion_task(*rd, req.data_json.value, input_files, req.is_chat.value, req.is_asr.value);

    res.success.value = true;
    res.req_id.value = register_reader(std::move(rd));
    return res;
  }

  void create_embedding_tasks(server_response_reader &rd, std::string &req_raw, std::vector<raw_buffer> &input_files)
  {
    json body = json::parse(req_raw);

    json prompt;
    if (body.count("input") != 0)
    {
      prompt = body.at("input");
    }
    else if (body.contains("content"))
    {
      prompt = body.at("content");
    }
    else
    {
      throw app_exception("\"input\" or \"content\" must be provided");
    }

    int embd_normalize = 2;
    if (body.count("embd_normalize") != 0)
    {
      embd_normalize = body.at("embd_normalize").get<int>();
    }

    std::vector<server_task> tasks;

    auto new_task = [&]() {
      server_task task = server_task(SERVER_TASK_TYPE_EMBEDDING);
      task.id = rd.get_new_id();
      task.params.res_type = TASK_RESPONSE_TYPE_OAI_EMBD;
      task.params.embd_normalize = embd_normalize;
      return task;
    };

    if (input_files.empty())
    {
      auto tokenized_prompts = tokenize_input_prompts(vocab, nullptr, prompt, true, true, mtmd_helper_init_opt_default());
      for (auto &tokens : tokenized_prompts)
      {
        if (tokens.empty())
        {
          throw app_exception("Input content cannot be empty");
        }
        server_task task = new_task();
        task.tokens = std::move(tokens);
        tasks.push_back(std::move(task));
      }
    }
    else
    {
      llama_context *lctx = ctx_server.get_llama_context();
      const bool has_memory = llama_get_memory(lctx) != nullptr;
      const bool is_last_pooling = llama_pooling_type(lctx) == LLAMA_POOLING_TYPE_LAST;

      // server decodes image in its own batch here, so pooled output has only the text tokens
      if (has_memory && !is_last_pooling && llama_get_causal_attn(lctx))
      {
        throw app_exception("Image embedding on a causal model requires pooling_type = last");
      }

      // cli mode lets the server tokenize with the mtmd context
      const std::string media_marker = get_media_marker();
      auto texts = prompt.is_array() ? prompt.get<std::vector<std::string>>() : std::vector<std::string>{prompt.get<std::string>()};
      size_t i_file = 0;
      for (auto &text : texts)
      {
        server_task task = new_task();
        for (size_t pos = text.find(media_marker); pos != std::string::npos; pos = text.find(media_marker, pos + media_marker.size()))
        {
          if (i_file >= input_files.size())
          {
            throw app_exception("Not enough media files");
          }
          task.cli_files.push_back(std::move(input_files[i_file++]));
        }
        // image can not be the last token here, server needs a text token for the final batch
        if (has_memory && is_last_pooling && string_ends_with(text, media_marker))
        {
          text += "\n";
        }
        task.cli_prompt = std::move(text);
        task.cli = true;
        tasks.push_back(std::move(task));
      }
      if (i_file != input_files.size())
      {
        throw app_exception("Unused media files");
      }
    }

    if (tasks.empty())
    {
      throw app_exception("Input content cannot be empty");
    }

    rd.post_tasks(std::move(tasks));
  }

  glue_msg_embedding_res action_embedding(const char *req_raw)
  {
    PARSE_REQ(glue_msg_embedding_req);
    glue_msg_embedding_res res;

    auto rd = std::make_unique<server_response_reader>(ctx_server.get_response_reader());
    try
    {
      create_embedding_tasks(*rd, req.data_json.value, req.files.arr);
    }
    catch (const std::exception &e)
    {
      res.success.value = false;
      res.message.value = e.what();
      return res;
    }

    res.success.value = true;
    res.req_id.value = register_reader(std::move(rd));
    return res;
  }

  glue_msg_rerank_res action_rerank(const char *req_raw)
  {
    PARSE_REQ(glue_msg_rerank_req);
    glue_msg_rerank_res res;

    json body = json::parse(req.data_json.value);
    if (!body.contains("query") || !body.at("query").is_string())
    {
      throw app_exception("\"query\" must be a string");
    }
    if (!body.contains("document") || !body.at("document").is_string())
    {
      throw app_exception("\"document\" must be a string");
    }
    std::string query = body.at("query");
    std::string document = body.at("document");

    auto rd = std::make_unique<server_response_reader>(ctx_server.get_response_reader());

    auto tokens = format_prompt_rerank(model, vocab, nullptr, query, document, {}, {}, mtmd_helper_init_opt_default());
    server_task task = server_task(SERVER_TASK_TYPE_RERANK);
    task.id = rd->get_new_id();
    task.index = 0;
    task.tokens = std::move(tokens);
    rd->post_task(std::move(task));

    res.success.value = true;
    res.req_id.value = register_reader(std::move(rd));
    return res;
  }

  glue_msg_systemone_res action_systemone(const char *req_raw)
  {
    PARSE_REQ(glue_msg_systemone_req);
    glue_msg_systemone_res res;

    if (decision.type == COMMON_DECISION_TYPE_NONE)
    {
      throw app_exception("This model is not a decision model");
    }

    json body = json::parse(req.data_json.value);
    decision_request dreq;
    dreq.questions = decision.parse_questions(body);

    std::vector<raw_buffer> files;
    const json state = decision.parse_state(body, files);
    if (!files.empty())
    {
      throw app_exception("Image input is not supported for systemone");
    }

    auto rd = std::make_unique<server_response_reader>(ctx_server.get_response_reader());

    // one task per variant of each question
    std::vector<server_task> tasks;
    for (const auto &question : dreq.questions)
    {
      for (size_t variant = 0; variant < decision.n_variants(question); variant++)
      {
        server_task task = server_task(SERVER_TASK_TYPE_DECISION);
        task.id = rd->get_new_id();
        decision.fill_task(state, dreq.questions, question, variant, files, nullptr, mtmd_helper_init_opt_default(), task);
        tasks.push_back(std::move(task));
      }
    }
    dreq.scores.resize(tasks.size());
    if (decision.can_share_prompt())
    {
      tasks = server_decision_group_tasks(std::move(tasks), params.n_parallel);
    }
    rd->post_tasks(std::move(tasks));

    res.success.value = true;
    res.req_id.value = register_reader(std::move(rd));
    decision_reqs.emplace(res.req_id.value, std::move(dreq));
    return res;
  }

  void create_tts_task(server_response_reader &rd, std::string &req_raw, std::vector<raw_buffer> &files)
  {
    if (!meta->has_cap_tts)
    {
      throw app_exception("This model does not support audio generation");
    }
    const auto info = mtmd_gen_audio_get_info(shared_mctx);

    json body = json::parse(req_raw);
    const std::string prompt = json_value(body, "input", std::string());
    if (prompt.empty())
    {
      throw app_exception("\"input\" must be a non-empty string");
    }

    server_task task(SERVER_TASK_TYPE_TTS);
    server_schema::eval_tts_schema(params, task, body);

    // a model with a "tts" chat template speaks the prompt rendered from it
    const std::string lang = json_value(body, "lang", std::string());
    const std::string voice = json_value(body, "voice", std::string());
    const std::string tts_prompt = common_chat_get_tts_prompt(meta->chat_params.tmpls.get(), prompt, lang, voice);
    task.tts_inp.set_prompt(tts_prompt.empty() ? prompt : tts_prompt);
    task.tts_inp.set_lang(lang);
    task.tts_inp.data.stream = task.params.stream;
    task.tts_inp.data.out_type = json_value(body, "response_format", std::string("wav")) == "pcm"
                                     ? MTMD_HELPER_GEN_AUDIO_OUTTYPE_PCM
                                     : MTMD_HELPER_GEN_AUDIO_OUTTYPE_WAV;
    // note: -1 is clamped to 0 by the sampler, it will disable the penalty
    task.params.sampling.penalty_last_n = task.params.n_predict > 0 ? task.params.n_predict : 512;
    task.params.sampling.seed = task.tts_inp.data.seed; // same seed for the codec RNG
    if (task.tts_inp.data.top_k > 0)
    {
      task.params.sampling.top_k = task.tts_inp.data.top_k;
    }
    if (task.tts_inp.data.top_p > 0)
    {
      task.params.sampling.top_p = task.tts_inp.data.top_p;
    }

    // a model with predefined voices only speaks with one of them
    mtmd::bitmap_ptr voice_ref;
    if (!voice.empty() && info.n_voices > 0)
    {
      voice_ref.reset(mtmd_gen_audio_get_voice(shared_mctx, voice.c_str()));
      if (!voice_ref)
      {
        throw app_exception("Unknown voice \"" + voice + "\"");
      }
    }

    if (!files.empty())
    {
      if (voice_ref)
      {
        throw app_exception("\"voice\" and \"speaker_ref\" cannot be used together");
      }
      if (!mtmd_support_audio(shared_mctx))
      {
        throw app_exception("This model has no voice encoder, \"speaker_ref\" cannot be used");
      }
      auto wrapper = mtmd_helper_bitmap_init_from_buf(shared_mctx, files[0].data(), files[0].size(), false, mtmd_helper_init_opt_default());
      if (!wrapper.bitmap)
      {
        throw app_exception("Failed to decode \"speaker_ref\"");
      }
      task.tts_inp.set_speaker_ref(mtmd::bitmap_ptr(wrapper.bitmap));
    }
    else if (voice_ref)
    {
      task.tts_inp.set_speaker_ref(std::move(voice_ref));
    }
    else if (info.n_voices > 0)
    {
      // without a voice prompt the model produces near silence: the first voice is the default
      task.tts_inp.set_speaker_ref(mtmd::bitmap_ptr(mtmd_gen_audio_get_voice(shared_mctx, mtmd_gen_audio_voice_name(shared_mctx, 0))));
    }

    task.id = rd.get_new_id();
    rd.post_task(std::move(task));
  }

  glue_msg_tts_res action_tts(const char *req_raw)
  {
    PARSE_REQ(glue_msg_tts_req);
    glue_msg_tts_res res;

    auto rd = std::make_unique<server_response_reader>(ctx_server.get_response_reader());
    try
    {
      create_tts_task(*rd, req.data_json.value, req.files.arr);
    }
    catch (const std::exception &e)
    {
      res.success.value = false;
      res.message.value = e.what();
      return res;
    }

    res.success.value = true;
    res.req_id.value = register_reader(std::move(rd));
    return res;
  }

  glue_msg_realtime_start_res action_realtime_start(const char *req_raw)
  {
    PARSE_REQ(glue_msg_realtime_start_req);
    glue_msg_realtime_start_res res;

    if (!meta->has_audio_stream)
    {
      res.success.value = false;
      res.message.value = "This model does not support realtime audio";
      return res;
    }

    // do not wait for a free slot: the worker cannot run anything else meanwhile, and a session never frees its slot by itself
    {
      server_response_reader rd_metrics = ctx_server.get_response_reader();
      server_task task(SERVER_TASK_TYPE_METRICS);
      task.id = rd_metrics.get_new_id();
      rd_metrics.post_task(std::move(task));
      run_queue_scope scope;
      auto result = rd_metrics.next(should_stop);
      auto *metrics = result ? dynamic_cast<server_task_result_metrics *>(result.get()) : nullptr;
      if (!metrics || metrics->n_processing_slots + metrics->n_tasks_deferred >= params.n_parallel)
      {
        res.success.value = false;
        res.message.value = "No free slot, wait for a request to finish or close a realtime session first";
        return res;
      }
    }

    json body = json::parse(req.data_json.value);
    common_params_sampling sampling = params.sampling;
    sampling.seed = json_value(body, "seed", sampling.seed);

    auto rd = std::make_unique<server_response_reader>(ctx_server.get_response_reader());

    // same choice as the /v1/realtime route of llama-server
    const bool live_asr = !asr_prompt.empty() && json_value(body, "intent", std::string()) == "transcription";
    const bool is_conversation = mtmd_cap.chat && !live_asr;
    std::unique_ptr<realtime_session> session;
    if (is_conversation)
    {
      auto conversation = std::make_unique<realtime_session_conversation>(*rd, *meta, sampling, mtmd_cap, vocab);
      if (!conversation->conversation.configure())
      {
        res.success.value = false;
        res.message.value = "The chat template does not fit the realtime turn format";
        return res;
      }
      session = std::move(conversation);
    }
    else
    {
      session = std::make_unique<realtime_session_live>(*rd, meta->model_name, mtmd_get_audio_sample_rate(shared_mctx));
    }

    server_task task(is_conversation ? SERVER_TASK_TYPE_REALTIME : SERVER_TASK_TYPE_TRANSCRIBE_RT);
    task.id = rd->get_new_id();
    task.params.sampling = sampling;
    task.params.stream = true;
    task.params.oaicompat_cmpl_id = session->get_id();
    task.params.oaicompat_model = meta->model_name;
    rd->post_task(std::move(task));

    // the slot answers once it holds the session
    {
      run_queue_scope scope;
      auto result = rd->next(should_stop);
      if (!result || result->is_error())
      {
        res.success.value = false;
        res.message.value = result ? json_value(result->to_json(), "message", std::string("server error")) : "server error";
        return res;
      }
    }
    session->send_event({{"type", "session.created"}, {"session", session->session_json()}});

    res.success.value = true;
    res.req_id.value = register_reader(std::move(rd));
    realtime_sessions.emplace(res.req_id.value, std::move(session));
    return res;
  }

  glue_msg_realtime_send_res action_realtime_send(const char *req_raw)
  {
    PARSE_REQ(glue_msg_realtime_send_req);
    glue_msg_realtime_send_res res;

    auto it = realtime_sessions.find(req.req_id.value);
    if (it == realtime_sessions.end())
    {
      res.success.value = false;
      return res;
    }

    run_queue_scope scope;
    try
    {
      it->second->handle(req.data_json.value);
    }
    catch (const std::exception &e)
    {
      it->second->send_error(e.what(), "server_error");
    }

    res.success.value = true;
    return res;
  }

  json format_decision_response(const decision_request &dreq)
  {
    json answers = json::object();
    size_t i_result = 0;
    for (const auto &question : dreq.questions)
    {
      std::vector<std::vector<float>> scores;
      for (size_t variant = 0; variant < decision.n_variants(question); variant++)
      {
        scores.push_back(dreq.scores[i_result++]);
      }
      answers[question.id] = decision.format_answer(question, scores);
    }
    return json{
        {"model", meta->model_name},
        {"answers", answers},
        {"usage", {
                      {"input_tokens", dreq.n_tokens},
                      {"output_tokens", 0},
                  }},
    };
  }

  glue_msg_get_result_res action_get_result(const char *req_raw)
  {
    PARSE_REQ(glue_msg_get_result_req);
    glue_msg_get_result_res res;

    auto it = readers.find(req.req_id.value);
    if (it == readers.end())
    {
      // request already finished or cancelled; report end-of-stream so late polls are harmless
      res.success.value = true;
      res.has_more.value = false;
      res.data_json.value = "";
      res.is_error.value = false;
      return res;
    }
    server_response_reader &rd = *it->second;

    run_loop();
    auto [result, is_error] = get_next_result(rd);
    // the task queue can be empty while the request is still streaming, keep polling until we get the stop result
    const bool has_more = rd.has_next();

    auto session = realtime_sessions.find(req.req_id.value);
    if (session != realtime_sessions.end())
    {
      if (is_error)
      {
        session->second->send_error(json_value(result->to_json(), "message", std::string("server error")), "server_error");
      }
      else if (result)
      {
        try
        {
          session->second->on_result(*result);
        }
        catch (const std::exception &e)
        {
          session->second->send_error(e.what(), "server_error");
        }
      }
      res.success.value = true;
      res.has_more.value = has_more;
      res.data_json.value = session->second->take_events();
      res.is_error.value = false; // errors go out as events
      if (!has_more)
      {
        realtime_sessions.erase(session);
        readers.erase(it);
      }
      return res;
    }

    json data_json;
    if (result)
    {
      if (auto *tts = dynamic_cast<server_task_result_tts *>(result.get()))
      {
        data_json = tts->to_json();
        res.data_raw.buf.assign(tts->audio.begin(), tts->audio.end());
      }
      else if (auto *embd = dynamic_cast<server_task_result_embd *>(result.get()))
      {
        (void)embd;
        // collect all embeddings of a batch and send them in one response
        auto &responses = embedding_reqs[req.req_id.value];
        if (responses.size() <= result->index)
        {
          responses.resize(result->index + 1);
        }
        responses[result->index] = result->to_json();
        if (has_more)
        {
          result = nullptr;
        }
        else
        {
          json body = {{"model", meta->model_name}};
          // TODO: support base64 output
          data_json = format_embeddings_response_oaicompat(body, meta->model_name, responses, false);
        }
      }
      else if (auto *rerank = dynamic_cast<server_task_result_rerank *>(result.get()))
      {
        data_json = json{
            {"score", rerank->score},
            {"tokens_evaluated", rerank->n_tokens},
        };
      }
      else if (auto *dres = dynamic_cast<server_task_result_decision *>(result.get()))
      {
        // only send the answers once all the tasks are done
        auto &dreq = decision_reqs.at(req.req_id.value);
        GGML_ASSERT(dres->index >= 0 && (size_t)dres->index < dreq.scores.size());
        dreq.scores[dres->index] = std::move(dres->scores);
        dreq.n_tokens += dres->n_tokens;
        if (has_more)
        {
          result = nullptr;
        }
        else
        {
          data_json = format_decision_response(dreq);
        }
      }
      else
      {
        // completion result
        data_json = result->to_json();
      }
    }

    res.success.value = true;
    res.has_more.value = has_more;
    res.data_json.value = result ? data_json.dump() : "";
    res.is_error.value = is_error;

    if (!has_more)
    {
      readers.erase(it);
      decision_reqs.erase(req.req_id.value);
      embedding_reqs.erase(req.req_id.value);
    }
    return res;
  }

  glue_msg_cancel_res action_cancel(const char *req_raw)
  {
    PARSE_REQ(glue_msg_cancel_req);
    glue_msg_cancel_res res;

    auto it = readers.find(req.req_id.value);
    if (it != readers.end())
    {
      // reader destructor posts cancel tasks; run one loop iteration to release the slot right away
      readers.erase(it);
      decision_reqs.erase(req.req_id.value);
      embedding_reqs.erase(req.req_id.value);
      realtime_sessions.erase(req.req_id.value);
      run_loop();
    }

    res.success.value = true;
    return res;
  }

  glue_msg_test_backend_ops_res action_test_backend_ops(const char *req_raw)
  {
    PARSE_REQ(glue_msg_test_backend_ops_req);
    glue_msg_test_backend_ops_res res;

    auto &args = req.args.arr;

    std::vector<char *> argv;
    argv.reserve(args.size());
    for (auto &s : args)
    {
      argv.push_back(const_cast<char *>(s.c_str()));
    }

    auto curr_log_lvl = log_level;
    log_level = GGML_LOG_LEVEL_DEBUG;
    force_single_thread = true;
    int retcode = main_test_backend_ops((int)argv.size(), argv.data());
    log_level = curr_log_lvl;    // restore log level
    force_single_thread = false; // restore threading

    res.retcode.value = retcode;
    res.success.value = retcode == 0;

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
  // cli input is already tokenized and cleared, do not tokenize it again on retry
  task.cli = false;
  queue_tasks_deferred.push_back(std::move(task));
}

void server_queue::pop_deferred_task(int id_slot)
{
  if (queue_tasks_deferred.empty())
  {
    return;
  }
  auto it = std::find_if(queue_tasks_deferred.begin(), queue_tasks_deferred.end(), [id_slot](const server_task &task)
                         { return task.id_slot == id_slot; });
  if (it == queue_tasks_deferred.end())
  {
    it = queue_tasks_deferred.begin();
  }
  queue_tasks.emplace_front(std::move(*it));
  queue_tasks_deferred.erase(it);
}

void server_queue::wait_until_no_sleep()
{
  // wllama never enters the sleeping state, so this is a no-op
}

void server_queue::terminate()
{
  running = false;
}

void server_queue::yield_to_queue(std::function<void()> &&work)
{
  // no worker thread in wllama, run the work inline
  work();
}

void server_queue::worker_stop()
{
  // no worker thread in wllama, so this is a no-op
}

void server_response::send(server_task_result_ptr &&result)
{
  if (test_stack_trace == TEST_STACK_TRACE_ABORT)
  {
    LOG_DBG("%s: force abort for testing\n", __func__);
    abort();
  }
  else if (test_stack_trace == TEST_STACK_TRACE_OOB)
  {
    LOG_DBG("%s: force out-of-bounds for testing\n", __func__);
    int *ptr = reinterpret_cast<int *>(0x40000000); // 1GB
    *ptr = 0;
  }
  LOG_DBG("%s\n", __func__);
  queue_results.push_back(std::move(result));
}

void server_response::add_waiting_task_id(int id)
{
  // no-op
}

void server_response::add_waiting_task_ids(const std::unordered_set<int> &id_tasks)
{
  // no-op
}

void server_response::remove_waiting_task_ids(const std::unordered_set<int> &id_tasks)
{
  // drop results that belong to these tasks; no reader will consume them anymore
  queue_results.erase(
      std::remove_if(queue_results.begin(), queue_results.end(),
                     [&](const server_task_result_ptr &res)
                     { return id_tasks.count(res->id) > 0; }),
      queue_results.end());
}

server_task_result_ptr server_response::recv(const std::unordered_set<int> &id_tasks)
{
  // multiple requests can be in-flight, only pick results belonging to this reader
  for (size_t i = 0; i < queue_results.size(); i++)
  {
    if (id_tasks.count(queue_results[i]->id) == 0)
    {
      continue;
    }
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
    // wllama never yields, so the task can never be declined
    GGML_ASSERT(callback_new_task(std::move(task), false));
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
  states.push_back(task_result_state());
  queue_results.add_waiting_task_id(task.id);
  queue_tasks.post(std::move(task), front);
}

void server_response_reader::post_tasks(std::vector<server_task> &&tasks, const common_chat_session &session, bool front)
{
  LOG_DBG("%s\n", __func__);
  GGML_ASSERT(id_tasks.empty() && "post_tasks() can only be called once per reader");
  id_tasks = server_task::get_list_id(tasks);
  states.reserve(tasks.size());
  size_t index = 0;
  for (auto &task : tasks)
  {
    task.index = index++;
    states.push_back(task_result_state(session));
    // for child tasks
    for (auto &child_task : task.child_tasks)
    {
      child_task.index = index++;
      states.push_back(task_result_state(session));
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
  while (!result && run_queue_in_next && !should_stop())
  {
    queue_tasks.start_loop();
    result = queue_results.recv(id_tasks);
  }
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
  if (result && result->is_stop())
  {
    received_count++;
  }
  return result;
}

void server_response_reader::stop()
{
  queue_results.remove_waiting_task_ids(id_tasks);
  if (!has_next())
  {
    // all results already received (or already cancelled), nothing left to cancel
    return;
  }
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

int common_log_get_verbosity_thold(void)
{
  return log_level;
}

void common_log_set_verbosity_thold(int verbosity)
{
  log_level = static_cast<ggml_log_level>(verbosity);
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

    const char *lvl = "@@DEBUG";
    const char *text = msg.data();
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
  return static_cast<int>(level);
}

void common_log_default_callback(enum ggml_log_level level, const char *text, void * /*user_data*/)
{
  auto verbosity = common_get_verbosity(level);
  if (verbosity >= static_cast<int>(log_level))
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
    const common_fit_extra_model *extra,
    ggml_log_level log_level)
{
  return COMMON_PARAMS_FIT_STATUS_FAILURE;
}

std::pair<long, std::vector<char>> common_remote_get_content(const std::string &url,
                                                             const common_remote_params &params)
{
  throw std::runtime_error("common_remote_get_content is not implemented in wllama");
}

#if 1
common_device_memory_data_vec common_get_device_memory_data(
    const char *path_model,
    const llama_model_params *mparams,
    const llama_context_params *cparams,
    std::vector<ggml_backend_dev_t> &devs,
    uint32_t &hp_ngl,
    uint32_t &hp_n_ctx_train,
    uint32_t &hp_n_expert,
    ggml_log_level log_level)
{
  throw std::runtime_error("common_get_device_memory_data is not implemented in wllama");
}
#else
std::vector<llama_device_memory_data> common_get_device_memory_data(
    const char *path_model,
    const struct llama_model_params *mparams,
    const struct llama_context_params *cparams,
    std::vector<ggml_backend_dev_t> &devs,
    uint32_t &hp_ngl,
    uint32_t &hp_n_ctx_train,
    uint32_t &hp_n_expert,
    enum ggml_log_level log_level)
{
  throw std::runtime_error("common_get_device_memory_data is not implemented in wllama");
}
#endif
