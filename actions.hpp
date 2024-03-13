#include <iostream>
#include <vector>
#include <string>
#include <sstream>
#include <stdio.h>

#include "llama.h"
#include "json.hpp"
#include "common.h"

/**
 * CCAMA project - A low-level llama.cpp API via JSON
 * https://github.com/ngxson/ccama
 */

using json = nlohmann::json;

#define LOG_JSON(str, ...)                                \
  {                                                       \
    char output[1024];                                    \
    sprintf(output, str.c_str(), __VA_ARGS__);            \
    send_response(json{{"debug" : std::string(output)}}); \
  }

#define ACTION(name)          \
  if (action == #name)        \
  {                           \
    action_##name(app, body); \
    continue;                 \
  }

struct app_t
{
  llama_model *model;
  llama_context *ctx;
  struct llama_sampling_context *ctx_sampling = nullptr;
  llama_batch batch = llama_batch_init(512, 0, 1);
  std::vector<llama_token> tokens;
  // group attention
  int32_t ga_i = 0; // group-attention state
  int32_t ga_n = 0; // group-attention factor
  int32_t ga_w = 0; // group-attention width
  int32_t n_past_self_extension = 0;
};

inline void send_response(json data)
{
  std::cout << data.dump() << "\n";
}

inline std::vector<unsigned int> convert_string_to_int_arr(std::string &input)
{
  std::vector<unsigned int> output;
  unsigned char *input_ptr = (unsigned char *)input.data();
  output.resize(input.length());
  for (size_t i = 0; i < input.length(); i++)
  {
    output[i] = static_cast<unsigned int>(input_ptr[i]);
  }
  return std::move(output);
}

inline static ggml_type kv_cache_type_from_str(const std::string &s)
{
  if (s == "f32")
  {
    return GGML_TYPE_F32;
  }
  if (s == "f16")
  {
    return GGML_TYPE_F16;
  }
  if (s == "q8_0")
  {
    return GGML_TYPE_Q8_0;
  }
  if (s == "q4_0")
  {
    return GGML_TYPE_Q4_0;
  }
  if (s == "q4_1")
  {
    return GGML_TYPE_Q4_1;
  }
  if (s == "q5_0")
  {
    return GGML_TYPE_Q5_0;
  }
  if (s == "q5_1")
  {
    return GGML_TYPE_Q5_1;
  }

  throw std::runtime_error("Invalid cache type: " + s);
}

//////////////////////////////////////////
//////////////////////////////////////////
//////////////////////////////////////////

json action_load(app_t &app, json &body)
{
  std::string model_path = body["model_path"];
  auto mparams = llama_model_default_params();
  auto cparams = llama_context_default_params();
  cparams.seed = body["seed"];
  cparams.n_ctx = body["n_ctx"];
  cparams.n_threads = body["n_threads"];
  cparams.n_threads_batch = cparams.n_threads;
  if (body.count("embeddings") > 0)
    cparams.embeddings = body["embeddings"];
  if (body.count("offload_kqv") > 0)
    cparams.offload_kqv = body["offload_kqv"];
  if (body.count("n_batch") > 0)
    cparams.n_batch = body["n_batch"];
  if (body.count("n_seq_max") > 0)
    cparams.n_seq_max = body["n_seq_max"];
  // context extending: https://github.com/ggerganov/llama.cpp/pull/2054
  if (body.count("rope_scaling_type") > 0)
    cparams.rope_scaling_type = body["rope_scaling_type"];
  if (body.count("rope_freq_base") > 0)
    cparams.rope_freq_base = body["rope_freq_base"];
  if (body.count("rope_freq_scale") > 0)
    cparams.rope_freq_scale = body["rope_freq_scale"];
  if (body.count("yarn_ext_factor") > 0)
    cparams.yarn_ext_factor = body["yarn_ext_factor"];
  if (body.count("yarn_attn_factor") > 0)
    cparams.yarn_attn_factor = body["yarn_attn_factor"];
  if (body.count("yarn_beta_fast") > 0)
    cparams.yarn_beta_fast = body["yarn_beta_fast"];
  if (body.count("yarn_beta_slow") > 0)
    cparams.yarn_beta_slow = body["yarn_beta_slow"];
  if (body.count("yarn_orig_ctx") > 0)
    cparams.yarn_orig_ctx = body["yarn_orig_ctx"];
  // group attention
  if (body.count("grp_attn_n") > 0)
    app.ga_n = body["grp_attn_n"];
  if (body.count("grp_attn_w") > 0)
    app.ga_w = body["grp_attn_w"];
  // optimizations
  if (body.count("cache_type_k") > 0)
    cparams.type_k = kv_cache_type_from_str(body["cache_type_k"]);
  if (body.count("cache_type_v") > 0)
    cparams.type_k = kv_cache_type_from_str(body["cache_type_v"]);
  app.model = llama_load_model_from_file(model_path.c_str(), mparams);
  app.ctx = llama_new_context_with_model(app.model, cparams);
  llama_batch_free(app.batch);
  app.batch = llama_batch_init(cparams.n_batch, 0, 1);
  return json{
      {"success", true},
      {"token_bos", llama_token_bos(app.model)},
      {"token_eos", llama_token_eos(app.model)},
  };
}

json action_sampling_init(app_t &app, json &body)
{
  // sampling
  llama_sampling_params sparams;
  if (body.count("mirostat") > 0)
    sparams.mirostat = body["mirostat"];
  if (body.count("mirostat_tau") > 0)
    sparams.mirostat_tau = body["mirostat_tau"];
  if (body.count("mirostat_eta") > 0)
    sparams.mirostat_eta = body["mirostat_eta"];
  if (body.count("temp") > 0)
    sparams.temp = body["temp"];
  if (body.count("top_p") > 0)
    sparams.top_p = body["top_p"];
  if (body.count("top_k") > 0)
    sparams.top_k = body["top_k"];
  if (body.count("penalty_last_n") > 0)
    sparams.penalty_last_n = body["penalty_last_n"];
  if (body.count("penalty_repeat") > 0)
    sparams.penalty_repeat = body["penalty_repeat"];
  if (body.count("penalty_freq") > 0)
    sparams.penalty_freq = body["penalty_freq"];
  if (body.count("penalty_present") > 0)
    sparams.penalty_present = body["penalty_present"];
  // if (body.count("samplers_sequence") > 0)
  //   sparams.samplers_sequence = body["samplers_sequence"];
  if (body.count("grammar") > 0)
    sparams.grammar = body["grammar"];
  if (body.count("n_prev") > 0)
    sparams.n_prev = body["n_prev"];
  if (body.count("n_probs") > 0)
    sparams.n_probs = body["n_probs"];
  if (body.count("min_p") > 0)
    sparams.min_p = body["min_p"];
  if (body.count("tfs_z") > 0)
    sparams.tfs_z = body["tfs_z"];
  if (body.count("typical_p") > 0)
    sparams.typical_p = body["typical_p"];
  // maybe free before creating a new one
  if (app.ctx_sampling != nullptr)
  {
    llama_sampling_free(app.ctx_sampling);
  }
  app.ctx_sampling = llama_sampling_init(sparams);
  if (body.count("tokens") > 0)
  {
    std::vector<llama_token> tokens = body["tokens"];
    for (auto id : tokens)
    {
      llama_sampling_accept(app.ctx_sampling, app.ctx, id, false);
    }
  }
  return json{{"success", true}};
}

// lookup single token (also be able to check if it exists or not)
json action_lookup_token(app_t &app, json &body)
{
  std::string piece = body["piece"];
  int32_t max_tokens = llama_n_vocab(app.model);
  for (int32_t id = 0; id < max_tokens; id++)
  {
    std::string token_as_str = llama_token_to_piece(app.ctx, id);
    if (token_as_str == piece)
    {
      return json{
          {"success", true},
          {"token", id},
      };
    }
  }
  // not found
  return json{{"success", false}};
}

// tokenize an input string
json action_tokenize(app_t &app, json &body)
{
  std::string text = body["text"];
  bool special = body.count("special") > 0;
  std::vector<llama_token> tokens_list;
  tokens_list = ::llama_tokenize(app.model, text, false, special);
  return json{
      {"success", true},
      {"tokens", tokens_list},
  };
}

// detokenize a list of tokens
json action_detokenize(app_t &app, json &body)
{
  std::vector<llama_token> tokens = body["tokens"];
  std::stringstream output;
  for (auto id : tokens)
  {
    output << llama_token_to_piece(app.ctx, id);
  }
  std::string parsed_str = output.str();
  return json{
      {"success", true},
      {"buffer", convert_string_to_int_arr(parsed_str)},
  };
}

// decode an array of tokens
json action_decode(app_t &app, json &body)
{
  std::vector<llama_token> tokens_list = body["tokens"];
  bool skip_logits = body.count("skip_logits") > 0;
  /*bool grp_attn_enabled = app.ga_n > 1;
  if (grp_attn_enabled)
  {
    group_attention_shift_context(app);
  }*/
  size_t i = 0;
  llama_batch_clear(app.batch);
  for (auto id : tokens_list)
  {
    bool grp_attn_enabled = false; // TODO: maybe remove grp_attn
    int32_t n_past = grp_attn_enabled
                         ? app.n_past_self_extension
                         : app.tokens.size();
    llama_batch_add(app.batch, id, n_past, {0}, false);
    app.tokens.push_back(id);
    i++;
    app.n_past_self_extension++;
  }
  // llama_decode will output logits only for the last token of the prompt
  if (!skip_logits)
  {
    app.batch.logits[app.batch.n_tokens - 1] = true;
  }
  if (llama_decode(app.ctx, app.batch) != 0)
  {
    return json{{"error", "llama_decode failed, maybe n_batch is too small?"}};
  }
  else
  {
    return json{
        {"success", true},
        {"n_past", app.tokens.size()},
    };
  }
}

// decode the current logits and sample the new token
json action_sampling_sample(app_t &app, json &body)
{
  int32_t idx = app.batch.n_tokens - 1;
  const llama_token new_token_id = llama_sampling_sample(app.ctx_sampling, app.ctx, NULL, idx);
  std::string piece = llama_token_to_piece(app.ctx, new_token_id);
  return json{
      {"success", true},
      {"piece", convert_string_to_int_arr(piece)},
      {"token", new_token_id},
  };
}

// accept this token
json action_sampling_accept(app_t &app, json &body)
{
  std::vector<llama_token> tokens_list = body["tokens"];
  for (auto id : tokens_list)
  {
    llama_sampling_accept(app.ctx_sampling, app.ctx, id, false);
  }
  return json{{"success", true}};
}

// get embeddings, this will call action_decode internally
json action_embeddings(app_t &app, json &body)
{
  std::vector<llama_token> tokens_list = body["tokens"];
  // allocate output
  const int n_embd = llama_n_embd(app.model);
  std::vector<float> embeddings(n_embd, 0); // single seq
  float *out = embeddings.data();
  // decode
  json req = json{{"tokens", tokens_list}};
  json res = action_decode(app, req);
  if (res.count("error")) {
    return res;
  }
  int32_t idx = app.batch.n_tokens - 1;
  const float *embd = llama_get_embeddings_seq(app.ctx, 0);
  if (embd == NULL) {
      embd = llama_get_embeddings_ith(app.ctx, idx);
      if (embd == NULL) {
        fprintf(stderr, "%s: failed to get embeddings for token %d\n", __func__, idx);
        return json{{"error", "failed to get embeddings"}};
      }
  }
  llama_embd_normalize(embd, out, n_embd);
  return json{
      {"success", true},
      {"embeddings", embeddings},
  };
}

// remove tokens in kv, for context-shifting
json action_kv_remove(app_t &app, json &body)
{
  const int n_keep = body["n_keep"];
  const int n_discard = body["n_discard"];
  const int n_past = app.tokens.size();
  llama_kv_cache_seq_rm(app.ctx, 0, n_keep, n_keep + n_discard);
  llama_kv_cache_seq_add(app.ctx, 0, n_keep + n_discard, n_past, -n_discard);
  app.tokens.erase(
      app.tokens.begin() + n_keep,
      app.tokens.begin() + n_keep + n_discard);
  return json{
      {"success", true},
      {"n_past", app.tokens.size()},
  };
}

// clear all tokens in kv
json action_kv_clear(app_t &app, json &body)
{
  llama_kv_cache_clear(app.ctx);
  app.tokens.clear();
  return json{
      {"success", true},
      {"n_past", app.tokens.size()},
  };
}

// save current session
json action_session_save(app_t &app, json &body)
{
  std::string session_path = body["session_path"];
  std::vector<llama_token> dummy;
  if (!llama_save_session_file(
          app.ctx,
          session_path.c_str(),
          dummy.data(),
          dummy.size()))
  {
    return json{{"error", "action_session_save failed"}};
  }
  return json{
      {"success", true},
      {"tokens", app.tokens},
  };
}

// load a session from disk
json action_session_load(app_t &app, json &body)
{
  std::string session_path = body["session_path"];
  std::vector<llama_token> saved_tokens = body["tokens"];
  auto n_ctx = llama_n_ctx(app.ctx);
  size_t n_token_count_out = 0;
  std::vector<llama_token> dummy;
  if (!llama_load_session_file(
          app.ctx,
          session_path.c_str(),
          dummy.data(),
          dummy.capacity(),
          &n_token_count_out))
  {
    return json{{"error", "llama_load_session_file failed"}};
  }
  // load tokens
  app.tokens.clear();
  app.tokens.reserve(saved_tokens.size());
  for (auto id : saved_tokens)
  {
    app.tokens.push_back(id);
  }
  return json{{"success", true}};
}

// get the current status
json action_current_status(app_t &app, json &body)
{
  return json{
      {"success", true},
      {"tokens", app.tokens},
  };
}
