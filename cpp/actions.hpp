#pragma once

#include <iostream>
#include <vector>
#include <string>
#include <sstream>
#include <stdio.h>
#include <cmath>

#include "llama.h"
#include "helpers/wcommon.h"
#include "helpers/wsampling.h"

#include "glue.hpp"

#define PARSE_REQ(msg_typename) \
  msg_typename req;             \
  glue_inbuf inbuf(req_raw);    \
  req.handler.deserialize(inbuf);

struct app_t
{
  llama_model *model;
  llama_context *ctx;
  const llama_vocab *vocab;
  wcommon_sampler *ctx_sampling = nullptr;
  llama_batch batch = llama_batch_init(512, 0, 1);
  llama_tokens tokens;
  int32_t seed = LLAMA_DEFAULT_SEED;
};

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

void free_all(app_t &app)
{
  if (app.ctx != nullptr)
    llama_free(app.ctx);
  if (app.model != nullptr)
    llama_model_free(app.model);
  if (app.ctx_sampling != nullptr)
    wcommon_sampler_free(app.ctx_sampling);
}

struct kv_dump
{
  std::vector<std::string> keys;
  std::vector<std::string> vals;
};

kv_dump dump_metadata(app_t &app)
{
  kv_dump output;
  int count = llama_model_meta_count(app.model);
  std::string key;
  std::string val;
  std::vector<char> buf(1024);
  int res = 0;
  for (int i = 0; i < count; i++)
  {
    res = llama_model_meta_val_str_by_index(app.model, i, buf.data(), buf.size());
    if (res < 0)
      continue;
    if (res > buf.size())
    {
      buf.resize(res + 1);
      res = llama_model_meta_val_str_by_index(app.model, i, buf.data(), buf.size());
    }
    val = std::string(buf.data(), res);
    res = llama_model_meta_key_by_index(app.model, i, buf.data(), buf.size());
    if (res < 0)
      continue;
    if (res > buf.size())
    {
      buf.resize(res + 1);
      res = llama_model_meta_key_by_index(app.model, i, buf.data(), buf.size());
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

glue_msg_load_res action_load(app_t &app, const char *req_raw)
{
  PARSE_REQ(glue_msg_load_req);
  free_all(app);
  std::vector<std::string> &model_paths = req.model_paths.arr;
  bool n_ctx_auto = req.n_ctx_auto.value;

  auto mparams = llama_model_default_params();
  if (req.use_mmap.not_null())
    mparams.use_mmap = req.use_mmap.value;
  if (req.use_mlock.not_null())
    mparams.use_mlock = req.use_mlock.value;
  if (req.n_gpu_layers.not_null())
    mparams.n_gpu_layers = req.n_gpu_layers.value;

  auto cparams = llama_context_default_params();
  app.seed = req.seed.value;
  cparams.n_ctx = req.n_ctx.value;
  cparams.n_threads = req.n_threads.value;
  cparams.n_threads_batch = cparams.n_threads;
  if (req.embeddings.not_null())
    cparams.embeddings = req.embeddings.value;
  if (req.offload_kqv.not_null())
    cparams.offload_kqv = req.offload_kqv.value;
  if (req.n_batch.not_null())
    cparams.n_batch = req.n_batch.value;
  if (req.n_seq_max.not_null())
    cparams.n_seq_max = req.n_seq_max.value;
  if (req.pooling_type.not_null())
    cparams.pooling_type = pooling_type_from_str(req.pooling_type.value);
  // context extending: https://github.com/ggerganov/llama.cpp/pull/2054
  if (req.rope_scaling_type.not_null())
    cparams.rope_scaling_type = rope_scaling_type_from_str(req.rope_scaling_type.value);
  if (req.rope_freq_base.not_null())
    cparams.rope_freq_base = req.rope_freq_base.value;
  if (req.rope_freq_scale.not_null())
    cparams.rope_freq_scale = req.rope_freq_scale.value;
  if (req.yarn_ext_factor.not_null())
    cparams.yarn_ext_factor = req.yarn_ext_factor.value;
  if (req.yarn_attn_factor.not_null())
    cparams.yarn_attn_factor = req.yarn_attn_factor.value;
  if (req.yarn_beta_fast.not_null())
    cparams.yarn_beta_fast = req.yarn_beta_fast.value;
  if (req.yarn_beta_slow.not_null())
    cparams.yarn_beta_slow = req.yarn_beta_slow.value;
  if (req.yarn_orig_ctx.not_null())
    cparams.yarn_orig_ctx = req.yarn_orig_ctx.value;
  // optimizations
  if (req.cache_type_k.not_null())
    cparams.type_k = kv_cache_type_from_str(req.cache_type_k.value);
  if (req.cache_type_v.not_null())
    cparams.type_v = kv_cache_type_from_str(req.cache_type_v.value);
  if (req.swa_full.not_null())
    cparams.swa_full = req.swa_full.value;
  if (req.flash_attn.not_null())
    cparams.flash_attn_type = req.flash_attn.value ? LLAMA_FLASH_ATTN_TYPE_AUTO : LLAMA_FLASH_ATTN_TYPE_DISABLED;

  // init threadpool
  ggml_threadpool_params_default(cparams.n_threads);

  // prepare model paths
  std::vector<const char *> model_paths_ptrs;
  for (auto &path : model_paths)
  {
    model_paths_ptrs.push_back(path.c_str());
  }

  // load model
  app.model = llama_model_load_from_splits(
      model_paths_ptrs.data(), model_paths_ptrs.size(), mparams);
  if (app.model == nullptr)
  {
    free_all(app);
    throw app_exception("Error while loading model");
  }
  app.vocab = llama_model_get_vocab(app.model);
  for (; cparams.n_ctx > 0; cparams.n_ctx -= 1024)
  {
    app.ctx = llama_init_from_model(app.model, cparams);
    if (app.ctx != nullptr)
    {
      break; // OK
    }
    if (!n_ctx_auto)
    {
      free_all(app);
      throw app_exception("Error while creating llama_context model");
    }
    else
    {
      std::cerr << "llama_context == nullptr, Retrying with n_ctx = " << cparams.n_ctx;
      continue;
    }
  }
  if (cparams.n_ctx < 0)
  {
    free_all(app);
    throw app_exception("Out of memory, cannot create llama_context model");
  }
  llama_batch_free(app.batch);
  app.batch = llama_batch_init(cparams.n_batch, 0, 1);
  auto decoder_start_token = llama_model_decoder_start_token(app.model);
  if (decoder_start_token < 0)
  {
    decoder_start_token = llama_vocab_bos(app.vocab);
  }
  int n_vocab = llama_vocab_n_tokens(app.vocab);
  llama_tokens list_tokens_eog;
  for (int i = 0; i < n_vocab; i++)
  {
    if (llama_vocab_is_eog(app.vocab, i))
    {
      list_tokens_eog.push_back(i);
    }
  }
  kv_dump metadata = dump_metadata(app);

  glue_msg_load_res res;
  res.success.value = true;
  res.n_ctx.value = cparams.n_ctx;
  res.n_batch.value = llama_n_batch(app.ctx);
  res.n_ubatch.value = llama_n_ubatch(app.ctx);
  res.n_vocab.value = n_vocab;
  res.n_ctx_train.value = llama_model_n_ctx_train(app.model);
  res.n_embd.value = llama_model_n_embd(app.model);
  res.n_layer.value = llama_model_n_layer(app.model);
  res.metadata_key.arr = metadata.keys;
  res.metadata_val.arr = metadata.vals;
  res.token_bos.value = llama_vocab_bos(app.vocab);
  res.token_eos.value = llama_vocab_eos(app.vocab);
  res.token_eot.value = llama_vocab_eot(app.vocab);
  res.list_tokens_eog.arr = std::move(list_tokens_eog);
  res.add_bos_token.value = llama_vocab_get_add_bos(app.vocab) == 1;
  res.add_eos_token.value = llama_vocab_get_add_eos(app.vocab) == 1;
  res.has_encoder.value = llama_model_has_encoder(app.model);
  res.token_decoder_start.value = llama_model_decoder_start_token(app.model);
  return res;
}

// set various options at runtime (after loading model)
glue_msg_set_options_res action_set_options(app_t &app, const char *req_raw)
{
  PARSE_REQ(glue_msg_set_options_req);
  if (req.embeddings.value)
  {
    llama_set_embeddings(app.ctx, true);
    llama_set_causal_attn(app.ctx, false);
  }
  else
  {
    llama_set_embeddings(app.ctx, false);
    llama_set_causal_attn(app.ctx, true);
  }
  glue_msg_set_options_res res;
  res.success.value = true;
  return res;
}

// init (or re-init) sampling context
glue_msg_sampling_init_res action_sampling_init(app_t &app, const char *req_raw)
{
  PARSE_REQ(glue_msg_sampling_init_req);
  // sampling
  wcommon_params_sampling sparams;
  sparams.seed = app.seed;
  if (sparams.seed == LLAMA_DEFAULT_SEED)
    sparams.seed = time(NULL);

  if (req.mirostat.not_null())
    sparams.mirostat = req.mirostat.value;
  if (req.mirostat_tau.not_null())
    sparams.mirostat_tau = req.mirostat_tau.value;
  if (req.mirostat_eta.not_null())
    sparams.mirostat_eta = req.mirostat_eta.value;
  if (req.temp.not_null())
    sparams.temp = req.temp.value;
  if (req.top_p.not_null())
    sparams.top_p = req.top_p.value;
  if (req.top_k.not_null())
    sparams.top_k = req.top_k.value;
  if (req.penalty_last_n.not_null())
    sparams.penalty_last_n = req.penalty_last_n.value;
  if (req.penalty_repeat.not_null())
    sparams.penalty_repeat = req.penalty_repeat.value;
  if (req.penalty_freq.not_null())
    sparams.penalty_freq = req.penalty_freq.value;
  if (req.penalty_present.not_null())
    sparams.penalty_present = req.penalty_present.value;
  if (req.dynatemp_range.not_null())
    sparams.dynatemp_range = req.dynatemp_range.value;
  if (req.dynatemp_exponent.not_null())
    sparams.dynatemp_exponent = req.dynatemp_exponent.value;
  // if (req.samplers_sequence.not_null())
  //   sparams.samplers_sequence = req.samplers_sequence.value;
  if (req.grammar.not_null())
    sparams.grammar = req.grammar.value;
  if (req.n_prev.not_null())
    sparams.n_prev = req.n_prev.value;
  if (req.n_probs.not_null())
    sparams.n_probs = req.n_probs.value;
  if (req.min_p.not_null())
    sparams.min_p = req.min_p.value;
  if (req.typical_p.not_null())
    sparams.typ_p = req.typical_p.value; // for compat
  if (req.typ_p.not_null())
    sparams.typ_p = req.typ_p.value;
  // logit bias
  if (req.logit_bias_vals.not_null() && req.logit_bias_toks.not_null())
  {
    std::vector<llama_token> tokens = std::move(req.logit_bias_toks.arr);
    std::vector<float> &bias = req.logit_bias_vals.arr;
    for (size_t i = 0; i < tokens.size(); i++)
    {
      sparams.logit_bias.push_back({tokens[i], bias[i]});
    }
  }
  // maybe free before creating a new one
  if (app.ctx_sampling != nullptr)
  {
    wcommon_sampler_free(app.ctx_sampling);
  }
  app.ctx_sampling = wcommon_sampler_init(app.model, sparams);
  if (req.tokens.not_null())
  {
    for (auto id : req.tokens.arr)
    {
      wcommon_sampler_accept(app.ctx_sampling, id, false);
    }
  }

  glue_msg_sampling_init_res res;
  res.success.value = true;
  return res;
}

// get map token ID to vocab (be careful, it is slow!)
glue_msg_get_vocab_res action_get_vocab(app_t &app, const char *req_raw)
{
  PARSE_REQ(glue_msg_get_vocab_req);
  int32_t max_tokens = llama_vocab_n_tokens(app.vocab);
  std::vector<std::vector<char>> vocab;
  vocab.resize(max_tokens);
  for (int32_t id = 0; id < max_tokens; id++)
  {
    std::string token_as_str = wcommon_token_to_piece(app.ctx, id);
    vocab.emplace_back(convert_string_to_buf(token_as_str));
  }

  glue_msg_get_vocab_res res;
  res.success.value = true;
  res.vocab.arr = vocab;
  return res;
}

// lookup single token (also be able to check if it exists or not)
glue_msg_lookup_token_res action_lookup_token(app_t &app, const char *req_raw)
{
  PARSE_REQ(glue_msg_lookup_token_req);
  std::string &piece = req.piece.value;
  int32_t max_tokens = llama_vocab_n_tokens(app.vocab);
  glue_msg_lookup_token_res res;
  for (int32_t id = 0; id < max_tokens; id++)
  {
    std::string token_as_str = wcommon_token_to_piece(app.ctx, id);
    if (token_as_str == piece)
    {
      res.success.value = true;
      res.token.value = id;
      return res;
    }
  }
  // not found
  res.success.value = false;
  return res;
}

// tokenize an input string
glue_msg_tokenize_res action_tokenize(app_t &app, const char *req_raw)
{
  PARSE_REQ(glue_msg_tokenize_req);
  std::string &text = req.text.value;
  bool special = req.special.value;
  llama_tokens tokens_list = wcommon_tokenize(app.vocab, text, false, special);

  glue_msg_tokenize_res res;
  res.success.value = true;
  res.tokens.arr = std::move(tokens_list);
  return res;
}

// detokenize a list of tokens
glue_msg_detokenize_res action_detokenize(app_t &app, const char *req_raw)
{
  PARSE_REQ(glue_msg_detokenize_req);
  llama_tokens tokens = std::move(req.tokens.arr);
  std::stringstream output;
  for (auto id : tokens)
  {
    output << wcommon_token_to_piece(app.ctx, id);
  }
  std::string parsed_str = output.str();

  glue_msg_detokenize_res res;
  res.success.value = true;
  res.buffer.buf = convert_string_to_buf(parsed_str);
  return res;
}

// decode an array of tokens
glue_msg_decode_res action_decode(app_t &app, const char *req_raw)
{
  PARSE_REQ(glue_msg_decode_req);
  llama_tokens tokens_list = std::move(req.tokens.arr);
  bool skip_logits = req.skip_logits.value;
  size_t i = 0;
  wcommon_batch_clear(app.batch);
  for (auto id : tokens_list)
  {
    bool grp_attn_enabled = false; // TODO: maybe remove grp_attn
    int32_t n_past = app.tokens.size();
    wcommon_batch_add(app.batch, id, n_past, {0}, false);
    app.tokens.push_back(id);
    i++;
  }
  // llama_decode will output logits only for the last token of the prompt
  if (!skip_logits)
  {
    app.batch.logits[app.batch.n_tokens - 1] = true;
  }
  glue_msg_decode_res res;
  if (llama_decode(app.ctx, app.batch) != 0)
  {
    res.success.value = false;
    res.message.value = "llama_decode failed, maybe n_batch is too small?";
    res.n_past.value = app.tokens.size();
  }
  else
  {
    res.success.value = true;
    res.n_past.value = app.tokens.size();
  }
  return res;
}

// encode an array of tokens
glue_msg_encode_res action_encode(app_t &app, const char *req_raw)
{
  PARSE_REQ(glue_msg_encode_req);
  llama_tokens tokens_list = std::move(req.tokens.arr);
  if (!llama_model_has_encoder(app.model))
  {
    glue_msg_encode_res res;
    res.success.value = false;
    res.message.value = "this model does not have an encoder";
    return res;
  }
  size_t n_past = 0;
  wcommon_batch_clear(app.batch);
  for (auto id : tokens_list)
  {
    wcommon_batch_add(app.batch, id, n_past, {0}, false);
    n_past++;
  }
  glue_msg_encode_res res;
  if (llama_encode(app.ctx, app.batch) != 0)
  {
    res.success.value = false;
    res.message.value = "llama_encode failed, maybe n_batch is too small?";
    res.n_past.value = n_past;
  }
  else
  {
    res.success.value = true;
    res.n_past.value = n_past;
  }
  return res;
}

// decode the current logits and sample the new token
glue_msg_sampling_sample_res action_sampling_sample(app_t &app, const char *req_raw)
{
  PARSE_REQ(glue_msg_sampling_sample_req);
  int32_t idx = app.batch.n_tokens - 1;
  const llama_token new_token_id = wcommon_sampler_sample(app.ctx_sampling, app.ctx, idx, false);
  std::string piece = wcommon_token_to_piece(app.ctx, new_token_id);

  glue_msg_sampling_sample_res res;
  res.success.value = true;
  res.piece.buf = convert_string_to_buf(piece);
  res.token.value = new_token_id;
  return res;
}

// accept this token
glue_msg_sampling_accept_res action_sampling_accept(app_t &app, const char *req_raw)
{
  PARSE_REQ(glue_msg_sampling_accept_req);
  llama_tokens tokens_list = std::move(req.tokens.arr);
  for (auto id : tokens_list)
  {
    wcommon_sampler_accept(app.ctx_sampling, id, false);
  }

  glue_msg_sampling_accept_res res;
  res.success.value = true;
  return res;
}

// get softmax-ed probability of logits, can be used for custom sampling. The output is always sorted
glue_msg_get_logits_res action_get_logits(app_t &app, const char *req_raw)
{
  PARSE_REQ(glue_msg_get_logits_req);
  int top_k = req.top_k.value; // if is -1, we take all logits (will be slow!)
  int32_t idx = app.batch.n_tokens - 1;
  float *logits = llama_get_logits_ith(app.ctx, idx);
  int32_t n_vocab = llama_vocab_n_tokens(app.vocab);
  auto sort_fn = [](llama_token_data &a, llama_token_data &b) -> bool
  {
    return b.logit < a.logit;
  };
  // get all candidates and sort
  std::vector<llama_token_data> candidates;
  candidates.reserve(n_vocab);
  float sum = 0.0f; // for softmax
  for (llama_token token_id = 0; token_id < n_vocab; token_id++)
  {
    float exp_val = exp(logits[token_id]);
    candidates.emplace_back(llama_token_data{token_id, logits[token_id], exp_val});
    sum += exp_val;
  }
  for (auto &c : candidates)
  {
    c.p /= sum; // calculate softmax
  }
  std::sort(candidates.begin(), candidates.end(), sort_fn);
  if (top_k >= 0)
  {
    candidates.erase(candidates.begin() + top_k, candidates.end());
  }
  // convert response to json
  std::vector<int32_t> output_tokens;
  std::vector<float> output_probs;
  output_tokens.reserve(candidates.size());
  output_probs.reserve(candidates.size());
  for (auto &c : candidates)
  {
    output_tokens.push_back(c.id);
    output_probs.push_back(c.p);
  }

  glue_msg_get_logits_res res;
  res.success.value = true;
  res.tokens.arr = std::move(output_tokens);
  res.probs.arr = std::move(output_probs);
  return res;
}

// get embeddings, this will call action_decode internally
glue_msg_get_embeddings_res action_embeddings(app_t &app, const char *req_raw)
{
  PARSE_REQ(glue_msg_get_embeddings_req);
  auto &tokens_list = req.tokens.arr;
  // allocate output
  const int n_embd = llama_model_n_embd(app.model);
  std::vector<float> embeddings(n_embd, 0); // single seq
  float *out = embeddings.data();
  // decode
  glue_msg_get_embeddings_res res;
  glue_msg_decode_req decode_req;
  decode_req.tokens.arr = std::move(tokens_list);
  decode_req.skip_logits.value = false;
  glue_outbuf decode_req_buf;
  decode_req.handler.serialize(decode_req_buf);
  auto decode_res = action_decode(app, decode_req_buf.data.data());
  if (decode_res.success.value == false)
  {
    res.success.value = false;
    res.message.value = std::move(decode_res.message.value);
    return res;
  }
  int32_t idx = app.batch.n_tokens - 1;
  const float *embd = llama_get_embeddings_seq(app.ctx, 0);
  if (embd == NULL)
  {
    embd = llama_get_embeddings_ith(app.ctx, idx);
    if (embd == NULL)
    {
      // fprintf(stderr, "%s: failed to get embeddings for token %d\n", __func__, idx);
      res.success.value = false;
      res.message.value = "failed to get embeddings";
      return res;
    }
  }
  wcommon_embd_normalize(embd, out, n_embd, 2);

  res.success.value = true;
  res.embeddings.arr = std::move(embeddings);
  return res;
}

// remove tokens in kv, for context-shifting
glue_msg_get_kv_remove_res action_kv_remove(app_t &app, const char *req_raw)
{
  PARSE_REQ(glue_msg_get_kv_remove_req);
  const int n_keep = req.n_keep.value;
  const int n_discard = req.n_discard.value;
  auto * mem = llama_get_memory(app.ctx);
  
  glue_msg_get_kv_remove_res res;
  bool & success = res.success.value;
  success = false;
  res.n_past.value = app.tokens.size();

  llama_pos pos_min = llama_memory_seq_pos_min(mem, 0);
  if (pos_min > 0) {
    // TODO: rm tokens from SWA is currently unsupported
    success = false;
    return res;
  }

  if (n_discard > 0)
  {
    // TODO: this code branch is kinda broken, to be fixed later
    const int n_past = app.tokens.size();
    success = llama_memory_seq_rm(mem, 0, n_keep, n_keep + n_discard);
    if (!success)
    {
      return res;
    }
    llama_memory_seq_add(mem, 0, n_keep + n_discard, n_past, -n_discard);
    app.tokens.erase(
        app.tokens.begin() + n_keep,
        app.tokens.begin() + n_keep + n_discard);
  }
  else if (n_discard < 0)
  {
    if (n_keep == 0)
    {
      llama_memory_clear(mem, true);
    }
    else
    {
      success = llama_memory_seq_rm(mem, 0, n_keep, -1);
      if (!success)
      {
        return res;
      }
      app.tokens.erase(
          app.tokens.begin() + n_keep,
          app.tokens.end());
    }
  }

  return res;
}

// clear all tokens in kv
glue_msg_get_kv_clear_res action_kv_clear(app_t &app, const char *req_raw)
{
  PARSE_REQ(glue_msg_get_kv_clear_req);
  auto * mem = llama_get_memory(app.ctx);
  llama_memory_clear(mem, true);
  app.tokens.clear();

  glue_msg_get_kv_clear_res res;
  res.success.value = true;
  res.n_past.value = app.tokens.size();
  return res;
}

/*
// save current session
json action_session_save(app_t &app, json &body)
{
  std::string session_path = body["session_path"];
  llama_tokens dummy;
  if (!llama_state_seq_save_file(
          app.ctx,
          session_path.c_str(),
          0,            // seq_id
          dummy.data(), // tokens
          dummy.size()  // n_token_count
          ))
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
  llama_tokens saved_tokens = body["tokens"];
  auto n_ctx = llama_n_ctx(app.ctx);
  size_t n_token_count_out = 0;
  llama_tokens dummy;
  if (!llama_state_seq_load_file(
          app.ctx,
          session_path.c_str(),
          0,                 // dest_seq_id
          dummy.data(),      // tokens_out
          dummy.capacity(),  // n_token_capacity
          &n_token_count_out // n_token_count_out
          ))
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
*/

// get the current status
glue_msg_status_res action_current_status(app_t &app, const char *req_raw)
{
  PARSE_REQ(glue_msg_status_req);
  glue_msg_status_res res;
  res.success.value = true;
  res.tokens.arr = app.tokens; // copy
  return res;
}

//
// benchmark & perplexity
//

glue_msg_test_benchmark_res action_test_benchmark(app_t &app, const char *req_raw)
{
  PARSE_REQ(glue_msg_test_benchmark_req);
  std::string type = req.type.value;   // "pp" (prompt proc) or "tg" (tok gen)
  int n_samples = req.n_samples.value; // n_batch in pp and n_predict in pg

  llama_memory_clear(llama_get_memory(app.ctx), true);
  int n_vocab = llama_vocab_n_tokens(app.vocab);
  int64_t t_start = ggml_time_ms();

  if (type == "pp")
  {
    llama_batch batch = llama_batch_init(n_samples, 0, 1);
    for (int i = 0; i < n_samples; i++)
    {
      wcommon_batch_add(batch, i % n_vocab, i, {0}, i == n_samples - 1);
    }
    int ret = llama_decode(app.ctx, batch);
    llama_batch_free(batch);
    if (ret != 0)
    {
      glue_msg_test_benchmark_res res;
      res.success.value = false;
      res.message.value = "llama_decode failed with status = " + std::to_string(ret);
      return res;
    }
  }
  else if (type == "tg")
  {
    llama_batch batch = llama_batch_init(1, 0, 1);
    for (int i = 0; i < n_samples; i++)
    {
      wcommon_batch_clear(batch);
      wcommon_batch_add(batch, i % n_vocab, i, {0}, true);
      int ret = llama_decode(app.ctx, batch);
      if (ret != 0)
      {
        glue_msg_test_benchmark_res res;
        res.success.value = false;
        res.message.value = "llama_decode failed with status = " + std::to_string(ret);
        return res;
      }
    }
    llama_batch_free(batch);
  }
  else
  {
    glue_msg_test_benchmark_res res;
    res.success.value = false;
    res.message.value = "unknown type: " + type;
    return res;
  }

  int64_t t_end = ggml_time_ms();
  glue_msg_test_benchmark_res res;
  res.success.value = true;
  res.t_ms.value = t_end - t_start;
  return res;
}

glue_msg_test_perplexity_res action_test_perplexity(app_t &app, const char *req_raw)
{
  PARSE_REQ(glue_msg_test_perplexity_req);
  llama_tokens input = std::move(req.tokens.arr);
  const size_t n = input.size();

  int64_t t_start = ggml_time_ms();

  if (n < 2)
  {
    glue_msg_test_perplexity_res res;
    res.success.value = false;
    res.message.value = "Input must contain at least two tokens";
    return res;
  }

  // Clear existing context to start fresh
  llama_memory_clear(llama_get_memory(app.ctx), true);
  app.tokens.clear();

  const int32_t n_vocab = llama_vocab_n_tokens(app.vocab);
  double nll = 0.0;

  static auto log_softmax = [](int n_vocab, const float *logits, int tok) -> double
  {
    float max_logit = logits[0];
    for (int i = 1; i < n_vocab; ++i)
    {
      max_logit = std::max(max_logit, logits[i]);
    }
    double sum_exp = 0.0;
    for (int i = 0; i < n_vocab; ++i)
    {
      sum_exp += expf(logits[i] - max_logit);
    }
    return logits[tok] - max_logit - log(sum_exp);
  };

  for (size_t i = 0; i < n - 1; ++i)
  {
    // Prepare batch with current token (input[i])
    wcommon_batch_clear(app.batch);
    wcommon_batch_add(app.batch, input[i], i, {0}, true); // Enable logits for this token

    if (llama_decode(app.ctx, app.batch) != 0)
    {
      glue_msg_test_perplexity_res res;
      res.success.value = false;
      res.message.value = "llama_decode failed at position " + std::to_string(i);
      return res;
    }

    float *logits = llama_get_logits_ith(app.ctx, 0);

    // Get true next token (input[i+1])
    const int32_t true_token = input[i + 1];

    nll += -log_softmax(n_vocab, logits, true_token);
  }

  // Calculate final metrics
  const double cross_entropy = nll / (n - 1);
  const double ppl = std::exp(cross_entropy);

  int64_t t_end = ggml_time_ms();

  glue_msg_test_perplexity_res res;
  res.success.value = true;
  res.ppl.value = ppl;
  res.nll.value = nll;
  res.cross_entropy.value = cross_entropy;
  res.n_tokens.value = n - 1;
  res.t_ms.value = t_end - t_start;
  return res;
}

glue_msg_chat_format_res action_chat_format(app_t &app, const char *req_raw)
{
  PARSE_REQ(glue_msg_chat_format_req);
  std::string tmpl = req.tmpl.not_null() ? req.tmpl.value : "";
  bool add_ass = req.add_ass.not_null() ? req.add_ass.value : false;
  std::vector<std::string> &roles = req.roles.arr;
  std::vector<std::string> &contents = req.contents.arr;
  std::vector<wcommon_chat_msg> chat;
  for (size_t i = 0; i < roles.size(); i++)
  {
    chat.push_back({roles[i], contents[i]});
  }
  try
  {
    std::string formatted_chat = wcommon_chat_apply_template(app.model, tmpl, chat, add_ass);
    glue_msg_chat_format_res res;
    res.success.value = true;
    res.formatted_chat.value = formatted_chat;
    return res;
  }
  catch (const std::exception &e)
  {
    glue_msg_chat_format_res res;
    res.success.value = true;
    res.message.value = std::string(e.what());
    return res;
  }
}
