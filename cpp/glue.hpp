#pragma once
/**
 * Simple serializer / deserializer inspired by protobuf
 *
 * Structure:
 * - 4 bytes magic number   (GLUE_MAGIC)
 * - 4 bytes version number (GLUE_VERSION)
 * - 8 bytes message prototype ID
 * - 4 bytes message length, unsigned number
 * - message data
 *
 * Each field in the message is encoded as:
 * - 4 bytes data type
 * - 4 bytes size, unsigned number (only for array and string)
 * - data
 */

#include <cstdint>
#include <string>
#include <vector>
#include <functional>

// increase when messages change
#define GLUE_VERSION 1

#define GLUE_MAGIC 0x45554c47 // "GLUE"
#define GLUE_PROTO_ID_LEN 8

#ifndef GLUE_DEBUG
#define GLUE_DEBUG(...)
#endif

#define BITS_TO_BYTES(x) ((x) / 8)

// Data types
// Note: we're doing polymorphism using enum to prevent using virtual functions

enum glue_dtype
{
  GLUE_DTYPE_NULL,
  GLUE_DTYPE_BOOL,
  GLUE_DTYPE_INT,
  GLUE_DTYPE_FLOAT,
  GLUE_DTYPE_STRING,
  GLUE_DTYPE_RAW,
  GLUE_DTYPE_ARRAY_BOOL,
  GLUE_DTYPE_ARRAY_INT,
  GLUE_DTYPE_ARRAY_FLOAT,
  GLUE_DTYPE_ARRAY_STRING,
  GLUE_DTYPE_ARRAY_RAW,
};

using glue_data_ptr = const char *;

struct glue_outbuf
{
  std::vector<char> data;
  glue_outbuf()
  {
    data.reserve(1024);
  }
  void append(const char *val, size_t size)
  {
    GLUE_DEBUG(" << offset = 0x%02zx\n", data.size());
    data.insert(data.end(), val, val + size);
  }
  void append_str(const std::string &val)
  {
    GLUE_DEBUG(" << offset = 0x%02zx\n", data.size());
    data.insert(data.end(), val.begin(), val.end());
  }
  void append_u32(uint32_t val)
  {
    GLUE_DEBUG(" << offset = 0x%02zx\n", data.size());
    data.insert(data.end(), (char *)&val, (char *)&val + BITS_TO_BYTES(32));
  }
  void append_i32(int32_t val)
  {
    GLUE_DEBUG(" << offset = 0x%02zx\n", data.size());
    data.insert(data.end(), (char *)&val, (char *)&val + BITS_TO_BYTES(32));
  }
  void append_f32(float val)
  {
    GLUE_DEBUG(" << offset = 0x%02zx\n", data.size());
    data.insert(data.end(), (char *)&val, (char *)&val + BITS_TO_BYTES(32));
  }
  void clear() {
    data.clear();
    data.reserve(1024);
  }
};

struct glue_inbuf
{
  glue_data_ptr base;
  glue_data_ptr cur;
  glue_inbuf(glue_data_ptr data) : base(data), cur(data) {}
  uint32_t read_u32()
  {
    GLUE_DEBUG(" >> offset = 0x%02zx\n", cur - base);
    uint32_t val = *(uint32_t *)cur;
    cur += BITS_TO_BYTES(32);
    return val;
  }
  int32_t read_i32()
  {
    GLUE_DEBUG(" >> offset = 0x%02zx\n", cur - base);
    int32_t val = *(int32_t *)cur;
    cur += BITS_TO_BYTES(32);
    return val;
  }
  float read_f32()
  {
    GLUE_DEBUG(" >> offset = 0x%02zx\n", cur - base);
    float val = *(float *)cur;
    cur += BITS_TO_BYTES(32);
    return val;
  }
  std::string read_str(uint32_t size)
  {
    GLUE_DEBUG(" >> offset = 0x%02zx\n", cur - base);
    std::string val(cur, size);
    cur += size;
    return val;
  }
  std::vector<char> read_raw(uint32_t size)
  {
    GLUE_DEBUG(" >> offset = 0x%02zx\n", cur - base);
    std::vector<char> val;
    val.reserve(size);
    val.insert(val.end(), cur, cur + size);
    cur += size;
    return val;
  }

  // for array
  void read(uint32_t &out) { out = read_u32(); }
  void read(int32_t &out) { out = read_i32(); }
  void read(float &out) { out = read_f32(); }
  void read(std::string &out)
  {
    uint32_t size = read_u32();
    out = read_str(size);
  }
  void read(std::vector<char> &out)
  {
    uint32_t size = read_u32();
    out = read_raw(size);
  }
};

struct glue_type_base;
struct glue_handler
{
  const char *name = nullptr;
  std::vector<glue_type_base *> fields;

  glue_handler(const char *name) : name(name) {}
  void register_field(glue_type_base *field)
  {
    fields.push_back(field);
  };
  void serialize(glue_outbuf &output);
  void deserialize(glue_inbuf &input);
};

struct glue_type_base
{
  const char *name = nullptr;
  glue_dtype dtype = GLUE_DTYPE_NULL;
  glue_handler handler;

  glue_type_base() = delete;
  glue_type_base(const char *name, glue_handler &handler, glue_dtype dtype) : name(name), handler(handler), dtype(dtype)
  {
    handler.register_field(this);
  }
  bool is_null() { return dtype == GLUE_DTYPE_NULL; }
  bool not_null() { return !is_null(); }
  void set_null() { dtype = GLUE_DTYPE_NULL; }
  bool parse_type(glue_inbuf &input)
  {
    dtype = (glue_dtype)input.read_u32();
    if (dtype == GLUE_DTYPE_NULL)
    {
      GLUE_DEBUG(" >> null\n");
      return true;
    }
    return false;
  }
};

struct glue_bool : glue_type_base
{
  bool value = false;

  glue_bool(const char *name, glue_handler &handler) : glue_type_base(name, handler, GLUE_DTYPE_BOOL) {}
  void parse(glue_inbuf &input)
  {
    if (parse_type(input))
      return;
    value = (bool)input.read_u32();
    GLUE_DEBUG(" >> bool %d\n", value);
  }
  void serialize(glue_outbuf &output)
  {
    GLUE_DEBUG(" << bool %d\n", value);
    output.append_u32(dtype);
    output.append_u32(value);
  }
};

struct glue_int : glue_type_base
{
  int32_t value = 0;

  glue_int(const char *name, glue_handler &handler) : glue_type_base(name, handler, GLUE_DTYPE_INT) {}
  void parse(glue_inbuf &input)
  {
    if (parse_type(input))
      return;
    value = input.read_i32();
    GLUE_DEBUG(" >> int %d\n", value);
  }
  void serialize(glue_outbuf &output)
  {
    GLUE_DEBUG(" << int %d\n", value);
    output.append_u32(dtype);
    output.append_i32(value);
  }
};

struct glue_float : glue_type_base
{
  float value = 0.0f;

  glue_float(const char *name, glue_handler &handler) : glue_type_base(name, handler, GLUE_DTYPE_FLOAT) {}
  void parse(glue_inbuf &input)
  {
    if (parse_type(input))
      return;
    value = input.read_f32();
    GLUE_DEBUG(" >> float %f\n", value);
  }
  void serialize(glue_outbuf &output)
  {
    GLUE_DEBUG(" << float %f\n", value);
    output.append_u32(dtype);
    output.append_f32(value);
  }
};

struct glue_str : glue_type_base
{
  std::string value;

  glue_str(const char *name, glue_handler &handler) : glue_type_base(name, handler, GLUE_DTYPE_STRING) {}
  void parse(glue_inbuf &input)
  {
    if (parse_type(input))
      return;
    uint32_t size = input.read_u32();
    value = input.read_str(size);
    GLUE_DEBUG(" >> string %s\n", value.c_str());
  }
  void serialize(glue_outbuf &output)
  {
    GLUE_DEBUG(" << string %s\n", value.c_str());
    output.append_u32(dtype);
    output.append_u32(value.size());
    output.append_str(value);
  }
};

struct glue_raw : glue_type_base
{
  std::vector<char> buf;

  glue_raw(const char *name, glue_handler &handler) : glue_type_base(name, handler, GLUE_DTYPE_RAW) {}
  void parse(glue_inbuf &input)
  {
    if (parse_type(input))
      return;
    uint32_t size = input.read_u32();
    buf = input.read_raw(size);
    GLUE_DEBUG(" >> raw, size = %zu\n", buf.size());
  }
  void serialize(glue_outbuf &output)
  {
    GLUE_DEBUG(" << raw, size = %zu\n", buf.size());
    output.append_u32(dtype);
    output.append_u32(buf.size());
    output.append(buf.data(), buf.size());
  }
};

template <typename T>
struct glue_arr : glue_type_base
{
  std::vector<T> arr;
  std::function<void(T &, glue_outbuf &)> serialize_elem;

  glue_arr(const char *name, glue_handler &handler, glue_dtype dtype) : glue_type_base(name, handler, dtype) {}
  void parse(glue_inbuf &input)
  {
    if (parse_type(input))
      return;
    uint32_t size = input.read_u32();
    GLUE_DEBUG(" >> array[%u]\n", size);
    arr.reserve(size);
    for (uint32_t i = 0; i < size; i++)
    {
      T elem;
      input.read(elem);
      arr.push_back(std::move(elem));
    }
  }
  void serialize(glue_outbuf &output)
  {
    GLUE_DEBUG(" << array[%zu]\n", arr.size());
    output.append_u32(dtype);
    output.append_u32(arr.size());
    for (auto elem : arr)
    {
      serialize_elem(elem, output);
    }
  }
};

#define DEF_GLUE_ARR(tname, dtype, enum_type, serialize_fn)                                                   \
  struct glue_arr_##tname : glue_arr<dtype>                                                                   \
  {                                                                                                           \
    glue_arr_##tname(const char *name, glue_handler &handler) : glue_arr<dtype>(name, handler, enum_type) \
    {                                                                                                         \
      serialize_elem = [](dtype & elem, glue_outbuf & output) serialize_fn;                                   \
    }                                                                                                         \
  };

DEF_GLUE_ARR(bool, uint32_t, GLUE_DTYPE_ARRAY_BOOL, {
  output.append_u32(elem);
})
DEF_GLUE_ARR(int, int32_t, GLUE_DTYPE_ARRAY_INT, {
  output.append_i32(elem);
})
DEF_GLUE_ARR(float, float, GLUE_DTYPE_ARRAY_FLOAT, {
  output.append_f32(elem);
})
DEF_GLUE_ARR(str, std::string, GLUE_DTYPE_ARRAY_STRING, {
  output.append_u32(elem.size());
  output.append_str(elem);
})
DEF_GLUE_ARR(raw, std::vector<char>, GLUE_DTYPE_ARRAY_RAW, {
  output.append_u32(elem.size());
  output.append(elem.data(), elem.size());
})

// Message base

void glue_handler::serialize(glue_outbuf &output)
{
  output.clear();
  output.append_u32(GLUE_MAGIC);
  output.append_u32(GLUE_VERSION);
  output.append(name, 8);
  GLUE_DEBUG("Serializing message %s\n", name);
  GLUE_DEBUG("Fields: %zu\n", fields.size());
  for (auto field : fields)
  {
    GLUE_DEBUG("Serializing field %s, type = %d\n", field->name, field->dtype);
    switch (field->dtype)
    {
    case GLUE_DTYPE_NULL:
      output.append_u32(GLUE_DTYPE_NULL);
      break;
    case GLUE_DTYPE_BOOL:
      ((glue_bool *)field)->serialize(output);
      break;
    case GLUE_DTYPE_INT:
      ((glue_int *)field)->serialize(output);
      break;
    case GLUE_DTYPE_FLOAT:
      ((glue_float *)field)->serialize(output);
      break;
    case GLUE_DTYPE_STRING:
      ((glue_str *)field)->serialize(output);
      break;
    case GLUE_DTYPE_RAW:
      ((glue_raw *)field)->serialize(output);
      break;
    case GLUE_DTYPE_ARRAY_BOOL:
      ((glue_arr_bool *)field)->serialize(output);
      break;
    case GLUE_DTYPE_ARRAY_INT:
      ((glue_arr_int *)field)->serialize(output);
      break;
    case GLUE_DTYPE_ARRAY_FLOAT:
      ((glue_arr_float *)field)->serialize(output);
      break;
    case GLUE_DTYPE_ARRAY_STRING:
      ((glue_arr_str *)field)->serialize(output);
      break;
    case GLUE_DTYPE_ARRAY_RAW:
      ((glue_arr_raw *)field)->serialize(output);
      break;
    }
  }
}

void glue_handler::deserialize(glue_inbuf &input)
{
  uint32_t magic = input.read_u32();
  if (magic != GLUE_MAGIC)
  {
    throw std::runtime_error("Invalid magic number");
  }

  uint32_t version = input.read_u32();
  if (version != GLUE_VERSION)
  {
    throw std::runtime_error("Version mismatch");
  }

  std::string proto_id = input.read_str(GLUE_PROTO_ID_LEN);
  if (proto_id != name)
  {
    throw std::runtime_error("Prototype ID mismatch " + proto_id + " != " + name);
  }

  GLUE_DEBUG("Deserializing message %s\n", name);
  for (auto field : fields)
  {
    GLUE_DEBUG("Deserializing field %s, type = %d\n", field->name, field->dtype);
    switch (field->dtype)
    {
    case GLUE_DTYPE_NULL:
      field->parse_type(input);
      break;
    case GLUE_DTYPE_BOOL:
      ((glue_bool *)field)->parse(input);
      break;
    case GLUE_DTYPE_INT:
      ((glue_int *)field)->parse(input);
      break;
    case GLUE_DTYPE_FLOAT:
      ((glue_float *)field)->parse(input);
      break;
    case GLUE_DTYPE_STRING:
      ((glue_str *)field)->parse(input);
      break;
    case GLUE_DTYPE_RAW:
      ((glue_raw *)field)->parse(input);
    case GLUE_DTYPE_ARRAY_BOOL:
      ((glue_arr_bool *)field)->parse(input);
      break;
    case GLUE_DTYPE_ARRAY_INT:
      ((glue_arr_int *)field)->parse(input);
      break;
    case GLUE_DTYPE_ARRAY_FLOAT:
      ((glue_arr_float *)field)->parse(input);
      break;
    case GLUE_DTYPE_ARRAY_STRING:
      ((glue_arr_str *)field)->parse(input);
      break;
    case GLUE_DTYPE_ARRAY_RAW:
      ((glue_arr_raw *)field)->parse(input);
    }
  }
}

template <std::size_t N>
constexpr auto &PROTO_ID(char const (&s)[N])
{
  static_assert(N == GLUE_PROTO_ID_LEN + 1, "Prototype ID must be 8 characters long");
  return s;
}
#define GLUE_FIELD(type, name) glue_##type name = glue_##type(#name, handler);
#define GLUE_FIELD_NULLABLE(type, name) glue_##type name = glue_##type(#name, handler);
#define GLUE_HANDLER(name) glue_handler handler = glue_handler(PROTO_ID(name));

// Message for events

struct glue_msg_error
{
  GLUE_HANDLER("erro_evt")
  GLUE_FIELD(str, message)
};

// Message for actions

struct glue_msg_load_req
{
  GLUE_HANDLER("load_req")
  GLUE_FIELD(arr_str, model_paths)
  GLUE_FIELD(bool, n_ctx_auto)
  GLUE_FIELD(bool, use_mmap)
  GLUE_FIELD(bool, use_mlock)
  GLUE_FIELD(int, n_gpu_layers)
  GLUE_FIELD(int, seed)
  GLUE_FIELD(int, n_ctx)
  GLUE_FIELD(int, n_threads)
  GLUE_FIELD_NULLABLE(bool, embeddings)
  GLUE_FIELD_NULLABLE(bool, offload_kqv)
  GLUE_FIELD_NULLABLE(int, n_batch)
  GLUE_FIELD_NULLABLE(int, n_seq_max)
  GLUE_FIELD_NULLABLE(str, pooling_type)
  GLUE_FIELD_NULLABLE(str, rope_scaling_type)
  GLUE_FIELD_NULLABLE(float, rope_freq_base)
  GLUE_FIELD_NULLABLE(float, rope_freq_scale)
  GLUE_FIELD_NULLABLE(float, yarn_ext_factor)
  GLUE_FIELD_NULLABLE(float, yarn_attn_factor)
  GLUE_FIELD_NULLABLE(float, yarn_beta_fast)
  GLUE_FIELD_NULLABLE(float, yarn_beta_slow)
  GLUE_FIELD_NULLABLE(int, yarn_orig_ctx)
  GLUE_FIELD_NULLABLE(str, cache_type_k)
  GLUE_FIELD_NULLABLE(str, cache_type_v)
  GLUE_FIELD_NULLABLE(bool, flash_attn)
  GLUE_FIELD_NULLABLE(bool, swa_full)
};

struct glue_msg_load_res
{
  GLUE_HANDLER("load_res")
  GLUE_FIELD(bool, success)
  GLUE_FIELD(int, n_ctx)
  GLUE_FIELD(int, n_batch)
  GLUE_FIELD(int, n_ubatch)
  GLUE_FIELD(int, n_vocab)
  GLUE_FIELD(int, n_ctx_train)
  GLUE_FIELD(int, n_embd)
  GLUE_FIELD(int, n_layer)
  GLUE_FIELD(arr_str, metadata_key)
  GLUE_FIELD(arr_str, metadata_val)
  GLUE_FIELD(int, token_bos)
  GLUE_FIELD(int, token_eos)
  GLUE_FIELD(int, token_eot)
  GLUE_FIELD(arr_int, list_tokens_eog)
  GLUE_FIELD(bool, add_bos_token)
  GLUE_FIELD(bool, add_eos_token)
  GLUE_FIELD(bool, has_encoder)
  GLUE_FIELD(int, token_decoder_start)
};

/////////

struct glue_msg_set_options_req
{
  GLUE_HANDLER("opti_req")
  GLUE_FIELD(bool, embeddings)
};

struct glue_msg_set_options_res
{
  GLUE_HANDLER("opti_res")
  GLUE_FIELD(bool, success)
};

/////////

struct glue_msg_sampling_init_req
{
  GLUE_HANDLER("sint_req")
  GLUE_FIELD_NULLABLE(int, mirostat)
  GLUE_FIELD_NULLABLE(float, mirostat_tau)
  GLUE_FIELD_NULLABLE(float, mirostat_eta)
  GLUE_FIELD_NULLABLE(float, temp)
  GLUE_FIELD_NULLABLE(float, top_p)
  GLUE_FIELD_NULLABLE(int, top_k)
  GLUE_FIELD_NULLABLE(int, penalty_last_n)
  GLUE_FIELD_NULLABLE(float, penalty_repeat)
  GLUE_FIELD_NULLABLE(float, penalty_freq)
  GLUE_FIELD_NULLABLE(float, penalty_present)
  GLUE_FIELD_NULLABLE(float, dynatemp_range)
  GLUE_FIELD_NULLABLE(float, dynatemp_exponent)
  GLUE_FIELD_NULLABLE(arr_str, samplers_sequence)
  GLUE_FIELD_NULLABLE(str, grammar)
  GLUE_FIELD_NULLABLE(int, n_prev)
  GLUE_FIELD_NULLABLE(int, n_probs)
  GLUE_FIELD_NULLABLE(float, min_p)
  GLUE_FIELD_NULLABLE(float, typical_p)
  GLUE_FIELD_NULLABLE(float, typ_p)
  GLUE_FIELD_NULLABLE(arr_int, logit_bias_toks)
  GLUE_FIELD_NULLABLE(arr_float, logit_bias_vals)
  GLUE_FIELD_NULLABLE(arr_int, tokens)
};

struct glue_msg_sampling_init_res
{
  GLUE_HANDLER("sint_res")
  GLUE_FIELD(bool, success)
};

/////////

struct glue_msg_get_vocab_req
{
  GLUE_HANDLER("gvoc_req")
};

struct glue_msg_get_vocab_res
{
  GLUE_HANDLER("gvoc_res")
  GLUE_FIELD(bool, success)
  GLUE_FIELD(arr_raw, vocab)
};

/////////

struct glue_msg_lookup_token_req
{
  GLUE_HANDLER("lkup_req")
  GLUE_FIELD(str, piece) // TODO: maybe use raw instead
};

struct glue_msg_lookup_token_res
{
  GLUE_HANDLER("lkup_res")
  GLUE_FIELD(bool, success)
  GLUE_FIELD(int, token)
};

/////////

struct glue_msg_tokenize_req
{
  GLUE_HANDLER("tokn_req")
  GLUE_FIELD(str, text)
  GLUE_FIELD(bool, special)
};

struct glue_msg_tokenize_res
{
  GLUE_HANDLER("tokn_res")
  GLUE_FIELD(bool, success)
  GLUE_FIELD(arr_int, tokens)
};

/////////

struct glue_msg_detokenize_req
{
  GLUE_HANDLER("dtkn_req")
  GLUE_FIELD(arr_int, tokens)
};

struct glue_msg_detokenize_res
{
  GLUE_HANDLER("dtkn_res")
  GLUE_FIELD(bool, success)
  GLUE_FIELD(raw, buffer)
};

/////////

struct glue_msg_decode_req
{
  GLUE_HANDLER("deco_req")
  GLUE_FIELD(arr_int, tokens)
  GLUE_FIELD(bool, skip_logits)
};

struct glue_msg_decode_res
{
  GLUE_HANDLER("deco_res")
  GLUE_FIELD(bool, success)
  GLUE_FIELD(str, message)
  GLUE_FIELD(int, n_past)
};

/////////

struct glue_msg_encode_req
{
  GLUE_HANDLER("enco_req")
  GLUE_FIELD(arr_int, tokens)
};

struct glue_msg_encode_res
{
  GLUE_HANDLER("enco_res")
  GLUE_FIELD(bool, success)
  GLUE_FIELD(str, message)
  GLUE_FIELD(int, n_past)
};

/////////

struct glue_msg_sampling_sample_req
{
  GLUE_HANDLER("ssam_req")
};

struct glue_msg_sampling_sample_res
{
  GLUE_HANDLER("ssam_res")
  GLUE_FIELD(bool, success)
  GLUE_FIELD(raw, piece)
  GLUE_FIELD(int, token)
};

/////////

struct glue_msg_sampling_accept_req
{
  GLUE_HANDLER("sacc_req")
  GLUE_FIELD(arr_int, tokens)
};

struct glue_msg_sampling_accept_res
{
  GLUE_HANDLER("sacc_res")
  GLUE_FIELD(bool, success)
};

/////////

struct glue_msg_get_logits_req
{
  GLUE_HANDLER("glog_req")
  GLUE_FIELD(int, top_k)
};

struct glue_msg_get_logits_res
{
  GLUE_HANDLER("glog_res")
  GLUE_FIELD(bool, success)
  GLUE_FIELD(arr_int, tokens)
  GLUE_FIELD(arr_float, probs)
};

/////////

struct glue_msg_get_embeddings_req
{
  GLUE_HANDLER("gemb_req")
  GLUE_FIELD(arr_int, tokens)
};

struct glue_msg_get_embeddings_res
{
  GLUE_HANDLER("gemb_res")
  GLUE_FIELD(bool, success)
  GLUE_FIELD(str, message)
  GLUE_FIELD(arr_float, embeddings)
};

/////////

struct glue_msg_get_kv_remove_req
{
  GLUE_HANDLER("kvcr_req")
  GLUE_FIELD(int, n_keep)
  GLUE_FIELD(int, n_discard)
};

struct glue_msg_get_kv_remove_res
{
  GLUE_HANDLER("kvcr_res")
  GLUE_FIELD(int, n_past)
  GLUE_FIELD(bool, success)
};

/////////

struct glue_msg_get_kv_clear_req
{
  GLUE_HANDLER("kvcc_req")
};

struct glue_msg_get_kv_clear_res
{
  GLUE_HANDLER("kvcc_res")
  GLUE_FIELD(int, n_past)
  GLUE_FIELD(bool, success)
};

/////////

struct glue_msg_session_save_req
{
  GLUE_HANDLER("sesa_req")
  GLUE_FIELD(str, session_path)
};

struct glue_msg_session_save_res
{
  GLUE_HANDLER("sesa_res")
  GLUE_FIELD(bool, success)
  GLUE_FIELD(arr_int, tokens)
};

/////////

struct glue_msg_session_load_req
{
  GLUE_HANDLER("sesl_req")
  GLUE_FIELD(str, session_path)
  GLUE_FIELD(arr_int, tokens)
};

struct glue_msg_session_load_res
{
  GLUE_HANDLER("sesl_res")
  GLUE_FIELD(bool, success)
};

/////////

struct glue_msg_status_req
{
  GLUE_HANDLER("stat_req")
};

struct glue_msg_status_res
{
  GLUE_HANDLER("stat_res")
  GLUE_FIELD(bool, success)
  GLUE_FIELD(arr_int, tokens)
};

/////////

struct glue_msg_test_benchmark_req
{
  GLUE_HANDLER("tben_req")
  GLUE_FIELD(str, type)
  GLUE_FIELD(int, n_samples)
};

struct glue_msg_test_benchmark_res
{
  GLUE_HANDLER("tben_res")
  GLUE_FIELD(bool, success)
  GLUE_FIELD(str, message)
  GLUE_FIELD(int, t_ms)
};

/////////

struct glue_msg_test_perplexity_req
{
  GLUE_HANDLER("tper_req")
  GLUE_FIELD(arr_int, tokens)
};

struct glue_msg_test_perplexity_res
{
  GLUE_HANDLER("tper_res")
  GLUE_FIELD(bool, success)
  GLUE_FIELD(str, message)
  GLUE_FIELD(float, ppl)
  GLUE_FIELD(float, nll)
  GLUE_FIELD(float, cross_entropy)
  GLUE_FIELD(int, n_tokens)
  GLUE_FIELD(int, t_ms)
};

/////////

struct glue_msg_chat_format_req
{
  GLUE_HANDLER("cfmt_req")
  GLUE_FIELD_NULLABLE(str, tmpl)
  GLUE_FIELD_NULLABLE(bool, add_ass)
  GLUE_FIELD(arr_str, roles)
  GLUE_FIELD(arr_str, contents)
};

struct glue_msg_chat_format_res
{
  GLUE_HANDLER("cfmt_res")
  GLUE_FIELD(bool, success)
  GLUE_FIELD(str, message)
  GLUE_FIELD(str, formatted_chat)
};
