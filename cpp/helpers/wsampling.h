#pragma once

#include "llama.h"

#include "wcommon.h"

#include <string>
#include <vector>

// wcommon_sampler extends llama_sampler with additional functionality:
//
//  - grammar support
//  - custom sampler logic based on the parameters
//  - history of the last accepted tokens
//  - performance metrics
//
// This goal is to have a common implementation of the sampling logic shared across the examples.
// For example, depending on the temperature, the sampling chain can be very simple (greedy) or more
// complex (top-k, top-p, etc).
//
// Another example is related to the grammar. In general, the grammar constraints applied on the full
// vocabulary can be very taxing. To improve performance, the grammar can be applied only to the sampled
// token in order to verify if it fits the grammar. And only if the token doesn't fit the grammar, the
// grammar constraints are applied to the full vocabulary and the token is resampled.
//
// The wcommon_sampler also maintains a container with the last accepted tokens. In the future, this can
// be moved into the core llama library.
//
// For convenience, the wcommon_sampler also maintains a container with the current candidate tokens.
// This can be used to access the probabilities of the rest of the non-sampled tokens.
//
// TODO: measure grammar performance
//

struct wcommon_sampler;

// llama_sampler API overloads

struct wcommon_sampler * wcommon_sampler_init(const struct llama_model * model, const struct wcommon_params_sampling & params);

void wcommon_sampler_free(struct wcommon_sampler * gsmpl);

// if accept_grammar is true, the token is accepted both by the sampling chain and the grammar
void                    wcommon_sampler_accept(struct wcommon_sampler * gsmpl, llama_token token, bool accept_grammar);
void                    wcommon_sampler_reset (struct wcommon_sampler * gsmpl);
struct wcommon_sampler * wcommon_sampler_clone (struct wcommon_sampler * gsmpl);

// arguments can be nullptr to skip printing
void wcommon_perf_print(const struct llama_context * ctx, const struct wcommon_sampler * gsmpl);

// extended sampling implementation:
//
// - set logits
// - apply the configured sampler chain
// - check if the token fits the grammar (if any)
// - if not: resample by first applying the grammar constraints and then sampling again (slower path)
//
// if grammar_first is true, the grammar is applied before the samplers (slower)
// useful in cases where all the resulting candidates (not just the sampled one) must fit the grammar
//
llama_token wcommon_sampler_sample(struct wcommon_sampler * gsmpl, struct llama_context * ctx, int idx, bool grammar_first = false);

// generalized version of wcommon_sampler_sample
//
// will cross-reference the sampled tokens with a batch of draft tokens and accept those that match
// if the sampler disagrees at some point, we stop and return the accepted tokens up to now
//
//      wcommon_sampler_sample_n(gsmpl, ctx, { idx }, {});
//
// is equivalent to
//
//      wcommon_sampler_sample(gsmpl, ctx, idx);
//      wcommon_sampler_accept(gsmpl, token, true);
//
// requires: idxs.size() == draft.size() + 1
//
// returns at least 1 token, up to idxs.size()
//
std::vector<llama_token> wcommon_sampler_sample_and_accept_n(struct wcommon_sampler * gsmpl, struct llama_context * ctx, const std::vector<int> & idxs, const llama_tokens & draft, bool grammar_first = false);

// assume idxs == [ 0, 1, 2, ..., draft.size() ]
std::vector<llama_token> wcommon_sampler_sample_and_accept_n(struct wcommon_sampler * gsmpl, struct llama_context * ctx, const llama_tokens & draft, bool grammar_first = false);

uint32_t wcommon_sampler_get_seed(const struct wcommon_sampler * gsmpl);

// helpers

// access the internal list of current candidate tokens
llama_token_data_array * wcommon_sampler_get_candidates(struct wcommon_sampler * gsmpl);

// get the last accepted token
llama_token wcommon_sampler_last(const struct wcommon_sampler * gsmpl);

// print the sampler chain into a string
std::string wcommon_sampler_print(const struct wcommon_sampler * gsmpl);

// get a string representation of the last accepted tokens
std::string wcommon_sampler_prev_str(wcommon_sampler * gsmpl, llama_context * ctx, int n);

char        wcommon_sampler_type_to_chr(enum wcommon_sampler_type cnstr);
std::string wcommon_sampler_type_to_str(enum wcommon_sampler_type cnstr);

std::vector<enum wcommon_sampler_type> wcommon_sampler_types_from_names(const std::vector<std::string> & names, bool allow_alt_names);
std::vector<enum wcommon_sampler_type> wcommon_sampler_types_from_chars(const std::string & chars);

llama_sampler * llama_sampler_init_llg(const llama_vocab * vocab,
                const char * grammar_kind, const char * grammar_data);
