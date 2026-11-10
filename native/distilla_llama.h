/*
 * distilla_llama: a narrow C surface over llama.cpp for jolt.ffi.
 *
 * llama.cpp passes its parameter structs and batches by value and changes
 * their layout between releases, so the jolt side binds these functions
 * instead: every argument is a scalar or a pointer, and the shim is rebuilt
 * against the pinned llama.cpp commit (`jolt native`).
 *
 * The decoder runs a prompt as consecutive segments into one KV cache, so a
 * prompt "prefix tokens, memory vectors, suffix tokens" is the same
 * computation as the Python version's scatter of memory vectors into the
 * <|memory|> placeholder embeddings.
 */
#ifndef DISTILLA_LLAMA_H
#define DISTILLA_LLAMA_H

#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef struct dl_model dl_model;
typedef struct dl_ctx dl_ctx;

/* Silence llama.cpp's logging (1) or let it through (0). */
void dl_quiet(int quiet);

/* Load a GGUF model. n_gpu_layers < 0 offloads every layer; vocab_only loads
 * just the tokenizer. NULL on failure. */
dl_model *dl_model_load(const char *path, int32_t n_gpu_layers, int32_t vocab_only);
void dl_model_free(dl_model *model);

int32_t dl_model_n_embd(const dl_model *model);
int32_t dl_model_n_ctx_train(const dl_model *model);
int32_t dl_token_eos(const dl_model *model);
int32_t dl_token_bos(const dl_model *model);
int32_t dl_token_is_eog(const dl_model *model, int32_t token);
int32_t dl_vocab_add_bos(const dl_model *model);
int32_t dl_vocab_add_eos(const dl_model *model);

/* Tokenize UTF-8 text. Returns the token count, or -needed when cap is too
 * small. parse_special lets special tokens such as <|im_start|> through. */
int32_t dl_tokenize(const dl_model *model, const char *text, int32_t text_len,
                    int32_t *out, int32_t cap, int32_t add_special,
                    int32_t parse_special);

/* Detokenize. Special tokens are dropped unless render_special is set.
 * Returns the byte count, or -needed when cap is too small. */
int32_t dl_detokenize(const dl_model *model, const int32_t *tokens,
                      int32_t n_tokens, char *out, int32_t cap,
                      int32_t render_special);

/*
 * A context.
 *   embeddings    1 for an embedding context (pooled vectors), 0 to generate
 *   pooling       llama_pooling_type, -1 = the model's own
 *   n_seq_max     sequences one embedding batch may hold
 *   yarn_factor   > 0 turns on YaRN RoPE scaling by that factor
 *   yarn_orig_ctx the context the model was trained with (YaRN only)
 */
dl_ctx *dl_ctx_new(dl_model *model, int32_t n_ctx, int32_t n_batch,
                   int32_t n_threads, int32_t embeddings, int32_t pooling,
                   int32_t n_seq_max, float yarn_factor,
                   int32_t yarn_orig_ctx);
void dl_ctx_free(dl_ctx *ctx);
int32_t dl_ctx_n_ctx(const dl_ctx *ctx);

/* Forget the KV cache. */
void dl_ctx_clear(dl_ctx *ctx);

/* Drop the KV cache from position pos on, keeping [0, pos). 1 = done, 0 =
 * the memory can't remove a partial sequence (then clear and re-decode). */
int32_t dl_kv_truncate(dl_ctx *ctx, int32_t pos);

/* Decode a token segment at positions pos0.. (in n_batch chunks). With
 * logits_last the last position's logits are kept for sampling. 0 = ok. */
int32_t dl_decode_tokens(dl_ctx *ctx, const int32_t *tokens, int32_t n,
                         int32_t pos0, int32_t logits_last);

/* Decode n input embeddings (row-major n x n_embd floats) at pos0.. */
int32_t dl_decode_embd(dl_ctx *ctx, const float *embd, int32_t n,
                       int32_t pos0, int32_t logits_last);

/* Sample up to max_new tokens after a decode whose last logits were kept,
 * the next position being pos. temperature <= 0 is greedy. Stops at an
 * end-of-generation token (not written). Returns the count written to out,
 * or < 0 on a decode failure. *n_decoded (when not NULL) receives how many of
 * the written tokens are in the KV cache (all but a last one sampled when the
 * budget ran out). *stop_token (when not NULL) receives the end-of-generation
 * token that stopped it, or -1 when the budget or the context ran out. */
int32_t dl_generate(dl_ctx *ctx, int32_t pos, int32_t max_new,
                    float temperature, float top_p, uint32_t seed,
                    int32_t *out, int32_t cap, int32_t *n_decoded, int32_t *stop_token);

/* Embed n_seqs token sequences in one batch, sequence i being
 * tokens[offsets[i] .. offsets[i]+lengths[i]). Writes n_seqs x n_embd pooled
 * floats to out (not normalised). 0 = ok. */
int32_t dl_embed(dl_ctx *ctx, const int32_t *tokens, const int32_t *offsets,
                 const int32_t *lengths, int32_t n_seqs, float *out);

/* The logits of the last decoded position (kept with logits_last); writes
 * n_vocab floats and returns n_vocab, or -1. */
int32_t dl_logits_last(dl_ctx *ctx, float *out);
int32_t dl_model_n_vocab(const dl_model *model);

#ifdef __cplusplus
}
#endif

#endif
