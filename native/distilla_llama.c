/* See distilla_llama.h. */
#include "distilla_llama.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "llama.h"

struct dl_model {
    struct llama_model *model;
    const struct llama_vocab *vocab;
};

struct dl_ctx {
    dl_model *model;
    struct llama_context *ctx;
    int32_t n_batch;
    int32_t n_embd;
};

static int dl_backend_ready = 0;
static int dl_is_quiet = 1;

static void dl_log(enum ggml_log_level level, const char *text, void *user) {
    (void)user;
    if (!dl_is_quiet || level >= GGML_LOG_LEVEL_ERROR) {
        fputs(text, stderr);
    }
}

static void dl_init(void) {
    if (!dl_backend_ready) {
        llama_log_set(dl_log, NULL);
        llama_backend_init();
        dl_backend_ready = 1;
    }
}

void dl_quiet(int quiet) {
    dl_is_quiet = quiet;
    llama_log_set(dl_log, NULL);
}

static bool dl_no_progress(float progress, void *user) {
    (void)progress;
    (void)user;
    return true;
}

dl_model *dl_model_load(const char *path, int32_t n_gpu_layers, int32_t vocab_only) {
    dl_init();
    struct llama_model_params params = llama_model_default_params();
    if (dl_is_quiet) {
        params.progress_callback = dl_no_progress;
    }
    params.n_gpu_layers = n_gpu_layers < 0 ? 999 : n_gpu_layers;
    params.vocab_only = vocab_only != 0;
    struct llama_model *model = llama_model_load_from_file(path, params);
    if (model == NULL) {
        return NULL;
    }
    dl_model *out = (dl_model *)calloc(1, sizeof(dl_model));
    out->model = model;
    out->vocab = llama_model_get_vocab(model);
    return out;
}

void dl_model_free(dl_model *model) {
    if (model != NULL) {
        llama_model_free(model->model);
        free(model);
    }
}

int32_t dl_model_n_embd(const dl_model *model) { return llama_model_n_embd(model->model); }
int32_t dl_model_n_ctx_train(const dl_model *model) { return llama_model_n_ctx_train(model->model); }
int32_t dl_token_eos(const dl_model *model) { return llama_vocab_eos(model->vocab); }
int32_t dl_token_bos(const dl_model *model) { return llama_vocab_bos(model->vocab); }
int32_t dl_token_is_eog(const dl_model *model, int32_t token) { return llama_vocab_is_eog(model->vocab, token) ? 1 : 0; }
int32_t dl_vocab_add_bos(const dl_model *model) { return llama_vocab_get_add_bos(model->vocab) ? 1 : 0; }
int32_t dl_vocab_add_eos(const dl_model *model) { return llama_vocab_get_add_eos(model->vocab) ? 1 : 0; }

int32_t dl_tokenize(const dl_model *model, const char *text, int32_t text_len,
                    int32_t *out, int32_t cap, int32_t add_special,
                    int32_t parse_special) {
    return llama_tokenize(model->vocab, text, text_len, out, cap,
                          add_special != 0, parse_special != 0);
}

int32_t dl_detokenize(const dl_model *model, const int32_t *tokens,
                      int32_t n_tokens, char *out, int32_t cap,
                      int32_t render_special) {
    return llama_detokenize(model->vocab, tokens, n_tokens, out, cap,
                            render_special == 0, render_special != 0);
}

dl_ctx *dl_ctx_new(dl_model *model, int32_t n_ctx, int32_t n_batch,
                   int32_t n_threads, int32_t embeddings, int32_t pooling,
                   int32_t n_seq_max, float yarn_factor,
                   int32_t yarn_orig_ctx) {
    dl_init();
    struct llama_context_params params = llama_context_default_params();
    params.n_ctx = (uint32_t)n_ctx;
    params.n_batch = (uint32_t)n_batch;
    params.n_ubatch = (uint32_t)n_batch;
    params.n_seq_max = (uint32_t)(n_seq_max > 0 ? n_seq_max : 1);
    if (n_threads > 0) {
        params.n_threads = n_threads;
        params.n_threads_batch = n_threads;
    }
    params.embeddings = embeddings != 0;
    if (embeddings) {
        params.pooling_type = (enum llama_pooling_type)pooling;
        params.kv_unified = true;
    }
    if (yarn_factor > 0.0f) {
        params.rope_scaling_type = LLAMA_ROPE_SCALING_TYPE_YARN;
        params.rope_freq_scale = 1.0f / yarn_factor;
        params.yarn_orig_ctx = (uint32_t)yarn_orig_ctx;
    }
    struct llama_context *ctx = llama_init_from_model(model->model, params);
    if (ctx == NULL) {
        return NULL;
    }
    dl_ctx *out = (dl_ctx *)calloc(1, sizeof(dl_ctx));
    out->model = model;
    out->ctx = ctx;
    out->n_batch = n_batch;
    out->n_embd = llama_model_n_embd(model->model);
    return out;
}

void dl_ctx_free(dl_ctx *ctx) {
    if (ctx != NULL) {
        llama_free(ctx->ctx);
        free(ctx);
    }
}

int32_t dl_ctx_n_ctx(const dl_ctx *ctx) { return (int32_t)llama_n_ctx(ctx->ctx); }

void dl_ctx_clear(dl_ctx *ctx) { llama_memory_clear(llama_get_memory(ctx->ctx), true); }

int32_t dl_kv_truncate(dl_ctx *ctx, int32_t pos) {
    if (pos <= 0) {
        llama_memory_clear(llama_get_memory(ctx->ctx), true);
        return 1;
    }
    return llama_memory_seq_rm(llama_get_memory(ctx->ctx), 0, pos, -1) ? 1 : 0;
}

/* Decode rows [0, n) of either tokens or embeddings, n_batch at a time. */
static int32_t dl_decode_rows(dl_ctx *ctx, const int32_t *tokens,
                              const float *embd, int32_t n, int32_t pos0,
                              int32_t logits_last) {
    int32_t chunk = ctx->n_batch;
    for (int32_t start = 0; start < n; start += chunk) {
        int32_t count = n - start < chunk ? n - start : chunk;
        struct llama_batch batch = llama_batch_init(count, embd ? ctx->n_embd : 0, 1);
        for (int32_t i = 0; i < count; i++) {
            if (embd) {
                memcpy(batch.embd + (size_t)i * ctx->n_embd,
                       embd + (size_t)(start + i) * ctx->n_embd,
                       sizeof(float) * (size_t)ctx->n_embd);
            } else {
                batch.token[i] = tokens[start + i];
            }
            batch.pos[i] = pos0 + start + i;
            batch.n_seq_id[i] = 1;
            batch.seq_id[i][0] = 0;
            batch.logits[i] = (logits_last && start + i == n - 1) ? 1 : 0;
        }
        batch.n_tokens = count;
        int32_t rc = llama_decode(ctx->ctx, batch);
        llama_batch_free(batch);
        if (rc != 0) {
            return rc;
        }
    }
    return 0;
}

int32_t dl_decode_tokens(dl_ctx *ctx, const int32_t *tokens, int32_t n,
                         int32_t pos0, int32_t logits_last) {
    return dl_decode_rows(ctx, tokens, NULL, n, pos0, logits_last);
}

int32_t dl_decode_embd(dl_ctx *ctx, const float *embd, int32_t n,
                       int32_t pos0, int32_t logits_last) {
    return dl_decode_rows(ctx, NULL, embd, n, pos0, logits_last);
}

int32_t dl_generate(dl_ctx *ctx, int32_t pos, int32_t max_new,
                    float temperature, float top_p, uint32_t seed,
                    int32_t *out, int32_t cap, int32_t *n_decoded, int32_t *stop_token) {
    struct llama_sampler *chain =
        llama_sampler_chain_init(llama_sampler_chain_default_params());
    if (temperature <= 0.0f) {
        llama_sampler_chain_add(chain, llama_sampler_init_greedy());
    } else {
        llama_sampler_chain_add(chain, llama_sampler_init_top_p(top_p, 1));
        llama_sampler_chain_add(chain, llama_sampler_init_temp(temperature));
        llama_sampler_chain_add(chain, llama_sampler_init_dist(seed));
    }
    int32_t n_ctx = (int32_t)llama_n_ctx(ctx->ctx);
    int32_t count = 0;
    int32_t decoded = 0;
    if (stop_token) {
        *stop_token = -1;
    }
    while (count < max_new && count < cap && pos < n_ctx) {
        int32_t token = llama_sampler_sample(chain, ctx->ctx, -1);
        if (llama_vocab_is_eog(ctx->model->vocab, token)) {
            if (stop_token) {
                *stop_token = token;
            }
            break;
        }
        out[count++] = token;
        if (count == max_new) {
            break;
        }
        int32_t rc = dl_decode_rows(ctx, &token, NULL, 1, pos, 1);
        if (rc != 0) {
            llama_sampler_free(chain);
            return -1;
        }
        decoded++;
        pos++;
    }
    llama_sampler_free(chain);
    if (n_decoded) {
        *n_decoded = decoded;
    }
    return count;
}

int32_t dl_embed(dl_ctx *ctx, const int32_t *tokens, const int32_t *offsets,
                 const int32_t *lengths, int32_t n_seqs, float *out) {
    int32_t total = 0;
    for (int32_t s = 0; s < n_seqs; s++) {
        total += lengths[s];
    }
    if (total > ctx->n_batch) {
        return -2;
    }
    llama_memory_clear(llama_get_memory(ctx->ctx), true);
    struct llama_batch batch = llama_batch_init(total, 0, 1);
    int32_t row = 0;
    for (int32_t s = 0; s < n_seqs; s++) {
        for (int32_t i = 0; i < lengths[s]; i++) {
            batch.token[row] = tokens[offsets[s] + i];
            batch.pos[row] = i;
            batch.n_seq_id[row] = 1;
            batch.seq_id[row][0] = s;
            batch.logits[row] = 1;
            row++;
        }
    }
    batch.n_tokens = total;
    int32_t rc = llama_decode(ctx->ctx, batch);
    llama_batch_free(batch);
    if (rc != 0) {
        return rc;
    }
    for (int32_t s = 0; s < n_seqs; s++) {
        const float *vec = llama_get_embeddings_seq(ctx->ctx, s);
        if (vec == NULL) {
            return -3;
        }
        memcpy(out + (size_t)s * ctx->n_embd, vec, sizeof(float) * (size_t)ctx->n_embd);
    }
    return 0;
}

int32_t dl_logits_last(dl_ctx *ctx, float *out) {
    const float *logits = llama_get_logits_ith(ctx->ctx, -1);
    if (logits == NULL) {
        return -1;
    }
    int32_t n_vocab = llama_vocab_n_tokens(ctx->model->vocab);
    memcpy(out, logits, sizeof(float) * (size_t)n_vocab);
    return n_vocab;
}

int32_t dl_model_n_vocab(const dl_model *model) { return llama_vocab_n_tokens(model->vocab); }
