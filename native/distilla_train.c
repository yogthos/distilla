/*
 * distilla_train: the frozen decoder's backward pass for training the memory
 * module.
 *
 * Training needs d(loss)/d(memory tokens): the answer-token cross-entropy of
 * the frozen decoder, differentiated with respect to the input embeddings at
 * the <|memory|> positions. llama.cpp's public API trains model tensors, not
 * inputs, so this file builds the decoder's forward graph itself from the
 * GGUF weights (Llama / Qwen2 / Qwen3 layouts: RMSNorm, optional QK-norm and
 * QKV biases, RoPE, grouped-query attention, SwiGLU) with the memory vectors
 * as the only parameter, and lets ggml's autodiff produce the gradient. The
 * decoder's weights are never updated.
 *
 * The weights are read with the gguf API, as lev's lev_clef.cpp reads rows
 * out of a GGUF, and placed on the GPU when there is one. ggml's gradient
 * through a weight matmul is out_prod(W, grad^T): the CPU computes it for any
 * weight type, CUDA only for F32 weights (load with f32_weights), otherwise
 * the backend scheduler runs those products on the CPU.
 */
#define _POSIX_C_SOURCE 200809L
#define _FILE_OFFSET_BITS 64
#include "distilla_train.h"

#include <math.h>
#include <stdarg.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "ggml-alloc.h"
#include "ggml-backend.h"
#include "ggml-cpu.h"
#include "ggml.h"
#include "gguf.h"

#define DT_GRAPH_SIZE 65536
#define DT_MAX_LAYERS 512

static char dt_error[512];

const char *dt_last_error(void) { return dt_error; }

static void dt_fail(const char *fmt, ...) {
    va_list ap;
    va_start(ap, fmt);
    vsnprintf(dt_error, sizeof dt_error, fmt, ap);
    va_end(ap);
}

struct dt_layer {
    struct ggml_tensor *attn_norm, *wq, *wk, *wv, *wo, *bq, *bk, *bv, *q_norm, *k_norm;
    struct ggml_tensor *ffn_norm, *w_gate, *w_up, *w_down;
};

struct dt_model {
    int32_t n_embd, n_layer, n_head, n_head_kv, head_dim, n_rot, n_vocab, rope_mode;
    float rms_eps, freq_base;
    struct ggml_context *wctx;
    ggml_backend_buffer_t wbuf;
    ggml_backend_t gpu, cpu;
    ggml_backend_sched_t sched;
    int32_t chunk;          /* query rows per chunk of the layer-wise backward */
    int32_t full_max;       /* sequences up to this length use one whole graph */
    struct ggml_tensor *tok_embd, *output_norm, *output;
    struct dt_layer layers[DT_MAX_LAYERS];
};

/* --- metadata -------------------------------------------------------------- */

static int64_t kv(struct gguf_context *g, const char *arch, const char *suffix) {
    char key[256];
    snprintf(key, sizeof key, "%s.%s", arch, suffix);
    return gguf_find_key(g, key);
}

static int dt_int(struct gguf_context *g, const char *arch, const char *suffix, int dflt) {
    int64_t id = kv(g, arch, suffix);
    if (id < 0) return dflt;
    switch (gguf_get_kv_type(g, id)) {
        case GGUF_TYPE_UINT32: return (int)gguf_get_val_u32(g, id);
        case GGUF_TYPE_INT32: return (int)gguf_get_val_i32(g, id);
        case GGUF_TYPE_UINT64: return (int)gguf_get_val_u64(g, id);
        default: return dflt;
    }
}

static float dt_float(struct gguf_context *g, const char *arch, const char *suffix, float dflt) {
    int64_t id = kv(g, arch, suffix);
    if (id < 0 || gguf_get_kv_type(g, id) != GGUF_TYPE_FLOAT32) return dflt;
    return gguf_get_val_f32(g, id);
}

/* --- weights ----------------------------------------------------------------- */

/* A tensor of the weight context mirroring `name` of the file (NULL when the
 * file lacks it and it is optional). f32 asks for a float32 copy. */
static struct ggml_tensor *dt_weight(struct dt_model *m, struct ggml_context *meta, const char *name,
                                     int required, int f32) {
    struct ggml_tensor *src = ggml_get_tensor(meta, name);
    if (!src) {
        if (required) dt_fail("tensor %s missing from the GGUF", name);
        return NULL;
    }
    enum ggml_type type = f32 && src->type != GGML_TYPE_F32 ? GGML_TYPE_F32 : src->type;
    struct ggml_tensor *t = ggml_new_tensor(m->wctx, type, GGML_MAX_DIMS, src->ne);
    ggml_set_name(t, name);
    return t;
}

static int dt_read_weights(struct dt_model *m, const char *path, struct gguf_context *g, struct ggml_context *meta) {
    FILE *f = fopen(path, "rb");
    if (!f) {
        dt_fail("cannot open %s", path);
        return -1;
    }
    size_t base = gguf_get_data_offset(g);
    void *raw = NULL;
    float *conv = NULL;
    size_t raw_cap = 0, conv_cap = 0;
    int rc = 0;
    for (struct ggml_tensor *t = ggml_get_first_tensor(m->wctx); t && rc == 0; t = ggml_get_next_tensor(m->wctx, t)) {
        struct ggml_tensor *src = ggml_get_tensor(meta, ggml_get_name(t));
        int64_t id = gguf_find_tensor(g, ggml_get_name(t));
        size_t nbytes = ggml_nbytes(src);
        if (nbytes > raw_cap) {
            free(raw);
            raw = malloc(nbytes);
            raw_cap = nbytes;
        }
        if (!raw || fseeko(f, (off_t)(base + gguf_get_tensor_offset(g, id)), SEEK_SET) != 0 ||
            fread(raw, 1, nbytes, f) != nbytes) {
            dt_fail("cannot read tensor %s", ggml_get_name(t));
            rc = -1;
            break;
        }
        if (t->type == src->type) {
            ggml_backend_tensor_set(t, raw, 0, nbytes);
        } else {
            const struct ggml_type_traits *tr = ggml_get_type_traits(src->type);
            size_t n = (size_t)ggml_nelements(src);
            if (!tr || !tr->to_float) {
                dt_fail("no float conversion for %s (%s)", ggml_get_name(t), ggml_type_name(src->type));
                rc = -1;
                break;
            }
            if (n * sizeof(float) > conv_cap) {
                free(conv);
                conv = malloc(n * sizeof(float));
                conv_cap = n * sizeof(float);
            }
            tr->to_float(raw, conv, (int64_t)n);
            ggml_backend_tensor_set(t, conv, 0, n * sizeof(float));
        }
    }
    free(raw);
    free(conv);
    fclose(f);
    return rc;
}

dt_model *dt_model_load(const char *path, int32_t use_gpu, int32_t n_threads, int32_t f32_weights) {
    struct ggml_context *meta = NULL;
    struct gguf_init_params gp = {true, &meta};
    struct gguf_context *g = gguf_init_from_file(path, gp);
    if (!g) {
        dt_fail("cannot read GGUF %s", path);
        return NULL;
    }
    dt_model *m = calloc(1, sizeof(dt_model));
    int64_t arch_id = gguf_find_key(g, "general.architecture");
    const char *arch = arch_id < 0 ? "" : gguf_get_val_str(g, arch_id);
    if (strcmp(arch, "llama") == 0) {
        m->rope_mode = 0; /* LLAMA_ROPE_TYPE_NORM */
    } else if (strcmp(arch, "qwen2") == 0 || strcmp(arch, "qwen3") == 0) {
        m->rope_mode = GGML_ROPE_TYPE_NEOX;
    } else {
        dt_fail("unsupported architecture '%s' (llama, qwen2, qwen3)", arch);
        goto fail;
    }
    m->n_embd = dt_int(g, arch, "embedding_length", 0);
    m->n_layer = dt_int(g, arch, "block_count", 0);
    m->n_head = dt_int(g, arch, "attention.head_count", 0);
    m->n_head_kv = dt_int(g, arch, "attention.head_count_kv", m->n_head);
    m->head_dim = dt_int(g, arch, "attention.key_length", m->n_head ? m->n_embd / m->n_head : 0);
    m->n_rot = dt_int(g, arch, "rope.dimension_count", m->head_dim);
    m->rms_eps = dt_float(g, arch, "attention.layer_norm_rms_epsilon", 1e-5f);
    m->freq_base = dt_float(g, arch, "rope.freq_base", 10000.0f);
    if (m->n_embd <= 0 || m->n_layer <= 0 || m->n_layer > DT_MAX_LAYERS || m->n_head <= 0) {
        dt_fail("incomplete hyperparameters for %s", arch);
        goto fail;
    }

    size_t n_tensors = (size_t)gguf_get_n_tensors(g) + 8;
    struct ggml_init_params wp = {ggml_tensor_overhead() * n_tensors, NULL, true};
    m->wctx = ggml_init(wp);

    m->tok_embd = dt_weight(m, meta, "token_embd.weight", 1, f32_weights);
    m->output_norm = dt_weight(m, meta, "output_norm.weight", 1, 1);
    m->output = dt_weight(m, meta, "output.weight", 0, f32_weights);
    if (!m->tok_embd || !m->output_norm) goto fail;
    if (!m->output) m->output = m->tok_embd; /* tied embeddings */
    m->n_vocab = (int32_t)m->tok_embd->ne[1];

    for (int il = 0; il < m->n_layer; il++) {
        char name[128];
        struct dt_layer *l = &m->layers[il];
#define W(field, suffix, req, f32)                                         \
    snprintf(name, sizeof name, "blk.%d." suffix, il);                     \
    l->field = dt_weight(m, meta, name, req, f32);                         \
    if ((req) && !l->field) goto fail;
        W(attn_norm, "attn_norm.weight", 1, 1)
        W(wq, "attn_q.weight", 1, f32_weights)
        W(wk, "attn_k.weight", 1, f32_weights)
        W(wv, "attn_v.weight", 1, f32_weights)
        W(wo, "attn_output.weight", 1, f32_weights)
        W(bq, "attn_q.bias", 0, 1)
        W(bk, "attn_k.bias", 0, 1)
        W(bv, "attn_v.bias", 0, 1)
        W(q_norm, "attn_q_norm.weight", 0, 1)
        W(k_norm, "attn_k_norm.weight", 0, 1)
        W(ffn_norm, "ffn_norm.weight", 1, 1)
        W(w_gate, "ffn_gate.weight", 1, f32_weights)
        W(w_up, "ffn_up.weight", 1, f32_weights)
        W(w_down, "ffn_down.weight", 1, f32_weights)
#undef W
    }

    if (use_gpu) {
        ggml_backend_dev_t dev = ggml_backend_dev_by_type(GGML_BACKEND_DEVICE_TYPE_GPU);
        if (dev) m->gpu = ggml_backend_dev_init(dev, NULL);
    }
    m->cpu = ggml_backend_cpu_init();
    if (!m->cpu) {
        dt_fail("no CPU backend");
        goto fail;
    }
    if (n_threads > 0) ggml_backend_cpu_set_n_threads(m->cpu, n_threads);

    m->wbuf = ggml_backend_alloc_ctx_tensors(m->wctx, m->gpu ? m->gpu : m->cpu);
    if (!m->wbuf) {
        dt_fail("cannot allocate the decoder weights");
        goto fail;
    }
    ggml_backend_buffer_set_usage(m->wbuf, GGML_BACKEND_BUFFER_USAGE_WEIGHTS);
    if (dt_read_weights(m, path, g, meta) != 0) goto fail;

    ggml_backend_t backends[2];
    int nb = 0;
    if (m->gpu) backends[nb++] = m->gpu;
    backends[nb++] = m->cpu;
    m->sched = ggml_backend_sched_new(backends, NULL, nb, DT_GRAPH_SIZE, false, true);
    m->chunk = 256;
    m->full_max = 1024;

    gguf_free(g);
    ggml_free(meta);
    return m;

fail:
    gguf_free(g);
    ggml_free(meta);
    dt_model_free(m);
    return NULL;
}

void dt_model_free(dt_model *m) {
    if (!m) return;
    if (m->sched) ggml_backend_sched_free(m->sched);
    if (m->wbuf) ggml_backend_buffer_free(m->wbuf);
    if (m->wctx) ggml_free(m->wctx);
    if (m->gpu) ggml_backend_free(m->gpu);
    if (m->cpu) ggml_backend_free(m->cpu);
    free(m);
}

int32_t dt_model_n_embd(const dt_model *m) { return m->n_embd; }
int32_t dt_model_n_vocab(const dt_model *m) { return m->n_vocab; }
int32_t dt_model_on_gpu(const dt_model *m) { return m->gpu != NULL; }

/* --- the graph ----------------------------------------------------------------- */

static struct ggml_tensor *dt_linear(struct ggml_context *ctx, struct ggml_tensor *w, struct ggml_tensor *b,
                                     struct ggml_tensor *x) {
    struct ggml_tensor *y = ggml_mul_mat(ctx, w, x);
    return b ? ggml_add(ctx, y, b) : y;
}

static int32_t dt_step_layerwise(dt_model *m, const int32_t *tokens, const int32_t *mem_index, const int32_t *targets,
                                 int32_t n, const float *mem, int32_t n_mem, float *grad_out, float *loss_out,
                                 int32_t want_grad);

void dt_set_chunking(dt_model *m, int32_t chunk, int32_t full_max) {
    if (chunk > 0) m->chunk = chunk;
    if (full_max >= 0) m->full_max = full_max;
}

int32_t dt_step(dt_model *m, const int32_t *tokens, const int32_t *mem_index, const int32_t *targets, int32_t n,
                const float *mem, int32_t n_mem, float *grad_out, float *loss_out, int32_t want_grad) {
    if (n > m->full_max) {
        return dt_step_layerwise(m, tokens, mem_index, targets, n, mem, n_mem, grad_out, loss_out, want_grad);
    }
    return dt_step_full(m, tokens, mem_index, targets, n, mem, n_mem, grad_out, loss_out, want_grad);
}

int32_t dt_step_full(dt_model *m, const int32_t *tokens, const int32_t *mem_index, const int32_t *targets, int32_t n,
                const float *mem, int32_t n_mem, float *grad_out, float *loss_out, int32_t want_grad) {
    int32_t n_t = 0;
    for (int32_t i = 0; i < n; i++) n_t += targets[i] >= 0;
    if (n <= 0 || n_t == 0) {
        dt_fail("nothing to score: %d positions, %d targets", n, n_t);
        return -1;
    }

    size_t ctx_size = ggml_tensor_overhead() * DT_GRAPH_SIZE * 2 + ggml_graph_overhead_custom(DT_GRAPH_SIZE, true);
    struct ggml_init_params ip = {ctx_size, NULL, true};
    struct ggml_context *ctx = ggml_init(ip);
    const int64_t E = m->n_embd, H = m->n_head, HKV = m->n_head_kv, D = m->head_dim;

    struct ggml_tensor *tok_ids = ggml_new_tensor_1d(ctx, GGML_TYPE_I32, n);
    struct ggml_tensor *mem_ids = ggml_new_tensor_1d(ctx, GGML_TYPE_I32, n);
    struct ggml_tensor *keep = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, 1, n);
    struct ggml_tensor *pos = ggml_new_tensor_1d(ctx, GGML_TYPE_I32, n);
    struct ggml_tensor *mask = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, n, n);
    struct ggml_tensor *target_pos = ggml_new_tensor_1d(ctx, GGML_TYPE_I32, n_t);
    struct ggml_tensor *labels = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, m->n_vocab, n_t);
    /* the memory vectors plus a zero column read at the token positions */
    struct ggml_tensor *memx = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, E, n_mem + 1);
    struct ggml_tensor *inputs[] = {tok_ids, mem_ids, keep, pos, mask, target_pos, labels, memx};
    for (size_t i = 0; i < sizeof inputs / sizeof inputs[0]; i++) ggml_set_input(inputs[i]);
    ggml_set_name(memx, "memory");
    if (want_grad) ggml_set_param(memx);

    struct ggml_tensor *x = ggml_add(ctx, ggml_mul(ctx, ggml_get_rows(ctx, m->tok_embd, tok_ids), keep),
                                     ggml_get_rows(ctx, memx, mem_ids));
    const float scale = 1.0f / sqrtf((float)D);
    for (int il = 0; il < m->n_layer; il++) {
        struct dt_layer *l = &m->layers[il];
        struct ggml_tensor *cur = ggml_mul(ctx, ggml_rms_norm(ctx, x, m->rms_eps), l->attn_norm);
        struct ggml_tensor *q = ggml_reshape_3d(ctx, dt_linear(ctx, l->wq, l->bq, cur), D, H, n);
        struct ggml_tensor *k = ggml_reshape_3d(ctx, dt_linear(ctx, l->wk, l->bk, cur), D, HKV, n);
        struct ggml_tensor *v = ggml_reshape_3d(ctx, dt_linear(ctx, l->wv, l->bv, cur), D, HKV, n);
        if (l->q_norm) q = ggml_mul(ctx, ggml_rms_norm(ctx, q, m->rms_eps), l->q_norm);
        if (l->k_norm) k = ggml_mul(ctx, ggml_rms_norm(ctx, k, m->rms_eps), l->k_norm);
        q = ggml_rope_ext(ctx, q, pos, NULL, m->n_rot, m->rope_mode, 0, m->freq_base, 1.0f, 0.0f, 1.0f, 32.0f, 1.0f);
        k = ggml_rope_ext(ctx, k, pos, NULL, m->n_rot, m->rope_mode, 0, m->freq_base, 1.0f, 0.0f, 1.0f, 32.0f, 1.0f);

        q = ggml_cont(ctx, ggml_permute(ctx, q, 0, 2, 1, 3));                  /* [D, n, H]   */
        k = ggml_cont(ctx, ggml_permute(ctx, k, 0, 2, 1, 3));                  /* [D, n, HKV] */
        struct ggml_tensor *vt = ggml_cont(ctx, ggml_permute(ctx, v, 1, 2, 0, 3)); /* [n, D, HKV] */
        struct ggml_tensor *kq = ggml_mul_mat(ctx, k, q);                      /* [n_k, n_q, H] */
        kq = ggml_soft_max_ext(ctx, kq, mask, scale, 0.0f);
        struct ggml_tensor *kqv = ggml_mul_mat(ctx, vt, kq);                   /* [D, n_q, H] */
        kqv = ggml_cont(ctx, ggml_permute(ctx, kqv, 0, 2, 1, 3));              /* [D, H, n] */
        cur = ggml_mul_mat(ctx, l->wo, ggml_reshape_2d(ctx, kqv, D * H, n));
        x = ggml_add(ctx, x, cur);

        cur = ggml_mul(ctx, ggml_rms_norm(ctx, x, m->rms_eps), l->ffn_norm);
        struct ggml_tensor *gate = ggml_silu(ctx, ggml_mul_mat(ctx, l->w_gate, cur));
        struct ggml_tensor *up = ggml_mul_mat(ctx, l->w_up, cur);
        x = ggml_add(ctx, x, ggml_mul_mat(ctx, l->w_down, ggml_mul(ctx, gate, up)));
    }
    x = ggml_mul(ctx, ggml_rms_norm(ctx, x, m->rms_eps), m->output_norm);
    struct ggml_tensor *logits = ggml_mul_mat(ctx, m->output, ggml_get_rows(ctx, x, target_pos));
    struct ggml_tensor *loss = ggml_cross_entropy_loss(ctx, logits, labels);
    ggml_set_name(loss, "loss");
    ggml_set_output(loss);
    ggml_set_loss(loss);

    struct ggml_cgraph *gf = ggml_new_graph_custom(ctx, DT_GRAPH_SIZE, true);
    ggml_build_forward_expand(gf, loss);
    struct ggml_cgraph *graph = gf;
    if (want_grad) {
        graph = ggml_graph_dup(ctx, gf, true);
        ggml_build_backward_expand(ctx, graph, NULL);
    }

    ggml_backend_sched_reset(m->sched);
    if (!ggml_backend_sched_alloc_graph(m->sched, graph)) {
        dt_fail("cannot allocate the training graph (%d positions)", n);
        ggml_free(ctx);
        return -1;
    }

    /* inputs */
    int32_t *buf_i = malloc(sizeof(int32_t) * (size_t)n);
    float *buf_f = malloc(sizeof(float) * (size_t)n * (size_t)n);
    ggml_backend_tensor_set(tok_ids, tokens, 0, sizeof(int32_t) * (size_t)n);
    for (int32_t i = 0; i < n; i++) buf_i[i] = mem_index[i] >= 0 ? mem_index[i] : n_mem;
    ggml_backend_tensor_set(mem_ids, buf_i, 0, sizeof(int32_t) * (size_t)n);
    for (int32_t i = 0; i < n; i++) buf_f[i] = mem_index[i] >= 0 ? 0.0f : 1.0f;
    ggml_backend_tensor_set(keep, buf_f, 0, sizeof(float) * (size_t)n);
    for (int32_t i = 0; i < n; i++) buf_i[i] = i;
    ggml_backend_tensor_set(pos, buf_i, 0, sizeof(int32_t) * (size_t)n);
    for (int32_t q = 0; q < n; q++)
        for (int32_t kk = 0; kk < n; kk++) buf_f[(size_t)q * n + kk] = kk <= q ? 0.0f : -INFINITY;
    ggml_backend_tensor_set(mask, buf_f, 0, sizeof(float) * (size_t)n * (size_t)n);
    int32_t j = 0;
    for (int32_t i = 0; i < n; i++)
        if (targets[i] >= 0) buf_i[j++] = i;
    ggml_backend_tensor_set(target_pos, buf_i, 0, sizeof(int32_t) * (size_t)n_t);
    ggml_backend_tensor_memset(labels, 0, 0, ggml_nbytes(labels));
    const float one = 1.0f;
    j = 0;
    for (int32_t i = 0; i < n; i++) {
        if (targets[i] < 0) continue;
        if (targets[i] >= m->n_vocab) {
            dt_fail("target %d outside the vocabulary of %d", targets[i], m->n_vocab);
            free(buf_i);
            free(buf_f);
            ggml_free(ctx);
            return -1;
        }
        ggml_backend_tensor_set(labels, &one, ((size_t)j * m->n_vocab + targets[i]) * sizeof(float), sizeof(float));
        j++;
    }
    ggml_backend_tensor_set(memx, mem, 0, sizeof(float) * (size_t)E * (size_t)n_mem);
    ggml_backend_tensor_memset(memx, 0, sizeof(float) * (size_t)E * (size_t)n_mem, sizeof(float) * (size_t)E);
    free(buf_i);
    free(buf_f);

    if (want_grad) ggml_graph_reset(graph);
    if (ggml_backend_sched_graph_compute(m->sched, graph) != GGML_STATUS_SUCCESS) {
        dt_fail("graph compute failed");
        ggml_free(ctx);
        return -1;
    }
    ggml_backend_tensor_get(loss, loss_out, 0, sizeof(float));
    if (want_grad) {
        struct ggml_tensor *grad = ggml_graph_get_grad(graph, memx);
        if (!grad) {
            dt_fail("no gradient reached the memory tokens");
            ggml_free(ctx);
            return -1;
        }
        ggml_backend_tensor_get(grad, grad_out, 0, sizeof(float) * (size_t)E * (size_t)n_mem);
    }
    ggml_free(ctx);
    return n_t;
}


/* --- the layer-wise backward ---------------------------------------------------------
 *
 * One graph over a whole sequence keeps every layer's attention probabilities
 * (n_head x n x n floats a layer) for its backward pass: gigabytes at a few
 * thousand tokens. Here the forward pass keeps only each layer's input on the
 * host; the backward pass then walks the layers from the top, rebuilding one
 * layer at a time for one chunk of query rows, and lets autodiff pull the
 * upstream gradient G through it as the gradient of sum(y (.) G). The keys and
 * values are computed once a layer and their gradients summed over the
 * chunks, then pulled back through the key/value projections in one more
 * small graph. Peak memory is n_head x chunk x n attention floats.
 */

struct dt_graph {
    struct ggml_context *ctx;
    struct ggml_cgraph *gf;
};

static struct dt_graph dt_graph_begin(void) {
    const size_t size = 8192;
    struct ggml_init_params ip = {ggml_tensor_overhead() * size * 2 + ggml_graph_overhead_custom(size, true), NULL,
                                  true};
    struct dt_graph g = {ggml_init(ip), NULL};
    return g;
}

/* Finish the graph at `out` (with its backward pass when grad), allocate it.
 * Returns 0 or -1. */
static int dt_graph_alloc(dt_model *m, struct dt_graph *g, struct ggml_tensor *out, int grad) {
    struct ggml_cgraph *gf = ggml_new_graph_custom(g->ctx, 8192, true);
    ggml_build_forward_expand(gf, out);
    if (grad) {
        struct ggml_cgraph *gb = ggml_graph_dup(g->ctx, gf, true);
        ggml_build_backward_expand(g->ctx, gb, NULL);
        gf = gb;
    }
    g->gf = gf;
    ggml_backend_sched_reset(m->sched);
    if (!ggml_backend_sched_alloc_graph(m->sched, gf)) {
        dt_fail("cannot allocate a layer graph");
        return -1;
    }
    return 0;
}

static int dt_graph_run(dt_model *m, struct dt_graph *g, int grad) {
    if (grad) ggml_graph_reset(g->gf);
    if (ggml_backend_sched_graph_compute(m->sched, g->gf) != GGML_STATUS_SUCCESS) {
        dt_fail("graph compute failed");
        return -1;
    }
    return 0;
}

static struct ggml_tensor *dt_input(struct ggml_context *ctx, enum ggml_type type, int64_t ne0, int64_t ne1,
                                    int64_t ne2) {
    struct ggml_tensor *t = ne2 > 0 ? ggml_new_tensor_3d(ctx, type, ne0, ne1, ne2)
                                    : (ne1 > 0 ? ggml_new_tensor_2d(ctx, type, ne0, ne1)
                                               : ggml_new_tensor_1d(ctx, type, ne0));
    ggml_set_input(t);
    return t;
}

/* keys [D, n, HKV] and transposed values [n, D, HKV] of layer il from its
 * input x [E, n], in the layouts the attention products take, so that no
 * parameter of a chunk graph is permuted (whose gradient would be a view) */
static void dt_build_kv(dt_model *m, struct ggml_context *ctx, int il, struct ggml_tensor *x, struct ggml_tensor *pos,
                        struct ggml_tensor **k_out, struct ggml_tensor **v_out) {
    struct dt_layer *l = &m->layers[il];
    int64_t n = x->ne[1];
    struct ggml_tensor *h = ggml_mul(ctx, ggml_rms_norm(ctx, x, m->rms_eps), l->attn_norm);
    struct ggml_tensor *k = ggml_reshape_3d(ctx, dt_linear(ctx, l->wk, l->bk, h), m->head_dim, m->n_head_kv, n);
    struct ggml_tensor *v = ggml_reshape_3d(ctx, dt_linear(ctx, l->wv, l->bv, h), m->head_dim, m->n_head_kv, n);
    if (l->k_norm) k = ggml_mul(ctx, ggml_rms_norm(ctx, k, m->rms_eps), l->k_norm);
    k = ggml_rope_ext(ctx, k, pos, NULL, m->n_rot, m->rope_mode, 0, m->freq_base, 1.0f, 0.0f, 1.0f, 32.0f, 1.0f);
    *k_out = ggml_cont(ctx, ggml_permute(ctx, k, 0, 2, 1, 3)); /* [D, n, HKV] */
    *v_out = ggml_cont(ctx, ggml_permute(ctx, v, 1, 2, 0, 3)); /* [n, D, HKV] */
}

/* layer il's output rows for the query rows xc [E, C] at positions posc, given
 * all keys kp [D, n, HKV], transposed values vt [n, D, HKV] and the mask [n, C] */
static struct ggml_tensor *dt_build_chunk(dt_model *m, struct ggml_context *ctx, int il, struct ggml_tensor *xc,
                                          struct ggml_tensor *posc, struct ggml_tensor *kp, struct ggml_tensor *vt,
                                          struct ggml_tensor *mask) {
    struct dt_layer *l = &m->layers[il];
    const int64_t C = xc->ne[1], D = m->head_dim, H = m->n_head;
    struct ggml_tensor *h = ggml_mul(ctx, ggml_rms_norm(ctx, xc, m->rms_eps), l->attn_norm);
    struct ggml_tensor *q = ggml_reshape_3d(ctx, dt_linear(ctx, l->wq, l->bq, h), D, H, C);
    if (l->q_norm) q = ggml_mul(ctx, ggml_rms_norm(ctx, q, m->rms_eps), l->q_norm);
    q = ggml_rope_ext(ctx, q, posc, NULL, m->n_rot, m->rope_mode, 0, m->freq_base, 1.0f, 0.0f, 1.0f, 32.0f, 1.0f);
    q = ggml_cont(ctx, ggml_permute(ctx, q, 0, 2, 1, 3));                       /* [D, C, H]   */
    struct ggml_tensor *kq = ggml_soft_max_ext(ctx, ggml_mul_mat(ctx, kp, q), mask, 1.0f / sqrtf((float)D), 0.0f);
    struct ggml_tensor *kqv = ggml_cont(ctx, ggml_permute(ctx, ggml_mul_mat(ctx, vt, kq), 0, 2, 1, 3));
    struct ggml_tensor *x = ggml_add(ctx, xc, ggml_mul_mat(ctx, l->wo, ggml_reshape_2d(ctx, kqv, D * H, C)));
    struct ggml_tensor *f = ggml_mul(ctx, ggml_rms_norm(ctx, x, m->rms_eps), l->ffn_norm);
    struct ggml_tensor *gate = ggml_silu(ctx, ggml_mul_mat(ctx, l->w_gate, f));
    struct ggml_tensor *up = ggml_mul_mat(ctx, l->w_up, f);
    return ggml_add(ctx, x, ggml_mul_mat(ctx, l->w_down, ggml_mul(ctx, gate, up)));
}

static void dt_fill_mask(float *mask, int32_t n, int32_t c0, int32_t C) {
    for (int32_t q = 0; q < C; q++)
        for (int32_t k = 0; k < n; k++) mask[(size_t)q * n + k] = k <= c0 + q ? 0.0f : -INFINITY;
}

static int32_t dt_step_layerwise(dt_model *m, const int32_t *tokens, const int32_t *mem_index, const int32_t *targets,
                                 int32_t n, const float *mem, int32_t n_mem, float *grad_out, float *loss_out,
                                 int32_t want_grad) {
    const int64_t E = m->n_embd, L = m->n_layer, KVW = (int64_t)m->head_dim * m->n_head_kv;
    const int32_t chunk = m->chunk;
    int32_t n_t = 0;
    for (int32_t i = 0; i < n; i++) n_t += targets[i] >= 0;
    if (n <= 0 || n_t == 0) {
        dt_fail("nothing to score: %d positions, %d targets", n, n_t);
        return -1;
    }
    int rc = -1;
    float *X = malloc(sizeof(float) * (size_t)(L + 1) * (size_t)n * (size_t)E); /* every layer's input */
    float *K = malloc(sizeof(float) * (size_t)n * (size_t)KVW);
    float *V = malloc(sizeof(float) * (size_t)n * (size_t)KVW);
    float *G = want_grad ? calloc((size_t)n * (size_t)E, sizeof(float)) : NULL;   /* upstream gradient */
    float *DX = want_grad ? calloc((size_t)n * (size_t)E, sizeof(float)) : NULL;  /* this layer's input gradient */
    float *DK = want_grad ? calloc((size_t)n * (size_t)KVW, sizeof(float)) : NULL;
    float *DV = want_grad ? calloc((size_t)n * (size_t)KVW, sizeof(float)) : NULL;
    float *maskbuf = malloc(sizeof(float) * (size_t)n * (size_t)chunk);
    int32_t *pos = malloc(sizeof(int32_t) * (size_t)n);
    int32_t *ibuf = malloc(sizeof(int32_t) * (size_t)n);
    float *fbuf = malloc(sizeof(float) * (size_t)n);
    if (!X || !K || !V || !maskbuf || !pos || !ibuf || !fbuf || (want_grad && (!G || !DX || !DK || !DV))) {
        dt_fail("out of host memory for %d positions", n);
        goto done;
    }
    for (int32_t i = 0; i < n; i++) pos[i] = i;
#define LAYER(l) (X + (size_t)(l) * (size_t)n * (size_t)E)

    /* the input embeddings */
    {
        struct dt_graph g = dt_graph_begin();
        struct ggml_tensor *ids = dt_input(g.ctx, GGML_TYPE_I32, n, 0, 0);
        struct ggml_tensor *mids = dt_input(g.ctx, GGML_TYPE_I32, n, 0, 0);
        struct ggml_tensor *keep = dt_input(g.ctx, GGML_TYPE_F32, 1, n, 0);
        struct ggml_tensor *memx = dt_input(g.ctx, GGML_TYPE_F32, E, n_mem + 1, 0);
        struct ggml_tensor *x0 = ggml_add(g.ctx, ggml_mul(g.ctx, ggml_get_rows(g.ctx, m->tok_embd, ids), keep),
                                          ggml_get_rows(g.ctx, memx, mids));
        ggml_set_output(x0);
        if (dt_graph_alloc(m, &g, x0, 0) != 0) {
            ggml_free(g.ctx);
            goto done;
        }
        ggml_backend_tensor_set(ids, tokens, 0, sizeof(int32_t) * (size_t)n);
        for (int32_t i = 0; i < n; i++) ibuf[i] = mem_index[i] >= 0 ? mem_index[i] : n_mem;
        ggml_backend_tensor_set(mids, ibuf, 0, sizeof(int32_t) * (size_t)n);
        for (int32_t i = 0; i < n; i++) fbuf[i] = mem_index[i] >= 0 ? 0.0f : 1.0f;
        ggml_backend_tensor_set(keep, fbuf, 0, sizeof(float) * (size_t)n);
        ggml_backend_tensor_set(memx, mem, 0, sizeof(float) * (size_t)E * (size_t)n_mem);
        ggml_backend_tensor_memset(memx, 0, sizeof(float) * (size_t)E * (size_t)n_mem, sizeof(float) * (size_t)E);
        int r = dt_graph_run(m, &g, 0);
        if (r == 0) ggml_backend_tensor_get(x0, LAYER(0), 0, sizeof(float) * (size_t)n * (size_t)E);
        ggml_free(g.ctx);
        if (r != 0) goto done;
    }

    /* forward, layer by layer */
    for (int il = 0; il < L; il++) {
        struct dt_graph g = dt_graph_begin();
        struct ggml_tensor *x = dt_input(g.ctx, GGML_TYPE_F32, E, n, 0);
        struct ggml_tensor *p = dt_input(g.ctx, GGML_TYPE_I32, n, 0, 0);
        struct ggml_tensor *k, *v;
        dt_build_kv(m, g.ctx, il, x, p, &k, &v);
        struct ggml_tensor *kv = ggml_concat(g.ctx, ggml_reshape_1d(g.ctx, k, KVW * n), ggml_reshape_1d(g.ctx, v, KVW * n), 0);
        ggml_set_output(kv);
        if (dt_graph_alloc(m, &g, kv, 0) != 0) {
            ggml_free(g.ctx);
            goto done;
        }
        ggml_backend_tensor_set(x, LAYER(il), 0, sizeof(float) * (size_t)n * (size_t)E);
        ggml_backend_tensor_set(p, pos, 0, sizeof(int32_t) * (size_t)n);
        int r = dt_graph_run(m, &g, 0);
        if (r == 0) {
            ggml_backend_tensor_get(kv, K, 0, sizeof(float) * (size_t)n * (size_t)KVW);
            ggml_backend_tensor_get(kv, V, sizeof(float) * (size_t)n * (size_t)KVW, sizeof(float) * (size_t)n * (size_t)KVW);
        }
        ggml_free(g.ctx);
        if (r != 0) goto done;

        for (int32_t c0 = 0; c0 < n; c0 += chunk) {
            int32_t C = n - c0 < chunk ? n - c0 : chunk;
            struct dt_graph h = dt_graph_begin();
            struct ggml_tensor *xc = dt_input(h.ctx, GGML_TYPE_F32, E, C, 0);
            struct ggml_tensor *pc = dt_input(h.ctx, GGML_TYPE_I32, C, 0, 0);
            struct ggml_tensor *kt = dt_input(h.ctx, GGML_TYPE_F32, m->head_dim, n, m->n_head_kv);
            struct ggml_tensor *vtn = dt_input(h.ctx, GGML_TYPE_F32, n, m->head_dim, m->n_head_kv);
            struct ggml_tensor *mk = dt_input(h.ctx, GGML_TYPE_F32, n, C, 0);
            struct ggml_tensor *y = dt_build_chunk(m, h.ctx, il, xc, pc, kt, vtn, mk);
            ggml_set_output(y);
            if (dt_graph_alloc(m, &h, y, 0) != 0) {
                ggml_free(h.ctx);
                goto done;
            }
            ggml_backend_tensor_set(xc, LAYER(il) + (size_t)c0 * E, 0, sizeof(float) * (size_t)C * (size_t)E);
            ggml_backend_tensor_set(pc, pos + c0, 0, sizeof(int32_t) * (size_t)C);
            ggml_backend_tensor_set(kt, K, 0, sizeof(float) * (size_t)n * (size_t)KVW);
            ggml_backend_tensor_set(vtn, V, 0, sizeof(float) * (size_t)n * (size_t)KVW);
            dt_fill_mask(maskbuf, n, c0, C);
            ggml_backend_tensor_set(mk, maskbuf, 0, sizeof(float) * (size_t)n * (size_t)C);
            int r2 = dt_graph_run(m, &h, 0);
            if (r2 == 0) ggml_backend_tensor_get(y, LAYER(il + 1) + (size_t)c0 * E, 0, sizeof(float) * (size_t)C * (size_t)E);
            ggml_free(h.ctx);
            if (r2 != 0) goto done;
        }
    }

    /* the loss over the scored positions, in chunks of targets; its gradient lands in G */
    {
        double loss_sum = 0.0;
        int32_t *tpos = malloc(sizeof(int32_t) * (size_t)n_t);
        int32_t j = 0;
        for (int32_t i = 0; i < n; i++)
            if (targets[i] >= 0) tpos[j++] = i;
        for (int32_t t0 = 0; t0 < n_t; t0 += chunk) {
            int32_t T = n_t - t0 < chunk ? n_t - t0 : chunk;
            struct dt_graph g = dt_graph_begin();
            struct ggml_tensor *xs = dt_input(g.ctx, GGML_TYPE_F32, E, T, 0);
            struct ggml_tensor *labels = dt_input(g.ctx, GGML_TYPE_F32, m->n_vocab, T, 0);
            if (want_grad) ggml_set_param(xs);
            struct ggml_tensor *f = ggml_mul(g.ctx, ggml_rms_norm(g.ctx, xs, m->rms_eps), m->output_norm);
            struct ggml_tensor *loss = ggml_scale(g.ctx, ggml_cross_entropy_loss(g.ctx, ggml_mul_mat(g.ctx, m->output, f), labels),
                                                  (float)T / (float)n_t);
            ggml_set_output(loss);
            ggml_set_loss(loss);
            if (dt_graph_alloc(m, &g, loss, want_grad) != 0) {
                ggml_free(g.ctx);
                free(tpos);
                goto done;
            }
            for (int32_t q = 0; q < T; q++)
                ggml_backend_tensor_set(xs, LAYER(L) + (size_t)tpos[t0 + q] * E, sizeof(float) * (size_t)q * E,
                                        sizeof(float) * (size_t)E);
            ggml_backend_tensor_memset(labels, 0, 0, ggml_nbytes(labels));
            const float one = 1.0f;
            for (int32_t q = 0; q < T; q++) {
                int32_t target = targets[tpos[t0 + q]];
                if (target >= m->n_vocab) {
                    dt_fail("target %d outside the vocabulary of %d", target, m->n_vocab);
                    ggml_free(g.ctx);
                    free(tpos);
                    goto done;
                }
                ggml_backend_tensor_set(labels, &one, ((size_t)q * m->n_vocab + target) * sizeof(float), sizeof(float));
            }
            int r = dt_graph_run(m, &g, want_grad);
            if (r == 0) {
                float l;
                ggml_backend_tensor_get(loss, &l, 0, sizeof(float));
                loss_sum += l;
                if (want_grad) {
                    struct ggml_tensor *gx = ggml_graph_get_grad(g.gf, xs);
                    for (int32_t q = 0; q < T; q++)
                        ggml_backend_tensor_get(gx, G + (size_t)tpos[t0 + q] * E, sizeof(float) * (size_t)q * E,
                                                sizeof(float) * (size_t)E);
                }
            }
            ggml_free(g.ctx);
            if (r != 0) {
                free(tpos);
                goto done;
            }
        }
        free(tpos);
        *loss_out = (float)loss_sum;
    }
    if (!want_grad) {
        rc = n_t;
        goto done;
    }

    /* backward, layer by layer from the top: G holds d(loss)/d(output of layer il) */
    for (int il = (int)L - 1; il >= 0; il--) {
        /* this layer's keys and values again */
        {
            struct dt_graph g = dt_graph_begin();
            struct ggml_tensor *x = dt_input(g.ctx, GGML_TYPE_F32, E, n, 0);
            struct ggml_tensor *p = dt_input(g.ctx, GGML_TYPE_I32, n, 0, 0);
            struct ggml_tensor *k, *v;
            dt_build_kv(m, g.ctx, il, x, p, &k, &v);
            struct ggml_tensor *kv = ggml_concat(g.ctx, ggml_reshape_1d(g.ctx, k, KVW * n), ggml_reshape_1d(g.ctx, v, KVW * n), 0);
            ggml_set_output(kv);
            if (dt_graph_alloc(m, &g, kv, 0) != 0) {
                ggml_free(g.ctx);
                goto done;
            }
            ggml_backend_tensor_set(x, LAYER(il), 0, sizeof(float) * (size_t)n * (size_t)E);
            ggml_backend_tensor_set(p, pos, 0, sizeof(int32_t) * (size_t)n);
            int r = dt_graph_run(m, &g, 0);
            if (r == 0) {
                ggml_backend_tensor_get(kv, K, 0, sizeof(float) * (size_t)n * (size_t)KVW);
                ggml_backend_tensor_get(kv, V, sizeof(float) * (size_t)n * (size_t)KVW, sizeof(float) * (size_t)n * (size_t)KVW);
            }
            ggml_free(g.ctx);
            if (r != 0) goto done;
        }
        memset(DK, 0, sizeof(float) * (size_t)n * (size_t)KVW);
        memset(DV, 0, sizeof(float) * (size_t)n * (size_t)KVW);

        /* each chunk of query rows: d/d(rows), d/dK, d/dV of sum(y (.) G) */
        for (int32_t c0 = 0; c0 < n; c0 += chunk) {
            int32_t C = n - c0 < chunk ? n - c0 : chunk;
            struct dt_graph h = dt_graph_begin();
            struct ggml_tensor *xc = dt_input(h.ctx, GGML_TYPE_F32, E, C, 0);
            struct ggml_tensor *pc = dt_input(h.ctx, GGML_TYPE_I32, C, 0, 0);
            struct ggml_tensor *kt = dt_input(h.ctx, GGML_TYPE_F32, m->head_dim, n, m->n_head_kv);
            struct ggml_tensor *vtn = dt_input(h.ctx, GGML_TYPE_F32, n, m->head_dim, m->n_head_kv);
            struct ggml_tensor *mk = dt_input(h.ctx, GGML_TYPE_F32, n, C, 0);
            struct ggml_tensor *gc = dt_input(h.ctx, GGML_TYPE_F32, E, C, 0);
            ggml_set_param(xc);
            ggml_set_param(kt);
            ggml_set_param(vtn);
            struct ggml_tensor *y = dt_build_chunk(m, h.ctx, il, xc, pc, kt, vtn, mk);
            struct ggml_tensor *obj = ggml_sum(h.ctx, ggml_mul(h.ctx, y, gc));
            ggml_set_loss(obj);
            if (dt_graph_alloc(m, &h, obj, 1) != 0) {
                ggml_free(h.ctx);
                goto done;
            }
            ggml_backend_tensor_set(xc, LAYER(il) + (size_t)c0 * E, 0, sizeof(float) * (size_t)C * (size_t)E);
            ggml_backend_tensor_set(pc, pos + c0, 0, sizeof(int32_t) * (size_t)C);
            ggml_backend_tensor_set(kt, K, 0, sizeof(float) * (size_t)n * (size_t)KVW);
            ggml_backend_tensor_set(vtn, V, 0, sizeof(float) * (size_t)n * (size_t)KVW);
            dt_fill_mask(maskbuf, n, c0, C);
            ggml_backend_tensor_set(mk, maskbuf, 0, sizeof(float) * (size_t)n * (size_t)C);
            ggml_backend_tensor_set(gc, G + (size_t)c0 * E, 0, sizeof(float) * (size_t)C * (size_t)E);
            int r = dt_graph_run(m, &h, 1);
            if (r == 0) {
                ggml_backend_tensor_get(ggml_graph_get_grad(h.gf, xc), DX + (size_t)c0 * E, 0,
                                        sizeof(float) * (size_t)C * (size_t)E);
                /* the key/value gradients of this chunk, added to the running sums */
                size_t kvn = (size_t)n * (size_t)KVW;
                float *tmp = malloc(sizeof(float) * kvn);
                struct ggml_tensor *gk = ggml_graph_get_grad(h.gf, kt);
                struct ggml_tensor *gv = ggml_graph_get_grad(h.gf, vtn);
                if (gk) {
                    ggml_backend_tensor_get(gk, tmp, 0, sizeof(float) * kvn);
                    for (size_t i = 0; i < kvn; i++) DK[i] += tmp[i];
                }
                if (gv) {
                    ggml_backend_tensor_get(gv, tmp, 0, sizeof(float) * kvn);
                    for (size_t i = 0; i < kvn; i++) DV[i] += tmp[i];
                }
                free(tmp);
            }
            ggml_free(h.ctx);
            if (r != 0) goto done;
        }

        /* the key/value path back to the layer input: d/dx of sum(K (.) DK) + sum(V (.) DV) */
        {
            struct dt_graph g = dt_graph_begin();
            struct ggml_tensor *x = dt_input(g.ctx, GGML_TYPE_F32, E, n, 0);
            struct ggml_tensor *p = dt_input(g.ctx, GGML_TYPE_I32, n, 0, 0);
            struct ggml_tensor *dk = dt_input(g.ctx, GGML_TYPE_F32, m->head_dim, n, m->n_head_kv);
            struct ggml_tensor *dv = dt_input(g.ctx, GGML_TYPE_F32, n, m->head_dim, m->n_head_kv);
            ggml_set_param(x);
            struct ggml_tensor *k, *v;
            dt_build_kv(m, g.ctx, il, x, p, &k, &v);
            struct ggml_tensor *obj = ggml_add(g.ctx, ggml_sum(g.ctx, ggml_mul(g.ctx, k, dk)),
                                               ggml_sum(g.ctx, ggml_mul(g.ctx, v, dv)));
            ggml_set_loss(obj);
            if (dt_graph_alloc(m, &g, obj, 1) != 0) {
                ggml_free(g.ctx);
                goto done;
            }
            ggml_backend_tensor_set(x, LAYER(il), 0, sizeof(float) * (size_t)n * (size_t)E);
            ggml_backend_tensor_set(p, pos, 0, sizeof(int32_t) * (size_t)n);
            ggml_backend_tensor_set(dk, DK, 0, sizeof(float) * (size_t)n * (size_t)KVW);
            ggml_backend_tensor_set(dv, DV, 0, sizeof(float) * (size_t)n * (size_t)KVW);
            int r = dt_graph_run(m, &g, 1);
            if (r == 0) {
                size_t xn = (size_t)n * (size_t)E;
                float *tmp = malloc(sizeof(float) * xn);
                ggml_backend_tensor_get(ggml_graph_get_grad(g.gf, x), tmp, 0, sizeof(float) * xn);
                for (size_t i = 0; i < xn; i++) G[i] = DX[i] + tmp[i];   /* becomes the next layer's upstream */
                free(tmp);
            }
            ggml_free(g.ctx);
            if (r != 0) goto done;
        }
    }

    /* G now holds d(loss)/d(input embeddings); the memory rows' share is the answer */
    memset(grad_out, 0, sizeof(float) * (size_t)E * (size_t)n_mem);
    for (int32_t i = 0; i < n; i++) {
        if (mem_index[i] >= 0 && mem_index[i] < n_mem) {
            float *dst = grad_out + (size_t)mem_index[i] * E;
            const float *src = G + (size_t)i * E;
            for (int64_t e = 0; e < E; e++) dst[e] += src[e];
        }
    }
    rc = n_t;

done:
#undef LAYER
    free(X);
    free(K);
    free(V);
    free(G);
    free(DX);
    free(DK);
    free(DV);
    free(maskbuf);
    free(pos);
    free(ibuf);
    free(fbuf);
    return rc;
}
