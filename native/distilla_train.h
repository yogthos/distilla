/* distilla_train: the frozen decoder's loss and its gradient with respect to
 * the memory tokens, for training the memory module (see distilla_train.c). */
#ifndef DISTILLA_TRAIN_H
#define DISTILLA_TRAIN_H

#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef struct dt_model dt_model;

/* The message of the last failure. */
const char *dt_last_error(void);

/* Load a GGUF decoder (llama, qwen2 or qwen3 layout). use_gpu puts the weights
 * on the first GPU when there is one; f32_weights widens matmul weights to
 * float32 (needed for the backward pass to stay on a CUDA GPU). NULL on
 * failure. */
dt_model *dt_model_load(const char *path, int32_t use_gpu, int32_t n_threads, int32_t f32_weights);
void dt_model_free(dt_model *m);
int32_t dt_model_n_embd(const dt_model *m);
int32_t dt_model_n_vocab(const dt_model *m);
int32_t dt_model_on_gpu(const dt_model *m);

/*
 * One sequence of n positions. Position i holds tokens[i], or memory row
 * mem_index[i] (>= 0) of mem (n_mem x n_embd floats, row-major). targets[i]
 * is the token position i must predict (-1: not scored). Writes the mean
 * cross-entropy over the scored positions to *loss_out and, with want_grad,
 * its gradient with respect to mem to grad_out (n_mem x n_embd). Returns the
 * number of scored positions, or -1 (see dt_last_error).
 */
int32_t dt_step(dt_model *m, const int32_t *tokens, const int32_t *mem_index, const int32_t *targets, int32_t n,
                const float *mem, int32_t n_mem, float *grad_out, float *loss_out, int32_t want_grad);

/* The single-graph step (every layer's activations kept at once). */
int32_t dt_step_full(dt_model *m, const int32_t *tokens, const int32_t *mem_index, const int32_t *targets, int32_t n,
                     const float *mem, int32_t n_mem, float *grad_out, float *loss_out, int32_t want_grad);

/* dt_step uses the single graph up to full_max positions (default 1024) and
 * beyond that the layer-wise backward, which rebuilds one layer at a time for
 * chunk query rows (default 256): peak memory n_head x chunk x n attention
 * floats instead of n_layer x n_head x n x n. Values <= 0 / < 0 keep the
 * current setting. */
void dt_set_chunking(dt_model *m, int32_t chunk, int32_t full_max);

#ifdef __cplusplus
}
#endif

#endif
