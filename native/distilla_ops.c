/*
 * distilla_ops: the optimizer kernels for training the memory module. A
 * module holds ~170M parameters, and an AdamW step touches each of them
 * several times, which is a loop for C rather than for the Scheme heap.
 * No dependencies; built by native/build.sh into libdistilla_ops.
 */
#include <math.h>
#include <stdint.h>
#include <string.h>

/* torch.optim.AdamW, one tensor: decoupled weight decay, then the Adam
 * update with bias corrections bc1 = 1 - beta1^t and bc2 = 1 - beta2^t. */
void dops_adamw(float *p, const float *g, float *m, float *v, int64_t n, float lr, float beta1, float beta2,
                float eps, float weight_decay, float bc1, float bc2) {
    const float decay = 1.0f - lr * weight_decay;
    const float step = lr / bc1;
    const float inv_sqrt_bc2 = 1.0f / sqrtf(bc2);
    for (int64_t i = 0; i < n; i++) {
        float gi = g[i];
        float mi = beta1 * m[i] + (1.0f - beta1) * gi;
        float vi = beta2 * v[i] + (1.0f - beta2) * gi * gi;
        m[i] = mi;
        v[i] = vi;
        p[i] = p[i] * decay - step * mi / (sqrtf(vi) * inv_sqrt_bc2 + eps);
    }
}

/* sum of squares, in double */
double dops_sumsq(const float *x, int64_t n) {
    double s = 0.0;
    for (int64_t i = 0; i < n; i++) s += (double)x[i] * (double)x[i];
    return s;
}

/* x *= a */
void dops_scale(float *x, int64_t n, float a) {
    for (int64_t i = 0; i < n; i++) x[i] *= a;
}

/* y += a * x */
void dops_axpy(float *y, const float *x, int64_t n, float a) {
    for (int64_t i = 0; i < n; i++) y[i] += a * x[i];
}

void dops_zero(float *x, int64_t n) { memset(x, 0, sizeof(float) * (size_t)n); }

void dops_copy(float *dst, const float *src, int64_t n) { memcpy(dst, src, sizeof(float) * (size_t)n); }
