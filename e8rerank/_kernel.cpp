/* E8 re-check kernel (C++, exported with a C interface for ctypes).
 *
 * For each query and each of its k1 candidates, the score is
 *     q . centre[bucket] + sum_levels scale_l * sum_blocks q_block . E8point
 * E8 points are stored as int8 at twice their size (every coordinate of an
 * E8 point is a multiple of 1/2), so the whole 65,536-point codebook takes
 * 512 KB and stays in cache. One 8-number block is exactly one 256-bit
 * vector register, which compilers vectorise from the plain loops below.
 */
#include <cstdint>
#include <cfloat>

/* 8 running sums (one per lane) so the loops vectorise without -ffast-math */
static inline float hsum8(const float *acc) {
    return ((acc[0] + acc[1]) + (acc[2] + acc[3])) + ((acc[4] + acc[5]) + (acc[6] + acc[7]));
}

static inline float dot_f(const float *a, const float *b, int n) {   /* n % 8 == 0 */
    float acc[8] = {0};
    for (int i = 0; i < n; i += 8)
        for (int t = 0; t < 8; t++) acc[t] += a[i + t] * b[i + t];
    return hsum8(acc);
}

extern "C" void e8_rerank(int nq, int k1, int D, int L, int Lstride, int NB,
               const float *Q, const int64_t *I,
               const float *cent, const int32_t *assign,
               const uint16_t *codes, const int8_t *C2,
               const float *half_scales, int k,
               float *out_s, int64_t *out_i)
{
    #pragma omp parallel for schedule(dynamic, 16)
    for (int qi = 0; qi < nq; qi++) {
        const float *q = Q + (int64_t)qi * D;
        const int64_t *cand = I + (int64_t)qi * k1;
        float *bs = out_s + (int64_t)qi * k;
        int64_t *bi = out_i + (int64_t)qi * k;
        for (int j = 0; j < k; j++) { bs[j] = -FLT_MAX; bi[j] = -1; }
        for (int j = 0; j < k1; j++) {
            int64_t v = cand[j];
            if (v < 0) continue;
            float s = dot_f(q, cent + (int64_t)assign[v] * D, D);
            const uint16_t *cv = codes + v * (int64_t)Lstride * NB;
            for (int l = 0; l < L; l++) {
                float acc[8] = {0};
                const uint16_t *cl = cv + (int64_t)l * NB;
                for (int b = 0; b < NB; b++) {
                    const int8_t *c = C2 + 8 * (int64_t)cl[b];
                    const float *qb = q + 8 * b;
                    for (int t = 0; t < 8; t++) acc[t] += qb[t] * (float)c[t];
                }
                s += half_scales[l] * hsum8(acc);
            }
            if (s > bs[k - 1]) {                 /* insert into sorted top-k */
                int p = k - 1;
                while (p > 0 && bs[p - 1] < s) { bs[p] = bs[p - 1]; bi[p] = bi[p - 1]; p--; }
                bs[p] = s; bi[p] = v;
            }
        }
    }
}
