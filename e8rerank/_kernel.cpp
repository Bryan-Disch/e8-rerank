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
#include <vector>
#include <algorithm>

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

static inline float level_dot(const float *q, const uint16_t *cl, const int8_t *C2, int NB) {
    float acc[8] = {0};
    for (int b = 0; b < NB; b++) {
        const int8_t *c = C2 + 8 * (int64_t)cl[b];
        const float *qb = q + 8 * b;
        for (int t = 0; t < 8; t++) acc[t] += qb[t] * (float)c[t];
    }
    return hsum8(acc);
}

/* k2 > 0 and L > 1: two-step re-check. Every candidate is scored with the
 * bucket centre + level 1; only the best k2 get the remaining levels. */
extern "C" void e8_rerank(int nq, int k1, int D, int L, int Lstride, int NB,
               const float *Q, const int64_t *I,
               const float *cent, const int32_t *assign,
               const uint16_t *codes, const int8_t *C2,
               const float *half_scales, int k, int k2,
               float *out_s, int64_t *out_i)
{
    std::vector<float> sc(k1);
    std::vector<int64_t> id(k1);
    std::vector<int> ord(k1);
    const int64_t row = (int64_t)Lstride * NB;
    const bool two_step = (k2 > 0 && k2 < k1 && L > 1);
    const int first_levels = two_step ? 1 : L;
    for (int qi = 0; qi < nq; qi++) {
        const float *q = Q + (int64_t)qi * D;
        const int64_t *cand = I + (int64_t)qi * k1;
        float *bs = out_s + (int64_t)qi * k;
        int64_t *bi = out_i + (int64_t)qi * k;
        for (int j = 0; j < k; j++) { bs[j] = -FLT_MAX; bi[j] = -1; }

        int m = 0;                                     /* step 1 */
        for (int j = 0; j < k1; j++) {
            if (j + 4 < k1 && cand[j + 4] >= 0) {      /* fetch ahead: rows are random in memory */
                int64_t w = cand[j + 4];
                __builtin_prefetch(codes + w * row);
                __builtin_prefetch(assign + w);
            }
            int64_t v = cand[j];
            if (v < 0) continue;
            float s = dot_f(q, cent + (int64_t)assign[v] * D, D);
            const uint16_t *cv = codes + v * row;
            for (int l = 0; l < first_levels; l++)
                s += half_scales[l] * level_dot(q, cv + (int64_t)l * NB, C2, NB);
            sc[m] = s; id[m] = v; ord[m] = m; m++;
        }

        int keep = m;
        if (two_step && m > k2) {                       /* step 2: best k2 only */
            keep = k2;
            std::nth_element(ord.begin(), ord.begin() + keep, ord.begin() + m,
                             [&](int a, int b) { return sc[a] > sc[b]; });
        }
        if (two_step) {
            for (int t = 0; t < keep; t++) {
                int a = ord[t];
                const uint16_t *cv = codes + id[a] * row;
                for (int l = 1; l < L; l++)
                    sc[a] += half_scales[l] * level_dot(q, cv + (int64_t)l * NB, C2, NB);
            }
        }
        for (int t = 0; t < keep; t++) {               /* sorted top-k */
            int a = ord[t];
            float s = sc[a];
            if (s > bs[k - 1]) {
                int p = k - 1;
                while (p > 0 && bs[p - 1] < s) { bs[p] = bs[p - 1]; bi[p] = bi[p - 1]; p--; }
                bs[p] = s; bi[p] = id[a];
            }
        }
    }
}


/* ------------------------------------------------------------------ */
/* Encoding (used when building an index)                             */
/* ------------------------------------------------------------------ */
#include <cmath>
#include <thread>

template <class F>
static void parallel_for(int64_t n, int nthreads, F f) {
    if (nthreads <= 1 || n < 1024) { f(0, n); return; }
    std::vector<std::thread> ts;
    int64_t chunk = (n + nthreads - 1) / nthreads;
    for (int t = 0; t < nthreads; t++) {
        int64_t a = t * chunk, b = std::min(n, a + chunk);
        if (a < b) ts.emplace_back(f, a, b);
    }
    for (auto &t : ts) t.join();
}

static inline void near_D8(const double *x, double *f) {
    double s = 0; int j = 0; double worst = -1;
    for (int t = 0; t < 8; t++) {
        f[t] = std::nearbyint(x[t]); s += f[t];
        double d = std::fabs(x[t] - f[t]);
        if (d > worst) { worst = d; j = t; }
    }
    if (std::fmod(std::fabs(s), 2.0) != 0) f[j] += (x[j] - f[j] >= 0) ? 1.0 : -1.0;
}

static inline void near_E8(const double *x, double *out) {
    double a[8], b[8], xs[8];
    near_D8(x, a);
    for (int t = 0; t < 8; t++) xs[t] = x[t] - 0.5;
    near_D8(xs, b);
    double da = 0, db = 0;
    for (int t = 0; t < 8; t++) {
        b[t] += 0.5;
        da += (x[t] - a[t]) * (x[t] - a[t]);
        db += (x[t] - b[t]) * (x[t] - b[t]);
    }
    for (int t = 0; t < 8; t++) out[t] = (da <= db) ? a[t] : b[t];
}

struct Book {
    const float *C; const int64_t *sk; const int32_t *sv; int64_t nk; const float *roots;
    int lookup(const double *p) const {            /* index of an E8 point, or -1 */
        int64_t key = 0, mult = 1;
        for (int t = 0; t < 8; t++) {
            long k = std::lround(p[t] * 2) + 8;
            if (k < 0 || k > 15) return -1;
            key += k * mult; mult *= 16;
        }
        const int64_t *it = std::lower_bound(sk, sk + nk, key);
        return (it != sk + nk && *it == key) ? sv[it - sk] : -1;
    }
};

extern "C" void e8_encode(int64_t n, const float *Y, const float *C, const int64_t *sk,
                          const int32_t *sv, int64_t nk, const float *roots, double r_ball,
                          int nthreads, uint16_t *out)
{
    Book bk{C, sk, sv, nk, roots};
    parallel_for(n, nthreads, [&](int64_t a, int64_t b) {
        double x[8], p[8];
        for (int64_t i = a; i < b; i++) {
            const float *y = Y + 8 * i;
            for (int t = 0; t < 8; t++) x[t] = y[t];
            near_E8(x, p);
            int idx = bk.lookup(p);
            if (idx >= 0) { out[i] = (uint16_t)idx; continue; }
            /* outside the codebook ball: pull toward the centre and retry */
            double r = 0; for (int t = 0; t < 8; t++) r += x[t] * x[t];
            r = std::sqrt(std::max(r, 1e-18));
            const double fs[] = {0.95, 0.9, 0.85, 0.8, 0.7, 0.6};
            for (double f : fs) {
                double xs[8];
                for (int t = 0; t < 8; t++) xs[t] = x[t] * (f * r_ball / r);
                near_E8(xs, p);
                idx = bk.lookup(p);
                if (idx >= 0) break;
            }
            if (idx < 0) {                              /* rare: brute force */
                double best = 1e300;
                for (int64_t j = 0; j < 65536; j++) {
                    double d = 0;
                    for (int t = 0; t < 8; t++) { double e = x[t] - C[8 * j + t]; d += e * e; }
                    if (d < best) { best = d; idx = (int)j; }
                }
                out[i] = (uint16_t)idx; continue;
            }
            /* walk to better codewords one E8 neighbour step at a time */
            double best = 0;
            for (int t = 0; t < 8; t++) { double e = x[t] - C[8 * idx + t]; best += e * e; }
            bool moved = true;
            while (moved) {
                moved = false;
                for (int j = 0; j < 240; j++) {
                    double v[8], d = 0;
                    for (int t = 0; t < 8; t++) {
                        v[t] = C[8 * idx + t] + roots[8 * j + t];
                        double e = x[t] - v[t]; d += e * e;
                    }
                    if (d >= best - 1e-9) continue;
                    int w = bk.lookup(v);
                    if (w >= 0) { best = d; idx = w; moved = true; }
                }
            }
            out[i] = (uint16_t)idx;
        }
    });
}

/* nbr[240 * c + j] = index of codeword c + root j, or -1 if outside the codebook */
extern "C" void e8_neighbours(const float *C, const int64_t *sk, const int32_t *sv, int64_t nk,
                              const float *roots, int nthreads, int32_t *nbr)
{
    Book bk{C, sk, sv, nk, roots};
    parallel_for(65536, nthreads, [&](int64_t a, int64_t b) {
        double v[8];
        for (int64_t c = a; c < b; c++)
            for (int j = 0; j < 240; j++) {
                for (int t = 0; t < 8; t++) v[t] = C[8 * c + t] + roots[8 * j + t];
                nbr[240 * c + j] = bk.lookup(v);
            }
    });
}

extern "C" void e8_score_aware(int64_t n, int D, const float *R, const float *XH, uint16_t *codes,
                               const float *C, float s, const int32_t *nbr,
                               const float *roots, float eta, int rounds, int nthreads)
{
    const int NB = D / 8;
    parallel_for(n, nthreads, [&](int64_t a, int64_t b) {
        std::vector<float> e(D);
        for (int64_t i = a; i < b; i++) {
            const float *r = R + i * D, *xh = XH + i * D;
            uint16_t *c = codes + i * NB;
            float par = 0;
            for (int bb = 0; bb < NB; bb++)
                for (int t = 0; t < 8; t++) {
                    e[8 * bb + t] = r[8 * bb + t] - s * C[8 * c[bb] + t];
                    par += e[8 * bb + t] * xh[8 * bb + t];
                }
            for (int rd = 0; rd < rounds; rd++) {
                for (int bb = 0; bb < NB; bb++) {
                    int cur = c[bb];
                    float pb = 0;
                    for (int t = 0; t < 8; t++) pb += e[8 * bb + t] * xh[8 * bb + t];
                    float rest = par - pb, best = 1e30f, bpb = pb;
                    int bidx = cur;
                    const int32_t *nb = nbr + 240 * (int64_t)cur;
                    for (int j = 0; j <= 240; j++) {      /* current point + 240 neighbours */
                        int w = (j < 240) ? nb[j] : cur;
                        if (w < 0) continue;
                        const float *v = C + 8 * (int64_t)w;
                        float nrm = 0, pp = 0;
                        for (int t = 0; t < 8; t++) {
                            float d = r[8 * bb + t] - s * v[t];
                            nrm += d * d; pp += d * xh[8 * bb + t];
                        }
                        float tot = rest + pp;
                        float loss = nrm + (eta - 1.f) * tot * tot;
                        if (loss < best) { best = loss; bidx = w; bpb = pp; }
                    }
                    if (bidx != cur) {
                        c[bb] = (uint16_t)bidx;
                        for (int t = 0; t < 8; t++) e[8 * bb + t] = r[8 * bb + t] - s * C[8 * bidx + t];
                        par = rest + bpb;
                    }
                }
            }
        }
    });
}
