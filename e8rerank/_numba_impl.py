"""Pure-Python (numba) fallbacks, used only when the C++ kernel is missing.
Imported lazily so that numba is not loaded otherwise."""
import numpy as np
import numba


@numba.njit(parallel=True, cache=True)
def _nearest_brute(Y, C):
    n = len(Y)
    out = np.empty(n, np.int32)
    cc = (C ** 2).sum(1)
    for i in numba.prange(n):
        best = 1e30
        bj = 0
        for j in range(len(C)):
            d = cc[j]
            for t in range(8):
                d -= 2 * Y[i, t] * C[j, t]
            if d < best:
                best = d
                bj = j
        out[i] = bj
    return out


@numba.njit(parallel=True, cache=True)
def _descend(Y, idx, C, sk, sv, roots):
    out = idx.copy()
    for i in numba.prange(len(Y)):
        cur = out[i]
        best = 0.0
        for t in range(8):
            best += (Y[i, t] - C[cur, t]) ** 2
        moved = True
        while moved:
            moved = False
            for j in range(240):
                key = 0
                mult = 1
                ok = True
                d = 0.0
                for t in range(8):
                    v = C[cur, t] + roots[j, t]
                    k = int(np.round(v * 2)) + 8
                    if k < 0 or k > 15:
                        ok = False
                    key += k * mult
                    mult *= 16
                    d += (Y[i, t] - v) ** 2
                if not ok or d >= best - 1e-9:
                    continue
                pos = np.searchsorted(sk, key)
                if pos < len(sk) and sk[pos] == key:
                    best = d
                    cur = sv[pos]
                    moved = True
        out[i] = cur
    return out


@numba.njit(parallel=True, cache=True)
def _score_aware(r, xh, codes, C, s, sk, sv, roots, eta, rounds):
    n, D = r.shape
    NB = D // 8
    out = codes.copy()
    for i in numba.prange(n):
        e = np.empty(D, np.float32)
        for b in range(NB):
            for t in range(8):
                e[8 * b + t] = r[i, 8 * b + t] - s * C[out[i, b], t]
        par = 0.0
        for t in range(D):
            par += e[t] * xh[i, t]
        for rd in range(rounds):
            for b in range(NB):
                cur = out[i, b]
                pb = 0.0
                for t in range(8):
                    pb += e[8 * b + t] * xh[i, 8 * b + t]
                rest = par - pb
                best = 1e30
                bidx = cur
                bpb = pb
                for j in range(241):  # current point + its 240 neighbours
                    key = 0
                    mult = 1
                    ok = True
                    nrm = 0.0
                    pp = 0.0
                    for t in range(8):
                        v = C[cur, t] + (roots[j, t] if j < 240 else 0.0)
                        k = int(np.round(v * 2)) + 8
                        if k < 0 or k > 15:
                            ok = False
                        key += k * mult
                        mult *= 16
                        d = r[i, 8 * b + t] - s * v
                        nrm += d * d
                        pp += d * xh[i, 8 * b + t]
                    if not ok:
                        continue
                    pos = np.searchsorted(sk, key)
                    if pos >= len(sk) or sk[pos] != key:
                        continue
                    tot = rest + pp
                    loss = nrm + (eta - 1.0) * tot * tot
                    if loss < best:
                        best = loss
                        bidx = sv[pos]
                        bpb = pp
                if bidx != cur:
                    out[i, b] = bidx
                    for t in range(8):
                        e[8 * b + t] = r[i, 8 * b + t] - s * C[bidx, t]
                    par = rest + bpb
    return out




@numba.njit(parallel=True, cache=True)
def _rerank(Q, I, qc, assign, codes, L, C, scales, k, out_s, out_i):
    nq, k1 = I.shape
    NB = codes.shape[2]
    for qi in numba.prange(nq):
        sc = np.full(k1, -np.inf, dtype=np.float32)
        for j in range(k1):
            v = I[qi, j]
            if v < 0:
                continue
            s = qc[qi, assign[v]]
            for lv in range(L):
                sl = 0.0
                for b in range(NB):
                    w = codes[v, lv, b]
                    for t in range(8):
                        sl += Q[qi, 8 * b + t] * C[w, t]
                s += scales[lv] * sl
            sc[j] = s
        order = np.argsort(-sc)[:k]
        for j in range(k):
            out_s[qi, j] = sc[order[j]]
            out_i[qi, j] = I[qi, order[j]]
