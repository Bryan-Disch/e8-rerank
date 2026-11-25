"""E8 lattice tools.

E8 is the set of 8-number points whose coordinates are all integers or all
half-integers, with an even coordinate sum. It is the densest way to pack
spheres in 8 dimensions (Viazovska, 2016), which also makes it a very good
grid for rounding 8 numbers at a time.

This module gives:
  * ``near_E8``: the exact nearest E8 point (Conway & Sloane's decoder),
  * ``E8Codebook``: a fixed-size codebook made of the 65,536 E8 points
    closest to the origin (16 bits per 8 numbers), with fast encoding,
  * ``score_aware_encode``: re-rounding that penalises the part of the
    error that changes inner-product scores (ScaNN's anisotropic loss).
"""
import numpy as np
import numba


def near_D8(x):
    """Nearest point of D8 (integer vectors with an even sum)."""
    f = np.round(x)
    bad = (f.sum(axis=-1) % 2) != 0
    if bad.any():
        xb, fb = x[bad], f[bad].copy()
        j = np.argmax(np.abs(xb - fb), axis=-1)
        r = np.arange(len(xb))
        d = xb[r, j] - fb[r, j]
        fb[r, j] += np.where(d >= 0, 1.0, -1.0)
        f[bad] = fb
    return f


def near_E8(x):
    """Nearest E8 point for each row of ``x`` (shape (n, 8))."""
    x = np.asarray(x, dtype=np.float64)
    a = near_D8(x)
    b = near_D8(x - 0.5) + 0.5
    da = ((x - a) ** 2).sum(-1)
    db = ((x - b) ** 2).sum(-1)
    return np.where((da <= db)[:, None], a, b)


def e8_points(max_norm2):
    """All E8 points with squared length <= ``max_norm2``."""
    R = int(np.floor(np.sqrt(max_norm2) + 1e-9))
    g = np.array(np.meshgrid(*[np.arange(-R, R + 1)] * 8, indexing="ij"),
                 dtype=np.int8).reshape(8, -1).T.astype(np.int32)
    g = g[(g.sum(1) % 2 == 0) & ((g ** 2).sum(1) <= max_norm2 + 1e-9)]
    Rh = int(np.floor(np.sqrt(max_norm2) - 0.5 + 1e-9))
    h = np.array(np.meshgrid(*[np.arange(-Rh - 1, Rh + 1) + 0.5] * 8,
                             indexing="ij")).reshape(8, -1).T
    h = h[(np.round(h.sum(1)) % 2 == 0) & ((h ** 2).sum(1) <= max_norm2 + 1e-9)]
    P = np.vstack([g, h]).astype(np.float64)
    return P[np.lexsort(P.T[::-1])]


def e8_codebook(size=65536, seed=0):
    """The ``size`` E8 points nearest the origin (a 'ball' of the lattice).
    The outermost, partly used shell is sampled at random with a fixed seed."""
    for m in range(2, 40, 2):
        P = e8_points(m)
        if len(P) >= size:
            break
    n2 = (P ** 2).sum(1).round(6)
    inner = P[n2 < m - 1e-9]
    shell = P[np.abs(n2 - m) < 1e-9]
    rng = np.random.default_rng(seed)
    take = shell[rng.choice(len(shell), size - len(inner), replace=False)]
    return np.vstack([inner, take])


_p = e8_points(2)
ROOTS = _p[(_p ** 2).sum(1) > 0].astype(np.float32)  # the 240 shortest vectors
assert len(ROOTS) == 240


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


def _keys(P):
    """Pack doubled coordinates (range -8..7) into one integer key per point."""
    k = np.round(P * 2).astype(np.int64) + 8
    ok = ((k >= 0) & (k < 16)).all(1)
    return np.where(ok, (np.clip(k, 0, 15) * (16 ** np.arange(8))).sum(1), -1)


class E8Codebook:
    """65,536 E8 points (16 bits per 8 numbers), scaled by ``scale``."""

    def __init__(self, size=65536, seed=0):
        self.C = e8_codebook(size, seed).astype(np.float32)
        k = _keys(self.C)
        o = np.argsort(k)
        self.sk = k[o]
        self.sv = o.astype(np.int32)
        self.scale = 1.0

    def _lookup(self, P):
        k = _keys(P)
        pos = np.minimum(np.searchsorted(self.sk, k), len(self.sk) - 1)
        return np.where(self.sk[pos] == k, self.sv[pos], -1).astype(np.int32)

    def encode_blocks(self, Y):
        """Codes for 8-number blocks ``Y`` (already divided by the scale)."""
        idx = self._lookup(near_E8(Y))
        bad = np.where(idx < 0)[0]
        miss = bad.copy()
        r_ball = np.sqrt((self.C.astype(np.float64) ** 2).sum(1).max())
        for f in (0.95, 0.9, 0.85, 0.8, 0.7, 0.6):
            # outside the ball: pull toward the centre and try again
            if not len(bad):
                break
            Yb = Y[bad].astype(np.float64)
            r = np.linalg.norm(Yb, axis=1, keepdims=True)
            got = self._lookup(near_E8(Yb * (f * r_ball / np.maximum(r, 1e-9))))
            ok = got >= 0
            idx[bad[ok]] = got[ok]
            bad = bad[~ok]
        if len(bad):
            idx[bad] = _nearest_brute(np.ascontiguousarray(Y[bad], np.float32), self.C)
        if len(miss):
            # the shrunken guess is valid but may not be the nearest codeword:
            # walk to better codewords one E8 neighbour step at a time
            idx[miss] = _descend(np.ascontiguousarray(Y[miss], np.float32), idx[miss],
                                 self.C, self.sk, self.sv, ROOTS)
        return idx

    def encode(self, X, chunk=200000):
        """X: (n, D) with D a multiple of 8 -> uint16 codes (n, D // 8)."""
        out = []
        for s in range(0, len(X), chunk):
            x = X[s:s + chunk]
            out.append(self.encode_blocks((x / self.scale).reshape(-1, 8)).reshape(len(x), -1))
        return np.vstack(out).astype(np.uint16)

    def decode(self, codes):
        n = len(codes)
        return (self.C[codes.astype(np.int64).ravel()] * self.scale).reshape(n, -1)

    def fit_scale(self, X, grid=np.linspace(0.7, 1.3, 7)):
        """Pick the scale with the lowest squared error on sample ``X``."""
        sd = float(X.std())
        best = None
        for f in grid:
            self.scale = f * sd
            e = float(((X - self.decode(self.encode(X))) ** 2).mean())
            if best is None or e < best[0]:
                best = (e, self.scale)
        self.scale = best[1]
        return best[0]


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


def score_aware_encode(book, residual, direction, codes, eta=4.0, rounds=2):
    """Improve ``codes`` so that the error along ``direction`` (unit vectors,
    the database vectors themselves) is penalised ``eta`` times more.
    Candidates per block: the current point and its 240 E8 neighbours."""
    out = _score_aware(np.ascontiguousarray(residual, np.float32),
                       np.ascontiguousarray(direction, np.float32),
                       codes.astype(np.int32), book.C, np.float32(book.scale),
                       book.sk, book.sv, ROOTS, np.float32(eta), rounds)
    return out.astype(np.uint16)
