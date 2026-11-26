"""E8RerankIndex: FAISS fast-scan finds candidates, E8 codes re-check them.

Search has two stages:
  1. First pass (FAISS ``IVF{nlist},PQ{d/2}x4fs``): very fast, rough scores,
     returns the top ``k1`` candidates.
  2. Re-check: each candidate's score is recomputed from its compact E8
     code (``levels`` x 2 bytes per 8 dimensions + a 2-byte bucket id), and
     the best ``k`` are returned.

The E8 codes store the vector minus its bucket centre (the residual), after
a fixed random rotation. Each level encodes what the previous levels missed.
Nothing about the codebook is trained: only one scale per level is fitted.
"""
import numpy as np
import faiss

from .lattice import E8Codebook, clear_neighbours, score_aware_encode
from ._native import KERNEL as _KERNEL


def _return_free_memory():
    """Hand freed build buffers back to the OS (glibc only; no-op elsewhere)."""
    try:
        import ctypes
        ctypes.CDLL("libc.so.6").malloc_trim(0)
    except (OSError, AttributeError):
        pass


def _rotation(D, seed):
    q, r = np.linalg.qr(np.random.default_rng(seed).normal(size=(D, D)))
    return (q * np.sign(np.diag(r))).astype(np.float32)


class E8RerankIndex:
    """Inner-product (or cosine, on normalised data) nearest-neighbour index.

    Parameters
    ----------
    d : int            vector dimension
    nlist : int        number of buckets (k-means clusters), at most 65,536
    levels : int       E8 re-check levels; memory is ``levels * ceil(d/8) * 2 + 2`` bytes
    score_aware : bool use score-aware (anisotropic) rounding, recommended
    eta : float        weight on the score-changing part of the error
    seed : int         random rotation seed
    """

    def __init__(self, d, nlist=4096, levels=2, score_aware=True, eta=4.0, seed=0):
        assert nlist <= 65536
        self.d, self.nlist, self.levels = d, nlist, levels
        self.score_aware, self.eta = score_aware, eta
        self.D = int(np.ceil(d / 8) * 8)
        self.R = _rotation(self.D, seed)
        m = d // 2 if d % 2 == 0 else d
        self.first = faiss.index_factory(d, f"IVF{nlist},PQ{m}x4fs", faiss.METRIC_INNER_PRODUCT)
        book = E8Codebook()
        self.books = [book.with_scale() for _ in range(levels)]
        self.codes = np.zeros((0, levels, self.D // 8), np.uint16)
        self.assign = np.zeros(0, np.int32)

    def _rot(self, X):
        P = np.zeros((len(X), self.D), np.float32)
        P[:, :self.d] = X
        return P @ self.R

    @staticmethod
    def _unit(X):
        return X / np.maximum(np.linalg.norm(X, axis=1, keepdims=True), 1e-12)

    def train(self, X, max_train=200000, seed=1):
        X = np.ascontiguousarray(X, np.float32)
        rng = np.random.default_rng(seed)
        if len(X) > max_train:
            X = X[rng.choice(len(X), max_train, replace=False)]
        self.first.train(X)
        # Bucket centres for the residuals: plain k-means in the rotated space.
        # (The first pass's own centres are trained for inner product and
        # leave ~25% larger residuals, which costs re-check accuracy.)
        km = faiss.Kmeans(self.D, self.nlist, niter=20, seed=seed)
        km.train(self._rot(X))
        self.cent = km.centroids
        self.centre_index = faiss.IndexFlatL2(self.D)
        self.centre_index.add(self.cent)
        # fit one scale per level on a sample of residuals
        S = X[rng.choice(len(X), min(8000, len(X)), replace=False)]
        Sr = self._rot(S)
        a = self.centre_index.search(Sr, 1)[1].ravel()
        r = Sr - self.cent[a]
        for book in self.books:
            book.fit_scale(r)
            r = r - book.decode(book.encode(r))

    def add(self, X):
        X = np.ascontiguousarray(X, np.float32)
        self.first.add(X)
        Xr = self._rot(X)
        a = self.centre_index.search(Xr, 1)[1].ravel().astype(np.int32)
        r = Xr - self.cent[a]
        direction = self._unit(Xr)
        new = np.zeros((len(X), self.levels, self.D // 8), np.uint16)
        for lv, book in enumerate(self.books):
            c = book.encode(r)
            if self.score_aware:
                c = score_aware_encode(book, r, direction, c, self.eta)
            new[:, lv] = c
            r = r - book.decode(c)
        self.codes = np.concatenate([self.codes, new])
        self.assign = np.concatenate([self.assign, a])
        clear_neighbours()  # 63 MB build-time table; not needed for search
        _return_free_memory()

    def search(self, Q, k=10, nprobe=64, k1=300, levels=None, use_c=True, k2=0):
        """Returns (scores, ids), each of shape (len(Q), k).
        ``levels`` < self.levels re-checks with only the first few levels.
        ``k2`` > 0 (C++ kernel only): score all k1 candidates with level 1,
        then apply the remaining levels to the best k2 only."""
        L = self.levels if levels is None else levels
        Q = np.ascontiguousarray(Q, np.float32)
        self.first.nprobe = nprobe
        _, I = self.first.search(Q, k1)
        out_s = np.empty((len(Q), k), np.float32)
        out_i = np.empty((len(Q), k), np.int64)
        scales = np.array([b.scale for b in self.books], np.float32)
        Qr = self._rot(Q)
        if _KERNEL is not None and use_c:
            if getattr(self, "_C2", None) is None:
                # E8 points doubled are integers in -8..7: an int8 table, 512 KB
                self._C2 = np.ascontiguousarray(np.round(self.books[0].C * 2), np.int8)
            half = (scales / 2).astype(np.float32)
            I = np.ascontiguousarray(I, np.int64)
            _KERNEL.e8_rerank(len(Q), I.shape[1], self.D, L, self.levels, self.D // 8,
                              Qr.ctypes.data, I.ctypes.data, self.cent.ctypes.data,
                              self.assign.ctypes.data, self.codes.ctypes.data,
                              self._C2.ctypes.data, half.ctypes.data, k, k2,
                              out_s.ctypes.data, out_i.ctypes.data)
            return out_s, out_i
        from ._numba_impl import _rerank
        qc = Qr @ self.cent.T  # query x bucket-centre scores, one matrix product
        _rerank(Qr, I, qc, self.assign, self.codes, L,
                self.books[0].C, scales, k, out_s, out_i)
        return out_s, out_i

    def bytes_per_vector(self, levels=None):
        """Memory per stored vector, split by stage."""
        L = self.levels if levels is None else levels
        n = max(len(self.assign), 1)
        first = len(faiss.serialize_index(self.first)) / n
        recheck = L * (self.D // 8) * 2 + 2
        return {"first_pass": first, "recheck": recheck, "total": first + recheck}
