import numpy as np
import faiss
from e8rerank import E8RerankIndex

from ..base.module import BaseANN


class E8Rerank(BaseANN):
    """FAISS IVF fast-scan first pass + E8 lattice re-check codes
    (https://github.com/Bryan-Disch/e8-rerank)."""

    def __init__(self, metric, n_list, levels):
        if metric not in ("angular", "euclidean"):
            raise NotImplementedError(metric)
        if metric == "euclidean":
            raise NotImplementedError("e8-rerank currently supports inner product / angular only")
        self._metric = metric
        self._n_list = n_list
        self._levels = levels

    def fit(self, X):
        X = np.ascontiguousarray(X, dtype=np.float32)
        faiss.normalize_L2(X)
        self.index = E8RerankIndex(X.shape[1], nlist=self._n_list, levels=self._levels)
        self.index.train(X)
        self.index.add(X)

    def set_query_arguments(self, n_probe, k1, levels):
        self._n_probe, self._k1 = n_probe, k1
        self._q_levels = min(levels, self._levels)

    def _search(self, X, n):
        X = np.ascontiguousarray(X, dtype=np.float32)
        faiss.normalize_L2(X)
        return self.index.search(X, n, self._n_probe, max(self._k1, n), self._q_levels)[1]

    def query(self, v, n):
        return self._search(v[None, :], n)[0]

    def batch_query(self, X, n):
        self.res = self._search(X, n)

    def get_batch_results(self):
        return self.res

    def __str__(self):
        return "E8Rerank(n_list=%d, levels=%d, n_probe=%d, k1=%d)" % (
            self._n_list, self._q_levels, self._n_probe, self._k1)
