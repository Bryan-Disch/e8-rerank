import numpy as np
import faiss
from e8rerank import E8Codebook, E8RerankIndex, near_E8, e8_codebook
from e8rerank.lattice import e8_points, ROOTS


def test_kissing_number():
    assert len(ROOTS) == 240
    assert np.allclose((ROOTS ** 2).sum(1), 2)


def test_near_E8_matches_brute_force():
    rng = np.random.default_rng(0)
    X = rng.normal(size=(500, 8)) * 0.8
    P = e8_points(8)  # every candidate within reach for these inputs
    brute = P[np.argmin(((X[:, None, :] - P[None]) ** 2).sum(-1), axis=1)]
    got = near_E8(X)
    assert np.all(((X - got) ** 2).sum(1) <= ((X - brute) ** 2).sum(1) + 1e-9)
    assert np.allclose(near_E8(got), got)


def test_codebook_points_are_in_E8():
    C = e8_codebook(65536)
    assert len(np.unique(C, axis=0)) == 65536
    assert np.allclose(near_E8(C), C)


def test_encode_is_nearest_codeword():
    rng = np.random.default_rng(1)
    book = E8Codebook()
    Y = rng.normal(size=(2000, 8)) * 1.2
    idx = book.encode_blocks(Y)
    d_got = ((Y - book.C[idx]) ** 2).sum(1)
    d_best = ((Y[:, None, :] - book.C[None]) ** 2).sum(-1).min(1)
    # exact for nearly every block; the rest are within a hair of the best
    assert np.mean(d_got <= d_best + 1e-5) > 0.97
    assert d_got.sum() / d_best.sum() < 1.01


def test_index_recall_on_clustered_data():
    rng = np.random.default_rng(2)
    d, n = 64, 20000
    centres = rng.normal(size=(50, d))
    X = centres[rng.integers(0, 50, n)] + 0.5 * rng.normal(size=(n, d))
    X = (X / np.linalg.norm(X, axis=1, keepdims=True)).astype(np.float32)
    Q, B = X[:200], X[200:]
    gt = faiss.IndexFlatIP(d)
    gt.add(B)
    _, G = gt.search(Q, 10)
    ix = E8RerankIndex(d, nlist=64, levels=3)
    ix.train(B)
    ix.add(B)
    def recall(I):
        return np.mean([len(set(a) & set(g)) / 10 for a, g in zip(I, G)])
    _, I = ix.search(Q, k=10, nprobe=64, k1=500)
    ix.first.nprobe = 64
    _, I0 = ix.first.search(Q, 10)
    # the E8 re-check must beat the rough first pass by a wide margin
    assert recall(I) > recall(I0) + 0.2, (recall(I), recall(I0))
    assert recall(I) > 0.85, recall(I)
    assert ix.bytes_per_vector()["recheck"] == 3 * 8 * 2 + 2
