"""Benchmark on GloVe-Twitter-100 (the data behind ann-benchmarks' glove-100-angular).

    python benchmarks/glove.py            # downloads ~390 MB on first run

Compares, on the SAME FAISS fast-scan candidates:
  * E8 re-check codes (2 and 3 levels, score-aware)      -- this package
  * FAISS IndexRefine with PQ (8-bit) and SQ8 re-check   -- standard FAISS
Reports recall@10, single-thread queries/second and bytes per vector.
"""
import argparse, gzip, os, time, urllib.request
import numpy as np
import faiss
from e8rerank import E8RerankIndex

URL = "https://github.com/RaRe-Technologies/gensim-data/releases/download/glove-twitter-100/glove-twitter-100.gz"


def prepare(cache):
    npz = os.path.join(cache, "glove100.npz")
    if os.path.exists(npz):
        z = np.load(npz)
        return z["base"], z["query"], z["gt"]
    os.makedirs(cache, exist_ok=True)
    gz = os.path.join(cache, "glove-twitter-100.gz")
    if not os.path.exists(gz):
        print("downloading", URL)
        urllib.request.urlretrieve(URL, gz)
    with gzip.open(gz, "rt", encoding="utf8", errors="ignore") as f:
        n, d = map(int, f.readline().split())
        X = np.zeros((n, d), np.float32)
        k = 0
        for line in f:
            p = line.rstrip().split(" ")
            if len(p) != d + 1:
                continue
            try:
                X[k] = np.asarray(p[1:], np.float32)
                k += 1
            except ValueError:
                pass
    X = X[:k]
    nrm = np.linalg.norm(X, axis=1)
    X = X[nrm > 0] / nrm[nrm > 0, None]          # angular -> inner product
    perm = np.random.default_rng(0).permutation(len(X))
    Q, B = X[perm[:10000]].copy(), X[perm[10000:]].copy()
    flat = faiss.IndexFlatIP(d)
    flat.add(B)
    _, GT = flat.search(Q, 100)
    np.savez(npz, base=B, query=Q, gt=GT.astype(np.int32))
    return B, Q, GT


def recall(I, GT):
    return float(np.mean([len(set(a[:10]) & set(g[:10])) / 10 for a, g in zip(I, GT)]))


def timed(fn, Q):
    faiss.omp_set_num_threads(1)
    fn(Q[:50])  # warm-up / compile
    t = time.perf_counter()
    I = fn(Q)
    qps = len(Q) / (time.perf_counter() - t)
    faiss.omp_set_num_threads(os.cpu_count())
    return I, qps


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache", default="data")
    ap.add_argument("--queries", type=int, default=2000)
    ap.add_argument("--nprobe", type=int, nargs="+", default=[64, 256])
    ap.add_argument("--k1", type=int, default=300)
    a = ap.parse_args()
    B, Q, GT = prepare(a.cache)
    Q, GT = Q[:a.queries], GT[:a.queries]
    n, d = B.shape
    print(f"GloVe-100: {n} vectors, {len(Q)} queries, recall@10, k1={a.k1} candidates\n")
    rows = []

    from numba import set_num_threads
    t = time.time()
    ix = E8RerankIndex(d, levels=3)  # levels 1-2 of a 3-level code are a 2-level code
    ix.train(B)
    ix.add(B)
    build = time.time() - t
    for levels in (2, 3):
        bpv = ix.bytes_per_vector(levels)
        for npb in a.nprobe:
            set_num_threads(1)
            I, qps = timed(lambda q: ix.search(q, 10, npb, a.k1, levels)[1], Q)
            set_num_threads(os.cpu_count())
            rows.append((f"E8 x{levels} score-aware", bpv["recheck"], npb, recall(I, GT), qps, build))
    first = ix.first  # reuse the same first pass for FAISS rivals

    for name, factory, nbytes in [("FAISS PQ 8-bit", f"PQ{d // 2}", d // 2), ("FAISS SQ8", "SQ8", d)]:
        t = time.time()
        ref = faiss.index_factory(d, factory, faiss.METRIC_INNER_PRODUCT)
        sample = B[np.random.default_rng(1).choice(n, 100000, replace=False)]
        if factory.startswith("PQ"):
            ref.pq.train(sample)  # much faster than IndexPQ.train
            ref.is_trained = True
        else:
            ref.train(sample)
        ref.add(B)
        rf = faiss.IndexRefine(first, ref)
        build = time.time() - t
        for npb in a.nprobe:
            params = faiss.IndexRefineSearchParameters(
                k_factor=a.k1 / 10, base_index_params=faiss.SearchParametersIVF(nprobe=npb))
            I, qps = timed(lambda q: rf.search(q, 10, params=params)[1], Q)
            rows.append((name, nbytes, npb, recall(I, GT), qps, build))

    print(f"{'re-check code':24s} {'bytes':>6s} {'nprobe':>6s} {'recall':>7s} {'QPS':>7s} {'build s':>8s}")
    for r in rows:
        print(f"{r[0]:24s} {r[1]:6.0f} {r[2]:6d} {r[3]:7.3f} {r[4]:7.0f} {r[5]:8.0f}")
    print(f"\nFirst pass (shared by all rows): {bpv['first_pass']:.1f} bytes/vector.")


if __name__ == "__main__":
    main()
