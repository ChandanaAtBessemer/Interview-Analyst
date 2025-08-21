from sentence_transformers import SentenceTransformer
from sklearn.cluster import AgglomerativeClustering
import numpy as np

_EMB = SentenceTransformer("all-mpnet-base-v2")

def cluster_terms(terms: list[str], thresh: float = .75) -> list[str]:
    vecs = _EMB.encode(terms, normalize_embeddings=True)
    dist = 1 - np.matmul(vecs, vecs.T)           # cosine distance matrix
    cl = AgglomerativeClustering(
        metric="precomputed", linkage="average",
        distance_threshold=1-thresh, n_clusters=None
    ).fit(dist)
    buckets: dict[int, list[str]] = {}
    for t, lbl in zip(terms, cl.labels_):
        buckets.setdefault(lbl, []).append(t)
    # canonical term = shortest in bucket
    return [sorted(b, key=len)[0] for b in buckets.values()]
