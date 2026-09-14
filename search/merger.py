"""
Merger / Reranker : fusionne les résultats dense et sparse quand le
routeur choisit "hybrid", via Reciprocal Rank Fusion (RRF).

RRF est préféré à une simple somme de scores car dense (cosine ~0-1) et
sparse (BM25, échelle arbitraire) ne sont pas directement comparables.
Le rang relatif dans chaque liste, lui, l'est.
"""
from config import RRF_K, TOP_K_DEFAULT


def _rrf_score(rank: int, k: int = RRF_K) -> float:
    return 1.0 / (k + rank)


def merge_results(
    dense_results: list[dict],
    sparse_results: list[dict],
    top_k: int = TOP_K_DEFAULT,
) -> list[dict]:
    """
    dense_results / sparse_results : listes ordonnées par pertinence,
    chaque item doit avoir une clé 'chunk_id' unique.
    """
    fused_scores: dict[str, float] = {}
    chunk_data: dict[str, dict] = {}

    for rank, item in enumerate(dense_results):
        cid = item["chunk_id"]
        fused_scores[cid] = fused_scores.get(cid, 0.0) + _rrf_score(rank)
        chunk_data[cid] = item

    for rank, item in enumerate(sparse_results):
        cid = item["chunk_id"]
        fused_scores[cid] = fused_scores.get(cid, 0.0) + _rrf_score(rank)
        chunk_data.setdefault(cid, item)  # garde les métadonnées si absent côté dense

    ranked_ids = sorted(fused_scores, key=lambda cid: fused_scores[cid], reverse=True)

    merged = []
    for cid in ranked_ids[:top_k]:
        entry = dict(chunk_data[cid])
        entry["fused_score"] = fused_scores[cid]
        merged.append(entry)
    return merged
