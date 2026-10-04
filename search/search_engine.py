"""
Point d'entrée unique de la recherche : orchestre routeur -> dense/sparse
-> merger -> (reranker si hybrid). C'est cette fonction que le serveur
MCP expose comme tool.
"""
from config import TOP_K_DEFAULT, TOP_K_CANDIDATES, RERANK_CONFIDENCE_THRESHOLD
from indexing.dense_index import search_dense
from indexing.sparse_index import SparseIndex
from routing.router import classify_query_fast
from search.merger import merge_results
from search.reranker import rerank
import sys
_sparse_index: SparseIndex | None = None


def _get_sparse_index() -> SparseIndex:
    global _sparse_index
    if _sparse_index is None:
        _sparse_index = SparseIndex.load()
    return _sparse_index


def _sparse_results_as_dicts(query: str, top_k: int) -> list[dict]:
    index = _get_sparse_index()
    ranked = index.search(query, top_k=top_k)
    return [
        {
            "chunk_id": chunk.chunk_id,
            "file_path": chunk.file_path,
            "name": chunk.name,
            "code": chunk.code,
            "start_line": chunk.start_line,
            "end_line": chunk.end_line,
            "score": float(score),
        }
        for chunk, score in ranked
    ]


def search_code(query: str, top_k: int = TOP_K_DEFAULT) -> dict:
    route, router_latency_ms, fallback_reason, source = classify_query_fast(query)
    low_confidence = False

    if route == "dense":
        results = search_dense(query, top_k=top_k)
    elif route == "sparse":
        results = _sparse_results_as_dicts(query, top_k=top_k)
    else:  # hybrid
        # RRF présélectionne TOP_K_CANDIDATES (20) candidats, le
        # cross-encoder les reclasse finement et ne garde que top_k.
        dense_results = search_dense(query, top_k=TOP_K_CANDIDATES)
        sparse_results = _sparse_results_as_dicts(query, top_k=TOP_K_CANDIDATES)
        merged_candidates = merge_results(dense_results, sparse_results, top_k=TOP_K_CANDIDATES)
        results = rerank(query, merged_candidates, top_k=top_k)

        # Signal de confiance : si même le meilleur résultat reste sous
        # le seuil, aucun résultat n'est réellement pertinent — on le
        # signale plutôt que de laisser croire à une bonne réponse.
        if results and results[0]["rerank_score"] < RERANK_CONFIDENCE_THRESHOLD:
            low_confidence = True

    return {
        "query": query,
        "route": route,
        "router_latency_ms": round(router_latency_ms, 1),
        "router_source": source,
        "router_fallback_reason": fallback_reason,
        "low_confidence": low_confidence,
        "results": results,
    }


if __name__ == "__main__":
    import sys

    query = sys.argv[1] if len(sys.argv) > 1 else "où est gérée l'authentification"
    output = search_code(query)
    print(f"Route: {output['route']} (source={output['router_source']}, {output['router_latency_ms']}ms)", file=sys.stderr)
    if output["router_fallback_reason"]:
        print(f"⚠️  Fallback: {output['router_fallback_reason']}", file=sys.stderr)
    if output["low_confidence"]:
        print("⚠️  Faible confiance : aucun résultat ne semble vraiment pertinent", file=sys.stderr)
    for r in output["results"]:
        score = r.get("rerank_score", r.get("fused_score", r.get("score")))
        print(f"- {r['name']} ({r['file_path']}:{r['start_line']}-{r['end_line']}) score={score}", file=sys.stderr)