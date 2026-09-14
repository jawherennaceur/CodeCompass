"""
Point d'entrée unique de la recherche : orchestre routeur -> dense/sparse
-> merger. C'est cette fonction que le serveur MCP expose comme tool.
"""
from config import TOP_K_DEFAULT
from indexing.dense_index import search_dense
from indexing.sparse_index import SparseIndex
from routing.router import classify_query
from search.merger import merge_results

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
    """
    Fonction centrale du pipeline. Retourne un dict avec :
    - route utilisée (dense/sparse/hybrid)
    - latence du routeur
    - liste des résultats (chunk_id, file_path, name, code, lignes, score)
    """
    route, router_latency_ms = classify_query(query)

    if route == "dense":
        results = search_dense(query, top_k=top_k)
    elif route == "sparse":
        results = _sparse_results_as_dicts(query, top_k=top_k)
    else:  # hybrid
        dense_results = search_dense(query, top_k=top_k * 2)
        sparse_results = _sparse_results_as_dicts(query, top_k=top_k * 2)
        results = merge_results(dense_results, sparse_results, top_k=top_k)

    return {
        "query": query,
        "route": route,
        "router_latency_ms": round(router_latency_ms, 1),
        "results": results,
    }


if __name__ == "__main__":
    import sys

    query = sys.argv[1] if len(sys.argv) > 1 else "où est gérée l'authentification"
    output = search_code(query)
    print(f"Route choisie: {output['route']} ({output['router_latency_ms']}ms)")
    for r in output["results"]:
        print(f"- {r['name']} ({r['file_path']}:{r['start_line']}-{r['end_line']}) score={r.get('score')}")
