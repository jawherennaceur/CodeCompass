"""
Évaluation : compare précision@5 entre dense seul / sparse seul / hybride
sur le jeu de requêtes annotées (eval_queries.json).

Usage : python tests/evaluate.py
"""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from indexing.dense_index import search_dense
from indexing.sparse_index import SparseIndex
from search.merger import merge_results

EVAL_FILE = Path(__file__).parent / "eval_queries.json"


def _hit(results: list[dict], expected_substring: str) -> bool:
    return any(expected_substring.lower() in r["code"].lower() for r in results)


def evaluate():
    with open(EVAL_FILE) as f:
        data = json.load(f)

    sparse_index = SparseIndex.load()

    scores = {"dense": 0, "sparse": 0, "hybrid": 0}
    total = len(data["queries"])

    for item in data["queries"]:
        query = item["query"]
        expected = item["expected_chunk_contains"]

        dense_results = search_dense(query, top_k=5)
        sparse_ranked = sparse_index.search(query, top_k=5)
        sparse_results = [
            {"chunk_id": c.chunk_id, "code": c.code, "file_path": c.file_path,
             "start_line": c.start_line, "end_line": c.end_line, "name": c.name}
            for c, _ in sparse_ranked
        ]
        hybrid_results = merge_results(dense_results, sparse_results, top_k=5)

        if _hit(dense_results, expected):
            scores["dense"] += 1
        if _hit(sparse_results, expected):
            scores["sparse"] += 1
        if _hit(hybrid_results, expected):
            scores["hybrid"] += 1

    print(f"Résultats sur {total} requêtes (precision@5) :")
    for method, count in scores.items():
        pct = 100 * count / total if total else 0
        print(f"  {method:8s} : {count}/{total}  ({pct:.0f}%)")


if __name__ == "__main__":
    evaluate()
