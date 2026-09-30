"""
Test du merger RRF avec de vraies recherches sparse (BM25) et dense
(Qdrant) — Phase 5, étape 3.

Usage : python scripts/test_merger_real.py "ta requête"
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from indexing.sparse_index import SparseIndex
from indexing.dense_index import search_dense
from search.merger import merge_results


def main():
    query = sys.argv[1] if len(sys.argv) > 1 else "connect_db"

    sparse_index = SparseIndex.load()
    sparse_ranked = sparse_index.search(query, top_k=10)
    sparse_results = [
        {
            "chunk_id": chunk.chunk_id,
            "name": chunk.name,
            "code": chunk.code,
            "file_path": chunk.file_path,
            "start_line": chunk.start_line,
            "end_line": chunk.end_line,
        }
        for chunk, score in sparse_ranked
    ]

    dense_results = search_dense(query, top_k=10)

    print(f"\nRequête : '{query}'\n")

    print("--- Résultats SPARSE (BM25) ---")
    for i, r in enumerate(sparse_results, 1):
        print(f"{i}. {r['name']}  [{r['chunk_id'][:8]}...]")

    print("\n--- Résultats DENSE (Qdrant) ---")
    for i, r in enumerate(dense_results, 1):
        print(f"{i}. {r['name']}  [{r['chunk_id'][:8]}...]")

    merged = merge_results(dense_results, sparse_results, top_k=5)

    print("\n--- Résultats FUSIONNÉS (RRF) ---")
    for i, r in enumerate(merged, 1):
        print(f"{i}. {r['name']}  (fused_score={r['fused_score']:.5f})  [{r['chunk_id'][:8]}...]")


if __name__ == "__main__":
    main()