"""
Test manuel de l'index sparse (BM25) — Phase 2, étape 4.

Usage : python scripts/test_sparse.py "ta requête"
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from indexing.sparse_index import SparseIndex


def main():
    query = sys.argv[1] if len(sys.argv) > 1 else "connect_db"
    index = SparseIndex.load()
    results = index.search(query, top_k=5)

    print(f"\nRequête : '{query}'\n")
    for i, (chunk, score) in enumerate(results, 1):
        print(f"{i}. {chunk.name}  (score={score:.3f})  [{chunk.file_path}:{chunk.start_line}]")


if __name__ == "__main__":
    main()