"""
Test manuel de la recherche dense (Qdrant) — Phase 3, étape 5.

Usage : python scripts/test_dense.py "ta requête"
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from indexing.dense_index import search_dense


def main():
    query = sys.argv[1] if len(sys.argv) > 1 else "comment se connecter à la base de données"
    results = search_dense(query, top_k=5)

    print(f"\nRequête : '{query}'\n")
    for i, r in enumerate(results, 1):
        print(f"{i}. {r['name']}  (score={r['score']:.3f})  [{r['file_path']}:{r['start_line']}]")


if __name__ == "__main__":
    main()