"""
Script d'indexation : à lancer une fois (ou à chaque mise à jour majeure
du repo) pour construire les index sparse et dense.

Usage : python scripts/index_repo.py [chemin_du_repo]
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from config import REPO_PATH
from ingestion.parser import parse_repo
from indexing.sparse_index import build_sparse_index
from indexing.dense_index import build_dense_index


def main():
    repo_path = sys.argv[1] if len(sys.argv) > 1 else REPO_PATH
    print(f"Indexation du repo: {repo_path}")

    chunks = parse_repo(repo_path)
    print(f"{len(chunks)} chunks extraits (fonctions/classes).")

    if not chunks:
        print("Aucun chunk trouvé — vérifie REPO_PATH et SUPPORTED_EXTENSIONS.")
        return

    sparse_index = build_sparse_index(chunks)
    sparse_index.save()
    print("Index sparse (BM25) construit et sauvegardé.")

    n_dense = build_dense_index(chunks)
    print(f"Index dense (Qdrant) construit : {n_dense} vecteurs insérés.")

    print("Indexation terminée.")


if __name__ == "__main__":
    main()
