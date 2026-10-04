"""
Test isolé de projects.py — utilise un registre TEMPORAIRE, jamais le
vrai data/projects.json, pour ne pas polluer les données réelles.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from projects import register_project, list_projects

TEST_REGISTRY = Path(__file__).resolve().parent.parent / "data" / "test_projects.json"


def main():
    # Nettoyage avant de commencer, au cas où un test précédent aurait laissé des traces
    if TEST_REGISTRY.exists():
        TEST_REGISTRY.unlink()

    p1 = register_project("projet-a", "local", "C:/fake/projet-a", registry_path=TEST_REGISTRY)
    p2 = register_project("projet-b", "local", "C:/fake/projet-b", registry_path=TEST_REGISTRY)

    print("Projet A:", p1)
    print("Projet B:", p2)
    print()
    print("Collections différentes ?", p1.qdrant_collection != p2.qdrant_collection)
    print("Fichiers BM25 différents ?", p1.bm25_index_path != p2.bm25_index_path)
    print()
    print("Liste des projets enregistrés (registre de test):")
    for p in list_projects(registry_path=TEST_REGISTRY):
        print(f"  - {p.name} ({p.source_type}) -> {p.qdrant_collection}")

    # Nettoyage après le test — le vrai data/projects.json n'a jamais été touché
    TEST_REGISTRY.unlink()
    print(f"\nFichier de test nettoyé: {TEST_REGISTRY}")


if __name__ == "__main__":
    main()