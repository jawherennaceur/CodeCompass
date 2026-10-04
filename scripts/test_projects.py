"""
Test isolé de projects.py — registre temporaire + test de collision de
noms (bug corrigé en revue).
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from projects import register_project, list_projects, ProjectError

TEST_REGISTRY = Path(__file__).resolve().parent.parent / "data" / "test_projects.json"


def main():
    if TEST_REGISTRY.exists():
        TEST_REGISTRY.unlink()

    p1 = register_project("projet-a", "local", "C:/fake/projet-a", registry_path=TEST_REGISTRY)
    p2 = register_project("projet-b", "local", "C:/fake/projet-b", registry_path=TEST_REGISTRY)

    print("Collections différentes ?", p1.qdrant_collection != p2.qdrant_collection)

    print("\nTest collision de noms (Mon-Projet vs mon projet)...")
    register_project("Mon-Projet", "local", "C:/fake/x", registry_path=TEST_REGISTRY)
    try:
        register_project("mon projet", "local", "C:/fake/y", registry_path=TEST_REGISTRY)
        print("ÉCHEC : la collision aurait dû être détectée !")
    except ProjectError as e:
        print(f"Collision correctement détectée: {e}")

    print("\nTest ré-enregistrement sans overwrite (doit échouer)...")
    try:
        register_project("projet-a", "local", "C:/fake/autre", registry_path=TEST_REGISTRY)
        print("ÉCHEC : l'écrasement silencieux aurait dû être bloqué !")
    except ProjectError as e:
        print(f"Écrasement correctement bloqué: {e}")

    TEST_REGISTRY.unlink()
    print(f"\nFichier de test nettoyé: {TEST_REGISTRY}")


if __name__ == "__main__":
    main()