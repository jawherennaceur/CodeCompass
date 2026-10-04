"""
Test isolé de source_manager.py — clone un petit repo de test, vérifie
le comportement, puis nettoie automatiquement après lui.
"""
import shutil
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from source_manager import clone_or_pull_github, resolve_local, SourceError

TEST_URL = "https://github.com/octocat/Hello-World"
TEST_TARGET = Path(__file__).resolve().parent.parent / "data" / "repos" / "test-hello-world"


def main():
    if TEST_TARGET.exists():
        shutil.rmtree(TEST_TARGET)

    print(f"Clonage de {TEST_URL}...")
    path = clone_or_pull_github(TEST_URL, str(TEST_TARGET))
    print(f"Cloné dans: {path}")

    print("\nRe-test (doit faire un pull, pas un re-clone)...")
    path2 = clone_or_pull_github(TEST_URL, str(TEST_TARGET))
    print(f"Mis à jour dans: {path2}")

    print("\nTest résolution locale...")
    resolved = resolve_local(path)
    print(f"Résolu: {resolved}")

    print("\nTest d'erreur (dossier inexistant)...")
    try:
        resolve_local("C:/ce/dossier/nexiste/pas")
    except SourceError as e:
        print(f"Erreur correctement levée: {e}")

    # Nettoyage automatique
    shutil.rmtree(TEST_TARGET)
    print(f"\nDossier de test nettoyé: {TEST_TARGET}")


if __name__ == "__main__":
    main()