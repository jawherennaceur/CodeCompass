"""
Test de la détection de fraîcheur : vérifie qu'une modification de
fichier déclenche bien needs_reindex() == True, et qu'un projet
fraîchement indexé renvoie False.
"""
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from projects import register_project, get_project
from indexer import index_project, ensure_fresh

TEST_PROJECT = "test-freshness"
TEST_FILE = Path(__file__).resolve().parent.parent / "sample_repo" / "example_module.py"


def main():
    register_project(TEST_PROJECT, "local", "sample_repo")

    print("Indexation initiale...")
    index_project(TEST_PROJECT)

    print("\nVérification juste après indexation (doit être False)...")
    reindexed = ensure_fresh(TEST_PROJECT)
    print(f"Ré-indexé ? {reindexed}")

    print("\nModification du fichier (on touche juste sa date)...")
    time.sleep(1)  # s'assurer que le timestamp est bien postérieur
    TEST_FILE.touch()

    print("\nVérification après modification (doit être True)...")
    reindexed = ensure_fresh(TEST_PROJECT)
    print(f"Ré-indexé ? {reindexed}")

    print("\nVérification juste après cette ré-indexation (doit être False à nouveau)...")
    reindexed = ensure_fresh(TEST_PROJECT)
    print(f"Ré-indexé ? {reindexed}")


if __name__ == "__main__":
    main()