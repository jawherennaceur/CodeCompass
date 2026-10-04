"""
Test d'isolation entre deux projets : vérifie que ré-indexer le
projet B ne touche PAS aux données du projet A dans Qdrant — c'est le
bug exact qu'on corrige avec cette refonte multi-projets.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from ingestion.parser import CodeChunk
from indexing.dense_index import build_dense_index, search_dense, get_client

COLLECTION_A = "test_isolation_project_a"
COLLECTION_B = "test_isolation_project_b"


def _fake_chunk(chunk_id_suffix: str, name: str, code: str) -> CodeChunk:
    return CodeChunk(
        chunk_id=f"test-{chunk_id_suffix}",
        file_path="fake.py",
        name=name,
        node_type="function_definition",
        code=code,
        start_line=1,
        end_line=2,
    )


def main():
    client = get_client()

    # Nettoyage avant de commencer, au cas où un test précédent aurait laissé des traces
    for col in (COLLECTION_A, COLLECTION_B):
        try:
            client.delete_collection(col)
        except Exception:
            pass

    chunks_a = [_fake_chunk("a1", "connect_db", "def connect_db(): pass")]
    chunks_b = [_fake_chunk("b1", "send_email", "def send_email(): pass")]

    print("Indexation du projet A...")
    build_dense_index(chunks_a, collection_name=COLLECTION_A)

    print("Indexation du projet B...")
    build_dense_index(chunks_b, collection_name=COLLECTION_B)

    print("\nRecherche 'database' dans le projet A :")
    results_a = search_dense("database", collection_name=COLLECTION_A, top_k=5)
    for r in results_a:
        print(f"  - {r['name']}")

    print("\nRecherche 'database' dans le projet B :")
    results_b = search_dense("database", collection_name=COLLECTION_B, top_k=5)
    for r in results_b:
        print(f"  - {r['name']}")

    # LE VRAI TEST : re-indexer le projet B ne doit RIEN changer au projet A
    print("\nRé-indexation du projet B (sans rien changer)...")
    build_dense_index(chunks_b, collection_name=COLLECTION_B)

    print("\nVérification : le projet A est-il toujours intact ?")
    results_a_after = search_dense("database", collection_name=COLLECTION_A, top_k=5)
    names_before = {r["name"] for r in results_a}
    names_after = {r["name"] for r in results_a_after}
    print(f"  Avant: {names_before}")
    print(f"  Après: {names_after}")
    print(f"  Isolation respectée ? {names_before == names_after}")

    # Nettoyage final
    client.delete_collection(COLLECTION_A)
    client.delete_collection(COLLECTION_B)
    print("\nCollections de test nettoyées.")


if __name__ == "__main__":
    main()