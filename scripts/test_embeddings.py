"""
Test manuel du modèle d'embeddings seul — Phase 3, étape 3.

Usage : python scripts/test_embeddings.py
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from indexing.dense_index import get_model


def main():
    model = get_model()

    text = "def connect_db(host, port): pass"
    vector = model.encode(text)

    print(f"Texte encodé : '{text}'")
    print(f"Dimension du vecteur : {len(vector)}")
    print(f"Premiers éléments : {vector[:5]}")
    print(f"Type : {type(vector)}")


if __name__ == "__main__":
    main()