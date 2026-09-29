"""
Liste les modèles actuellement disponibles sur NVIDIA NIM, avec ta clé.

Utile car le catalogue change régulièrement (modèles retirés/ajoutés) —
plutôt que de deviner un nom de modèle, on interroge l'API directement.

Usage : python scripts/list_nim_models.py [filtre optionnel, ex: llama]
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from openai import OpenAI
from config import NVIDIA_API_KEY, NIM_BASE_URL


def main():
    filter_str = sys.argv[1].lower() if len(sys.argv) > 1 else None

    client = OpenAI(base_url=NIM_BASE_URL, api_key=NVIDIA_API_KEY)
    models = client.models.list()

    ids = sorted(m.id for m in models.data)
    if filter_str:
        ids = [i for i in ids if filter_str in i.lower()]

    print(f"{len(ids)} modèle(s) trouvé(s) :\n")
    for model_id in ids:
        print(f"- {model_id}")


if __name__ == "__main__":
    main()