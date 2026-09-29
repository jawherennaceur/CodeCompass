"""
Teste plusieurs modèles NVIDIA NIM d'un coup avec un appel minimal, pour
identifier rapidement lesquels fonctionnent réellement sur ce compte
(GET /v1/models peut lister un modèle sans que POST /v1/chat/completions
fonctionne pour autant — voir les erreurs rencontrées en Phase 4).

Usage : python scripts/probe_nim_models.py
"""
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from openai import OpenAI
from config import NVIDIA_API_KEY, NIM_BASE_URL

# Modèles namespace nvidia/ trouvés dans le catalogue, du plus petit/rapide
# au plus gros, sans modèles de raisonnement (pour éviter le problème
# rencontré avec nemotron-3.5-lightning).
CANDIDATES = [
    "nvidia/mistral-nemo-minitron-8b-8k-instruct",
    "nvidia/llama-3.1-nemotron-51b-instruct",
    "nvidia/llama-3.1-nemotron-70b-instruct",
    "nvidia/nemotron-4-340b-instruct",
    # Quelques modèles tiers, pour confirmer s'ils sont VRAIMENT tous
    # bloqués ou si certains passent malgré tout.
    "mistralai/mistral-7b-instruct-v0.3",
    "meta/llama-3.2-11b-vision-instruct",
    "ibm/granite-3.0-8b-instruct",
    "deepseek-ai/deepseek-coder-6.7b-instruct",
]

client = OpenAI(base_url=NIM_BASE_URL, api_key=NVIDIA_API_KEY)


def probe(model_id: str) -> tuple[bool, str, float]:
    start = time.perf_counter()
    try:
        response = client.chat.completions.create(
            model=model_id,
            max_tokens=5,
            temperature=0,
            messages=[{"role": "user", "content": "dis juste 'ok'"}],
        )
        latency_ms = (time.perf_counter() - start) * 1000
        text = response.choices[0].message.content
        return True, f"OK -> '{text}'", latency_ms
    except Exception as e:
        latency_ms = (time.perf_counter() - start) * 1000
        return False, str(e)[:150], latency_ms


def main():
    print(f"Test de {len(CANDIDATES)} modèles...\n")
    working = []
    for model_id in CANDIDATES:
        success, detail, latency_ms = probe(model_id)
        status = "✅" if success else "❌"
        print(f"{status} {model_id}  [{latency_ms:.0f}ms]")
        print(f"   {detail}\n")
        if success:
            working.append(model_id)

    print("=" * 60)
    if working:
        print(f"Modèles fonctionnels ({len(working)}) :")
        for m in working:
            print(f"  - {m}")
    else:
        print("Aucun modèle testé ne fonctionne sur ce compte.")


if __name__ == "__main__":
    main()