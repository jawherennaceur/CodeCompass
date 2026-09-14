"""
Routeur : classifie une requête utilisateur en "dense", "sparse" ou "hybrid".

Utilise Claude Haiku (petit modèle rapide) pour la classification.
Budget de latence cible : < 200ms (critère de validation du Projet 1).
"""
import time

import anthropic

from config import ANTHROPIC_API_KEY, ROUTER_MODEL, ROUTER_MAX_TOKENS, ROUTER_LATENCY_BUDGET_MS

_ROUTER_SYSTEM_PROMPT = """Tu classifies des requêtes de recherche de code en une seule catégorie parmi :
- "sparse" : la requête contient un nom exact (fonction, classe, variable, fichier) à rechercher tel quel.
- "dense" : la requête décrit une intention/un concept sans nom précis (ex: "où est géré le retry ?").
- "hybrid" : la requête mélange un terme précis ET une intention plus large.

Réponds UNIQUEMENT par un seul mot : sparse, dense, ou hybrid. Aucune explication.

Exemples :
Q: "où est définie la fonction parse_config" -> sparse
Q: "comment est gérée l'authentification" -> dense
Q: "tous les endroits qui utilisent le cache Redis pour les sessions" -> hybrid
"""

_client = anthropic.Anthropic(api_key=ANTHROPIC_API_KEY)

_VALID_ROUTES = {"sparse", "dense", "hybrid"}


def classify_query(query: str) -> tuple[str, float]:
    """Retourne (route, latence_ms). Fallback sur 'hybrid' en cas d'erreur
    ou de réponse inattendue, pour ne jamais bloquer la recherche."""
    start = time.perf_counter()
    try:
        response = _client.messages.create(
            model=ROUTER_MODEL,
            max_tokens=ROUTER_MAX_TOKENS,
            system=_ROUTER_SYSTEM_PROMPT,
            messages=[{"role": "user", "content": query}],
        )
        raw = response.content[0].text.strip().lower()
        route = raw if raw in _VALID_ROUTES else "hybrid"
    except Exception as e:
        print(f"[WARN] Routeur en échec ({e}), fallback sur 'hybrid'.")
        route = "hybrid"

    latency_ms = (time.perf_counter() - start) * 1000
    if latency_ms > ROUTER_LATENCY_BUDGET_MS:
        print(f"[WARN] Routeur au-delà du budget latence: {latency_ms:.0f}ms")

    return route, latency_ms


if __name__ == "__main__":
    test_queries = [
        "où est définie la fonction parse_config",
        "comment est gérée l'authentification",
        "tous les endroits qui utilisent le cache Redis",
    ]
    for q in test_queries:
        route, latency = classify_query(q)
        print(f"'{q}' -> {route} ({latency:.0f}ms)")
