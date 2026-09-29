"""
Routeur : classifie une requête utilisateur en "dense", "sparse" ou "hybrid".

Approche retenue (Option C) : une heuristique légère tranche d'abord les
cas évidents sans aucun appel réseau. Le modèle NIM (NVIDIA) n'est
appelé qu'en repli, pour les requêtes réellement ambiguës.

Fournisseur : NVIDIA NIM (API compatible OpenAI), modèle
meta/llama-3.1-8b-instruct — remplace l'API Anthropic (Haiku) utilisée
initialement, décision explicite pour accéder à un tier gratuit/moins
coûteux.
"""
import re
import time

from openai import OpenAI

from config import NVIDIA_API_KEY, NIM_BASE_URL, ROUTER_MODEL, ROUTER_MAX_TOKENS, ROUTER_LATENCY_BUDGET_MS

_ROUTER_SYSTEM_PROMPT = """Tu classifies des requêtes de recherche de code en une seule catégorie parmi :
- "sparse" : la requête contient un nom exact (fonction, classe, variable, fichier) à rechercher tel quel.
- "dense" : la requête décrit une intention/un concept sans nom précis (ex: "où est géré le retry ?").
- "hybrid" : la requête mélange un terme précis ET une intention plus large.

Réponds UNIQUEMENT par un seul mot : sparse, dense, ou hybrid. Aucune explication, aucune ponctuation.

Exemples :
Q: "où est définie la fonction parse_config" -> sparse
Q: "comment est gérée l'authentification" -> dense
Q: "tous les endroits qui utilisent le cache Redis pour les sessions" -> hybrid
"""

_client = OpenAI(base_url=NIM_BASE_URL, api_key=NVIDIA_API_KEY)

_VALID_ROUTES = {"sparse", "dense", "hybrid"}

# --- Heuristique de pré-filtrage (Option C) --------------------------------

_IDENTIFIER_PATTERN = re.compile(r"_|[a-z][A-Z]")

_NATURAL_LANGUAGE_MARKERS = {
    "comment", "où", "pourquoi", "quoi", "quand", "qui", "que",
    "est", "sont", "gérée", "géré", "gère", "fonctionne", "utilisé",
    "utilisée", "tous", "toutes", "les", "endroits",
    "how", "where", "why", "what", "when", "who", "which",
    "is", "are", "does", "do", "handled", "used",
}


def _looks_like_identifier(word: str) -> bool:
    return bool(_IDENTIFIER_PATTERN.search(word.strip("?,.:;'\"")))


_MAX_WORDS_FOR_SPARSE = 3
_MIN_WORDS_FOR_DENSE = 5


def _heuristic_classify(query: str) -> str | None:
    """Retourne 'sparse', 'dense', ou None si le cas est ambigu (auquel
    cas on doit appeler le modèle NIM pour trancher)."""
    words = query.strip().split()
    num_words = len(words)

    if num_words == 0:
        return None

    identifier_flags = [_looks_like_identifier(w) for w in words]
    identifier_ratio = sum(identifier_flags) / num_words

    lowered = {w.lower().strip("?,.:;'\"") for w in words}
    has_nl_marker = bool(lowered & _NATURAL_LANGUAGE_MARKERS)

    if identifier_ratio == 1.0 and num_words <= _MAX_WORDS_FOR_SPARSE:
        return "sparse"

    if identifier_ratio == 0.0 and has_nl_marker and num_words >= _MIN_WORDS_FOR_DENSE:
        return "dense"

    return None


def classify_query(query: str) -> tuple[str, float, str | None]:
    """Classification via le modèle NIM uniquement (pas d'heuristique).
    Appelée en repli par classify_query_fast() pour les cas ambigus.

    Retourne (route, latence_ms, fallback_reason).
    """
    start = time.perf_counter()

    if not NVIDIA_API_KEY:
        latency_ms = (time.perf_counter() - start) * 1000
        return "hybrid", latency_ms, "missing_api_key"

    try:
        response = _client.chat.completions.create(
            model=ROUTER_MODEL,
            max_tokens=ROUTER_MAX_TOKENS,
            temperature=0,
            messages=[
                {"role": "system", "content": _ROUTER_SYSTEM_PROMPT},
                {"role": "user", "content": query},
            ],
        )
        raw = response.choices[0].message.content.strip().lower()
        # Les modèles ouverts sont parfois moins disciplinés que Haiku
        # sur la consigne "un seul mot" — on tolère la ponctuation
        # résiduelle et on cherche le mot valide dans la réponse.
        raw_clean = raw.strip(" .!\"'")
        route = raw_clean if raw_clean in _VALID_ROUTES else "hybrid"
        fallback_reason = None if raw_clean in _VALID_ROUTES else "unexpected_response"
    except Exception as e:
        print(f"[WARN] Routeur en échec ({e}), fallback sur 'hybrid'.")
        route = "hybrid"
        fallback_reason = "router_error"

    latency_ms = (time.perf_counter() - start) * 1000
    if latency_ms > ROUTER_LATENCY_BUDGET_MS:
        print(f"[WARN] Routeur au-delà du budget latence: {latency_ms:.0f}ms")

    return route, latency_ms, fallback_reason


def classify_query_fast(query: str) -> tuple[str, float, str | None, str]:
    """Point d'entrée principal du routeur (Option C).

    Retourne (route, latence_ms, fallback_reason, source) où source
    vaut "heuristic" ou "llm".
    """
    start = time.perf_counter()
    heuristic_result = _heuristic_classify(query)

    if heuristic_result is not None:
        latency_ms = (time.perf_counter() - start) * 1000
        return heuristic_result, latency_ms, None, "heuristic"

    route, llm_latency_ms, fallback_reason = classify_query(query)
    total_latency_ms = (time.perf_counter() - start) * 1000
    return route, total_latency_ms, fallback_reason, "llm"


if __name__ == "__main__":
    test_queries = [
        "connect_db",
        "getUserData",
        "où est définie la fonction parse_config",
        "comment est gérée l'authentification des utilisateurs",
        "gestion du cache",
        "UserAuth flow",
        "tous les endroits qui utilisent le cache Redis pour les sessions",
    ]
    for q in test_queries:
        route, latency, fallback_reason, source = classify_query_fast(q)
        status = f"(fallback: {fallback_reason})" if fallback_reason else "(choix réel)"
        print(f"'{q}' -> {route} {status} [source={source}, {latency:.1f}ms]")