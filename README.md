"""
Routeur : classifie une requête utilisateur en "dense", "sparse" ou "hybrid".

Approche retenue (Option C, discutée en revue) : une heuristique légère
tranche d'abord les cas évidents sans aucun appel réseau. Haiku n'est
appelé qu'en repli, pour les requêtes réellement ambiguës. Ça évite de
payer une latence de 300-800ms et une dépendance réseau pour des
requêtes comme "connect_db" qui n'ont besoin d'aucune "intelligence"
pour être classifiées correctement.
"""
import re
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


# CONCEPTION PRODUCTION : on remplace "un seul mot suffit à trancher"
# (fragile — pris en défaut en test réel sur "UserAuth flow" et sur
# "où est définie la fonction parse_config") par un calcul de
# PROPORTION sur l'ensemble de la requête, volontairement conservateur.
# Principe : l'heuristique ne tranche QUE les cas sans ambiguïté
# structurelle ; au moindre doute, elle renvoie None et laisse Haiku
# décider. Le coût d'un appel Haiku de plus est minime comparé au coût
# d'une mauvaise classification silencieuse dans un outil quotidien.
_MAX_WORDS_FOR_SPARSE = 3
_MIN_WORDS_FOR_DENSE = 5


def _heuristic_classify(query: str) -> str | None:
    """Retourne 'sparse', 'dense', ou None si le cas est ambigu (auquel
    cas on doit appeler Haiku pour trancher)."""
    words = query.strip().split()
    num_words = len(words)

    if num_words == 0:
        return None

    identifier_flags = [_looks_like_identifier(w) for w in words]
    identifier_ratio = sum(identifier_flags) / num_words

    lowered = {w.lower().strip("?,.:;'\"") for w in words}
    has_nl_marker = bool(lowered & _NATURAL_LANGUAGE_MARKERS)

    # Cas évident -> sparse : TOUS les mots ressemblent à un
    # identifiant (ratio 100%, pas "au moins un"), requête courte.
    # "UserAuth flow" (ratio 50%) n'est PAS ce cas — "flow" suggère une
    # intention plus large que "juste trouver ce symbole".
    if identifier_ratio == 1.0 and num_words <= _MAX_WORDS_FOR_SPARSE:
        return "sparse"

    # Cas évident -> dense : AUCUN mot ne ressemble à un identifiant, un
    # marqueur de langage naturel est présent, requête assez longue.
    if identifier_ratio == 0.0 and has_nl_marker and num_words >= _MIN_WORDS_FOR_DENSE:
        return "dense"

    # Tout le reste (mélange identifiant + langage naturel, longueur
    # intermédiaire, pas de marqueur clair...) : ambigu -> Haiku.
    return None


def classify_query(query: str) -> tuple[str, float, str | None]:
    """Classification via Haiku uniquement (pas d'heuristique). Appelée
    en repli par classify_query_fast() pour les cas ambigus.

    Retourne (route, latence_ms, fallback_reason).
    fallback_reason est None si le routeur a réellement classifié la
    requête ; sinon il indique la panne ("missing_api_key",
    "router_error", "unexpected_response").
    """
    start = time.perf_counter()

    if not ANTHROPIC_API_KEY:
        latency_ms = (time.perf_counter() - start) * 1000
        return "hybrid", latency_ms, "missing_api_key"

    try:
        response = _client.messages.create(
            model=ROUTER_MODEL,
            max_tokens=ROUTER_MAX_TOKENS,
            system=_ROUTER_SYSTEM_PROMPT,
            messages=[{"role": "user", "content": query}],
        )
        raw = response.content[0].text.strip().lower()
        route = raw if raw in _VALID_ROUTES else "hybrid"
        fallback_reason = None if raw in _VALID_ROUTES else "unexpected_response"
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