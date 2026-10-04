"""
Reranker cross-encoder : affine le classement du top-20 fusionné (RRF)
en un top-5 final, en évaluant chaque paire (requête, code) ensemble
plutôt que séparément.

Différence avec l'embedding de la Phase 3 (bi-encoder) : celui-ci encode
requête et code séparément puis compare des vecteurs déjà calculés —
rapide, mais moins précis. Le cross-encoder regarde les deux textes en
même temps, donc plus précis, mais trop lent pour tout l'index — d'où
son usage uniquement sur les 20 candidats déjà présélectionnés par RRF.
"""
from sentence_transformers import CrossEncoder
import sys
from config import TOP_K_DEFAULT

_CROSS_ENCODER_MODEL_NAME = "cross-encoder/ms-marco-MiniLM-L-6-v2"

_model: CrossEncoder | None = None


def get_cross_encoder() -> CrossEncoder:
    """Charge le cross-encoder une seule fois (singleton), comme le
    modèle d'embeddings en Phase 3."""
    global _model
    if _model is None:
        _model = CrossEncoder(_CROSS_ENCODER_MODEL_NAME)
    return _model


def rerank(query: str, candidates: list[dict], top_k: int = TOP_K_DEFAULT) -> list[dict]:
    """
    Prend les candidats déjà fusionnés par RRF (idéalement ~20), les
    reclasse selon un score de pertinence cross-encoder, et retourne
    les top_k meilleurs.
    """
    if not candidates:
        return []

    model = get_cross_encoder()

    # Le cross-encoder attend des paires (requête, texte) — on construit
    # une paire par candidat, avec le code du chunk comme texte.
    pairs = [(query, c["code"]) for c in candidates]
    scores = model.predict(pairs)

    for candidate, score in zip(candidates, scores):
        candidate["rerank_score"] = float(score)

    reranked = sorted(candidates, key=lambda c: c["rerank_score"], reverse=True)
    return reranked[:top_k]


if __name__ == "__main__":
    # Test rapide avec des exemples construits à la main.
    query = "comment se connecter à la base de données"
    candidates = [
        {"chunk_id": "a", "name": "connect_db", "code": "def connect_db(host, port):\n    print('connexion')\n    return True"},
        {"chunk_id": "b", "name": "retry_api_call", "code": "def retry_api_call(func, max_retries=3):\n    for i in range(max_retries):\n        func()"},
        {"chunk_id": "c", "name": "login", "code": "def login(self, username, password):\n    return self._check(username, password)"},
    ]
    results = rerank(query, candidates, top_k=3)
    for i, r in enumerate(results, 1):
        print(f"{i}. {r['name']}  (rerank_score={r['rerank_score']:.3f})", file=sys.stderr)