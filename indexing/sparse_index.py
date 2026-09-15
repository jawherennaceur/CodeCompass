"""
Index sparse (mots-clés exacts) basé sur BM25.

Utilisé quand le routeur détecte une requête "exacte" (nom de fonction,
symbole précis) plutôt qu'une intention sémantique floue.
"""
import json
import re
from dataclasses import dataclass, asdict

from rank_bm25 import BM25Okapi

from config import BM25_INDEX_PATH
from ingestion.parser import CodeChunk

_TOKEN_RE = re.compile(r"[A-Za-z_][A-Za-z0-9_]*")

# CORRECTIF (test Phase 2) : sans filtrage, des mots grammaticaux banals
# ("des", "le", "et"...) ou des mots-clés de langage très fréquents
# ("self", "return", "def") gonflaient artificiellement des scores BM25
# sans rapport avec la vraie pertinence de la requête — observé en test
# réel avec la requête "gestion des identifiants utilisateur", où "des"
# faisait remonter UserAuth sans lien sémantique réel.
# Liste volontairement courte et ciblée (pas une liste NLTK complète) :
# juste ce qui posait un problème concret + les évidences du même genre.
_STOPWORDS = {
    # Français — articles, prépositions, connecteurs très fréquents
    "de", "des", "du", "le", "la", "les", "un", "une", "et", "ou",
    "est", "en", "à", "au", "aux", "pour", "par", "sur", "dans", "ce",
    "cette", "ces", "son", "sa", "ses", "qui", "que", "avec",
    # Anglais — équivalents
    "the", "a", "an", "of", "and", "or", "is", "in", "on", "for",
    "to", "this", "that", "with", "as", "if", "else",
    # Mots-clés de code très fréquents, peu discriminants
    "self", "return", "def", "class", "none", "true", "false",
    "import", "from", "pass",
}


def _tokenize(text: str) -> list[str]:
    """Tokenisation simple adaptée au code : découpe aussi le snake_case
    et le camelCase pour améliorer le rappel sur les noms composés.
    Filtre aussi les stopwords (mots trop fréquents et peu informatifs)."""
    raw_tokens = _TOKEN_RE.findall(text)
    tokens: list[str] = []
    for tok in raw_tokens:
        tok_lower = tok.lower()
        if tok_lower in _STOPWORDS:
            continue
        tokens.append(tok_lower)
        # CORRECTIF (test Phase 2) : snake_case et camelCase étaient tous
        # les deux appliqués sur un même token comme "connect_db" — la
        # regex camelCase ne reconnaît pas "_" comme séparateur de mot et
        # "revoyait" donc les mêmes sous-mots ("connect", "db"), qui
        # étaient alors ajoutés une seconde fois. On les rend mutuellement
        # exclusifs : un token a soit des underscores, soit des majuscules
        # internes, rarement les deux à la fois dans du code réel.
        if "_" in tok:
            tokens.extend(
                p.lower() for p in tok.split("_")
                if p and p.lower() not in _STOPWORDS
            )
        else:
            camel_parts = re.findall(r"[A-Z]?[a-z0-9]+|[A-Z]+(?=[A-Z]|$)", tok)
            if len(camel_parts) > 1:
                tokens.extend(
                    p.lower() for p in camel_parts
                    if p.lower() not in _STOPWORDS
                )
    return tokens


@dataclass
class SparseIndex:
    bm25: BM25Okapi
    chunks: list[CodeChunk]

    def search(self, query: str, top_k: int = 5) -> list[tuple[CodeChunk, float]]:
        tokenized_query = _tokenize(query)
        scores = self.bm25.get_scores(tokenized_query)
        ranked = sorted(zip(self.chunks, scores), key=lambda x: x[1], reverse=True)
        return ranked[:top_k]

    def save(self, path: str = BM25_INDEX_PATH):
        """
        CORRECTIF (revue + décision explicite) : on ne sérialise plus
        l'objet BM25Okapi ni l'objet SparseIndex directement via pickle —
        pickle peut exécuter du code arbitraire au chargement si le
        fichier est un jour corrompu ou modifié par un tiers. On stocke
        à la place uniquement des données pures (JSON), et on reconstruit
        l'index BM25 au chargement. Bénéfice secondaire : on reconstruit
        toujours avec la version actuelle de _tokenize(), donc pas de
        risque de "tokenisation figée" périmée si le code évolue.
        """
        data = {
            "chunks": [asdict(c) for c in self.chunks],
        }
        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False)

    @staticmethod
    def load(path: str = BM25_INDEX_PATH) -> "SparseIndex":
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        chunks = [CodeChunk(**c) for c in data["chunks"]]
        return build_sparse_index(chunks)


def build_sparse_index(chunks: list[CodeChunk]) -> SparseIndex:
    tokenized_corpus = [_tokenize(c.code) for c in chunks]
    bm25 = BM25Okapi(tokenized_corpus)
    return SparseIndex(bm25=bm25, chunks=chunks)


if __name__ == "__main__":
    from ingestion.parser import parse_repo

    chunks = parse_repo()
    index = build_sparse_index(chunks)
    index.save()
    print(f"Index BM25 construit et sauvegardé ({len(chunks)} chunks).")