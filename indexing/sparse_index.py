"""
Index sparse (mots-clés exacts) basé sur BM25.

Utilisé quand le routeur détecte une requête "exacte" (nom de fonction,
symbole précis) plutôt qu'une intention sémantique floue.
"""
import pickle
import re
from dataclasses import dataclass

from rank_bm25 import BM25Okapi

from config import BM25_INDEX_PATH
from ingestion.parser import CodeChunk

_TOKEN_RE = re.compile(r"[A-Za-z_][A-Za-z0-9_]*")


def _tokenize(text: str) -> list[str]:
    """Tokenisation simple adaptée au code : découpe aussi le snake_case
    et le camelCase pour améliorer le rappel sur les noms composés."""
    raw_tokens = _TOKEN_RE.findall(text)
    tokens: list[str] = []
    for tok in raw_tokens:
        tokens.append(tok.lower())
        # snake_case -> sous-tokens
        if "_" in tok:
            tokens.extend(p.lower() for p in tok.split("_") if p)
        # camelCase -> sous-tokens
        camel_parts = re.findall(r"[A-Z]?[a-z0-9]+|[A-Z]+(?=[A-Z]|$)", tok)
        if len(camel_parts) > 1:
            tokens.extend(p.lower() for p in camel_parts)
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
        with open(path, "wb") as f:
            pickle.dump(self, f)

    @staticmethod
    def load(path: str = BM25_INDEX_PATH) -> "SparseIndex":
        with open(path, "rb") as f:
            return pickle.load(f)


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
