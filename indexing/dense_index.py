"""
Index dense (recherche sémantique) : embeddings locaux (bge-small-en)
stockés dans Qdrant.

Justification du choix local : latence quasi nulle (pas d'appel réseau à
chaque recherche) + confidentialité du code (rien ne quitte la machine).
"""
import uuid

from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, PointStruct
from sentence_transformers import SentenceTransformer

from config import (
    EMBEDDING_MODEL_NAME,
    EMBEDDING_DIM,
    QDRANT_HOST,
    QDRANT_PORT,
    QDRANT_COLLECTION,
)
from ingestion.parser import CodeChunk

_model: SentenceTransformer | None = None

# Namespace fixe pour dériver un UUID stable à partir de chunk_id.
# Sans ça, chaque ré-indexation créerait de nouveaux points au lieu
# d'écraser les existants (doublons infinis dans Qdrant).
_POINT_ID_NAMESPACE = uuid.UUID("7c1f2b2a-0000-4000-8000-000000000001")


def _point_id_for(chunk_id: str) -> str:
    """UUID déterministe : le même chunk_id produit toujours le même ID,
    donc upsert() écrase bien l'ancien point au lieu d'en créer un nouveau."""
    return str(uuid.uuid5(_POINT_ID_NAMESPACE, chunk_id))


def get_model() -> SentenceTransformer:
    """Charge le modèle d'embeddings local une seule fois (singleton)."""
    global _model
    if _model is None:
        _model = SentenceTransformer(EMBEDDING_MODEL_NAME)
    return _model


def get_client() -> QdrantClient:
    return QdrantClient(host=QDRANT_HOST, port=QDRANT_PORT)


def ensure_collection(client: QdrantClient):
    existing = [c.name for c in client.get_collections().collections]
    if QDRANT_COLLECTION not in existing:
        client.create_collection(
            collection_name=QDRANT_COLLECTION,
            vectors_config=VectorParams(size=EMBEDDING_DIM, distance=Distance.COSINE),
        )


def build_dense_index(chunks: list[CodeChunk], batch_size: int = 32):
    """Calcule les embeddings de chaque chunk et les insère dans Qdrant."""
    model = get_model()
    client = get_client()
    ensure_collection(client)

    points: list[PointStruct] = []
    for i in range(0, len(chunks), batch_size):
        batch = chunks[i:i + batch_size]
        texts = [c.code for c in batch]
        embeddings = model.encode(texts, show_progress_bar=False)

        for chunk, vector in zip(batch, embeddings):
            points.append(PointStruct(
                id=_point_id_for(chunk.chunk_id),
                vector=vector.tolist(),
                payload={
                    "chunk_id": chunk.chunk_id,
                    "file_path": chunk.file_path,
                    "name": chunk.name,
                    "node_type": chunk.node_type,
                    "code": chunk.code,
                    "start_line": chunk.start_line,
                    "end_line": chunk.end_line,
                },
            ))

    client.upsert(collection_name=QDRANT_COLLECTION, points=points)
    return len(points)


def search_dense(query: str, top_k: int = 5) -> list[dict]:
    model = get_model()
    client = get_client()
    query_vector = model.encode(query).tolist()

    try:
        response = client.query_points(
            collection_name=QDRANT_COLLECTION,
            query=query_vector,
            limit=top_k,
        )
        results = response.points
    except Exception as e:
        print(f"[ERROR] Recherche dense indisponible (Qdrant injoignable ?): {e}")
        return []

    return [
        {
            "chunk_id": r.payload["chunk_id"],
            "file_path": r.payload["file_path"],
            "name": r.payload["name"],
            "code": r.payload["code"],
            "start_line": r.payload["start_line"],
            "end_line": r.payload["end_line"],
            "score": r.score,
        }
        for r in results
    ]


if __name__ == "__main__":
    from ingestion.parser import parse_repo

    chunks = parse_repo()
    n = build_dense_index(chunks)
    print(f"Index dense construit et inséré dans Qdrant ({n} chunks).")