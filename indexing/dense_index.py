"""
Index dense (recherche sémantique) : embeddings locaux (bge-small-en)
stockés dans Qdrant.

Chaque fonction accepte maintenant collection_name en paramètre — pour
que chaque projet utilise sa PROPRE collection Qdrant, isolée des
autres (corrige le bug : le nettoyage des orphelins d'un projet ne doit
jamais affecter un autre projet).
"""
import uuid

from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, PointStruct
from sentence_transformers import SentenceTransformer

from config import EMBEDDING_MODEL_NAME, EMBEDDING_DIM, QDRANT_HOST, QDRANT_PORT, QDRANT_COLLECTION
from ingestion.parser import CodeChunk
from logging_config import get_logger

logger = get_logger(__name__)

_model: SentenceTransformer | None = None

_POINT_ID_NAMESPACE = uuid.UUID("7c1f2b2a-0000-4000-8000-000000000001")


def _point_id_for(chunk_id: str) -> str:
    return str(uuid.uuid5(_POINT_ID_NAMESPACE, chunk_id))


def get_model() -> SentenceTransformer:
    global _model
    if _model is None:
        _model = SentenceTransformer(EMBEDDING_MODEL_NAME)
    return _model


def get_client() -> QdrantClient:
    return QdrantClient(host=QDRANT_HOST, port=QDRANT_PORT)


def ensure_collection(client: QdrantClient, collection_name: str = QDRANT_COLLECTION):
    existing = [c.name for c in client.get_collections().collections]
    if collection_name not in existing:
        client.create_collection(
            collection_name=collection_name,
            vectors_config=VectorParams(size=EMBEDDING_DIM, distance=Distance.COSINE),
        )


def _existing_point_ids(client: QdrantClient, collection_name: str) -> set[str]:
    ids: set[str] = set()
    offset = None
    while True:
        points, offset = client.scroll(
            collection_name=collection_name,
            with_payload=False,
            with_vectors=False,
            limit=256,
            offset=offset,
        )
        ids.update(str(p.id) for p in points)
        if offset is None:
            break
    return ids


def build_dense_index(
    chunks: list[CodeChunk],
    collection_name: str = QDRANT_COLLECTION,
    batch_size: int = 32,
):
    """Calcule les embeddings de chaque chunk et les insère dans la
    collection Qdrant spécifiée (une collection par projet)."""
    model = get_model()
    client = get_client()
    ensure_collection(client, collection_name)

    current_ids: set[str] = set()
    points: list[PointStruct] = []
    for i in range(0, len(chunks), batch_size):
        batch = chunks[i:i + batch_size]
        texts = [c.code for c in batch]
        embeddings = model.encode(texts, show_progress_bar=False)

        for chunk, vector in zip(batch, embeddings):
            point_id = _point_id_for(chunk.chunk_id)
            current_ids.add(point_id)
            points.append(PointStruct(
                id=point_id,
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

    client.upsert(collection_name=collection_name, points=points)

    existing_ids = _existing_point_ids(client, collection_name)
    orphan_ids = existing_ids - current_ids
    if orphan_ids:
        client.delete(collection_name=collection_name, points_selector=list(orphan_ids))

    return len(points), len(orphan_ids)


def search_dense(query: str, collection_name: str = QDRANT_COLLECTION, top_k: int = 5) -> list[dict]:
    model = get_model()
    client = get_client()
    query_vector = model.encode(query).tolist()

    try:
        response = client.query_points(
            collection_name=collection_name,
            query=query_vector,
            limit=top_k,
        )
        results = response.points
    except Exception as e:
        logger.error(f"Recherche dense indisponible sur '{collection_name}' (Qdrant injoignable ?): {e}")
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