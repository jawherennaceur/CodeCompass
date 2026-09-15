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

# CORRECTIF (revue, bug critique #1) : namespace fixe pour dériver un UUID
# stable à partir de chunk_id. Avant, l'ID de chaque point Qdrant était un
# uuid4() aléatoire généré à chaque indexation — donc chaque ré-indexation
# (même sans aucun changement de code) créait de NOUVEAUX points au lieu
# d'écraser les existants via upsert. Résultat : le nombre de vecteurs
# doublait à chaque ré-indexation, à l'infini, avec des doublons obsolètes
# jamais nettoyés.
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


def _existing_point_ids(client: QdrantClient) -> set[str]:
    """Récupère tous les IDs de points actuellement stockés dans la
    collection, en paginant (scroll) pour ne rien manquer même sur un
    gros repo."""
    ids: set[str] = set()
    offset = None
    while True:
        points, offset = client.scroll(
            collection_name=QDRANT_COLLECTION,
            with_payload=False,
            with_vectors=False,
            limit=256,
            offset=offset,
        )
        ids.update(str(p.id) for p in points)
        if offset is None:
            break
    return ids


def build_dense_index(chunks: list[CodeChunk], batch_size: int = 32):
    """Calcule les embeddings de chaque chunk et les insère dans Qdrant.

    AMÉLIORATION (Phase 3, étape 8) : supprime aussi les vecteurs
    "orphelins" — ceux qui correspondaient à du code qui a été supprimé
    ou renommé depuis la dernière indexation. Sans ça, une fonction
    supprimée du code source restait indéfiniment retrouvable dans les
    résultats de recherche dense, pointant vers du code qui n'existe
    plus.
    """
    model = get_model()
    client = get_client()
    ensure_collection(client)

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

    client.upsert(collection_name=QDRANT_COLLECTION, points=points)

    # Nettoyage des orphelins : tout point déjà présent AVANT cette
    # indexation mais absent du jeu de chunks actuel correspond à du
    # code qui n'existe plus — on le supprime.
    existing_ids = _existing_point_ids(client)
    orphan_ids = existing_ids - current_ids
    if orphan_ids:
        client.delete(
            collection_name=QDRANT_COLLECTION,
            points_selector=list(orphan_ids),
        )

    return len(points), len(orphan_ids)


def search_dense(query: str, top_k: int = 5) -> list[dict]:
    model = get_model()
    client = get_client()
    query_vector = model.encode(query).tolist()

    try:
        # CORRECTIF (revue, bug high #9) : client.search() est déprécié
        # dans les versions récentes de qdrant-client au profit de
        # query_points(). On a déjà été mordus une fois par une API
        # dépréciée qui change sous nos pieds (tree-sitter) — autant
        # corriger celle-ci maintenant qu'on le sait.
        response = client.query_points(
            collection_name=QDRANT_COLLECTION,
            query=query_vector,
            limit=top_k,
        )
        results = response.points
    except Exception as e:
        # Qdrant peut être down (pas démarré, Docker arrêté...) — on ne
        # veut pas planter tout le pipeline de recherche pour ça, mais on
        # ne veut pas non plus masquer silencieusement l'erreur.
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
    n_inserted, n_orphans_removed = build_dense_index(chunks)
    print(f"Index dense construit et inséré dans Qdrant ({n_inserted} chunks).")
    if n_orphans_removed:
        print(f"{n_orphans_removed} vecteur(s) orphelin(s) supprimé(s) (code obsolète).")