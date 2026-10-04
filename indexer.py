"""
Point d'entrée unique pour (ré-)indexer un projet du registre — relie
projects.py (config isolée par projet), ingestion/parser.py (chunking),
indexing/sparse_index.py et indexing/dense_index.py (construction des
deux index, chacun dans son espace isolé).
"""
from ingestion.parser import parse_repo
from indexing.sparse_index import build_sparse_index
from indexing.dense_index import build_dense_index
from projects import get_project, update_last_indexed
from logging_config import get_logger
from freshness import needs_reindex

logger = get_logger(__name__)


def ensure_fresh(name: str) -> bool:
    """Ré-indexe le projet SEULEMENT si nécessaire. Retourne True si une
    ré-indexation a eu lieu, False si l'index était déjà à jour."""
    project = get_project(name)
    if project is None:
        raise ValueError(f"Projet inconnu: '{name}'")

    if needs_reindex(project):
        logger.info(f"Projet '{name}' modifié depuis la dernière indexation — ré-indexation automatique.")
        index_project(name)
        return True

    return False
def index_project(name: str) -> dict:
    """(Ré-)indexe un projet déjà enregistré dans le registre. Retourne
    un résumé (nombre de chunks, orphelins supprimés)."""
    project = get_project(name)
    if project is None:
        raise ValueError(f"Projet inconnu: '{name}'. Utilise register_project() d'abord.")

    logger.info(f"Indexation du projet '{name}' ({project.repo_path})...")
    chunks = parse_repo(project.repo_path)

    if not chunks:
        logger.warning(f"Aucun chunk trouvé pour '{name}' — vérifie le chemin: {project.repo_path}")
        return {"project": name, "chunks": 0, "orphans_removed": 0}

    sparse_index = build_sparse_index(chunks)
    sparse_index.save(path=project.bm25_index_path)

    n_dense, n_orphans = build_dense_index(chunks, collection_name=project.qdrant_collection)

    update_last_indexed(name)

    logger.info(f"Projet '{name}' indexé: {len(chunks)} chunks, {n_orphans} orphelins supprimés.")
    return {"project": name, "chunks": len(chunks), "orphans_removed": n_orphans}


if __name__ == "__main__":
    import sys
    if len(sys.argv) < 2:
        print("Usage: python indexer.py <nom_du_projet>")
        sys.exit(1)

    result = index_project(sys.argv[1])
    print(f"Projet '{result['project']}' indexé : {result['chunks']} chunks, {result['orphans_removed']} orphelins supprimés.")