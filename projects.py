"""
Registre des projets indexés — chaque projet a sa propre collection
Qdrant et son propre fichier BM25, complètement isolés les uns des
autres (corrige le bug découvert : le nettoyage des orphelins d'un
projet ne doit jamais affecter un autre projet).
"""
import json
import re
from dataclasses import dataclass, asdict, field
from datetime import datetime, timezone
from pathlib import Path

from config import DATA_DIR

REGISTRY_PATH = DATA_DIR / "projects.json"
REPOS_DIR = DATA_DIR / "repos"  # clones Git locaux


class ProjectError(Exception):
    """Erreur liée au registre de projets — message pensé pour être
    lisible tel quel par Claude et transmis à l'utilisateur."""
    pass


def _sanitize(name: str) -> str:
    safe = re.sub(r"[^a-zA-Z0-9_]", "_", name.strip().lower())
    return safe or "project"


@dataclass
class Project:
    name: str
    source_type: str
    source: str
    repo_path: str
    qdrant_collection: str
    bm25_index_path: str
    last_indexed_at: str | None = None
    created_at: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())


def _load_registry(registry_path: Path = REGISTRY_PATH) -> dict:
    if not registry_path.exists():
        return {}
    with open(registry_path, "r", encoding="utf-8") as f:
        return json.load(f)


def _save_registry(registry: dict, registry_path: Path = REGISTRY_PATH):
    registry_path.parent.mkdir(parents=True, exist_ok=True)
    with open(registry_path, "w", encoding="utf-8") as f:
        json.dump(registry, f, ensure_ascii=False, indent=2)


def list_projects(registry_path: Path = REGISTRY_PATH) -> list[Project]:
    registry = _load_registry(registry_path)
    return [Project(**data) for data in registry.values()]


def get_project(name: str, registry_path: Path = REGISTRY_PATH) -> Project | None:
    registry = _load_registry(registry_path)
    data = registry.get(name)
    return Project(**data) if data else None


def project_exists(name: str, registry_path: Path = REGISTRY_PATH) -> bool:
    return name in _load_registry(registry_path)


def register_project(
    name: str,
    source_type: str,
    source: str,
    registry_path: Path = REGISTRY_PATH,
    overwrite: bool = False,
) -> Project:
    """Enregistre un nouveau projet.

    CORRECTIF (revue) : lève désormais ProjectError si 'name' existe déjà
    (sauf overwrite=True explicite) — avant, un ré-enregistrement
    écrasait silencieusement l'entrée existante, perdant son historique
    d'indexation sans aucun avertissement.
    """
    registry = _load_registry(registry_path)

    if name in registry and not overwrite:
        raise ProjectError(
            f"Un projet nommé '{name}' existe déjà. Utilise un autre nom, "
            f"ou overwrite=True pour le remplacer explicitement."
        )

    safe_name = _sanitize(name)

    # CORRECTIF (revue, bug 🟠) : deux noms différents peuvent produire
    # le même safe_name après nettoyage (ex: "Mon-Projet" et "mon projet"
    # donnent tous les deux "mon_projet") — sans ce contrôle, les deux
    # projets se retrouveraient avec la MÊME collection Qdrant et le
    # MÊME fichier BM25, réintroduisant par un autre chemin le bug de
    # pollution croisée qu'on vient de corriger.
    for other_name, other_data in registry.items():
        if other_name == name:
            continue  # c'est le projet qu'on met à jour (cas overwrite), pas une collision
        if _sanitize(other_name) == safe_name:
            raise ProjectError(
                f"Le nom '{name}' entre en collision avec le projet existant "
                f"'{other_name}' une fois normalisé ('{safe_name}'). "
                f"Choisis un nom plus distinct."
            )

    if source_type == "local":
        repo_path = source
    elif source_type == "github":
        repo_path = str(REPOS_DIR / safe_name)
    else:
        raise ValueError(f"source_type invalide: {source_type} (attendu: 'local' ou 'github')")

    project = Project(
        name=name,
        source_type=source_type,
        source=source,
        repo_path=repo_path,
        qdrant_collection=f"project_{safe_name}",
        bm25_index_path=str(DATA_DIR / f"bm25_{safe_name}.json"),
    )

    registry[name] = asdict(project)
    _save_registry(registry, registry_path)
    return project


def update_last_indexed(name: str, timestamp: str | None = None, registry_path: Path = REGISTRY_PATH):
    registry = _load_registry(registry_path)
    if name not in registry:
        raise ProjectError(f"Projet inconnu: '{name}'")
    registry[name]["last_indexed_at"] = timestamp or datetime.now(timezone.utc).isoformat()
    _save_registry(registry, registry_path)


def delete_project(name: str, registry_path: Path = REGISTRY_PATH, cleanup_data: bool = True) -> None:
    """CORRECTIF (revue) : nouvelle fonction — il était jusqu'ici
    impossible de retirer un projet du registre. Par défaut, nettoie
    aussi sa collection Qdrant et son fichier BM25 (cleanup_data=True) ;
    les échecs de nettoyage (ex: Qdrant injoignable) sont loggés mais
    n'empêchent pas la suppression de l'entrée du registre."""
    registry = _load_registry(registry_path)
    if name not in registry:
        raise ProjectError(f"Projet inconnu: '{name}'")

    project = Project(**registry[name])

    if cleanup_data:
        from logging_config import get_logger
        logger = get_logger(__name__)
        try:
            from indexing.dense_index import get_client
            get_client().delete_collection(project.qdrant_collection)
        except Exception as e:
            logger.warning(f"Échec suppression collection Qdrant '{project.qdrant_collection}': {e}")
        try:
            Path(project.bm25_index_path).unlink(missing_ok=True)
        except Exception as e:
            logger.warning(f"Échec suppression fichier BM25 '{project.bm25_index_path}': {e}")

    del registry[name]
    _save_registry(registry, registry_path)