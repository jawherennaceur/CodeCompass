"""
Vérification de fraîcheur d'un projet : détecte si son code source a
été modifié depuis la dernière indexation, pour déclencher une
ré-indexation automatique avant la recherche (projets locaux
uniquement — les projets GitHub se rafraîchissent via un outil dédié,
pas automatiquement, pour éviter un appel réseau à chaque recherche).
"""
from datetime import datetime, timezone
from pathlib import Path

from config import SUPPORTED_EXTENSIONS
from projects import Project
from logging_config import get_logger

logger = get_logger(__name__)


def _latest_mtime(repo_path: str) -> datetime | None:
    """Date de modification la plus récente parmi les fichiers de code
    du projet. None si le dossier est vide/inexistant."""
    root = Path(repo_path)
    if not root.exists():
        return None

    latest: float | None = None
    for path in root.rglob("*"):
        if path.is_file() and path.suffix in SUPPORTED_EXTENSIONS:
            if any(part in {"node_modules", ".git", "venv", "__pycache__", "dist", "build"}
                   for part in path.parts):
                continue
            mtime = path.stat().st_mtime
            if latest is None or mtime > latest:
                latest = mtime

    return datetime.fromtimestamp(latest, tz=timezone.utc) if latest is not None else None


def needs_reindex(project: Project) -> bool:
    """True si le projet doit être ré-indexé avant une recherche :
    jamais indexé, ou code modifié depuis la dernière indexation.
    Ne s'applique qu'aux projets locaux — les projets GitHub ne sont
    jamais ré-indexés automatiquement (voir docstring du module)."""
    if project.source_type != "local":
        return False

    if project.last_indexed_at is None:
        return True

    latest_change = _latest_mtime(project.repo_path)
    if latest_change is None:
        return False  # dossier vide/inexistant, rien à ré-indexer

    last_indexed = datetime.fromisoformat(project.last_indexed_at)
    return latest_change > last_indexed