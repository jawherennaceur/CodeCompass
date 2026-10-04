"""
Vérification de fraîcheur d'un projet : détecte si son code source a
été modifié depuis la dernière indexation, pour déclencher une
ré-indexation automatique avant la recherche (projets locaux
uniquement — les projets GitHub se rafraîchissent via un outil dédié).

LIMITE CONNUE, NON CORRIGÉE (revue) : _latest_mtime() parcourt TOUS les
fichiers du repo à chaque recherche. Sur un petit repo, c'est instantané ;
sur un très gros repo (milliers de fichiers), ça peut ajouter une
latence perceptible à chaque recherche. Pas de correctif simple et
correct disponible : utiliser la date de modification du DOSSIER parent
au lieu de chaque fichier semble tentant, mais c'est FAUX sous Windows/
NTFS — modifier le contenu d'un fichier ne met pas à jour la date de
modification de son dossier parent (celle-ci ne change qu'à l'ajout/
suppression/renommage d'une entrée). Une vraie solution demanderait soit
un file watcher permanent (complexité qu'on a délibérément écartée),
soit un cache de hashs de fichiers (complexité supplémentaire, non
implémentée pour l'instant). À surveiller si l'usage sur un gros repo
réel montre une latence gênante.
"""
from datetime import datetime, timezone
from pathlib import Path

from config import SUPPORTED_EXTENSIONS
from projects import Project
from logging_config import get_logger

logger = get_logger(__name__)


def _latest_mtime(repo_path: str) -> datetime | None:
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
    if project.source_type != "local":
        return False

    if project.last_indexed_at is None:
        return True

    latest_change = _latest_mtime(project.repo_path)
    if latest_change is None:
        return False

    last_indexed = datetime.fromisoformat(project.last_indexed_at)
    return latest_change > last_indexed