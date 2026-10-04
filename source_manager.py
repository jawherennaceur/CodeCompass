"""
Gestion de l'origine du code source d'un projet : validation d'un
dossier local, ou clone/pull d'un repo GitHub public.
"""
import subprocess
from pathlib import Path

from logging_config import get_logger

logger = get_logger(__name__)


class SourceError(Exception):
    """Levée quand la source (dossier local ou repo GitHub) est invalide
    ou inaccessible — message pensé pour être lisible tel quel par Claude
    et transmis à l'utilisateur."""
    pass


def resolve_local(path: str) -> str:
    """Valide qu'un chemin local existe et est bien un dossier."""
    p = Path(path)
    if not p.exists():
        raise SourceError(f"Le dossier '{path}' n'existe pas.")
    if not p.is_dir():
        raise SourceError(f"'{path}' n'est pas un dossier.")
    return str(p.resolve())


def clone_or_pull_github(url: str, target_dir: str) -> str:
    """Clone un repo GitHub public dans target_dir s'il n'existe pas
    encore, ou le met à jour (git pull) s'il existe déjà. Repos publics
    uniquement pour l'instant — pas de gestion d'authentification."""
    target = Path(target_dir)

    if target.exists() and (target / ".git").exists():
        logger.info(f"Mise à jour du repo existant: {url}")
        result = subprocess.run(
            ["git", "-C", str(target), "pull"],
            capture_output=True, text=True,
        )
    else:
        target.parent.mkdir(parents=True, exist_ok=True)
        logger.info(f"Clonage du repo: {url}")
        result = subprocess.run(
            ["git", "clone", url, str(target)],
            capture_output=True, text=True,
        )

    if result.returncode != 0:
        raise SourceError(
            f"Échec de l'opération Git sur '{url}': {result.stderr.strip()}"
        )

    return str(target.resolve())


