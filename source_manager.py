"""
Gestion de l'origine du code source d'un projet : validation d'un
dossier local, ou clone/pull d'un repo GitHub public.
"""
import subprocess
from pathlib import Path

from logging_config import get_logger

logger = get_logger(__name__)

# CORRECTIF (revue, bug 🟠) : sans timeout, un réseau lent/bloqué pouvait
# figer indéfiniment le processus serveur MCP entier sur cette requête.
GIT_TIMEOUT_SECONDS = 120


class SourceError(Exception):
    """Levée quand la source (dossier local ou repo GitHub) est invalide
    ou inaccessible — message pensé pour être lisible tel quel par Claude
    et transmis à l'utilisateur."""
    pass


def resolve_local(path: str) -> str:
    p = Path(path)
    if not p.exists():
        raise SourceError(f"Le dossier '{path}' n'existe pas.")
    if not p.is_dir():
        raise SourceError(f"'{path}' n'est pas un dossier.")
    return str(p.resolve())


def _validate_github_url(url: str):
    """CORRECTIF (revue, bug 🟢) : garde-fou minimal — pas une validation
    complète, juste un rejet rapide des entrées manifestement invalides
    avant de lancer un subprocess Git."""
    if not url.startswith(("http://", "https://")):
        raise SourceError(f"URL invalide (doit commencer par http:// ou https://): '{url}'")


def clone_or_pull_github(url: str, target_dir: str) -> str:
    """Clone un repo GitHub public dans target_dir s'il n'existe pas
    encore, ou le met à jour (git pull) s'il existe déjà."""
    _validate_github_url(url)
    target = Path(target_dir)

    if target.exists() and (target / ".git").exists():
        cmd = ["git", "-C", str(target), "pull"]
        action = "Mise à jour"
    else:
        target.parent.mkdir(parents=True, exist_ok=True)
        cmd = ["git", "clone", url, str(target)]
        action = "Clonage"

    logger.info(f"{action} du repo: {url}")

    try:
        result = subprocess.run(
            cmd, capture_output=True, text=True, timeout=GIT_TIMEOUT_SECONDS
        )
    except subprocess.TimeoutExpired:
        raise SourceError(
            f"L'opération Git sur '{url}' a dépassé {GIT_TIMEOUT_SECONDS}s "
            f"(réseau lent ou bloqué ?)."
        )
    except FileNotFoundError:
        # CORRECTIF (revue, bug 🟡) : git absent du PATH levait une
        # FileNotFoundError brute et peu claire ; message explicite désormais.
        raise SourceError(
            "La commande 'git' est introuvable. Vérifie que Git est "
            "installé et accessible depuis le PATH."
        )

    if result.returncode != 0:
        raise SourceError(f"Échec de l'opération Git sur '{url}': {result.stderr.strip()}")

    return str(target.resolve())