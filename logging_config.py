"""
Configuration centralisée du logging — écrit à la fois vers stderr
(visible dans "View logs" de Claude Desktop / terminal) et vers un
fichier persistant (survit aux redémarrages, consultable après coup).
"""
import logging
import sys
from logging.handlers import RotatingFileHandler

from config import PROJECT_ROOT

LOG_FILE = PROJECT_ROOT / "data" / "server.log"


def get_logger(name: str) -> logging.Logger:
    logger = logging.getLogger(name)
    if logger.handlers:  # déjà configuré (évite les doublons de handlers)
        return logger

    logger.setLevel(logging.INFO)

    # Vers stderr — jamais stdout (romprait le protocole MCP en stdio)
    stderr_handler = logging.StreamHandler(sys.stderr)
    stderr_handler.setFormatter(logging.Formatter("[%(levelname)s] %(name)s: %(message)s"))
    logger.addHandler(stderr_handler)

    # Vers fichier persistant, avec rotation (max 1 Mo, 3 fichiers d'historique)
    try:
        file_handler = RotatingFileHandler(LOG_FILE, maxBytes=1_000_000, backupCount=3)
        file_handler.setFormatter(
            logging.Formatter("%(asctime)s [%(levelname)s] %(name)s: %(message)s")
        )
        logger.addHandler(file_handler)
    except PermissionError:
        # Même logique que pour DATA_DIR.mkdir() — certains environnements
        # (Claude Desktop packagé MSIX) peuvent refuser l'écriture.
        pass

    return logger