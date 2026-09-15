"""
Configuration centrale du projet "Fast & Precise" — Scénario A.

Toutes les valeurs peuvent être surchargées via variables d'environnement
(voir .env.example). Ce fichier ne contient AUCUN secret.
"""
import os
import sys
from pathlib import Path

from dotenv import load_dotenv

# CRITIQUE (revue): .env n'était jamais chargé — os.getenv() lisait
# uniquement les variables déjà exportées manuellement dans le shell.
# Sans cette ligne, ANTHROPIC_API_KEY était systématiquement vide en
# usage normal, et le routeur tombait en fallback silencieux à chaque
# requête sans que rien ne le signale.
load_dotenv()

# --- Repo cible ---------------------------------------------------------
# Chemin absolu vers le repo de code à indexer (le tien, en usage réel).
REPO_PATH = os.getenv("REPO_PATH", "./sample_repo")

# Extensions de fichiers à indexer (adapter selon ton langage principal)
SUPPORTED_EXTENSIONS = {".py", ".ts", ".tsx", ".js", ".jsx"}

# --- Chunking / parsing --------------------------------------------------
# Nombre max de lignes tolérées pour un chunk avant découpage forcé
# (garde-fou pour les très grosses fonctions/classes)
MAX_CHUNK_LINES = 200

# --- Embeddings (modèle LOCAL retenu pour le Scénario A) -----------------
# Justification : latence quasi nulle + confidentialité du code (rien ne
# quitte la machine). Voir discussion stack pour le détail des critères.
EMBEDDING_MODEL_NAME = os.getenv("EMBEDDING_MODEL_NAME", "BAAI/bge-small-en-v1.5")
EMBEDDING_DIM = 384  # dimension native de bge-small-en-v1.5

# --- Base vectorielle : Qdrant (Docker) -----------------------------------
QDRANT_HOST = os.getenv("QDRANT_HOST", "localhost")
QDRANT_PORT = int(os.getenv("QDRANT_PORT", "6333"))
QDRANT_COLLECTION = os.getenv("QDRANT_COLLECTION", "search_code_chunks")

# --- Index sparse (BM25) --------------------------------------------------
BM25_INDEX_PATH = os.getenv("BM25_INDEX_PATH", "./data/bm25_index.json")

# --- Routeur (Haiku) -------------------------------------------------------
# Nécessite ANTHROPIC_API_KEY dans l'environnement (.env)
ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY", "")
if not ANTHROPIC_API_KEY:
    print(
        "[CONFIG WARNING] ANTHROPIC_API_KEY est vide — le routeur tombera "
        "systématiquement en fallback 'hybrid' sur chaque requête, sans "
        "appeler Haiku. Vérifie ton fichier .env.",
        file=sys.stderr,
    )
ROUTER_MODEL = os.getenv("ROUTER_MODEL", "claude-haiku-4-5-20251001")
ROUTER_MAX_TOKENS = 50  # réponse courte attendue: "dense" | "sparse" | "hybrid"
ROUTER_LATENCY_BUDGET_MS = 200  # critère de validation du projet

# --- Recherche / fusion -----------------------------------------------------
TOP_K_DEFAULT = 5
TOP_K_CANDIDATES = 20  # nombre de candidats avant reranking éventuel
RRF_K = 60  # constante standard pour Reciprocal Rank Fusion

# --- Chemins internes --------------------------------------------------------
DATA_DIR = Path(os.getenv("DATA_DIR", "./data"))
DATA_DIR.mkdir(parents=True, exist_ok=True)