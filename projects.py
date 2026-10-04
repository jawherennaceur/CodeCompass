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
    name: str, source_type: str, source: str, registry_path: Path = REGISTRY_PATH
) -> Project:
    if source_type not in {"local", "github"}:
        raise ValueError(f"source_type invalide: {source_type} (attendu: 'local' ou 'github')")

    safe_name = _sanitize(name)

    if source_type == "local":
        repo_path = source
    else:
        repo_path = str(REPOS_DIR / safe_name)

    project = Project(
        name=name,
        source_type=source_type,
        source=source,
        repo_path=repo_path,
        qdrant_collection=f"project_{safe_name}",
        bm25_index_path=str(DATA_DIR / f"bm25_{safe_name}.json"),
    )

    registry = _load_registry(registry_path)
    registry[name] = asdict(project)
    _save_registry(registry, registry_path)
    return project


def update_last_indexed(name: str, timestamp: str | None = None, registry_path: Path = REGISTRY_PATH):
    registry = _load_registry(registry_path)
    if name not in registry:
        raise KeyError(f"Projet inconnu: {name}")
    registry[name]["last_indexed_at"] = timestamp or datetime.now(timezone.utc).isoformat()
    _save_registry(registry, registry_path)