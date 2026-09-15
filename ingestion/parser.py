"""
Ingestion & chunking — parse le code source avec tree-sitter et découpe
en chunks au niveau fonction/classe (pas par nombre de lignes fixe).

Chaque chunk retourné contient : le code, le nom, le type (function/class),
le chemin du fichier, et les lignes de départ/fin.
"""
import hashlib
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator

import tree_sitter_languages  # wrapper simple pour charger les grammaires

from config import REPO_PATH, SUPPORTED_EXTENSIONS, MAX_CHUNK_LINES

# Mapping extension -> nom de grammaire tree-sitter
_LANGUAGE_BY_EXT = {
    ".py": "python",
    ".ts": "typescript",
    ".tsx": "tsx",
    ".js": "javascript",
    ".jsx": "javascript",
}

# Types de nœuds AST considérés comme des "unités" de chunk par langage.
# À étendre si tu ajoutes d'autres langages.
_CHUNK_NODE_TYPES = {
    "python": {"function_definition", "class_definition"},
    "typescript": {"function_declaration", "class_declaration", "method_definition"},
    "tsx": {"function_declaration", "class_declaration", "method_definition"},
    "javascript": {"function_declaration", "class_declaration", "method_definition"},
}

# CORRECTIF (revue, bug critique #4) : en Python, une fonction/classe
# décorée (@app.route, @staticmethod, @property, @pytest.fixture...) a
# pour nœud racine "decorated_definition", PAS "function_definition"
# directement — le décorateur est un sibling en dehors du nœud capturé
# jusqu'ici. Sans ça, le décorateur (souvent l'info la plus importante,
# ex: la route d'un endpoint) était silencieusement absent du chunk.
# NOTE: TS/JS gèrent les décorateurs différemment dans leur grammaire —
# non couvert ici, limitation connue et documentée (pas une correction
# silencieuse).
_DECORATOR_WRAPPER_TYPES = {
    "python": {"decorated_definition"},
}


@dataclass
class CodeChunk:
    chunk_id: str
    file_path: str
    name: str
    node_type: str
    code: str
    start_line: int
    end_line: int


def _iter_source_files(repo_path: str) -> Iterator[Path]:
    root = Path(repo_path)
    for path in root.rglob("*"):
        if path.is_file() and path.suffix in SUPPORTED_EXTENSIONS:
            # Ignore les dossiers usuels de dépendances/build
            if any(part in {"node_modules", ".git", "venv", "__pycache__", "dist", "build"}
                   for part in path.parts):
                continue
            yield path


def _extract_name(node, source_bytes: bytes) -> str:
    """Essaie de récupérer le nom (identifier) d'un nœud fonction/classe."""
    for child in node.children:
        if child.type == "identifier":
            return source_bytes[child.start_byte:child.end_byte].decode("utf-8")
    return "<anonymous>"


def _make_chunk_id(file_path: str, name: str, start_line: int) -> str:
    """
    CORRECTIF (revue, bug high #5) : l'ancien format f"{file_path}:{name}:{start_line}"
    cassait sur Windows, où file_path contient déjà un ':' (ex: "C:\\Users\\...").
    On utilise un hash déterministe à la place — stable, sans ambiguïté de
    séparateur, et réutilisable tel quel comme ID de point Qdrant (voir
    indexing/dense_index.py).
    """
    raw = f"{file_path}|{name}|{start_line}"
    return hashlib.sha1(raw.encode("utf-8")).hexdigest()[:16]


def _make_chunk(node, node_type: str, file_path: Path, source_bytes: bytes) -> CodeChunk:
    code = source_bytes[node.start_byte:node.end_byte].decode("utf-8", errors="replace")
    start_line = node.start_point[0] + 1
    end_line = node.end_point[0] + 1
    name = _extract_name(node, source_bytes)
    chunk_id = _make_chunk_id(str(file_path), name, start_line)
    return CodeChunk(
        chunk_id=chunk_id,
        file_path=str(file_path),
        name=name,
        node_type=node_type,
        code=code,
        start_line=start_line,
        end_line=end_line,
    )


def parse_file(file_path: Path) -> list[CodeChunk]:
    """Parse un seul fichier et retourne la liste de ses chunks."""
    ext = file_path.suffix
    lang_name = _LANGUAGE_BY_EXT.get(ext)
    if lang_name is None:
        return []

    parser = tree_sitter_languages.get_parser(lang_name)
    source_bytes = file_path.read_bytes()
    tree = parser.parse(source_bytes)

    chunk_types = _CHUNK_NODE_TYPES.get(lang_name, set())
    wrapper_types = _DECORATOR_WRAPPER_TYPES.get(lang_name, set())
    chunks: list[CodeChunk] = []

    def walk(node, skip_ids: frozenset = frozenset()):
        if node.type in wrapper_types:
            # Le wrapper (ex: decorated_definition) capture décorateur(s)
            # + définition en une seule fois, sur toute la plage du nœud.
            inner = next(
                (c for c in node.children if c.type in chunk_types), None
            )
            inner_type = inner.type if inner is not None else node.type
            chunks.append(_make_chunk(node, inner_type, file_path, source_bytes))
            # On empêche le nœud interne (function_definition brut) d'être
            # capturé une seconde fois — sinon on aurait deux chunks quasi
            # identiques : un avec décorateur, un sans.
            if inner is not None:
                skip_ids = skip_ids | {id(inner)}
        elif node.type in chunk_types and id(node) not in skip_ids:
            chunks.append(_make_chunk(node, node.type, file_path, source_bytes))

        for child in node.children:
            walk(child, skip_ids)

    walk(tree.root_node)
    return chunks


def parse_repo(repo_path: str = REPO_PATH) -> list[CodeChunk]:
    """Parse tout le repo et retourne la liste complète des chunks."""
    all_chunks: list[CodeChunk] = []
    for file_path in _iter_source_files(repo_path):
        try:
            all_chunks.extend(parse_file(file_path))
        except Exception as e:
            print(f"[WARN] Échec parsing {file_path}: {e}")
    return all_chunks


if __name__ == "__main__":
    chunks = parse_repo()
    print(f"{len(chunks)} chunks extraits depuis {REPO_PATH}")
    for c in chunks[:5]:
        print(f"- {c.node_type} '{c.name}' ({c.file_path}:{c.start_line}-{c.end_line})")