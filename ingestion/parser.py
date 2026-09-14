"""
Ingestion & chunking — parse le code source avec tree-sitter et découpe
en chunks au niveau fonction/classe (pas par nombre de lignes fixe).

Chaque chunk retourné contient : le code, le nom, le type (function/class),
le chemin du fichier, et les lignes de départ/fin.
"""
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
    chunks: list[CodeChunk] = []

    def walk(node):
        if node.type in chunk_types:
            code = source_bytes[node.start_byte:node.end_byte].decode("utf-8", errors="replace")
            start_line = node.start_point[0] + 1
            end_line = node.end_point[0] + 1

            # Garde-fou: si un chunk est énorme, on le garde quand même
            # mais on pourrait choisir de le sous-découper ici plus tard.
            name = _extract_name(node, source_bytes)
            chunk_id = f"{file_path}:{name}:{start_line}"
            chunks.append(CodeChunk(
                chunk_id=chunk_id,
                file_path=str(file_path),
                name=name,
                node_type=node.type,
                code=code,
                start_line=start_line,
                end_line=end_line,
            ))
            # Ne pas descendre dans les enfants d'un chunk déjà capturé
            # évite les doublons méthode-dans-classe si non désiré.
            # (Ici on choisit de continuer à descendre pour capturer aussi
            # les méthodes internes comme chunks séparés.)
        for child in node.children:
            walk(child)

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
