# search-code-mcp — Projet 1 "Fast & Precise"

Moteur de recherche de code hybride (dense + sparse), exposé via MCP,
pour être utilisé directement par Claude Desktop pendant que tu codes.

**Scénario retenu :** usage personnel sur ton propre repo (Scénario A).

## Architecture

```
Requête
   │
   ▼
[Routeur Haiku] ──► dense | sparse | hybrid
   │
   ├──► Dense (embeddings locaux + Qdrant)
   └──► Sparse (BM25)
   │
   ▼
[Merger RRF] ──► fusionne si hybrid
   │
   ▼
[Serveur MCP] ──► tool "search_code"
   │
   ▼
Claude Desktop
```

## Stack retenue (et pourquoi)

| Composant | Choix | Raison |
|---|---|---|
| Parsing AST | tree-sitter | Standard multi-langage |
| Embeddings | **Local** (`bge-small-en`) | Latence quasi nulle + confidentialité du code (rien n'est envoyé à un tiers) |
| Base vectorielle | Qdrant (Docker) | Disponible et plus robuste que Chroma pour un usage continu |
| Sparse | rank_bm25 | Léger, suffisant pour un repo perso |
| Routeur | Claude Haiku | Classification rapide et fiable dense/sparse/hybrid |
| Merger | Reciprocal Rank Fusion (RRF) | Ne nécessite pas de comparer des scores sur des échelles différentes |

## Setup

### 1. Dépendances

```bash
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

### 2. Qdrant (Docker)

```bash
docker compose up -d
```

### 3. Configuration

```bash
cp .env.example .env
# Éditer .env : REPO_PATH, ANTHROPIC_API_KEY
```

### 4. Indexer ton repo

```bash
python scripts/index_repo.py
```

### 5. Tester en ligne de commande (avant de connecter Claude)

```bash
python scripts/test_search.py "où est gérée l'authentification"
```

### 6. Connecter à Claude Desktop

Ajouter dans la config MCP de Claude Desktop (`claude_desktop_config.json`) :

```json
{
  "mcpServers": {
    "search-code": {
      "command": "python",
      "args": ["-m", "mcp_server.server"],
      "cwd": "/chemin/absolu/vers/search-code-mcp"
    }
  }
}
```

Redémarrer Claude Desktop. Le tool `search_code` doit apparaître disponible.

## Évaluation

Compléter `tests/eval_queries.json` avec des requêtes réelles sur ton
repo (une fois indexé), puis :

```bash
python tests/evaluate.py
```

Affiche precision@5 pour dense seul / sparse seul / hybride.

## Ré-indexer après modification du repo

Relancer simplement `python scripts/index_repo.py` — l'index sparse est
régénéré entièrement, l'index dense est mis à jour (upsert) dans Qdrant.

## Prochaines étapes (hors squelette)

- [ ] Reranker cross-encoder optionnel sur le top-20 avant top-5
- [ ] UI web de démo (bonus, pour présentation/LinkedIn)
- [ ] Ré-indexation incrémentale (ne re-parser que les fichiers modifiés)
