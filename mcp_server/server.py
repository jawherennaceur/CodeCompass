"""
Serveur MCP : expose le tool `search_code` pour que Claude Desktop (ou
tout client MCP) puisse interroger le pipeline de recherche hybride.

Lancement : python -m mcp_server.server
Puis configurer Claude Desktop pour s'y connecter (voir README).
"""
import asyncio

from mcp.server import Server
from mcp.server.stdio import stdio_server
from mcp.types import Tool, TextContent

from search.search_engine import search_code as run_search_code

app = Server("search-code-mcp")

# CORRECTIF (revue critique, bug high #7) : sans ces bornes, une requête
# malformée ou un top_k excessif (ex: 50000) pourrait faire remonter des
# quantités de code déraisonnables dans le contexte de Claude.
MAX_TOP_K = 20
MAX_QUERY_LENGTH = 500


@app.list_tools()
async def list_tools() -> list[Tool]:
    return [
        Tool(
            name="search_code",
            description=(
                "Recherche du code pertinent dans le repo indexé, en combinant "
                "recherche sémantique (dense) et par mots-clés (sparse). "
                "Retourne les chunks (fonctions/classes) les plus pertinents "
                "avec leur fichier, leurs numéros de ligne et leur code."
            ),
            inputSchema={
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "Question ou terme à rechercher dans le code.",
                    },
                    "top_k": {
                        "type": "integer",
                        "description": f"Nombre de résultats à retourner (défaut: 5, max: {MAX_TOP_K}).",
                        "default": 5,
                    },
                },
                "required": ["query"],
            },
        )
    ]


@app.call_tool()
async def call_tool(name: str, arguments: dict) -> list[TextContent]:
    if name != "search_code":
        raise ValueError(f"Tool inconnu: {name}")

    # CORRECTIF (revue critique, bug high #7) : validation des entrées,
    # absente jusqu'ici — un "query" manquant levait un KeyError brut,
    # et top_k n'avait aucune limite.
    query = arguments.get("query", "").strip()
    if not query:
        return [TextContent(type="text", text="Erreur : le paramètre 'query' est requis et ne peut pas être vide.")]
    if len(query) > MAX_QUERY_LENGTH:
        query = query[:MAX_QUERY_LENGTH]

    top_k = arguments.get("top_k", 5)
    if not isinstance(top_k, int) or top_k < 1:
        top_k = 5
    top_k = min(top_k, MAX_TOP_K)

    # CORRECTIF (revue critique, bug high #8) : si l'index sparse n'a
    # jamais été construit (premier lancement sans indexation préalable),
    # on donne un message clair plutôt qu'un crash avec traceback brut.
    try:
        output = run_search_code(query, top_k=top_k)
    except FileNotFoundError:
        return [TextContent(
            type="text",
            text=(
                "Erreur : aucun index trouvé. Lance d'abord "
                "`python scripts/index_repo.py` pour indexer ton repo "
                "avant de pouvoir le rechercher."
            ),
        )]
    except Exception as e:
        return [TextContent(type="text", text=f"Erreur inattendue pendant la recherche : {e}")]

    lines = [f"Route: {output['route']} ({output['router_latency_ms']}ms)\n"]

    if output.get("low_confidence"):
        lines.append(
            "⚠️ Aucun résultat vraiment pertinent trouvé dans le code indexé "
            "pour cette requête. Les résultats ci-dessous sont les plus "
            "proches disponibles, mais aucun n'est un bon match confirmé — "
            "à mentionner avec cette réserve plutôt que comme une réponse "
            "fiable.\n"
        )

    if not output["results"]:
        lines.append("Aucun résultat trouvé.")

    for r in output["results"]:
        lines.append(
            f"--- {r['name']} ({r['file_path']}:{r['start_line']}-{r['end_line']}) ---\n"
            f"{r['code']}\n"
        )
    return [TextContent(type="text", text="\n".join(lines))]


async def main():
    async with stdio_server() as (read_stream, write_stream):
        await app.run(read_stream, write_stream, app.create_initialization_options())


if __name__ == "__main__":
    asyncio.run(main())