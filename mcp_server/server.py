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
                        "description": "Nombre de résultats à retourner (défaut: 5).",
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

    query = arguments["query"]
    top_k = arguments.get("top_k", 5)

    output = run_search_code(query, top_k=top_k)

    lines = [f"Route: {output['route']} ({output['router_latency_ms']}ms)\n"]
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
