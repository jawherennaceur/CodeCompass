"""
Script de test manuel : permet de tester la recherche en ligne de
commande, sans passer par Claude Desktop / MCP.

Usage : python scripts/test_search.py "où est gérée l'authentification"
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from search.search_engine import search_code


def main():
    if len(sys.argv) < 2:
        print('Usage: python scripts/test_search.py "ta requête"')
        return

    query = " ".join(sys.argv[1:])
    output = search_code(query)

    print(f"\nRequête : {query}")
    print(f"Route choisie : {output['route']} (routeur: {output['router_latency_ms']}ms)\n")

    for i, r in enumerate(output["results"], 1):
        print(f"{i}. {r['name']}  ({r['file_path']}:{r['start_line']}-{r['end_line']})  score={r.get('score', r.get('fused_score'))}")
        print("   " + r["code"].strip().splitlines()[0][:100])
        print()


if __name__ == "__main__":
    main()
