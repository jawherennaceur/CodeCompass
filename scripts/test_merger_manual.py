"""
Test manuel du merger RRF avec des listes construites à la main — Phase 5,
étape 2. Permet de vérifier la logique de fusion sur un cas dont on peut
calculer le résultat attendu nous-mêmes, avant de dépendre de vraies
recherches sparse/dense.

Usage : python scripts/test_merger_manual.py
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from search.merger import merge_results

# Cas construit à la main :
# - "chunk_A" est 1er en dense ET 1er en sparse -> doit gagner (trouvé 2x)
# - "chunk_B" est 2e en dense seulement
# - "chunk_C" est 2e en sparse seulement
# Calcul attendu (k=60, valeur par défaut) :
#   chunk_A = 1/(60+0) + 1/(60+0) = 0.01667 + 0.01667 = 0.03333
#   chunk_B = 1/(60+1)                                = 0.01639
#   chunk_C =                        1/(60+1)         = 0.01639
# -> Attendu : chunk_A en premier, chunk_B et chunk_C ex-aequo ensuite

dense_results = [
    {"chunk_id": "chunk_A", "name": "func_a", "code": "..."},
    {"chunk_id": "chunk_B", "name": "func_b", "code": "..."},
]

sparse_results = [
    {"chunk_id": "chunk_A", "name": "func_a", "code": "..."},
    {"chunk_id": "chunk_C", "name": "func_c", "code": "..."},
]

merged = merge_results(dense_results, sparse_results, top_k=5)

print("Résultat de la fusion :\n")
for i, r in enumerate(merged, 1):
    print(f"{i}. {r['chunk_id']}  (fused_score={r['fused_score']:.5f})")

print("\nAttendu : chunk_A en 1er (~0.03333), puis chunk_B et chunk_C ex-aequo (~0.01639)")