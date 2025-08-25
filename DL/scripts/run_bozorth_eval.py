# scripts/run_bozorth_eval.py
from __future__ import annotations
from pathlib import Path
from fp.bozorth_eval import evaluate_soco

# Chemins (adapte-les si besoin)
OUT_DIR = Path("out")              # là où tes minuties sont déjà écrites: out/<stem>/<stem>.json
SAVE_DIR = OUT_DIR / "_eval_bozorth"

# Hyperparamètres Bozorth-like (les mêmes que ton moteur)
TD = 12.0          # tolérance distance (pixels)
TA_DEG = 10.0      # tolérance angle (degrés)
MIN_CLUSTER = 5    # taille minimale du cluster cohérent
MAX_PAIRS = None   # ou un entier si tu veux échantillonner les paires intra-table

if __name__ == "__main__":
    SAVE_DIR.mkdir(parents=True, exist_ok=True)
    res = evaluate_soco(
        out_dir=OUT_DIR,
        td=TD,
        ta_deg=TA_DEG,
        min_cluster=MIN_CLUSTER,
        max_pairs=MAX_PAIRS,
        save_dir=SAVE_DIR
    )
    print("=== Bozorth-like evaluation on SOCOFing ===")
    print(f"items: {res['n_items']}, genuine pairs: {res['n_genuine_pairs']}, impostor pairs: {res['n_impostor_pairs']}")
    print("\nVerification metrics:")
    for k, v in res["verification"].items():
        print(f"  {k}: {v:.4f}")
    print("\nIdentification metrics:")
    for k, v in res["identification"].items():
        print(f"  {k}: {v:.4f}")
    print(f"\nRaw scores and metrics saved to: {SAVE_DIR}")
