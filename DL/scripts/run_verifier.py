# verify_soco_minutiae.py
from __future__ import annotations
from pathlib import Path
import re, numpy as np, csv
from typing import Dict, List, Tuple
from fp.verifier import MinutiaeVerifier

# --------------------------- CONFIG ---------------------------
OUT_DIR = Path("out")        # where out/<stem>/<stem>.json live
WRITE_PAIR_SCORES_CSV = True
PAIR_SCORES_CSV = Path("pair_scores.csv")

# Caps (set to None to use all)
MAX_GENUINE  = 3000
MAX_IMPOSTOR = 3000

# --------------------------- SOCOFing parsing ---------------------------
_ID_RE = re.compile(r"(?P<id>\d+)")

def parse_id_finger(stem: str) -> Tuple[str, str]:
    """
    SOCOFing-like stems: 101__M_Left_index_finger[...]
    Returns (subject_id, finger_key) e.g. ("101", "Left_index_finger")
    """
    m = _ID_RE.match(stem)
    sid = m.group("id") if m else stem.split("__")[0]
    parts = stem.split("__")
    if len(parts) >= 3:
        finger = parts[2]
    else:
        m2 = re.search(r"(Left|Right)_[A-Za-z]+_finger", stem)
        finger = m2.group(0) if m2 else "unknown"
    return sid, finger

def build_index(out_dir: Path) -> Dict[str, Dict[str, List[str]]]:
    """
    idx[subject_id][finger] = [stems...]
    from subfolders out/<stem> containing <stem>.json or <stem>.csv
    """
    idx: Dict[str, Dict[str, List[str]]] = {}
    for sub in out_dir.iterdir():
        if not sub.is_dir():
            continue
        stem = sub.name
        if not (sub / f"{stem}.json").exists() and not (sub / f"{stem}.csv").exists():
            continue
        sid, finger = parse_id_finger(stem)
        idx.setdefault(sid, {}).setdefault(finger, []).append(stem)
    return idx

# --------------------------- metrics ---------------------------
def sweep_thresholds(genuine_scores: List[float], impostor_scores: List[float]):
    """
    Compute EER (min |FAR-FRR|), ROC AUC, and precision/recall/F1 at best-F1 threshold.
    """
    g = np.asarray(genuine_scores, float)
    i = np.asarray(impostor_scores, float)
    thr = np.unique(np.concatenate([g, i]))  # all achievable thresholds
    if thr.size == 0:
        return None

    FAR_list, FRR_list = [], []
    for t in thr:
        FAR_list.append(np.mean(i >= t))  # impostor accepted
        FRR_list.append(np.mean(g <  t))  # genuine rejected
    FAR = np.asarray(FAR_list); FRR = np.asarray(FRR_list)

    # EER
    k = int(np.argmin(np.abs(FAR - FRR)))
    EER = float((FAR[k] + FRR[k]) / 2.0); thr_eer = float(thr[k])

    # ROC AUC
    TPR = 1.0 - FRR; FPR = FAR
    order = np.argsort(FPR)
    AUC = float(np.trapz(TPR[order], FPR[order]))

    # Best F1 (report precision/recall at that threshold)
    bestF1, bestT, bestP, bestR = -1.0, thr[0], 0.0, 0.0
    for t in thr:
        TP = float(np.sum(g >= t)); FP = float(np.sum(i >= t)); FN = float(np.sum(g < t))
        P  = TP / (TP + FP + 1e-12)
        R  = TP / (TP + FN + 1e-12)
        F1 = 2*P*R / (P + R + 1e-12)
        if F1 > bestF1:
            bestF1, bestT, bestP, bestR = float(F1), float(t), float(P), float(R)

    return {"EER":EER, "thr_eer":thr_eer, "ROC_AUC":AUC,
            "best_F1":bestF1, "thr_best_F1":bestT,
            "precision_at_best_F1":bestP, "recall_at_best_F1":bestR}

# --------------------------- main ---------------------------
def main():
    idx = build_index(OUT_DIR)
    total_templates = sum(len(v) for d in idx.values() for v in d.values())
    print(f"Indexed {total_templates} templates from {len(idx)} subjects.")

    # Build genuine / impostor pairs (stems only)
    genuines, impostors = [], []

    # Genuine: same subject & finger, different impressions
    for sid, fingers in idx.items():
        for finger, stems in fingers.items():
            S = sorted(stems)
            for a in range(len(S)):
                for b in range(a+1, len(S)):
                    genuines.append((S[a], S[b]))

    # Impostor: different subjects, same finger (first impression for each)
    subs = sorted(idx.keys())
    for a in range(len(subs)):
        for b in range(a+1, len(subs)):
            fa, fb = idx[subs[a]], idx[subs[b]]
            common = set(fa.keys()) & set(fb.keys())
            for finger in common:
                if not fa[finger] or not fb[finger]:
                    continue
                impostors.append((sorted(fa[finger])[0], sorted(fb[finger])[0]))

    # Optional caps to keep runtime quick
    rng = np.random.default_rng(0)
    if MAX_GENUINE is not None and len(genuines) > MAX_GENUINE:
        genuines = list(map(tuple, rng.choice(genuines, size=MAX_GENUINE, replace=False)))
    if MAX_IMPOSTOR is not None and len(impostors) > MAX_IMPOSTOR:
        impostors = list(map(tuple, rng.choice(impostors, size=MAX_IMPOSTOR, replace=False)))

    print(f"Genuine pairs: {len(genuines)} | Impostor pairs: {len(impostors)}")

    # Run matcher
    verifier = MinutiaeVerifier(out_dir=OUT_DIR, dist_tol_px=15.0, ang_tol_deg=20.0, use_scale=False)
    g_scores, i_scores = [], []
    pairs_rows = []

    for A, B in genuines:
        r = verifier.score_pair(A, B)
        g_scores.append(r["score"])
        if WRITE_PAIR_SCORES_CSV:
            pairs_rows.append((A, B, 1, r["score"], r["matches"]))

    for A, B in impostors:
        r = verifier.score_pair(A, B)
        i_scores.append(r["score"])
        if WRITE_PAIR_SCORES_CSV:
            pairs_rows.append((A, B, 0, r["score"], r["matches"]))

    # Metrics
    stats = sweep_thresholds(g_scores, i_scores)
    if stats is None:
        print("Not enough pairs to evaluate.")
        return

    print("\n=== Minutiae-based Verification Metrics (SOCOFing) ===")
    print(f"EER:       {stats['EER']:.4f}  at thr={stats['thr_eer']:.4f}")
    print(f"ROC AUC:   {stats['ROC_AUC']:.4f}")
    print(f"Best F1:   {stats['best_F1']:.4f}  at thr={stats['thr_best_F1']:.4f}")
    print(f"Precision@bestF1: {stats['precision_at_best_F1']:.4f}")
    print(f"Recall@bestF1:    {stats['recall_at_best_F1']:.4f}")

    # Optional CSV of pair scores (for analysis)
    if WRITE_PAIR_SCORES_CSV:
        with open(PAIR_SCORES_CSV, "w", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            w.writerow(["stemA", "stemB", "label", "score", "matches"])
            w.writerows(pairs_rows)
        print(f"\nSaved pair scores to {PAIR_SCORES_CSV}")
