# scripts/run_soco_verify.py
# SOCOFing runner:
# - scans out/<stem>/ for minutiae files
# - builds genuine & impostor pairs (same finger key)
# - scores pairs with MinutiaeVerifier
# - computes metrics (ROC-AUC, EER, best-F1, precision, recall)
# - saves CSV, JSON and plots in results/

from __future__ import annotations
from pathlib import Path
from typing import Dict, Tuple, List, Optional
import re, json, math, random, csv
import numpy as np
import matplotlib.pyplot as plt

# import the engine
from fp.verifier import MinutiaeVerifier

# ---------- Paths & settings (edit if needed) ----------
OUT_DIR     = Path("out")
RESULTS_DIR = Path("results"); RESULTS_DIR.mkdir(parents=True, exist_ok=True)

MAX_GENUINE  = 3000     # cap (None = all)
MAX_IMPOSTOR = 3000
RNG_SEED     = 123

# Matching hyper-params (must match your engine)
DIST_TOL_PX  = 15.0
ANG_TOL_DEG  = 20.0
USE_SCALE    = False
# -------------------------------------------------------


# ---------------- Stem parsing for SOCOFing --------------
SOCO_RE = re.compile(
    r"^(?P<subject>\d+)__([MF])_(?P<hand>Left|Right)_(?P<finger>(thumb|index_finger|middle_finger|ring_finger|little_finger))",
    re.IGNORECASE
)

def parse_soco_stem(stem: str) -> Optional[Tuple[str, str]]:
    """
    Return (subject_id, finger_key) from a SOCOFing-like stem or None.
    finger_key = "<Hand>_<finger>", e.g. "Left_index_finger"
    """
    m = SOCO_RE.match(stem)
    if not m:
        return None
    subject = m.group("subject")
    finger_key = f"{m.group('hand').capitalize()}_{m.group('finger').lower()}"
    return subject, finger_key


def collect_stems(out_dir: Path = OUT_DIR) -> Dict[Tuple[str,str], List[str]]:
    """
    Group stems by (subject, finger_key) if minutiae file exists.
    Returns: {(subject, finger_key): [stem, ...], ...}
    """
    groups: Dict[Tuple[str,str], List[str]] = {}
    for p in out_dir.iterdir():
        if not p.is_dir():
            continue
        stem = p.name
        meta = parse_soco_stem(stem)
        if meta is None:
            continue
        # ensure minutiae file exists
        j = p / f"{stem}.json"
        c = p / f"{stem}.csv"
        if not j.exists() and not c.exists():
            continue
        key = (meta[0], meta[1])
        groups.setdefault(key, []).append(stem)

    for k in groups:
        groups[k] = sorted(groups[k])
    return groups


def make_pairs(groups: Dict[Tuple[str,str], List[str]],
               max_genuine: Optional[int],
               max_impostor: Optional[int]) -> Tuple[List[Tuple[str,str]], List[Tuple[str,str]]]:
    """Build genuine pairs (same subject+finger) and impostor pairs (different subjects, same finger)."""
    rng = random.Random(RNG_SEED)

    # Genuine
    genuines: List[Tuple[str,str]] = []
    finger_to_subjects: Dict[str, List[Tuple[str, List[str]]]] = {}
    for (subject, finger), stems in groups.items():
        if len(stems) >= 2:
            for i in range(len(stems)):
                for j in range(i+1, len(stems)):
                    genuines.append((stems[i], stems[j]))
        finger_to_subjects.setdefault(finger, []).append((subject, stems))

    # Impostors (sampled): choose one stem from subject A and one from subject B for the same finger
    impostors: List[Tuple[str,str]] = []
    for finger, items in finger_to_subjects.items():
        if len(items) < 2:
            continue
        for i in range(len(items)):
            subjA, stemsA = items[i]
            for j in range(i+1, len(items)):
                subjB, stemsB = items[j]
                a = rng.choice(stemsA)
                b = rng.choice(stemsB)
                impostors.append((a, b))

    rng.shuffle(genuines)
    rng.shuffle(impostors)
    if max_genuine is not None and len(genuines) > max_genuine:
        genuines = genuines[:max_genuine]
    if max_impostor is not None and len(impostors) > max_impostor:
        impostors = impostors[:max_impostor]

    return genuines, impostors


# ---------------------- Metrics (no sklearn) ----------------------
def _roc_points(scores: np.ndarray, labels: np.ndarray):
    """Compute ROC points (FPR, TPR, thresholds) sweeping unique thresholds high->low."""
    order = np.argsort(scores)[::-1]
    s = scores[order]; y = labels[order]
    P = float((y == 1).sum()); N = float((y == 0).sum())
    tprs, fprs, thrs = [], [], []
    tp = fp = 0.0; last = None

    for i in range(len(s)):
        if last is None or s[i] != last:
            if last is not None:
                tprs.append(tp / P if P > 0 else 0.0)
                fprs.append(fp / N if N > 0 else 0.0)
                thrs.append(last)
            last = s[i]
        if y[i] == 1: tp += 1.0
        else:         fp += 1.0

    tprs.append(tp / P if P > 0 else 0.0)
    fprs.append(fp / N if N > 0 else 0.0)
    thrs.append(last if last is not None else 1.0)
    return np.array(fprs), np.array(tprs), np.array(thrs)

def _auc_trapz(fpr: np.ndarray, tpr: np.ndarray) -> float:
    order = np.argsort(fpr)
    return float(np.trapz(tpr[order], fpr[order]))

def _eer_from_roc(fpr: np.ndarray, tpr: np.ndarray, thr: np.ndarray):
    fnr = 1.0 - tpr
    idx = int(np.argmin(np.abs(fpr - fnr)))
    eer = 0.5 * (fpr[idx] + fnr[idx])
    return float(eer), float(thr[idx])

def _best_f1(scores: np.ndarray, labels: np.ndarray):
    thrs = np.unique(scores)[::-1]
    best = (0.0, 0.0, 0.0, 0.0)  # F1, thr, P, R
    for t in thrs:
        pred = (scores >= t).astype(int)
        tp = int(((pred == 1) & (labels == 1)).sum())
        fp = int(((pred == 1) & (labels == 0)).sum())
        fn = int(((pred == 0) & (labels == 1)).sum())
        if tp + fp == 0 or tp + fn == 0: 
            continue
        P = tp / float(tp + fp)
        R = tp / float(tp + fn)
        if P + R == 0: 
            continue
        F1 = 2 * P * R / (P + R)
        if F1 > best[0]:
            best = (float(F1), float(t), float(P), float(R))
    return best  # F1, thr, P, R


# ----------------------------- Runner ------------------------------
def main():
    random.seed(RNG_SEED); np.random.seed(RNG_SEED)

    groups = collect_stems(OUT_DIR)
    if not groups:
        print(f"❌ No minutiae found under {OUT_DIR}/<stem>/<stem>.json|.csv")
        return

    genuines, impostors = make_pairs(groups, MAX_GENUINE, MAX_IMPOSTOR)
    print(f"📦 Pairs → genuine: {len(genuines)}, impostor: {len(impostors)}")

    verifier = MinutiaeVerifier(dist_tol_px=DIST_TOL_PX,
                                ang_tol_deg=ANG_TOL_DEG,
                                use_scale=USE_SCALE)

    rows = []
    scores = []
    labels = []

    # Score genuine
    for A, B in genuines:
        r = verifier.score_pair(OUT_DIR, A, B)
        rows.append([A, B, 1, r["score"], r["matches"]])
        scores.append(r["score"]); labels.append(1)

    # Score impostor
    for A, B in impostors:
        r = verifier.score_pair(OUT_DIR, A, B)
        rows.append([A, B, 0, r["score"], r["matches"]])
        scores.append(r["score"]); labels.append(0)

    scores = np.array(scores, dtype=float)
    labels = np.array(labels, dtype=int)

    # Metrics
    fpr, tpr, thr = _roc_points(scores, labels)
    auc = _auc_trapz(fpr, tpr)
    eer, thr_eer = _eer_from_roc(fpr, tpr, thr)
    bestF1, thr_f1, P_f1, R_f1 = _best_f1(scores, labels)

    # Save CSV
    csv_path = RESULTS_DIR / "soco_pairs_scores.csv"
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["stemA", "stemB", "label_genuine1", "score", "matches"])
        w.writerows(rows)

    # Save metrics JSON
    metrics = {
        "pairs": {"genuine": int((labels==1).sum()), "impostor": int((labels==0).sum())},
        "matching_params": {
            "dist_tol_px": DIST_TOL_PX,
            "ang_tol_deg": ANG_TOL_DEG,
            "use_scale": USE_SCALE
        },
        "metrics": {
            "ROC_AUC": auc,
            "EER": eer,
            "threshold_at_EER": thr_eer,
            "best_F1": bestF1,
            "threshold_at_best_F1": thr_f1,
            "precision_at_best_F1": P_f1,
            "recall_at_best_F1": R_f1
        }
    }
    (RESULTS_DIR / "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")

    # Plots
    # ROC
    plt.figure(figsize=(5,4))
    plt.plot(fpr, tpr, lw=2)
    plt.plot([0,1], [0,1], linestyle="--")
    plt.xlabel("FPR"); plt.ylabel("TPR"); plt.title(f"ROC (AUC={auc:.3f})")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(RESULTS_DIR / "roc.png", dpi=140)
    plt.close()

    # score hist
    plt.figure(figsize=(5,4))
    plt.hist(scores[labels==1], bins=40, alpha=0.6, label="genuine")
    plt.hist(scores[labels==0], bins=40, alpha=0.6, label="impostor")
    plt.xlabel("score"); plt.ylabel("count"); plt.title("Score distribution")
    plt.legend(); plt.tight_layout()
    plt.savefig(RESULTS_DIR / "hist.png", dpi=140)
    plt.close()

    print(f"✅ Done. Saved:\n  - {csv_path}\n  - {RESULTS_DIR/'metrics.json'}\n  - {RESULTS_DIR/'roc.png'}\n  - {RESULTS_DIR/'hist.png'}")

if __name__ == "__main__":
    main()
