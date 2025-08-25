# scripts/run_verifier_nist.py
from __future__ import annotations
from pathlib import Path
from typing import Dict, Tuple, List, Optional
import re, json, csv, random
import numpy as np

from fp.minutiae_verifier import MinutiaeVerifier

# ---------- Chemins ----------
OUT_DIR  = Path("out")                                   # minuties: out/<stem>/<stem>.json|.csv
SAVE_DIR = OUT_DIR / "_eval_minutiae_verifier_nist"      # résultats ici
SAVE_DIR.mkdir(parents=True, exist_ok=True)

# ---------- Parsing SOCOFing (NIST-like grouping) ----------
SOCO_RE = re.compile(
    r"""
    ^(?P<sid>\d+)__        # subject id (avant __)
      [MF]_                # genre (ignoré)
      (?P<hand>Left|Right)_
      (?P<finger>thumb|index|middle|ring|little)_finger
    """, re.IGNORECASE | re.VERBOSE
)

def parse_stem(stem: str):
    m = SOCO_RE.match(stem)
    if not m: return None
    sid = m.group("sid")
    hand = m.group("hand").capitalize()
    finger = m.group("finger").lower()
    return sid, hand, finger  # identité = sid, "doigt" = (hand,finger)

def discover_stems(out_dir: Path = OUT_DIR) -> List[str]:
    stems = []
    for d in out_dir.iterdir():
        if not d.is_dir(): continue
        stem = d.name
        if (d / f"{stem}.json").exists() or (d / f"{stem}.csv").exists():
            if parse_stem(stem) is not None:
                stems.append(stem)
    return sorted(stems)

# ---------- Construction des paires (définition NIST/FVC) ----------
def make_pairs_nist(stems: List[str],
                    max_genuine: Optional[int] = None,
                    max_impostor: Optional[int] = None,
                    seed: int = 123):
    """
    Genuine  : même (sid, hand, finger), stems différents
    Impostor : (sid différents) ET (même hand,finger)
    """
    rng = random.Random(seed)

    # Regroupe par (sid,hand,finger)
    by_shf: Dict[Tuple[str,str,str], List[str]] = {}
    for s in stems:
        sid, hand, finger = parse_stem(s)
        by_shf.setdefault((sid, hand, finger), []).append(s)

    # Genuine: combinaisons intra-groupe
    genuines: List[Tuple[str,str,int]] = []
    for key, lst in by_shf.items():
        L = sorted(lst)
        for i in range(len(L)):
            for j in range(i+1, len(L)):
                genuines.append((L[i], L[j], 1))
    rng.shuffle(genuines)
    if max_genuine is not None:
        genuines = genuines[:max_genuine]

    # Impostor: même (hand,finger) mais sid différents
    by_hf: Dict[Tuple[str,str], Dict[str, List[str]]] = {}
    for (sid,hand,finger), lst in by_shf.items():
        by_hf.setdefault((hand,finger), {}).setdefault(sid, []).extend(lst)

    impostors: List[Tuple[str,str,int]] = []
    for (hand,finger), by_sid in by_hf.items():
        sids = sorted(by_sid.keys())
        for i in range(len(sids)):
            for j in range(i+1, len(sids)):
                A, B = by_sid[sids[i]], by_sid[sids[j]]
                # Pour limiter la combinatoire, on échantillonne 2×2 par paire de sujets (modifie si tu veux tout)
                a_samp = A if len(A) <= 2 else rng.sample(A, 2)
                b_samp = B if len(B) <= 2 else rng.sample(B, 2)
                for a in a_samp:
                    for b in b_samp:
                        impostors.append((a, b, 0))
    rng.shuffle(impostors)
    if max_impostor is not None:
        impostors = impostors[:max_impostor]

    return genuines, impostors

# ---------- Métriques ----------
def roc_points(scores: np.ndarray, labels: np.ndarray):
    order = np.argsort(scores)[::-1]  # similarité: plus grand = mieux
    s = scores[order]; y = labels[order]
    P = float((y==1).sum()); N = float((y==0).sum())
    tp = fp = 0.0; last = None
    tpr = []; fpr = []; thr = []
    for i in range(len(s)):
        if last is None or s[i] != last:
            if last is not None:
                tpr.append(tp / P if P>0 else 0.0)
                fpr.append(fp / N if N>0 else 0.0)
                thr.append(last)
            last = s[i]
        if y[i]==1: tp += 1.0
        else:       fp += 1.0
    tpr.append(tp / P if P>0 else 0.0)
    fpr.append(fp / N if N>0 else 0.0)
    thr.append(last if last is not None else 1.0)
    return np.array(fpr), np.array(tpr), np.array(thr)

def auc_trapz(fpr: np.ndarray, tpr: np.ndarray) -> float:
    idx = np.argsort(fpr)
    return float(np.trapz(tpr[idx], fpr[idx]))

def eer_from_roc(fpr: np.ndarray, tpr: np.ndarray):
    fnr = 1.0 - tpr
    i = int(np.argmin(np.abs(fpr - fnr)))
    return float(0.5*(fpr[i]+fnr[i]))

def tar_at_far(scores_imp: np.ndarray, scores_gen: np.ndarray, far_target: float):
    if scores_imp.size == 0 or scores_gen.size == 0:
        return 0.0, 0.0, 0.0
    thr = float(np.quantile(scores_imp, 1.0 - far_target))
    tar = float((scores_gen >= thr).mean())
    far = float((scores_imp >= thr).mean())  # FAR réalisé (peut différer un peu à cause des ex æquo)
    return tar, far, thr

# ---------- Run ----------
def main():
    # Paramètres du moteur géométrique
    verifier = MinutiaeVerifier(
        dist_tol_px=15.0,   # ajuste si tes images sont plus grandes/petites
        ang_tol_deg=20.0,   # tolérance d'orientation
        use_scale=False     # True si tu veux autoriser un scale global léger
    )

    stems = discover_stems(OUT_DIR)
    if not stems:
        print("❌ No minutiae found in out/<stem>/<stem>.json|.csv")
        return

    genuines, impostors = make_pairs_nist(stems, max_genuine=None, max_impostor=None)
    print(f"NIST pairs → genuine={len(genuines)}, impostor={len(impostors)}")

    rows = []
    scores = []
    labels = []

    for (A,B,y) in genuines + impostors:
        res = verifier.score_pair(OUT_DIR, A, B)
        s = res["score"]
        rows.append([A, B, y, s, res["matches"], res["theta_deg"], res["tx"], res["ty"], res["scale"]])
        scores.append(s); labels.append(y)

    scores = np.asarray(scores, float)
    labels = np.asarray(labels, int)
    s_imp = scores[labels==0]
    s_gen = scores[labels==1]

    # ROC / AUC / EER
    fpr, tpr, thr = roc_points(scores, labels)
    auc = auc_trapz(fpr, tpr)
    eer = eer_from_roc(fpr, tpr)

    # TAR@FAR (1% et 0.1%)
    tar1, far1, thr1 = tar_at_far(s_imp, s_gen, 0.01)
    tar01, far01, thr01 = tar_at_far(s_imp, s_gen, 0.001)

    # Sauvegardes
    csv_path = SAVE_DIR / "pair_scores_nist.csv"
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["A","B","label_genuine1","score","matches","theta_deg","tx","ty","scale"])
        w.writerows(rows)

    metrics = {
        "protocol": "NIST-like (genuine: same subject+finger; impostor: diff subjects, same finger)",
        "pairs": {"genuine": int((labels==1).sum()), "impostor": int((labels==0).sum())},
        "metrics": {
            "ROC_AUC": auc,
            "EER": eer,
            "TAR_at_FAR_1pct": tar1,   "FAR_real_1pct": far1,   "thr_at_FAR_1pct": thr1,
            "TAR_at_FAR_0p1pct": tar01,"FAR_real_0p1pct": far01,"thr_at_FAR_0p1pct": thr01,
        }
    }
    (SAVE_DIR / "metrics_nist.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")

    print("✅ Done.")
    print(f"  Scores CSV : {csv_path}")
    print(f"  Metrics    : {SAVE_DIR/'metrics_nist.json'}")
    print(f"  AUC={auc:.4f}  EER={eer:.4f}  TAR@FAR1%={tar1:.4f}  TAR@FAR0.1%={tar01:.4f}")

if __name__ == "__main__":
    main()
