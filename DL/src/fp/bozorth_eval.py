# src/fp/bozorth_eval.py
from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
from typing import List, Dict, Tuple, Optional
import json, csv, re, math
import numpy as np

# On réutilise TON implémentation de Bozorth-like (garde ton fichier existant)
from bozorth3_like import match_stems, _load_minutiae_from_out

# --------------------------- Parsing SOCOFing ---------------------------

_SOCO_STEM_RE = re.compile(
    r"""
    ^\s*
    (?P<sid>\d+)        # subject id (ex: 101)
    __
    (?P<gender>[MF])    # M / F
    _
    (?P<hand>Left|Right)
    _
    (?P<finger>thumb|index|middle|ring|little)_finger
    (?:[_\-]?(?P<suffix>.*))?
    \s*$
    """,
    re.IGNORECASE | re.VERBOSE,
)

@dataclass(frozen=True)
class Item:
    stem: str
    sid: str
    hand: str
    finger: str

def parse_soco_stem(stem: str) -> Optional[Item]:
    m = _SOCO_STEM_RE.match(stem)
    if not m:
        return None
    sid = m.group("sid")
    hand = m.group("hand").capitalize()
    finger = m.group("finger").lower()
    return Item(stem=stem, sid=sid, hand=hand, finger=finger)

# --------------------------- Découverte des fichiers ---------------------------

def discover_out(out_dir: str | Path) -> List[Item]:
    """
    Cherche les fichiers minuties dans: out/<stem>/<stem>.json|.csv
    et renvoie la liste des Items parsés (SOCOFing).
    """
    out_dir = Path(out_dir)
    items: List[Item] = []
    for sub in out_dir.iterdir():
        if not sub.is_dir():
            continue
        stem = sub.name
        j = sub / f"{stem}.json"
        c = sub / f"{stem}.csv"
        if j.exists() or c.exists():
            it = parse_soco_stem(stem)
            if it is not None:
                items.append(it)
    # ordre stable
    items.sort(key=lambda x: x.stem)
    return items

# --------------------------- Construction des paires ---------------------------

def make_pairs_for_verification(items: List[Item]) -> Tuple[List[Tuple[str,str]], List[Tuple[str,str]]]:
    """
    Construit toutes les paires genuine / impostor en tenant compte du doigt:
      - Genuine: même (sid, hand, finger), stems différents
      - Impostor: même (hand, finger), sid différent
    Retourne (genuine_pairs, impostor_pairs) de stems.
    """
    by_key: Dict[Tuple[str,str,str], List[Item]] = {}
    for it in items:
        by_key.setdefault((it.sid, it.hand, it.finger), []).append(it)

    # genuine = intra (même cluster) combinaisons 2 à 2
    genuine: List[Tuple[str,str]] = []
    for key, group in by_key.items():
        stems = [g.stem for g in group]
        for i in range(len(stems)):
            for j in range(i+1, len(stems)):
                genuine.append((stems[i], stems[j]))

    # impostor: pour chaque cluster A, contre clusters B partageant (hand,finger) mais sid ≠
    impostor: List[Tuple[str,str]] = []
    by_hf: Dict[Tuple[str,str], List[Item]] = {}
    for it in items:
        by_hf.setdefault((it.hand, it.finger), []).append(it)

    for (hand,finger), group in by_hf.items():
        # partition par sid
        by_sid: Dict[str, List[str]] = {}
        for it in group:
            by_sid.setdefault(it.sid, []).append(it.stem)
        sids = sorted(by_sid.keys())
        for i in range(len(sids)):
            for j in range(i+1, len(sids)):
                for sa in by_sid[sids[i]]:
                    for sb in by_sid[sids[j]]:
                        impostor.append((sa, sb))

    return genuine, impostor

# --------------------------- Scoring avec Bozorth-like ---------------------------

def score_pairs(out_dir: str | Path,
                pairs: List[Tuple[str,str]],
                td: float = 12.0,
                ta_deg: float = 10.0,
                min_cluster: int = 5,
                max_pairs: Optional[int] = None) -> List[Dict]:
    out_dir = Path(out_dir)
    scores: List[Dict] = []
    for a, b in pairs:
        res = match_stems(out_dir, a, b, td=td, ta_deg=ta_deg, min_cluster=min_cluster, max_pairs=max_pairs)
        scores.append({"A": a, "B": b, "score": float(res["score"]), "matches": int(res["matches"])})
    return scores

# --------------------------- Metrics (ROC/EER/TAR, CMC) ---------------------------

def _roc_metrics(gen: np.ndarray, imp: np.ndarray) -> Dict[str, float]:
    """Retourne EER, AUC approx (trapèzes), TAR@FAR {1%,0.1%}."""
    # Concatène pour seuils
    all_scores = np.concatenate([gen, imp])
    # tri décroissant (seuils)
    thresholds = np.unique(np.sort(all_scores)[::-1])

    # Comptes cumulés pour gagner du temps
    gen_sorted = np.sort(gen)[::-1]
    imp_sorted = np.sort(imp)[::-1]

    def tar_far_at(t):
        # TAR = P(gen >= t)
        tar = (gen_sorted >= t).mean() if gen_sorted.size else 0.0
        # FAR = P(imp >= t)
        far = (imp_sorted >= t).mean() if imp_sorted.size else 0.0
        return tar, far

    # Courbe ROC discrète
    fars, tars = [], []
    for t in thresholds:
        tar, far = tar_far_at(t)
        tars.append(tar); fars.append(far)

    # EER ≈ point où FAR ~ FRR; FRR = 1 - TAR
    frrs = [1.0 - t for t in tars]
    # recherche du croisement (linéaire entre deux points)
    eer = 1.0
    for i in range(1, len(thresholds)):
        x0, y0 = fars[i-1], frrs[i-1]
        x1, y1 = fars[i],   frrs[i]
        # intersection du segment avec y=x
        # param s: y = y0 + s*(y1-y0), x = x0 + s*(x1-x0), on cherche y=x
        denom = (y1 - y0) - (x1 - x0)
        if abs(denom) < 1e-12:
            continue
        s = (x0 - y0) / denom
        if 0.0 <= s <= 1.0:
            x = x0 + s*(x1 - x0)
            eer = min(eer, x)
    eer = float(max(0.0, min(1.0, eer)))

    # AUC (approx) via trapèzes sur ROC (FAR→TAR)
    # Il faut trier par FAR croissant
    order = np.argsort(fars)
    fars_arr = np.array(fars)[order]
    tars_arr = np.array(tars)[order]
    auc = float(np.trapz(tars_arr, fars_arr))

    # TAR @ FAR k
    def tar_at_far(target_far: float) -> float:
        # interpolation linéaire sur FAR
        if len(fars_arr) == 0:
            return 0.0
        if target_far <= fars_arr[0]:
            return float(tars_arr[0])
        if target_far >= fars_arr[-1]:
            return float(tars_arr[-1])
        i = np.searchsorted(fars_arr, target_far)
        x0, x1 = fars_arr[i-1], fars_arr[i]
        y0, y1 = tars_arr[i-1], tars_arr[i]
        if x1 == x0:
            return float(y0)
        y = y0 + (y1 - y0) * (target_far - x0) / (x1 - x0)
        return float(y)

    tar_far_1 = tar_at_far(0.01)
    tar_far_01 = tar_at_far(0.001)

    return {"EER": eer, "AUC": auc, "TAR@FAR=1%": tar_far_1, "TAR@FAR=0.1%": tar_far_01}

def _cmc_metrics(gallery: Dict[Tuple[str,str], str],
                 probes: List[Item],
                 out_dir: str | Path,
                 td: float, ta_deg: float, min_cluster: int, max_pairs: Optional[int]) -> Dict[str, float]:
    """
    Identification (1:N) par doigt: on construit un gallery avec 1 image par (sid,hand,finger).
    Pour chaque probe (même (hand,finger)), on classe les scores contre toutes les entrées gallery
    de ce doigt; Rank-k correct si la bonne identité est dans le top-k.
    """
    ranks = []
    for p in probes:
        key = (p.hand, p.finger)
        # candidates du même doigt
        cand = [(sid, stemG) for (sid_h, hand, finger), stemG in gallery.items()
                if (hand, finger) == key for sid in [sid_h]]
        if not cand:
            continue
        # score vs chaque candidat
        scored = []
        for sid_g, stem_g in cand:
            r = match_stems(out_dir, p.stem, stem_g,
                            td=td, ta_deg=ta_deg, min_cluster=min_cluster, max_pairs=max_pairs)
            scored.append((sid_g, r["score"]))
        scored.sort(key=lambda x: x[1], reverse=True)
        # rang de la bonne identité
        true_sid = p.sid
        rank = 1
        for (sid_g, _) in scored:
            if sid_g == true_sid:
                break
            rank += 1
        ranks.append(rank)

    if not ranks:
        return {"rank1": 0.0, "rank5": 0.0, "rank10": 0.0}

    ranks = np.array(ranks)
    n = float(len(ranks))
    return {
        "rank1":  float((ranks <= 1).sum() / n),
        "rank5":  float((ranks <= 5).sum() / n),
        "rank10": float((ranks <= 10).sum() / n),
    }

# --------------------------- Évaluation end-to-end ---------------------------

def evaluate_soco(out_dir: str | Path,
                  td: float = 12.0,
                  ta_deg: float = 10.0,
                  min_cluster: int = 5,
                  max_pairs: Optional[int] = None,
                  save_dir: Optional[str | Path] = None) -> Dict:
    """
    - Scanne out/… pour trouver toutes les images SOCOFing disponibles
    - Construit genuine/impostor
    - Score toutes les paires via Bozorth-like
    - Calcule ROC/EER/TAR
    - Construit un gallery 1-échantillon/identité/doigt et calcule CMC (rank-1/5/10)
    - Sauvegarde CSV/JSON
    """
    out_dir = Path(out_dir)
    save_dir = Path(save_dir) if save_dir is not None else (out_dir / "_eval_bozorth")
    save_dir.mkdir(parents=True, exist_ok=True)

    items = discover_out(out_dir)
    if len(items) < 2:
        raise RuntimeError("Not enough SOCOFing items discovered in out/. Make sure out/<stem>/<stem>.json exist.")

    # pairs
    gen_pairs, imp_pairs = make_pairs_for_verification(items)

    # scoring
    gen_scores = score_pairs(out_dir, gen_pairs, td=td, ta_deg=ta_deg, min_cluster=min_cluster, max_pairs=max_pairs)
    imp_scores = score_pairs(out_dir, imp_pairs, td=td, ta_deg=ta_deg, min_cluster=min_cluster, max_pairs=max_pairs)

    # save raw scores
    with open(save_dir / "scores_genuine.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["A","B","score","matches"]); w.writeheader(); w.writerows(gen_scores)
    with open(save_dir / "scores_impostor.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["A","B","score","matches"]); w.writeheader(); w.writerows(imp_scores)

    gen = np.array([s["score"] for s in gen_scores], dtype=float)
    imp = np.array([s["score"] for s in imp_scores], dtype=float)
    verif = _roc_metrics(gen, imp)

    # identification gallery/probes
    # gallery: premier stem rencontré pour chaque (sid,hand,finger)
    gallery: Dict[Tuple[str,str,str], str] = {}
    for it in items:
        key = (it.sid, it.hand, it.finger)
        if key not in gallery:
            gallery[key] = it.stem
    # probes: tous les autres (et ≠ gallery stem)
    probes = [it for it in items if gallery[(it.sid, it.hand, it.finger)] != it.stem]

    cmc = _cmc_metrics(
        {(sid,hand,finger): stem for (sid,hand,finger), stem in gallery.items()},
        probes, out_dir, td, ta_deg, min_cluster, max_pairs
    )

    # pack + save metrics
    results = {
        "n_items": len(items),
        "n_genuine_pairs": len(gen_pairs),
        "n_impostor_pairs": len(imp_pairs),
        "verification": verif,
        "identification": cmc,
        "params": dict(td=td, ta_deg=ta_deg, min_cluster=min_cluster, max_pairs=max_pairs),
    }
    with open(save_dir / "metrics.json", "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
    return results
