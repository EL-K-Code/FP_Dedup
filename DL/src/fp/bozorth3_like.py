# bozorth_like.py
from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
from typing import List, Dict, Tuple, Optional, Iterable, Set
import json, csv, math
import numpy as np

# --------------------------- Data models ---------------------------

@dataclass
class Minutia:
    x: float
    y: float
    theta: float  # radians (image coords: 0→→, +π/2↓)

@dataclass
class PairFeat:
    i: int
    j: int
    d: float         # distance between i,j
    phi_i: float     # relative angle of minutia i to the line i->j
    phi_j: float     # relative angle of minutia j to the line j->i (same line, opposite dir)

# --------------------------- Utils ---------------------------

def _wrap_pi(a: float) -> float:
    return (a + math.pi) % (2.0*math.pi) - math.pi

def _ang_diff(a: float, b: float) -> float:
    return abs(_wrap_pi(a - b))

def _load_minutiae_from_out(out_dir: str | Path, stem: str) -> List[Minutia]:
    """
    Expect: out/<stem>/<stem>.json (or .csv) with keys x,y,theta_img_deg or theta.
    Returns a list of Minutia with theta in radians.
    """
    out_dir = Path(out_dir)
    base = out_dir / stem / stem
    j = base.with_suffix(".json"); c = base.with_suffix(".csv")
    mins: List[Minutia] = []
    if j.exists():
        data = json.loads(j.read_text(encoding="utf-8"))
        for m in data:
            th = float(m.get("theta_img_deg", m.get("theta", 0.0)))
            mins.append(Minutia(float(m["x"]), float(m["y"]), math.radians(th)))
        return mins
    if c.exists():
        with open(c, "r", newline="", encoding="utf-8") as f:
            for row in csv.DictReader(f):
                th = float(row.get("theta_img_deg", row.get("theta", 0.0)))
                mins.append(Minutia(float(row["x"]), float(row["y"]), math.radians(th)))
        return mins
    raise FileNotFoundError(f"No minutiae file for stem={stem} in {out_dir}")

# --------------------------- Bozorth-like intra table ---------------------------

def build_intra_table(mins: List[Minutia],
                      max_pairs: Optional[int] = None) -> List[PairFeat]:
    """
    For all pairs (i<j), compute Bozorth-style features:
      - d      : Euclidean distance
      - phi_i  : (theta_i - alpha_ij) wrapped to (-pi,pi]
      - phi_j  : (theta_j - alpha_ji) = (theta_j - (alpha_ij+pi))
    These features are invariant to global rotation/translation (and robust across impressions).
    """
    n = len(mins)
    if n < 2: return []
    feats: List[PairFeat] = []
    # optional sampling if very dense
    indices = range(n)
    if max_pairs is None or (n*(n-1))//2 <= (max_pairs or 0):
        # full
        for i in range(n-1):
            xi, yi, thi = mins[i].x, mins[i].y, mins[i].theta
            for j in range(i+1, n):
                xj, yj, thj = mins[j].x, mins[j].y, mins[j].theta
                dx, dy = (xj - xi), (yj - yi)
                d = math.hypot(dx, dy)
                if d <= 0.0: 
                    continue
                alpha_ij = math.atan2(dy, dx)
                # relative orientations
                phi_i = _wrap_pi(thi - alpha_ij)
                phi_j = _wrap_pi(thj - (alpha_ij + math.pi))  # reverse direction at j
                feats.append(PairFeat(i, j, d, phi_i, phi_j))
        return feats

    # light random subsampling of pairs (rarely needed)
    rng = np.random.default_rng(0)
    pairs = [(i, j) for i in range(n-1) for j in range(i+1, n)]
    pairs = [pairs[k] for k in rng.choice(len(pairs), size=max_pairs, replace=False)]
    for i, j in pairs:
        xi, yi, thi = mins[i].x, mins[i].y, mins[i].theta
        xj, yj, thj = mins[j].x, mins[j].y, mins[j].theta
        dx, dy = (xj - xi), (yj - yi)
        d = math.hypot(dx, dy)
        if d <= 0.0: 
            continue
        alpha_ij = math.atan2(dy, dx)
        phi_i = _wrap_pi(thi - alpha_ij)
        phi_j = _wrap_pi(thj - (alpha_ij + math.pi))
        feats.append(PairFeat(i, j, d, phi_i, phi_j))
    return feats

# --------------------------- Pair compatibility ---------------------------

@dataclass(frozen=True)
class CorrNode:
    """A node in the correspondence graph == hypothesized mapping i (A) <-> k (B)."""
    i: int
    k: int

def compatible_pairs(pA: PairFeat, pB: PairFeat,
                     td: float, ta: float) -> List[Tuple[CorrNode, CorrNode]]:
    """
    Return up to 2 consistent correspondence edges implied by compatible pair-pairs:
    pA:(i,j) vs pB:(k,l) are compatible if:
      |dA-dB| <= td  AND
      |phi_iA - phi_kB| <= ta AND |phi_jA - phi_lB| <= ta
    Also check the swapped case (k<->l).
    """
    out = []
    if abs(pA.d - pB.d) <= td:
        # direct mapping: (i->k, j->l)
        if (_ang_diff(pA.phi_i, pB.phi_i) <= ta) and (_ang_diff(pA.phi_j, pB.phi_j) <= ta):
            out.append((CorrNode(pA.i, pB.i), CorrNode(pA.j, pB.j)))
        # swapped mapping: (i->l, j->k)
        if (_ang_diff(pA.phi_i, pB.phi_j) <= ta) and (_ang_diff(pA.phi_j, pB.phi_i) <= ta):
            out.append((CorrNode(pA.i, pB.j), CorrNode(pA.j, pB.i)))
    return out

# --------------------------- Graph build & clique-like scoring ---------------------------

def bozorth_like_match(minsA: List[Minutia],
                       minsB: List[Minutia],
                       td: float = 12.0,          # distance tolerance (pixels)
                       ta_deg: float = 10.0,      # angle tolerance (degrees)
                       min_cluster: int = 5,
                       max_pairs: Optional[int] = None) -> Dict[str, float]:
    """
    Build two intra tables, generate a compatibility graph of correspondences,
    then greedily grow dense, one-to-one, clique-like clusters. Return a score.
    """
    ta = math.radians(ta_deg)
    IA = build_intra_table(minsA, max_pairs=max_pairs)
    IB = build_intra_table(minsB, max_pairs=max_pairs)
    if not IA or not IB:
        return {"score": 0.0, "matches": 0}

    # 1) Build edges (corr graph)
    # nodes: CorrNode(i,k)
    # edges: between (i,k) and (j,l) if some (i,j)~(k,l) compatible-pair condition holds
    neighbors: Dict[CorrNode, Set[CorrNode]] = {}
    for a in IA:
        for b in IB:
            edges = compatible_pairs(a, b, td=td, ta=ta)
            for (u, v) in edges:
                if u == v: 
                    continue
                neighbors.setdefault(u, set()).add(v)
                neighbors.setdefault(v, set()).add(u)

    if not neighbors:
        return {"score": 0.0, "matches": 0}

    # 2) Greedy clique-like expansion with one-to-one constraint
    nodes = list(neighbors.keys())
    best = 0
    seen: Set[CorrNode] = set()

    # Helper to check 1-1 consistency in a cluster
    def consistent(cluster: List[CorrNode], cand: CorrNode) -> bool:
        for n in cluster:
            if n.i == cand.i or n.k == cand.k:
                return False
        return True

    # For speed: sort seeds by degree
    nodes_sorted = sorted(nodes, key=lambda u: len(neighbors[u]), reverse=True)

    for seed in nodes_sorted:
        # simple pruning: if even perfect growth cannot beat best, break
        if len(neighbors[seed]) + 1 <= best:
            break

        cluster = [seed]
        frontier = sorted(neighbors[seed], key=lambda u: len(neighbors[u]), reverse=True)

        # greedy add: candidate must connect to all current cluster members AND satisfy 1-1
        while frontier:
            cand = frontier.pop(0)
            if not consistent(cluster, cand):
                continue
            # must be adjacent to all nodes in current cluster
            if all((cand in neighbors[n]) for n in cluster):
                cluster.append(cand)
                # shrink frontier to those still adjacent to the new cluster member
                frontier = [u for u in frontier if u in neighbors[cand]]

        best = max(best, len(cluster))

        if best >= min_cluster and best >= 0.8 * (len(minsA)) and best >= 0.8 * (len(minsB)):
            # early out in very strong matches
            break

    # 3) Score mapping (monotone, bounded)
    m = best
    score = 1.0 - math.exp(-m / 8.0)  # faster saturation than simple count
    return {"score": score, "matches": int(m)}

# --------------------------- Convenience: from stems in out/ ---------------------------

def match_stems(out_dir: str | Path, stemA: str, stemB: str,
                **kwargs) -> Dict[str, float]:
    minsA = _load_minutiae_from_out(out_dir, stemA)
    minsB = _load_minutiae_from_out(out_dir, stemB)
    return bozorth_like_match(minsA, minsB, **kwargs)

