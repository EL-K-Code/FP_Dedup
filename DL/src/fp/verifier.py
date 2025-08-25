# src/fp/minutiae_verifier.py
# Reusable minutiae matching engine:
# - loads minutiae (x, y, theta_img_deg) from out/<stem>/<stem>.json|.csv
# - estimates global alignment (rotation + translation, optional scale)
# - greedy 1-to-1 matching with distance + orientation checks
# - returns a similarity score in [0,1] from the number of matches

from __future__ import annotations
from pathlib import Path
from dataclasses import dataclass
from typing import List, Tuple, Dict
import csv, json, math
import numpy as np

# --------------------- Small angle utils ---------------------
def _wrap_pi(a: float) -> float:
    """Wrap angle to (-pi, pi]."""
    return (a + math.pi) % (2.0 * math.pi) - math.pi

def _ang_diff(a: float, b: float) -> float:
    """Smallest absolute difference between two angles (radians)."""
    return abs(_wrap_pi(a - b))

def _deg2rad(d: float) -> float:
    return float(d) * math.pi / 180.0

def _rad2deg(r: float) -> float:
    return float(r) * 180.0 / math.pi


# --------------------- Data structure ------------------------
@dataclass
class Minutia:
    x: float
    y: float
    theta_rad: float  # radians in image coordinates

# ------------------------ I/O loader -------------------------
def load_minutiae(out_dir: Path, stem: str) -> List[Minutia]:
    """
    Load minutiae from out/<stem>/<stem>.json or .csv.
    Expected angle key: 'theta_img_deg' (deg). Fallback to 'theta'.
    """
    base = out_dir / stem / stem
    j = base.with_suffix(".json")
    c = base.with_suffix(".csv")
    mins: List[Minutia] = []

    if j.exists():
        data = json.loads(j.read_text(encoding="utf-8"))
        for m in data:
            th_deg = float(m.get("theta_img_deg", m.get("theta", 0.0)))
            mins.append(Minutia(float(m["x"]), float(m["y"]), _deg2rad(th_deg)))
        return mins

    if c.exists():
        with open(c, "r", newline="", encoding="utf-8") as f:
            for row in csv.DictReader(f):
                th_deg = float(row.get("theta_img_deg", row.get("theta", 0.0)))
                mins.append(Minutia(float(row["x"]), float(row["y"]), _deg2rad(th_deg)))
        return mins

    raise FileNotFoundError(f"No minutiae file for '{stem}' (looked for {j} / {c}).")


# ------------------ Step B: global alignment -----------------
def vote_rotation(minsA: List[Minutia],
                  minsB: List[Minutia],
                  bin_deg: float = 2.0,
                  top_k: int = 3,
                  rng_seed: int = 123) -> List[float]:
    """
    Hough-like voting over orientation differences to propose global rotation(s) in radians.
    Returns up to top_k rotation candidates, highest votes first.
    """
    if not minsA or not minsB:
        return [0.0]

    bin_rad = math.radians(bin_deg)
    nbins   = max(4, int(round((2.0 * math.pi) / bin_rad)))
    hist    = np.zeros(nbins, dtype=np.int32)

    rng = np.random.default_rng(rng_seed)
    # Light random sampling for speed (2*sqrt(n) from each side, at least 10)
    IA = rng.choice(len(minsA), size=min(len(minsA), max(10, int(2 * math.sqrt(len(minsA))))), replace=False)
    IB = rng.choice(len(minsB), size=min(len(minsB), max(10, int(2 * math.sqrt(len(minsB))))), replace=False)

    for i in IA:
        ai = minsA[i]
        for j in IB:
            bj = minsB[j]
            dth = _wrap_pi(bj.theta_rad - ai.theta_rad)  # (-pi,pi]
            b   = int(((dth + math.pi) / (2.0 * math.pi)) * nbins) % nbins
            hist[b] += 1

    cand_bins = np.argsort(hist)[::-1][:max(1, top_k)]
    cands = []
    for b in cand_bins:
        center = (-math.pi) + (b + 0.5) * (2.0 * math.pi / nbins)
        cands.append(center)
    return cands


def estimate_alignment(minsA: List[Minutia],
                       minsB: List[Minutia],
                       use_scale: bool = False) -> Tuple[float, np.ndarray, float]:
    """
    Estimate (theta, t, s) such that A' = s * R(theta) * A + t approximately aligns A to B.
    - theta: radians
    - t: (tx, ty)
    - s: ~1.0 (only if use_scale=True)
    """
    if not minsA or not minsB:
        return 0.0, np.zeros(2, dtype=float), 1.0

    thetas = vote_rotation(minsA, minsB)
    Axy = np.array([[m.x, m.y] for m in minsA], dtype=float)
    Bxy = np.array([[m.x, m.y] for m in minsB], dtype=float)

    best = None
    for th in thetas:
        c, s = math.cos(th), math.sin(th)
        R = np.array([[c, -s], [s, c]], dtype=float)
        A_rot = (R @ Axy.T).T

        if use_scale:
            cenA = A_rot.mean(axis=0)
            cenB = Bxy.mean(axis=0)
            rA = np.median(np.linalg.norm(A_rot - cenA, axis=1))
            rB = np.median(np.linalg.norm(Bxy - cenB, axis=1))
            sc = 1.0 if rA <= 1e-6 else np.clip(rB / rA, 0.85, 1.15)
        else:
            sc = 1.0

        t = Bxy.mean(axis=0) - sc * A_rot.mean(axis=0)
        # simple selection: keep the candidate with higher vote (order of thetas)
        cand = (th, t, sc)
        best = cand if best is None else best

    return best[0], best[1], best[2]


# ----------- Step C: greedy 1-to-1 minutiae matching --------
def greedy_match(minsA: List[Minutia], minsB: List[Minutia],
                 theta: float, t: np.ndarray, s: float,
                 dist_tol_px: float = 15.0,
                 ang_tol_deg: float = 20.0) -> Tuple[int, List[Tuple[int,int]]]:
    """
    Apply A' = s*R(theta)*A + t. Greedy nearest-neighbor 1-to-1 matching with:
      - Euclidean distance <= dist_tol_px,
      - orientation residual <= ang_tol_deg (accounting for global rotation).
    Returns (match_count, list_of_pairs).
    """
    if not minsA or not minsB:
        return 0, []

    c, si = math.cos(theta), math.sin(theta)
    R = np.array([[c, -si], [si, c]], dtype=float)

    Axy = np.array([[m.x, m.y] for m in minsA], dtype=float)
    Axy = (s * (R @ Axy.T).T) + t
    Bxy = np.array([[m.x, m.y] for m in minsB], dtype=float)

    used_B = np.zeros(len(minsB), dtype=bool)
    pairs: List[Tuple[int,int]] = []
    ang_tol = math.radians(ang_tol_deg)

    for i, a in enumerate(Axy):
        dists = np.linalg.norm(Bxy - a, axis=1)
        j = int(np.argmin(dists))
        if used_B[j]:
            continue
        if dists[j] > dist_tol_px:
            continue

        # expected: theta_B ≈ theta_A + theta (mod 2π)
        res = _ang_diff(minsB[j].theta_rad, minsA[i].theta_rad + theta)
        if res > ang_tol:
            continue

        used_B[j] = True
        pairs.append((i, j))

    return len(pairs), pairs


# -------------------- High-level API class -------------------
class MinutiaeVerifier:
    """
    Reusable engine:
      verifier = MinutiaeVerifier(dist_tol_px=15, ang_tol_deg=20, use_scale=False)
      res = verifier.score_pair(out_dir, "stemA", "stemB")
    """
    def __init__(self,
                 dist_tol_px: float = 15.0,
                 ang_tol_deg: float = 20.0,
                 use_scale: bool = False):
        self.dist_tol_px = float(dist_tol_px)
        self.ang_tol_deg = float(ang_tol_deg)
        self.use_scale   = bool(use_scale)

    def score_pair(self, out_dir: Path, stemA: str, stemB: str) -> Dict[str, float]:
        minsA = load_minutiae(out_dir, stemA)
        minsB = load_minutiae(out_dir, stemB)
        theta, t, s = estimate_alignment(minsA, minsB, use_scale=self.use_scale)
        m, pairs = greedy_match(minsA, minsB, theta, t, s,
                                dist_tol_px=self.dist_tol_px,
                                ang_tol_deg=self.ang_tol_deg)
        score = 1.0 - math.exp(- m / 10.0)
        return {
            "score": score,
            "matches": int(m),
            "theta_deg": _rad2deg(theta),
            "tx": float(t[0]),
            "ty": float(t[1]),
            "scale": float(s)
        }
