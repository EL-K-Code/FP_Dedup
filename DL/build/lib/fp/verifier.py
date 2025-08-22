# minutiae_verifier.py
from __future__ import annotations
from pathlib import Path
from typing import List, Dict, Tuple, Optional, Union
import json, csv, math, re
import numpy as np

# --------------------------- I/O ---------------------------

def load_minutiae_from_out(out_dir: Union[str, Path], stem: str) -> Optional[List[Dict]]:
    """
    Load minutiae list for a given image 'stem' from:
      out/<stem>/<stem>.json  (preferred)
      out/<stem>/<stem>.csv   (fallback)
    Each minutia should contain at least: x, y, theta_img_deg (or theta), score, class.
    """
    out_dir = Path(out_dir)
    base = out_dir / stem / stem
    j = base.with_suffix(".json")
    c = base.with_suffix(".csv")
    if j.exists():
        with open(j, "r", encoding="utf-8") as f:
            return json.load(f)
    if c.exists():
        mins = []
        with open(c, "r", newline="", encoding="utf-8") as f:
            for row in csv.DictReader(f):
                mins.append({
                    "x": int(float(row.get("x", 0))),
                    "y": int(float(row.get("y", 0))),
                    "theta_img_deg": float(row.get("theta_img_deg", row.get("theta", 0.0))),
                    "score": float(row.get("score", 0.0)),
                    "class": row.get("class", row.get("type", "")),
                })
        return mins
    return None

# --------------------------- geometry ---------------------------

def _deg2rad(a: float) -> float: return a * math.pi / 180.0
def _wrap_pi(a: float) -> float: return (a + math.pi) % (2.0 * math.pi) - math.pi
def _R(theta: float) -> np.ndarray:
    c, s = math.cos(theta), math.sin(theta)
    return np.array([[c, -s], [s, c]], dtype=np.float64)
def _rot(R: np.ndarray, X: np.ndarray) -> np.ndarray:
    return (R @ X.T).T

# --------------------------- alignment (vote on rotation) ---------------------------

def estimate_similarity(
    A: np.ndarray, thA: np.ndarray,
    B: np.ndarray, thB: np.ndarray,
    use_scale: bool = False, scale_tol: float = 0.10,
    top_k: int = 200
) -> Tuple[float, np.ndarray, float]:
    """
    Return (theta, t, s). If use_scale=False, s=1.
    Strategy:
      - sample up to 'top_k' pairs (i,j) to vote a global rotation
      - pick a few top rotation bins, compute t (and optional s) from centroids
    """
    nA, nB = len(A), len(B)
    if nA == 0 or nB == 0:
        return 0.0, np.zeros(2), 1.0

    # choose seeds
    k = max(1, int(math.sqrt(top_k)))
    ia = np.random.choice(nA, size=min(nA, 2*k), replace=False)
    ib = np.random.choice(nB, size=min(nB, 2*k), replace=False)

    # rotation accumulator (2° bins)
    bins = np.linspace(-math.pi, math.pi, 181)
    acc = np.zeros(len(bins), dtype=np.int32)
    for i in ia:
        for j in ib:
            dtheta = _wrap_pi(_deg2rad(thB[j]) - _deg2rad(thA[i]))
            b = int(np.digitize([dtheta], bins)[0]) - 1
            b = max(0, min(b, len(bins)-1))
            acc[b] += 1

    if acc.max() == 0:
        return 0.0, np.zeros(2), 1.0

    # try a few best rotations
    top_bins = np.argsort(acc)[::-1][:3]
    best = None
    for b in top_bins:
        theta = float((bins[b] + (bins[1] - bins[0]) / 2.0))
        R = _R(theta)
        if use_scale:
            cA, cB = A.mean(0), B.mean(0)
            dA = np.linalg.norm(A - cA, axis=1) + 1e-6
            dB = np.linalg.norm(B - cB, axis=1) + 1e-6
            s = float(np.median(dB) / np.median(dA))
            s = float(np.clip(s, 1.0 - scale_tol, 1.0 + scale_tol))
        else:
            s = 1.0
        t = B.mean(0) - s * _rot(R, A).mean(0)
        vote = acc[b]
        if best is None or vote > best[3]:
            best = (theta, t, s, vote)

    theta, t, s, _ = best
    return float(theta), t.astype(np.float64), float(s)

# --------------------------- matching after alignment ---------------------------

def greedy_match(
    A: np.ndarray, thA: np.ndarray,
    B: np.ndarray, thB: np.ndarray,
    theta: float, t: np.ndarray, s: float,
    dist_tol: float = 15.0, ang_tol_deg: float = 20.0
) -> List[Tuple[int, int]]:
    """
    Greedy 1-to-1 pairing after applying the similarity transform.
    Match if distance <= dist_tol and angular residual <= ang_tol_deg.
    """
    R = _R(theta)
    A2 = s * _rot(R, A) + t
    usedB = np.zeros(len(B), dtype=bool)
    pairs = []
    for i in range(len(A2)):
        d = np.linalg.norm(B - A2[i], axis=1)
        j = int(np.argmin(d))
        if usedB[j]:
            continue
        if d[j] <= dist_tol:
            dth = abs(_wrap_pi(_deg2rad(thB[j]) - _deg2rad(thA[i]) - theta))
            if dth <= _deg2rad(ang_tol_deg):
                usedB[j] = True
                pairs.append((i, j))
    return pairs

# --------------------------- public API ---------------------------

class MinutiaeVerifier:
    """
    Minutiae-based verifier using rigid alignment (R, t) and greedy matching.
    Returns a similarity score in [0,1] from raw match count with a soft cap.
    """
    def __init__(
        self,
        out_dir: Union[str, Path] = "out",
        dist_tol_px: float = 15.0,
        ang_tol_deg: float = 20.0,
        use_scale: bool = False,
        scale_tol: float = 0.10
    ):
        self.out_dir = Path(out_dir)
        self.dist_tol_px = float(dist_tol_px)
        self.ang_tol_deg = float(ang_tol_deg)
        self.use_scale = bool(use_scale)
        self.scale_tol = float(scale_tol)

    def _load(self, stem: str) -> List[Dict]:
        mins = load_minutiae_from_out(self.out_dir, stem) or []
        return mins

    def score_pair(self, stemA: str, stemB: str) -> Dict[str, float]:
        """
        Score two templates identified by their stems (folder names under out/).
        Returns:
            {
              "score": float in [0,1],
              "matches": int,
              "theta": float (rad),
              "tx": float, "ty": float, "s": float
            }
        """
        mA = self._load(stemA)
        mB = self._load(stemB)
        if not mA or not mB:
            return {"score": 0.0, "matches": 0, "theta": 0.0, "tx": 0.0, "ty": 0.0, "s": 1.0}

        A = np.array([[m["x"], m["y"]] for m in mA], dtype=np.float64)
        B = np.array([[m["x"], m["y"]] for m in mB], dtype=np.float64)
        thA = np.array([m.get("theta_img_deg", m.get("theta", 0.0)) for m in mA], dtype=np.float64)
        thB = np.array([m.get("theta_img_deg", m.get("theta", 0.0)) for m in mB], dtype=np.float64)

        theta, t, s = estimate_similarity(A, thA, B, thB, use_scale=self.use_scale, scale_tol=self.scale_tol)
        pairs = greedy_match(A, thA, B, thB, theta, t, s, dist_tol=self.dist_tol_px, ang_tol_deg=self.ang_tol_deg)
        m = len(pairs)

        # map raw match count to [0,1] smoothly (soft cap ~ 20 matches)
        score = float(1.0 - math.exp(-m / 10.0))
        return {"score": score, "matches": int(m), "theta": float(theta), "tx": float(t[0]), "ty": float(t[1]), "s": float(s)}
