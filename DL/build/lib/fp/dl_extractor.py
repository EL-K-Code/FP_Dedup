# src/fp/dl_extractor.py
from __future__ import annotations
from pathlib import Path
from typing import List, Dict, Optional, Union

import json
from dataclasses import dataclass

import numpy as np
import pandas as pd
import cv2

# --- Shims de compatibilité (NumPy/SciPy/scikit-image) -----------------------
# np.int/np.bool/np.object retirés en NumPy >=1.24 -> recrée les alias si absents
for _n, _t in (('int', int), ('bool', bool), ('float', float), ('complex', complex), ('object', object)):
    if _n not in np.__dict__:
        setattr(np, _n, _t)

# SciPy >=2 a déplacé gaussian -> assure l'alias
try:
    import scipy.signal as _sig
    if "gaussian" not in _sig.__dict__:
        from scipy.signal.windows import gaussian as _gaussian
        _sig.gaussian = _gaussian
except Exception:
    pass

# scikit-image >=0.20 a remplacé multichannel -> channel_axis
try:
    import skimage.filters as _filters
    _gauss_orig = _filters.gaussian
    def _gaussian_compat(image, sigma, *args, **kwargs):
        if "multichannel" in kwargs:
            mc = kwargs.pop("multichannel")
            kwargs.setdefault("channel_axis", (-1 if mc else None))
        return _gauss_orig(image, sigma, *args, **kwargs)
    _filters.gaussian = _gaussian_compat
except Exception:
    pass

# --- TensorFlow: éviter fusion BN+Mish (CPU) ---------------------------------
import tensorflow as tf
tf.config.optimizer.set_jit(False)        # pas de XLA JIT
tf.config.run_functions_eagerly(True)     # exécution eager

from fingerflow.extractor import Extractor


# ============================== Modèles de données ===========================

@dataclass
class Minutie:
    """Représente une minutie normalisée (repère image)."""
    x: int
    y: int
    angle_deg: float          # orientation en degrés [0,360), 0° →, 90° ↓
    score: float = 0.0
    classe: str = ""          # "ending", "bifurcation", ...

    @property
    def angle_rad(self) -> float:
        return float(np.deg2rad(self.angle_deg))

    def to_dict(self) -> Dict:
        return {
            "x": int(self.x),
            "y": int(self.y),
            "theta_img_deg": float(self.angle_deg),
            "theta_img_rad": float(self.angle_rad),
            "score": float(self.score),
            "class": str(self.classe),
        }


@dataclass
class ExtractionResult:
    """Résultat d'extraction DL."""
    minutiae: List[Minutie]
    overlay_path: Optional[Path] = None

    def save_json_csv_xyt(self, out_dir: Union[str, Path], stem: str) -> None:
        """Exporte JSON / CSV / XYT (x y angle_deg)."""
        out = Path(out_dir); out.mkdir(parents=True, exist_ok=True)
        # JSON
        with open(out / f"{stem}.json", "w", encoding="utf-8") as f:
            json.dump([m.to_dict() for m in self.minutiae], f, ensure_ascii=False, indent=2)
        # CSV
        import csv
        fields = ["x","y","theta_img_deg","theta_img_rad","score","class"]
        with open(out / f"{stem}.csv", "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=fields); w.writeheader()
            for m in self.minutiae:
                w.writerow(m.to_dict())
        # XYT
        with open(out / f"{stem}.xyt", "w", encoding="utf-8") as f:
            for m in self.minutiae:
                f.write(f"{m.x} {m.y} {m.angle_deg:.2f}\n")


# ============================== Helpers I/O & visu ===========================

def _ensure_bgr(img: np.ndarray) -> np.ndarray:
    """Normalise une image numpy en BGR uint8 (accepte gray/float)."""
    arr = np.asarray(img)
    if arr.dtype != np.uint8:
        if arr.max() <= 1.0:
            arr = (arr * 255.0).round().astype(np.uint8)
        else:
            arr = np.clip(arr, 0, 255).astype(np.uint8)
    if arr.ndim == 2:  # grayscale
        return cv2.cvtColor(arr, cv2.COLOR_GRAY2BGR)
    if arr.ndim == 3 and arr.shape[2] == 3:
        return arr  # on assume BGR/RGB – FingerFlow est tolérant ici
    raise ValueError(f"Format d’image non supporté (shape={arr.shape}, dtype={arr.dtype})")

def _angle_to_img_deg(a: float) -> float:
    # accepte radians ou degrés, normalise en repère image
    a = float(a)
    deg = np.degrees(a) if abs(a) <= 2*np.pi else a
    return (deg + 360.0) % 360.0

def _draw_overlay(gray: np.ndarray, mins: List[Minutie], out_path: Path, max_points: int = 300):
    vis = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
    L = 12
    ms = sorted(mins, key=lambda m: -float(getattr(m, "score", 0.0)))[:max_points]
    for m in ms:
        x, y = int(m.x), int(m.y)
        th = np.deg2rad(float(m.angle_deg))
        x2 = int(round(x + L*np.cos(th)))
        y2 = int(round(y + L*np.sin(th)))
        cv2.circle(vis, (x, y), 2, (0, 255, 0), -1)
        cv2.line(vis, (x, y), (x2, y2), (0, 0, 255), 1)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(out_path), vis)
    return out_path


# ------------------------ Normalisation de la sortie FF ----------------------

def _as_minutiae_dataframe(res) -> pd.DataFrame:
    """Normalise la sortie FingerFlow en pandas.DataFrame."""
    if hasattr(res, "minutiae"):              # objet avec .minutiae
        df = res.minutiae
    elif isinstance(res, dict):               # dict avec différentes clés
        df = None
        for k in ("minutiae", "minutiae_df", "points", "minutiae_points"):
            if k in res:
                df = res[k]; break
        if df is None:
            raise ValueError(f"Sortie FingerFlow inattendue (keys={list(res.keys())})")
    else:
        df = res

    if isinstance(df, pd.DataFrame):
        return df

    if isinstance(df, (list, tuple)):
        if not df:
            return pd.DataFrame(columns=["x","y","angle","score","class"])
        if isinstance(df[0], dict):
            return pd.DataFrame(df)
        arr = np.asarray(df)
        cols = ["x","y","angle","score","class"][: (arr.shape[1] if arr.ndim > 1 else 1)]
        return pd.DataFrame(arr, columns=cols)

    if isinstance(df, np.ndarray):
        cols = ["x","y","angle","score","class"][: (df.shape[1] if df.ndim > 1 else 1)]
        return pd.DataFrame(df, columns=cols)

    if hasattr(df, "iterrows"):
        return df

    raise TypeError(f"Impossible de convertir la sortie FingerFlow ({type(df)}) en DataFrame")


# ============================== Extracteur principal =========================

class DLMinutiaeExtractor:
    """
    Extracteur DL (FingerFlow) prêt à l'emploi.
    Usage:
        extr = DLMinutiaeExtractor(
            coarse="models/CoarseNet.h5",
            fine="models/FineNet.h5",
            classify="models/ClassifyNet_6_classes.h5",
            core="models/CoreNet.weights"
        )
        res = extr.extract(img_np, out_dir="out", stem="mon_image")
        # res est un ExtractionResult -> res.minutiae == List[Minutie]
    """
    def __init__(self, coarse: str, fine: str, classify: str, core: str):
        self.extractor = Extractor(coarse, fine, classify, core)

    def extract(self,
                img: np.ndarray,                  # ← objet image uniquement (np.ndarray)
                out_dir: Union[str, Path],
                stem: Optional[str] = None,
                save_overlay: bool = True,
                max_points_overlay: int = 300) -> ExtractionResult:
        # normalise l'image
        img_bgr = _ensure_bgr(img)
        gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)

        out_dir = Path(out_dir); out_dir.mkdir(parents=True, exist_ok=True)
        if stem is None:
            stem = "image"

        # --- inference FingerFlow
        res_ff = self.extractor.extract_minutiae(img_bgr)
        df = _as_minutiae_dataframe(res_ff)

        # --- to List[Minutie]
        mins: List[Minutie] = []
        for _, r in df.iterrows():
            x = int(r.get("x", r.get("col", r.get("cx", r.get("u", 0)))))
            y = int(r.get("y", r.get("row", r.get("cy", r.get("v", 0)))))
            ang_val = r.get("angle", r.get("theta", r.get("direction", r.get("theta_img_deg", 0.0))))
            score = float(r.get("score", r.get("confidence", r.get("prob", r.get("probability", 0.0)))))
            cls = str(r.get("class", r.get("type", r.get("label", ""))))
            mins.append(
                Minutie(
                    x=x,
                    y=y,
                    angle_deg=float(_angle_to_img_deg(float(ang_val))),
                    score=score,
                    classe=cls
                )
            )

        # --- exports + overlay
        result = ExtractionResult(minutiae=mins, overlay_path=None)
        result.save_json_csv_xyt(out_dir, stem)

        if save_overlay:
            result.overlay_path = _draw_overlay(gray, mins, out_dir / f"{stem}_overlay.png",
                                                max_points=max_points_overlay)
        return result
