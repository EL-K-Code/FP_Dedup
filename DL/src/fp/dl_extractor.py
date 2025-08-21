# src/fp/dl_extractor.py
from __future__ import annotations
from pathlib import Path
from typing import List, Dict, Optional, Union, Tuple
import json
from dataclasses import dataclass

import numpy as np
import pandas as pd
import cv2

# ---------- Shims compat (NumPy/SciPy/skimage) ----------
for _n, _t in (('int', int), ('bool', bool), ('float', float), ('complex', complex), ('object', object)):
    if _n not in np.__dict__:
        setattr(np, _n, _t)

try:
    import scipy.signal as _sig
    if "gaussian" not in _sig.__dict__:
        from scipy.signal.windows import gaussian as _gaussian
        _sig.gaussian = _gaussian
except Exception:
    pass

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

# ---------- TensorFlow (eager, sans XLA) ----------
import tensorflow as tf
tf.config.optimizer.set_jit(False)
tf.config.run_functions_eagerly(True)

from fingerflow.extractor import Extractor


# ===================== Modèles de données =====================

@dataclass
class Minutie:
    x: int
    y: int
    angle_deg: float          # [0,360)
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
    minutiae: List[Minutie]
    overlay_path: Optional[Path] = None

    def save_json_csv_xyt(self, out_dir: Union[str, Path], stem: str) -> None:
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


# ===================== Helpers I/O & visu =====================

def _load_any_as_gray_u8(img_or_path: Union[str, Path, np.ndarray]) -> Tuple[np.ndarray, bool]:
    """
    Charge n'importe quel input en Gray uint8.
    Retourne (gray_u8, loaded_from_path)
    """
    if isinstance(img_or_path, (str, Path)):
        img = cv2.imread(str(img_or_path), cv2.IMREAD_UNCHANGED)
        if img is None:
            raise ValueError(f"Impossible de lire: {img_or_path}")
        loaded_from_path = True
    else:
        arr = np.asarray(img_or_path)
        if arr.ndim == 2:         # déjà gray
            img = arr
        elif arr.ndim == 3:
            if arr.shape[2] == 4: # BGRA/RGBA
                img = cv2.cvtColor(arr.astype(np.uint8), cv2.COLOR_BGRA2BGR)
                img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            else:                 # BGR/RGB
                # On ne sait pas l'ordre; on passe par BGR2GRAY qui est robuste
                img = cv2.cvtColor(arr.astype(np.uint8), cv2.COLOR_BGR2GRAY)
        else:
            raise ValueError(f"Format d’image non supporté: shape={arr.shape}, dtype={arr.dtype}")
        loaded_from_path = False

    # 16-bit -> uint8
    if img.dtype != np.uint8:
        img = cv2.normalize(img, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    if img.ndim == 3:  # au cas improbable
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    return img, loaded_from_path


def _angle_to_img_deg(a: float) -> float:
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


# --------------- Prétraitement FingerFlow-friendly ----------------

def _prep_rgb_for_ff(gray_u8: np.ndarray,
                     min_short: int = 320,
                     pad_mode: int = cv2.BORDER_REFLECT_101) -> Tuple[np.ndarray, float, float, int, int]:
    """
    - CLAHE
    - Upscale si côté court < min_short (interp bicubique)
    - Padding fin de dimension pour multiple de 8 (bas/droite)
    - Convertit en RGB float32 [0,1]
    Retourne: (rgb, sx, sy, pad_right, pad_bottom)
    avec sx, sy = facteurs d'échelle (new/old).
    """
    # CLAHE
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    g = clahe.apply(gray_u8)

    H0, W0 = g.shape
    short = min(H0, W0)
    sx = sy = 1.0
    if short < min_short:
        scale = float(min_short) / float(short)
        W1, H1 = int(round(W0*scale)), int(round(H0*scale))
        g = cv2.resize(g, (W1, H1), interpolation=cv2.INTER_CUBIC)
        sx = float(W1) / float(W0)
        sy = float(H1) / float(H0)

    # padding multiple de 8
    H, W = g.shape
    H8, W8 = int(np.ceil(H/8.0)*8), int(np.ceil(W/8.0)*8)
    pad_bottom = H8 - H
    pad_right  = W8 - W
    if pad_bottom or pad_right:
        g = cv2.copyMakeBorder(g, 0, pad_bottom, 0, pad_right, pad_mode)

    rgb = cv2.cvtColor(g, cv2.COLOR_GRAY2RGB).astype(np.float32) / 255.0
    return rgb, sx, sy, pad_right, pad_bottom


def _maybe_invert(gray_u8: np.ndarray) -> np.ndarray:
    # Heuristique simple basé sur les bords
    H, W = gray_u8.shape
    bg = np.median(np.concatenate([
        gray_u8[:max(1,H//8), :].ravel(),
        gray_u8[-max(1,H//8):, :].ravel(),
        gray_u8[:, :max(1,W//8)].ravel(),
        gray_u8[:, -max(1,W//8):].ravel()
    ]))
    med = np.median(gray_u8)
    return gray_u8 if med < bg else (255 - gray_u8)


# --------------- Normalisation de la sortie FF -------------------

def _as_minutiae_dataframe(res) -> pd.DataFrame:
    """Uniformise la sortie Extractor (dict/obj/DataFrame) en DataFrame."""
    if hasattr(res, "minutiae"):
        df = res.minutiae
    elif isinstance(res, dict):
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


# ===================== Extracteur principal =====================

class DLMinutiaeExtractor:
    """
    DL minutiae extractor (FingerFlow backend).
    Usage:
        extr = DLMinutiaeExtractor(coarse=..., fine=..., classify=..., core=...)
        res = extr.extract(img_or_path, out_dir="out", stem="my_image")
    """
    def __init__(self, coarse: str, fine: str, classify: str, core: str):
        self.extractor = Extractor(coarse, fine, classify, core)

    def extract(self,
                img_or_path: Union[str, Path, np.ndarray],
                out_dir: Union[str, Path],
                stem: Optional[str] = None,
                save_overlay: bool = True,
                max_points_overlay: int = 300,
                min_short: int = 320) -> ExtractionResult:

        # 0) Robust load -> grayscale uint8 (handles paths or arrays, 8/16-bit, RGB/BGRA)
        gray0, _ = _load_any_as_gray_u8(img_or_path)
        H0, W0 = gray0.shape
        if stem is None:
            stem = "image"

        # Ensure per-image directory: out/<stem>/
        out_root = Path(out_dir)
        out_img_dir = out_root / stem
        out_img_dir.mkdir(parents=True, exist_ok=True)

        # 1) Preprocess (CLAHE + optional upscale + pad to /8 + RGB float32 [0,1])
        rgb, sx, sy, pad_r, pad_b = _prep_rgb_for_ff(gray0, min_short=min_short)

        # 2) First pass inference
        res_ff = self.extractor.extract_minutiae(rgb)
        df = _as_minutiae_dataframe(res_ff)

        # 3) Fallback: try inverted if zero detections
        if len(df) == 0:
            gray_inv = _maybe_invert(gray0)
            rgb, sx, sy, pad_r, pad_b = _prep_rgb_for_ff(gray_inv, min_short=min_short)
            res_ff = self.extractor.extract_minutiae(rgb)
            df = _as_minutiae_dataframe(res_ff)

        # 4) Convert to List[Minutie], remap coordinates back to original size
        mins: List[Minutie] = []
        for _, r in df.iterrows():
            x_p = float(r.get("x", r.get("col", r.get("cx", r.get("u", 0.0)))))
            y_p = float(r.get("y", r.get("row", r.get("cy", r.get("v", 0.0)))))
            ang_val = r.get("angle", r.get("theta", r.get("direction", r.get("theta_img_deg", 0.0))))
            score = float(r.get("score", r.get("confidence", r.get("prob", r.get("probability", 0.0)))))
            cls = str(r.get("class", r.get("type", r.get("label", ""))))

            # We upscaled isotropically and padded bottom/right only.
            x0 = int(np.clip(round(x_p / sx), 0, W0 - 1))
            y0 = int(np.clip(round(y_p / sy), 0, H0 - 1))

            mins.append(
                Minutie(
                    x=x0,
                    y=y0,
                    angle_deg=float(_angle_to_img_deg(float(ang_val))),
                    score=score,
                    classe=cls
                )
            )

        # 5) Save outputs inside out/<stem>/
        result = ExtractionResult(minutiae=mins, overlay_path=None)
        result.save_json_csv_xyt(out_img_dir, stem)
        if save_overlay:
            result.overlay_path = _draw_overlay(
                gray0, mins, out_img_dir / f"{stem}_overlay.png",
                max_points=max_points_overlay
            )
        return result
