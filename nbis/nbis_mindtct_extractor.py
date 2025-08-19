# nbis_mindtct_extractor.py
from __future__ import annotations
import subprocess, tempfile, shutil, json, csv
from pathlib import Path
from typing import List, Tuple, Optional, Union

import cv2
import numpy as np

# ---------------- Enhancement ----------------
_HAS_FPE = False
try:
    from fingerprint_enhancer import enhance_Fingerprint
    _HAS_FPE = True
except Exception:
    _HAS_FPE = False

def enhance_image(
    img_or_path: Union[str, Path, np.ndarray],
    mode: str = "auto",                  # "auto" | "gabor" | "clahe" | "none"
    clahe_clip: float = 2.0,
    clahe_grid: Tuple[int,int] = (8,8),
) -> np.ndarray:
    """Retourne une image 8-bit (H,W) améliorée."""
    if isinstance(img_or_path, (str, Path)):
        img = cv2.imread(str(img_or_path), cv2.IMREAD_GRAYSCALE)
        if img is None:
            raise ValueError(f"Impossible de lire: {img_or_path}")
    else:
        img = np.asarray(img_or_path)
        if img.ndim == 3:  # RGB/BGR -> Gray
            img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        if img.dtype != np.uint8:
            img = cv2.normalize(img, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)

    if mode == "none":
        return img

    if mode in ("auto", "gabor") and _HAS_FPE:
        try:
            return enhance_Fingerprint(img)
        except Exception:
            pass  # fallback CLAHE

    clahe = cv2.createCLAHE(clipLimit=clahe_clip, tileGridSize=clahe_grid)
    out = clahe.apply(img)
    out = cv2.GaussianBlur(out, (3,3), 0)
    return out


# ---------------- NBIS helpers ----------------
def _check_bin(name: str):
    try:
        subprocess.run([name, "-h"], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=False)
    except FileNotFoundError:
        raise RuntimeError(f"Binaire '{name}' introuvable. Installe NBIS et ajoute-le au PATH.")

def _run_mindtct(enhanced_png: Path, out_base: Path, extra_args: Optional[List[str]] = None) -> Path:
    """Exécute mindtct(enhanced_png, out_base) et renvoie le chemin .xyt."""
    _check_bin("mindtct")
    cmd = ["mindtct"]
    if extra_args:
        cmd += list(extra_args)
    cmd += [str(enhanced_png), str(out_base)]
    res = subprocess.run(cmd, capture_output=True, text=True)
    if res.returncode != 0:
        raise RuntimeError(f"mindtct a échoué:\n{res.stderr}")

    xyt = out_base.with_suffix(".xyt")
    if not xyt.exists():
        raise FileNotFoundError(
            f"{xyt.name} introuvable. Ta version NBIS requiert peut-être un flag (ex: extra_args=['-m1'])."
        )
    return xyt

def _parse_xyt(xyt_path: Path) -> List[dict]:
    """
    Lit un .xyt : lignes 'x y angle [quality]'.
    Angle = convention NBIS (0°=droite, 90°=HAUT). Qualité facultative.
    """
    mins = []
    with open(xyt_path, "r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            t = line.strip().split()
            if len(t) < 3:
                continue
            item = {"x": int(float(t[0])), "y": int(float(t[1])), "theta_nbis_deg": float(t[2])}
            if len(t) >= 4:
                try: item["quality"] = float(t[3])
                except: pass
            mins.append(item)
    return mins

# Conversion d’angle NBIS -> repère image (0°=→, 90°=↓)
def nbis_to_image_deg(theta_nbis: float) -> float:
    # NBIS: 0=→, 90=↑ ; Image: 0=→, 90=↓
    # => inversion verticale : theta_img = (360 - theta_nbis) % 360
    return (360.0 - float(theta_nbis)) % 360.0

# ---------------- Overlay ----------------
def _draw_overlay(gray_u8: np.ndarray, minutiae: List[dict], out_path: Path, max_points: int = 300):
    img = cv2.cvtColor(gray_u8, cv2.COLOR_GRAY2BGR)
    ms = minutiae
    if ms and "quality" in ms[0]:
        ms = sorted(ms, key=lambda m: -m.get("quality", 0.0))
    ms = ms[:max_points]
    L = 12
    for m in ms:
        x, y = int(m["x"]), int(m["y"])
        th_img = m.get("theta_img_deg")
        if th_img is None:
            th_img = nbis_to_image_deg(m["theta_nbis_deg"])
        rad = np.deg2rad(th_img)
        x2 = int(round(x + L*np.cos(rad)))
        y2 = int(round(y + L*np.sin(rad)))
        cv2.circle(img, (x, y), 2, (0, 255, 0), -1)
        cv2.line(img, (x, y), (x2, y2), (0, 0, 255), 1)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(out_path), img)

# ---------------- API principale ----------------
def extract_minutiae_mindtct(
    img_or_path: Union[str, Path, np.ndarray],
    out_dir: Union[str, Path],
    enhance_mode: str = "auto",                   # "auto" | "gabor" | "clahe" | "none"
    resize_to: Optional[Tuple[int,int]] = (512, 512),
    mindtct_args: Optional[List[str]] = None,     # ex: ["-m1"]
    save_overlay: bool = True,
    stem: Optional[str] = None
) -> dict:
    """
    Enhancement -> MINDTCT -> parsing -> conversion d’angles -> exports.
    Retourne un dict:
      {
        "minutiae": [{"x","y","theta_img_deg","theta_nbis_deg","theta_img_rad","quality"?}, ...],
        "xyt_path": Path,
        "overlay_path": Path|None
      }
    Les coordonnées (x,y) sont dans l'IMAGE D’ORIGINE (remappage si resize).
    """
    out_dir = Path(out_dir); out_dir.mkdir(parents=True, exist_ok=True)

    # lecture image d'origine
    if isinstance(img_or_path, (str, Path)):
        orig = cv2.imread(str(img_or_path), cv2.IMREAD_GRAYSCALE)
        if orig is None:
            raise ValueError(f"Impossible de lire: {img_or_path}")
        if stem is None:
            stem = Path(img_or_path).stem
    else:
        arr = np.asarray(img_or_path)
        if arr.ndim == 3:
            arr = cv2.cvtColor(arr, cv2.COLOR_BGR2GRAY)
        if arr.dtype != np.uint8:
            arr = cv2.normalize(arr, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
        orig = arr
        if stem is None: stem = "image"

    H0, W0 = orig.shape[:2]

    # enhancement
    enh = enhance_image(orig, mode=enhance_mode)

    # resize (souvent 512x512 pour stabiliser MINDTCT)
    if resize_to is not None:
        enh_for_nbis = cv2.resize(enh, (resize_to[0], resize_to[1]), interpolation=cv2.INTER_LINEAR)
        Hw, Ww = resize_to[1], resize_to[0]
    else:
        enh_for_nbis = enh
        Hw, Ww = H0, W0

    tmp = Path(tempfile.mkdtemp(prefix="nbis_"))
    overlay_path = None
    try:
        enh_png = tmp / "enh.png"
        cv2.imwrite(str(enh_png), enh_for_nbis)

        out_base = out_dir / stem

        # 1er essai
        try:
            xyt_path = _run_mindtct(enh_png, out_base, mindtct_args)
        except FileNotFoundError:
            # si .xyt manquant, on retente avec -m1
            if not mindtct_args or "-m1" not in mindtct_args:
                xyt_path = _run_mindtct(enh_png, out_base, (mindtct_args or []) + ["-m1"])
            else:
                raise

        mins = _parse_xyt(xyt_path)

        # remap coords vers image d'origine si resize
        if resize_to is not None:
            sx = W0 / float(Ww); sy = H0 / float(Hw)
            for m in mins:
                m["x"] = int(round(m["x"] * sx))
                m["y"] = int(round(m["y"] * sy))

        # angles convertis pour l'overlay & usage image
        for m in mins:
            th_img = nbis_to_image_deg(m["theta_nbis_deg"])
            m["theta_img_deg"] = th_img
            m["theta_img_rad"] = float(np.deg2rad(th_img))

        # exports JSON / CSV (en plus du .xyt de NBIS)
        with open(out_dir / f"{stem}.json", "w", encoding="utf-8") as f:
            json.dump(mins, f, ensure_ascii=False, indent=2)

        with open(out_dir / f"{stem}.csv", "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=["x","y","theta_img_deg","theta_img_rad","theta_nbis_deg","quality"])
            w.writeheader()
            for m in mins:
                w.writerow({
                    "x": m["x"], "y": m["y"],
                    "theta_img_deg": m["theta_img_deg"],
                    "theta_img_rad": m["theta_img_rad"],
                    "theta_nbis_deg": m["theta_nbis_deg"],
                    "quality": m.get("quality","")
                })

        if save_overlay:
            overlay_path = out_dir / f"{stem}_overlay.png"
            _draw_overlay(orig, mins, overlay_path)

        return {"minutiae": mins, "xyt_path": xyt_path, "overlay_path": overlay_path}
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
