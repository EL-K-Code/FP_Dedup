# -*- coding: utf-8 -*-
"""
Extraction de minuties après enhancement et thinning (sans NBIS).
- Enhancement: Gabor (fingerprint_enhancer si dispo) sinon CLAHE
- Binarisation + segmentation légère
- Thinning: skeletonize (Zhang-Suen)
- Minuties: terminasons & bifurcations via comptage de voisins (8-connexité)
- Orientation: suivi local du squelette (vecteur directionnel)
- Post-traitement: filtrage bord, dédoublonnage par distance, masque

API principale:
    detect_minutiae(img_or_path, params=...) -> dict avec:
        {
          "minutiae": [
             {"type":"ending"/"bifurcation","x":int,"y":int,
              "theta_rad":float,"theta_deg":float,"quality":float},
             ...
          ],
          "overlay": np.ndarray BGR (si requested),
          "skeleton": np.ndarray bool,
          "binary": np.ndarray uint8 (0/255),
        }
"""

from __future__ import annotations
from dataclasses import dataclass
from typing import Optional, Tuple, List, Dict, Union
from pathlib import Path

import numpy as np
import cv2
from skimage.morphology import skeletonize

# ---------------------------- Enhancement ----------------------------
_HAS_FPE = False
try:
    from fingerprint_enhancer import enhance_Fingerprint
    _HAS_FPE = True
except Exception:
    _HAS_FPE = False

def enhance_image(gray_u8: np.ndarray, mode: str = "auto") -> np.ndarray:
    """Enhancement simple: Gabor si lib dispo, sinon CLAHE + léger lissage."""
    if mode == "none":
        return gray_u8
    if mode in ("auto", "gabor") and _HAS_FPE:
        try:
            return enhance_Fingerprint(gray_u8)
        except Exception:
            pass
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    out = clahe.apply(gray_u8)
    out = cv2.GaussianBlur(out, (3,3), 0)
    return out

# ---------------------------- Utils I/O ----------------------------
def _to_gray_u8(img_or_path: Union[str, Path, np.ndarray]) -> np.ndarray:
    if isinstance(img_or_path, (str, Path)):
        g = cv2.imread(str(img_or_path), cv2.IMREAD_GRAYSCALE)
        if g is None:
            raise ValueError(f"Impossible de lire: {img_or_path}")
        return g
    arr = np.asarray(img_or_path)
    if arr.ndim == 3:
        arr = cv2.cvtColor(arr, cv2.COLOR_BGR2GRAY)
    if arr.dtype != np.uint8:
        arr = cv2.normalize(arr, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    return arr

# ---------------------------- Binarisation / Segmentation ----------------------------
def binarize_and_segment(enh_u8: np.ndarray, invert_auto: bool = True) -> Tuple[np.ndarray, np.ndarray]:
    # Otsu
    th, bin0 = cv2.threshold(enh_u8, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    # Inversion auto pour avoir crêtes=255
    if invert_auto:
        if np.mean(enh_u8[bin0 == 255]) > np.mean(enh_u8[bin0 == 0]):
            bin0 = 255 - bin0

    # Nettoyage morpho
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3,3))
    binary = cv2.morphologyEx(bin0, cv2.MORPH_OPEN, k, iterations=1)

    # Masque doigt (fermeture + plus grand composant)
    mask = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, k, iterations=2)
    # ⬇️  cast en uint8 pour OpenCV
    nb, labels = cv2.connectedComponents((mask > 0).astype(np.uint8))
    if nb > 1:
        counts = [(labels == i).sum() for i in range(1, nb)]
        i_max = 1 + int(np.argmax(counts))
        mask = np.where(labels == i_max, 255, 0).astype(np.uint8)
    else:
        mask = ((mask > 0).astype(np.uint8)) * 255

    return binary, mask

# ---------------------------- Thinning ----------------------------
def to_skeleton(binary_u8: np.ndarray) -> np.ndarray:
    """Convertit binaire (crêtes=255) -> squelette booléen."""
    skel = skeletonize((binary_u8 > 0)).astype(np.uint8)  # 0/1
    return skel

# ---------------------------- Minutiae detection ----------------------------
NB_KERNEL = np.array([[1,1,1],
                      [1,10,1],   # on met 10 au centre juste pour le préserver
                      [1,1,1]], dtype=np.uint8)

def neighbor_count(skel01: np.ndarray) -> np.ndarray:
    """Renvoie, pour chaque pixel squelette=1, le nombre de voisins 8-connexes."""
    conv = cv2.filter2D(skel01, -1, NB_KERNEL, borderType=cv2.BORDER_CONSTANT)
    # retirer la "valeur centre" (10) et garder uniquement la somme des voisins
    counts = conv - (skel01 * 10)
    return counts

def _follow_direction(skel01: np.ndarray, y: int, x: int, max_steps: int = 20, stop_on_branch=True) -> Tuple[float, float, int]:
    """
    Suit le squelette depuis (y,x) pour estimer la direction:
    - renvoie (dx, dy, steps_effectifs)
    - s'arrête si rencontre bifurcation (>=3 voisins) si stop_on_branch=True
    """
    H, W = skel01.shape
    prev = (y, x)
    # voisins 8 directions
    nbrs = [(-1,-1),(-1,0),(-1,1),(0,1),(1,1),(1,0),(1,-1),(0,-1)]
    # premier voisin (unique pour une terminaison)
    vs = []
    for dy, dx in nbrs:
        yy, xx = y+dy, x+dx
        if 0 <= yy < H and 0 <= xx < W and skel01[yy, xx]:
            vs.append((yy, xx))
    if not vs:
        return 0.0, 0.0, 0
    # choisir un voisin de départ
    cur = vs[0]
    steps = 0
    while steps < max_steps:
        steps += 1
        # chercher le prochain voisin (différent de prev)
        y0, x0 = cur
        # si bifurcation au milieu, on s'arrête si demandé
        deg = int(neighbor_count(skel01)[y0, x0])
        if stop_on_branch and deg >= 3 and steps > 1:
            break
        nxt = None
        cnt = 0
        for dy, dx in nbrs:
            yy, xx = y0+dy, x0+dx
            if 0 <= yy < H and 0 <= xx < W and skel01[yy, xx]:
                if (yy, xx) != prev:
                    nxt = (yy, xx)
                    cnt += 1
        if nxt is None or cnt == 0:
            prev, cur = cur, cur
            break
        prev, cur = cur, nxt
    dx = float(cur[1] - x)
    dy = float(cur[0] - y)
    return dx, dy, steps

def _angle_from_vec(dx: float, dy: float) -> float:
    # angle image: x vers la droite, y vers le bas -> atan2(dy, dx)
    return float(np.arctan2(dy, dx))

def extract_minutiae_from_skeleton(skel01: np.ndarray,
                                   mask_u8: Optional[np.ndarray] = None,
                                   border: int = 8,
                                   dedupe_radius: int = 10,
                                   dir_trace_steps: int = 18) -> List[Dict]:
    """
    Détecte terminasons (deg==1) et bifurcations (deg>=3) puis estime l'orientation.
    - mask_u8: si fourni, on garde seulement les minuties dans la zone doigt
    - border: supprime celles trop près du bord
    - dedupe_radius: NMS par distance (garde la meilleure 'quality')
    """
    H, W = skel01.shape
    counts = neighbor_count(skel01)

    # candidates
    ys, xs = np.where(skel01 > 0)
    items = []
    for y, x in zip(ys, xs):
        deg = int(counts[y, x])
        if deg == 1:
            dx, dy, steps = _follow_direction(skel01, y, x, max_steps=dir_trace_steps, stop_on_branch=True)
            if steps == 0: 
                continue
            theta = _angle_from_vec(dx, dy)
            quality = float(steps)
            items.append({"type":"ending","x":x,"y":y,"theta_rad":theta,"theta_deg":np.degrees(theta),"quality":quality})
        elif deg >= 3:
            # mesure simple: direction moyenne des 3 premières branches
            nbrs = [(-1,-1),(-1,0),(-1,1),(0,1),(1,1),(1,0),(1,-1),(0,-1)]
            dirs = []
            for dy0, dx0 in nbrs:
                yy, xx = y+dy0, x+dx0
                if 0 <= yy < H and 0 <= xx < W and skel01[yy, xx]:
                    dx, dy, steps = _follow_direction(skel01, y, x, max_steps=dir_trace_steps, stop_on_branch=False)
                    if steps > 0:
                        dirs.append((dx, dy, steps))
                    # on ne double pas le suivi par voisin ici pour rester rapide
                    break
            if not dirs:
                continue
            # orientation représentative
            dxm = np.mean([d[0] for d in dirs]); dym = np.mean([d[1] for d in dirs])
            theta = _angle_from_vec(dxm, dym)
            quality = float(np.mean([d[2] for d in dirs]))
            items.append({"type":"bifurcation","x":x,"y":y,"theta_rad":theta,"theta_deg":np.degrees(theta),"quality":quality})

    # filtrages
    out = []
    for m in items:
        if m["x"] < border or m["x"] >= W-border or m["y"] < border or m["y"] >= H-border:
            continue
        if mask_u8 is not None and mask_u8[m["y"], m["x"]] == 0:
            continue
        out.append(m)

    # dédoublonnage spatial (NMS)
    if dedupe_radius > 0 and out:
        out = _nms_by_distance(out, dedupe_radius)

    return out

def _nms_by_distance(points: List[Dict], radius: int) -> List[Dict]:
    """Regroupe les minuties trop proches; garde la meilleure 'quality'."""
    pts = np.array([[p["x"], p["y"]] for p in points], dtype=np.float32)
    kept = []
    used = np.zeros(len(points), dtype=bool)
    for i in np.argsort([-p.get("quality", 0.0) for p in points]):
        if used[i]: 
            continue
        kept.append(points[i])
        d = np.linalg.norm(pts - pts[i], axis=1)
        used |= (d <= radius)
    return kept

# ---------------------------- Overlay helper ----------------------------
def draw_overlay(gray_u8: np.ndarray, minutiae: List[Dict], max_draw: int = 400) -> np.ndarray:
    """BGR avec points + vecteurs d'orientation (verts=endings, rouges=bifurcations)."""
    vis = cv2.cvtColor(gray_u8, cv2.COLOR_GRAY2BGR)
    L = 12
    mins = sorted(minutiae, key=lambda m: -m.get("quality", 0.0))[:max_draw]
    for m in mins:
        x, y = int(m["x"]), int(m["y"])
        th = float(m["theta_rad"])
        x2 = int(round(x + L*np.cos(th)))
        y2 = int(round(y + L*np.sin(th)))
        color = (0,255,0) if m["type"]=="ending" else (0,0,255)
        cv2.circle(vis, (x,y), 2, color, -1)
        cv2.line(vis, (x,y), (x2,y2), color, 1)
    return vis

# ---------------------------- Pipeline haut-niveau ----------------------------
@dataclass
class DetectParams:
    enhance_mode: str = "auto"         # "auto"|"gabor"|"clahe"|"none"
    resize_to: Optional[Tuple[int,int]] = (512, 512)  # recommandé
    invert_auto: bool = True
    border: int = 8
    dedupe_radius: int = 10
    dir_trace_steps: int = 18
    return_overlay: bool = True

def detect_minutiae(img_or_path: Union[str, Path, np.ndarray],
                    params: DetectParams = DetectParams()) -> Dict:
    """Renvoie minuties + images intermédiaires pour debug/visualisation."""
    gray = _to_gray_u8(img_or_path)
    H0, W0 = gray.shape[:2]

    # enhancement
    enh = enhance_image(gray, params.enhance_mode)

    # resize (pour stabiliser le squelette)
    if params.resize_to is not None:
        enh_r = cv2.resize(enh, params.resize_to[::-1], interpolation=cv2.INTER_LINEAR)
    else:
        enh_r = enh

    # binarisation/segmentation
    binary, mask = binarize_and_segment(enh_r, invert_auto=params.invert_auto)

    # thinning
    skel01 = to_skeleton(binary)

    # minuties dans l'espace redimensionné
    mins = extract_minutiae_from_skeleton(
        skel01,
        mask_u8=mask,
        border=params.border,
        dedupe_radius=params.dedupe_radius,
        dir_trace_steps=params.dir_trace_steps
    )

    # remap coords vers taille originale si resize
    if params.resize_to is not None:
        sx = W0 / float(params.resize_to[0]); sy = H0 / float(params.resize_to[1])
        for m in mins:
            m["x"] = int(round(m["x"] * sx))
            m["y"] = int(round(m["y"] * sy))

    # overlay
    overlay = draw_overlay(gray, mins) if params.return_overlay else None

    return {
        "minutiae": mins,
        "overlay": overlay,
        "skeleton": (skel01 * 255).astype(np.uint8),
        "binary": binary
    }
