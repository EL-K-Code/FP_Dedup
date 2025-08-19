#!/usr/bin/env python3
# Télécharge les poids FingerFlow dans ./models (GDrive/Dropbox).
import argparse, hashlib
from pathlib import Path
import requests, gdown
from tqdm import tqdm

GDRIVE = {
    "CoarseNet.h5": "1alvw_kAyY4sxdzAkGABQR7waux-rgJKm",
    "FineNet.h5":   "1wdGZKNNDAyN-fajjVKJoiyDtXAvl-4zq",
    "ClassifyNet_6_classes.h5": "1dfQDW8yxjmFPVu0Ddui2voxdngOrU3rc",
    "CoreNet.weights": "1v091s0eY4_VOLU9BqDXVSaZcFnA9qJPl",
    "VerifyNet-10.h5": "1cEz3oCYS4JCUiZxpU5o8lYesMOVgR0rt",
    "VerifyNet-14.h5": "1CI7z1r99AEV6Lrm2bQeGEFmVdQ8colUW",
    "VerifyNet-20.h5": "1lP1zDHTa7TemWPluv89ueFWCa95RnLF-",
    "VerifyNet-24.h5": "1h2RwuM1-mgiF4dfwslbgiI7-K8F4aw2A",
    "VerifyNet-30.h5": "1gQEzJKlCmUqe7Sx-W-6H1w1NGY8M98bX",
}
DROPBOX = {
    "CoarseNet.h5": "https://www.dropbox.com/s/gppil4wybdjcihy/CoarseNet.h5?dl=1",
    "FineNet.h5":   "https://www.dropbox.com/s/k7q2vs9255jf2dh/FineNet.h5?dl=1",
}

def download_gdrive(file_id: str, dst: Path):
    url = f"https://drive.google.com/uc?id={file_id}"
    tmp = str(dst) + ".part"
    gdown.download(url=url, output=tmp, quiet=False)
    Path(tmp).rename(dst)

def download_dropbox(url: str, dst: Path):
    if "dl=0" in url: url = url.replace("dl=0", "dl=1")
    tmp = str(dst) + ".part"
    with requests.get(url, stream=True, timeout=60) as r:
        r.raise_for_status()
        total = int(r.headers.get("content-length", 0))
        with open(tmp, "wb") as f, tqdm(total=total, unit="B", unit_scale=True, desc=dst.name) as p:
            for chunk in r.iter_content(chunk_size=1<<16):
                if chunk: f.write(chunk); p.update(len(chunk))
    Path(tmp).rename(dst)

def ensure_file(name: str, out_dir: Path, force=False, prefer_dropbox=False) -> Path:
    out = out_dir / name
    if out.exists() and not force:
        print(f"[skip] {name} existe déjà")
        return out
    out_dir.mkdir(parents=True, exist_ok=True)
    if prefer_dropbox and name in DROPBOX:
        print(f"[dl] {name} depuis Dropbox…")
        download_dropbox(DROPBOX[name], out)
    else:
        if name in GDRIVE:
            print(f"[dl] {name} depuis Google Drive…")
            download_gdrive(GDRIVE[name], out)
        elif name in DROPBOX:
            print(f"[dl] {name} depuis Dropbox…")
            download_dropbox(DROPBOX[name], out)
        else:
            raise ValueError(f"Pas d’URL connue pour {name}")
    return out

def main():
    ap = argparse.ArgumentParser(description="Télécharge les poids FingerFlow dans ./models")
    ap.add_argument("--models_dir", default="models")
    ap.add_argument("--only", choices=["all","extractor","matcher"], default="all")
    ap.add_argument("--precision", type=int, choices=[10,14,20,24,30])
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--prefer-dropbox", action="store_true")
    args = ap.parse_args()

    out = Path(args.models_dir)
    extractor = ["CoarseNet.h5","FineNet.h5","ClassifyNet_6_classes.h5","CoreNet.weights"]
    matcher = [f"VerifyNet-{p}.h5" for p in [10,14,20,24,30]]

    todo = []
    if args.only in ("all","extractor"): todo += extractor
    if args.only in ("all","matcher"):
        todo += [f"VerifyNet-{args.precision}.h5"] if args.precision else matcher

    for f in todo: ensure_file(f, out, force=args.force, prefer_dropbox=args.prefer_dropbox)
    print(f"[✓] Modèles dans: {out.resolve()}")

if __name__ == "__main__":
    main()
