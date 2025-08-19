# scripts/run_extract.py
import argparse
from pathlib import Path
from nbis_mindtct_extractor import extract_minutiae_mindtct

def main():
    ap = argparse.ArgumentParser(description="Enhance + MINDTCT + exports")
    ap.add_argument("image", help="Chemin image (PNG/JPG/TIF...)")
    ap.add_argument("--out_dir", default="out", help="Dossier de sortie")
    ap.add_argument("--enhance", default="auto", choices=["auto","gabor","clahe","none"])
    ap.add_argument("--size", default="512x512", help="Taille pour NBIS (ex: 512x512, ou 'none')")
    ap.add_argument("--mindtct_args", default="", help="Args passés à mindtct (ex: '-m1')")
    ap.add_argument("--no_overlay", action="store_true", help="Ne pas générer l'overlay")
    args = ap.parse_args()

    resize = None if args.size.lower()=="none" else tuple(map(int, args.size.lower().split("x")))
    flags = [a for a in args.mindtct_args.split() if a]

    res = extract_minutiae_mindtct(
        args.image,
        out_dir=args.out_dir,
        enhance_mode=args.enhance,
        resize_to=resize,
        mindtct_args=flags or None,
        save_overlay=not args.no_overlay
    )
    print(f"{len(res['minutiae'])} minuties | XYT: {res['xyt_path']}")
    if res["overlay_path"]:
        print("Overlay:", res["overlay_path"])

if __name__ == "__main__":
    main()
