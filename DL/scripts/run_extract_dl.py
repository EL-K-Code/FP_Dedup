# run_extract_dl.py
import glob
from pathlib import Path
from fp.dl_extractor import DLMinutiaeExtractor

# Initialize extractor
extr = DLMinutiaeExtractor(
    coarse="models/CoarseNet.h5",
    fine="models/FineNet.h5",
    classify="models/ClassifyNet_6_classes.h5",
    core="models/CoreNet.weights",
)

DATA_DIR = Path("data")
OUT_DIR = Path("out")
OUT_DIR.mkdir(exist_ok=True, parents=True)

EXTS = ("*.png", "*.jpg", "*.jpeg", "*.tif", "*.tiff", "*.bmp", "*.BMP")

# Collect images (non-recursive; change to rglob for recursion)
paths = []
for pat in EXTS:
    paths.extend(glob.glob(str(DATA_DIR / pat)))
paths = sorted(set(paths))

print(f"Found {len(paths)} images in {DATA_DIR}")

for p in paths:
    print(f"\n▶ Processing: {p}")
    stem = Path(p).stem

    # Pass the PATH directly — extractor loads & normalizes internally
    res = extr.extract(
        img_or_path=p,
        out_dir=OUT_DIR,  # results go to out/<stem>/
        stem=stem,
        save_overlay=True,
        min_short=320,
    )

    print(f"✅ Extracted {len(res.minutiae)} minutiae for {stem}")
    for m in res.minutiae[:10]:
        print(m.x, m.y, f"{m.angle_deg:.2f}", f"{m.score:.3f}", m.classe)
