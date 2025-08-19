import cv2
import glob
from pathlib import Path
from fp.dl_extractor import DLMinutiaeExtractor


extr = DLMinutiaeExtractor(
    coarse="models/CoarseNet.h5",
    fine="models/FineNet.h5",
    classify="models/ClassifyNet_6_classes.h5",
    core="models/CoreNet.weights",
)
data_dir = Path("data")
out_dir = Path("out")
out_dir.mkdir(exist_ok=True, parents=True)
image_paths = []
for ext in ["*.png", "*.jpg", "*.jpeg", "*.tif", "*.tiff"]:
    image_paths.extend(glob.glob(str(data_dir / ext)))

print(f"🔍 {len(image_paths)} images trouvées dans {data_dir}")
for img_path in image_paths:
    print(f"\n📂 Traitement : {img_path}")
    img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        print(f"⚠️ Impossible de lire {img_path}, on saute.")
        continue
    stem = Path(img_path).stem
    res = extr.extract(img, out_dir=out_dir, stem=stem, save_overlay=True)

    print(f"✅ {len(res.minutiae)} minuties extraites pour {stem}")
    for m in res.minutiae:
        print(m.x, m.y, m.angle_deg, m.score, m.classe)
