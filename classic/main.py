from pathlib import Path
import json, cv2
from minutiae_skeleton import detect_minutiae, DetectParams

res = detect_minutiae("data/fing1.png", DetectParams(return_overlay=True))

out_dir = Path("out")
out_dir.mkdir(parents=True, exist_ok=True)  # <-- crée le dossier

# enregistrements
if res["overlay"] is not None:
    cv2.imwrite(str(out_dir / "overlay.png"), res["overlay"])
cv2.imwrite(str(out_dir / "skeleton.png"), res["skeleton"])
cv2.imwrite(str(out_dir / "binary.png"), res["binary"])

with open(out_dir / "minutiae.json", "w", encoding="utf-8") as f:
    json.dump(res["minutiae"], f, ensure_ascii=False, indent=2)
