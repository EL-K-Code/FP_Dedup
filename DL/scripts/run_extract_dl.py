import cv2
from fp.dl_extractor import DLMinutiaeExtractor

# quelque part plus haut dans ton pipeline tu as déjà img (np.ndarray)
img = cv2.imread("data/fing1.png", cv2.IMREAD_GRAYSCALE)  # exemple ; dans ton cas tu as déjà 'img'

extr = DLMinutiaeExtractor(
    coarse="models/CoarseNet.h5",
    fine="models/FineNet.h5",
    classify="models/ClassifyNet_6_classes.h5",
    core="models/CoreNet.weights",
)

res = extr.extract(img, out_dir="out", stem="fing1", save_overlay=True)
print(len(res.minutiae), "minuties")
for m in res.minutiae:
    print(m.x, m.y, m.angle_deg, m.score, m.classe)
