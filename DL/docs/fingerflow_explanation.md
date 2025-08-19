````markdown
# FingerFlow: Coarse-to-Fine Fingerprint Minutiae Extraction

Fingerprint **minutiae** (ridge endings and bifurcations) are the core features used in identification and **deduplication**. Extracting them reliably is difficult on low-quality or partial prints. **FingerFlow** addresses this with a *coarse-to-fine* deep-learning pipeline: a fast network proposes candidate minutiae across the whole image (high recall), then a patch-based verifier confirms or rejects each candidate (high precision). The final output is a clean list of minutiae with stable orientations, ready for indexing or matching in a dedup system.

---

## High-Level Workflow

```mermaid
%% FingerFlow — minutiae extraction (coarse-to-fine)
flowchart TD
    %% ===== 1) PREPROCESSING =====
    subgraph PREP["🔧 Preprocessing"]
    direction TB
        A["🖼️ Fingerprint image"] --> B["✨ Texture/contrast enhancement"]
        B --> C["🧭 Global orientation field (STFT)\ndir_map"]
    end

    %% ===== 2) COARSE DETECTION (CoarseNet) =====
    subgraph COARSE["🧠 Coarse detection — CoarseNet"]
    direction TB
        C --> D[["CoarseNet (FCN)"]]
        D --> D1["🗺️ seg_out (finger/background segmentation)"]
        D --> D2["📈 mnt_s_out (minutiae score per pixel)"]
        D --> D3["🧭 mnt_o_out (local orientation)"]
        D --> D4["📦 mnt_w/h_out (local size)"]
    end

    %% ===== 3) POST-PROCESSING + NMS + ADAPTIVE THRESHOLDS =====
    subgraph POST["🧹 Post-processing"]
    direction TB
        D1 --> E1["🧼 Morphology (close → open → dilate)"]
        D2 --> E2["🔧 label2mnt(score × mask)"]
        E2 --> E3["🧹 NMS #1 (IoU 0.5)"]
        E2 --> E4["🧹 NMS #2 (custom)"]
        E3 --> E5(("🔗 NMS fusion"))
        E4 --> E5
        E5 --> F{"Enough minutiae?"}
        F -- "No → lower thresholds" --> E2
        F -- "Yes" --> G["🔎 Filtered candidates (score ≥ early)"]
    end

    %% ===== 4) FINE VERIFICATION (FineNet) =====
    subgraph FINE["🧠 Patch-based refinement — FineNet"]
    direction TB
        G --> H["🧩 Extract centered patches (x,y)"]
        H --> I[["FineNet (binary)"]]
        I --> J["⚖️ Score fusion\n(4×Coarse + 1×Fine)/5"]
        J --> K["🧪 Final filter (score ≥ threshold)"]
    end

    %% ===== 5) FINAL ORIENTATION =====
    subgraph ORI["🧭 Final orientation"]
    direction TB
        K --> L["🔗 Fuse θ with dir_map (STFT)"]
    end

    L --> M["📤 Output: Minutiae list\n(x, y, θ, score, type)"]

    %% Styles
    style D stroke:#00aaff,stroke-width:3px,fill:#eaf7ff,rx:8,ry:8
    style I stroke:#00cc88,stroke-width:3px,fill:#eafff6,rx:8,ry:8
````

---

## 1) Preprocessing

### 1.1 Texture / Contrast Enhancement

**Goal.** Make ridge/valley structure more distinct and reduce noise so the networks see clean patterns.

**Typical operations.**

* Local contrast normalization (e.g., CLAHE) to balance bright/dark zones.
* Ridge enhancement with filters tuned to fingerprint spatial frequencies, or a fast texture-enhancement routine.
* Light denoising (median/gaussian) that preserves thin ridges.

**Notes.** Avoid over-sharpening (can break ridges into fragments). Enhancement strength often depends on the sensor and DPI.

### 1.2 Global Orientation Field (STFT)

**Goal.** Estimate a smooth **orientation map** `dir_map(x,y)` describing local ridge flow, used later to stabilize minutiae angles.

**How it works (intuition).**

* Partition the image into overlapping blocks (e.g., 64×64 with 16-px stride).
* Apply Short-Time Fourier Transform (windowed DFT) per block; the dominant spectral ellipse indicates ridge direction.
* Smooth the resulting field (vector averaging) to remove jitter; mask out background.

---

## 2) Coarse Detection — CoarseNet

**What it is.** A fully-convolutional CNN that scans the entire image **once** and outputs dense maps:

* `seg_out` — segmentation (finger vs. background).
* `mnt_s_out` — **minutiae score map** (per-pixel confidence).
* `mnt_o_out` — local orientation near candidate minutiae.
* `mnt_w_out`, `mnt_h_out` — local “scale” (width/height) to guide suppression.

**Why “coarse”.** It prioritizes **recall** (find as many candidates as possible) with global context. The output is a heatmap of potential minutiae, plus orientation cues.

---

## 3) Post-Processing: Mask, NMS, Adaptive Thresholds

### 3.1 Mask Cleanup (Morphology)

Use `seg_out` to remove spurious responses outside the finger area:

* **Close** fills small holes inside the finger region.
* **Open** removes small noise specks.
* **Dilate** slightly expands the valid region to avoid cutting off valid minutiae at the boundary.

### 3.2 Heatmap → Candidate Points + Dual NMS

Convert the score map into discrete candidates:

* Identify local peaks via `label2mnt(score × mask)` (score weighted by the segmentation mask).
* Apply **two forms of NMS** (e.g., IoU-based and a custom variant), then **fuse** their survivors.
  This collapses clusters of close detections into single, well-localized candidates.

### 3.3 Adaptive Thresholding

Some images are poor (dry fingers, smudges). A single fixed threshold can drop valid minutiae.
Strategy:

* Start with an “early” threshold (e.g., \~0.50) and a final threshold (e.g., \~0.45).
* If you get too few candidates (e.g., ≤ 4), **lower thresholds** step-by-step (−0.05) and retry.
* Stop once a reasonable count is reached or a floor is hit.

**Trade-off.** Lower thresholds → more candidates for FineNet (higher compute, but better recall). Higher thresholds → fewer false positives but risk missing minutiae.

---

## 4) Fine Verification — FineNet (Patch-Based)

**Goal.** **Confirm** each coarse candidate using a high-resolution local view and refine its orientation.

**How it works.**

* Crop a centered patch around each candidate (x, y) **from the original image**.
* Resize/normalize to FineNet’s input size.
* A binary CNN (**minutia vs. non-minutia**) classifies the patch and can regress a precise orientation.

**Score Fusion.**

* Combine CoarseNet’s score with FineNet’s probability, e.g.:
  `final_score = (4 × coarse_score + 1 × fine_prob) / 5`.
  CoarseNet remains the main signal; FineNet acts as a validator.

**Outcome.**

* True minutiae pass with strong confidence.
* Spurious candidates (blobs, noisy specks) are rejected.

---

## 5) Final Orientation Fusion

Even with local estimates, minutia angle can be noisy. The pipeline **fuses** the candidate’s angle with the **smoothed STFT orientation field** at that location.

* Produces angles consistent with global ridge flow.
* Reduces random flips and improves stability for downstream use (e.g., constructing features like `cosθ`/`sinθ`).

**Angle convention (image frame).** `0° →` right, `90° →` down, in `[0, 360)`.

---

## Output: Minutiae List

The final output is a structured list of minutiae, each with:

* **(x, y)** pixel coordinates,
* **θ** (stable orientation in degrees),
* **score** (confidence after fusion),
* optionally **type** (e.g., ending/bifurcation) if classification is enabled.

You may also produce:

* **JSON/CSV/XYT** files for interoperability,
* an **overlay image** (minutiae markers/arrows over the grayscale) for quality control.

---

## Practical Tuning (Cheat-Sheet)

* **Enhancement**: mild to moderate; avoid over-sharpening that breaks ridges.
* **STFT**: patch 64 / stride 16 is a good default; increase patch for very noisy images.
* **Coarse thresholds**: early ≈ 0.50, final ≈ 0.45; lower in steps of 0.05 if too few candidates.
* **NMS**: IoU ≈ 0.5; raise to merge more, lower to keep denser points.
* **Fine fusion**: start at 4:1 (Coarse\:Fine); move towards 3:2 if you trust FineNet more.
* **Final acceptance**: set a score threshold (e.g., 0.45–0.55) based on your data and desired precision/recall.

---

## Typical Pitfalls & Remedies

* **Very few minutiae (dry/partial prints)** → enable adaptive thresholds; rely more on FineNet; slightly stronger enhancement.
* **Many false positives on background/noise** → improve segmentation (morphology), raise final threshold modestly, strengthen NMS.
* **Unstable angles** → ensure the STFT orientation field is well smoothed; apply fusion after FineNet filtering.

---

## Why This Matters for Deduplication

For dedup, you eventually compare fingerprints to detect duplicates. FingerFlow’s pipeline gives you a **robust, consistent set of minutiae with reliable orientations**, which are ideal for building fixed-length descriptors or for graph-based matchers. Better extraction → better candidate retrieval and fewer false alarms when you scale to millions of prints.

---
