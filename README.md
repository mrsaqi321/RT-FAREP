# RT-FAREP: Real-Time Fog-Adaptive Face Recognition

**A research-oriented extension of FAREP (FYP), evaluating whether single-image dehazing improves face recognition robustness under fog/haze — with a precisely quantified operational range and honest real-time performance analysis.**

> 🏙️ **Motivation:** In cities like Lahore and Delhi, dense winter smog severely degrades CCTV and surveillance face recognition accuracy. This project asks a concrete question: *does dehazing actually help recognition, and if so, under what conditions does it stop helping?*

---

## Pipeline Overview

```
[Foggy Image/Frame] → [AOD-Net Dehazing] → [RetinaFace Detection] → [ArcFace Recognition] → [Identity]
```

- **Dehazing:** [AOD-Net](https://arxiv.org/abs/1707.06543) (Li et al., ICCV 2017) — lightweight CNN (1,761 parameters) based on the atmospheric scattering model.
- **Face Detection + Recognition:** [InsightFace](https://github.com/deepinsight/insightface) `buffalo_l` pack — RetinaFace detector + ArcFace recognizer (512-d embeddings).

---

## Key Results

### 1. Baseline Dehazing Quality (SOTS-Outdoor, 500 image pairs)

| Metric | Value |
|---|---|
| Average PSNR | **19.63 dB** |
| Average SSIM | **0.8946** |

Consistent with the original AOD-Net paper's reported range (~19–20 dB PSNR on SOTS-outdoor), confirming a correctly reproduced baseline. Fog-density (β) vs. quality shows the expected negative correlation — performance degrades as fog gets denser (see `results/fog_density_correlation.png`).

### 2. Does Dehazing Improve Face Recognition Under Fog?

Using LFW face images with synthetically applied fog (atmospheric scattering model, β swept from 0 to 6.0), we compared **raw recognition** vs. **recognition after AOD-Net dehazing**:

| Fog Density (β) | Raw Detection | Dehazed Detection | Verdict |
|---|:---:|:---:|---|
| 0.5 – 2.5 | ✅ Works | ✅ Works | Dehazing gives marginal improvement |
| **3.0 – 3.3** | ❌ **Fails** | ✅ **Recovers** | **Dehazing's effective operating zone** |
| ≥ 3.6 | ❌ Fails | ❌ Fails | Beyond recovery |

**Finding:** AOD-Net dehazing extends the effective operating range of face recognition from β ≈ 2.5 up to β ≈ 3.3 — a critical zone where raw recognition completely fails but recovers after dehazing. Beyond β ≈ 3.6, transmission drops below a recoverable threshold (mean pixel intensity plateaus near the atmospheric light value, A = 0.9) and single-image dehazing cannot reconstruct lost scene information. This is a fundamental limitation of the physical imaging model, not a model-specific failure — it applies to any single-image dehazing method.

Full data: `results/fog_recognition_experiment.csv`, `results/fog_recognition_zoom.csv`
Visualization: `results/fog_vs_recognition_robustness.png`, `results/extreme_fog_example.png`

### 3. Real-Time Performance (CPU inference)

| Configuration | FPS | Notes |
|---|:---:|---|
| Detection only (no dehazing) | 2.02 | RetinaFace + ArcFace, CPU |
| Dehaze + Detection | 1.22 | Full pipeline, CPU |

**Note on hardware:** These benchmarks were run on CPU due to a CUDA/cuDNN toolkit version mismatch in the evaluation environment (driver reported CUDA 13.0, but linked libraries were built against CUDA 12.x, causing `onnxruntime-gpu` to silently fall back to CPU). The dehazing step itself is extremely lightweight (AOD-Net: 1,761 parameters) and adds minimal overhead relative to the detection/recognition models, which dominate inference cost. On a properly configured GPU or edge accelerator (e.g., Jetson Nano, Coral TPU), this pipeline is expected to reach real-time speeds (15+ FPS), consistent with AOD-Net's published real-time benchmarks on GPU hardware.

---

## Repository Structure

```
rt-farep/
├── README.md
├── notebooks/
│   └── rt_farep_pipeline.ipynb        # Full pipeline: dehazing, recognition, fog experiments
├── results/
│   ├── aodnet_baseline_results.csv    # Per-image PSNR/SSIM (500 images)
│   ├── fog_density_correlation.png
│   ├── best_case.png / worst_case.png
│   ├── fog_recognition_experiment.csv
│   ├── fog_recognition_zoom.csv
│   ├── fog_vs_recognition_robustness.png
│   ├── extreme_fog_example.png
│   └── video_pipeline_results_optimized.csv
└── weights/
    └── aod_net.pth                     # Pretrained AOD-Net weights
```

---

## Datasets Used

- **[SOTS-Outdoor (RESIDE)](https://www.kaggle.com/datasets/balraj98/synthetic-objective-testing-set-sots-reside)** — 500 hazy/clear image pairs, used for dehazing baseline evaluation.
- **[LFW (Labeled Faces in the Wild)](https://www.kaggle.com/datasets/atulanandjha/lfwpeople)** — used with synthetically applied fog for face recognition robustness testing.

---

## How to Reproduce

1. Open `notebooks/rt_farep_pipeline.ipynb` in a Python environment with GPU access (Kaggle/Colab recommended).
2. Install dependencies: `torch`, `insightface`, `onnxruntime-gpu`, `opencv-python`, `scikit-image`, `pandas`.
3. Run cells sequentially — dataset paths assume Kaggle's `/kaggle/input/` structure; adjust paths if running locally.
4. Pretrained AOD-Net weights are fetched automatically from the source repository.

---

## Limitations & Future Work

- **Dense fog ceiling:** Single-image dehazing cannot recover identity beyond β ≈ 3.3 — a physics-based limitation. Future work could explore multi-frame temporal fusion or learned priors conditioned on estimated fog density (FiLM-style conditioning), which was part of the original FAREP FYP design.
- **Real-time deployment:** Current benchmarks are CPU-only due to environment constraints; GPU/edge-accelerator deployment is expected to close the real-time gap.
- **Single-subject validation:** The fog-robustness experiment uses one identity (George W. Bush, LFW) as a controlled case study. Scaling to multi-identity verification (full LFW pairs protocol) is a natural next step.

---

## Acknowledgments

- AOD-Net: Li, B., et al. *"AOD-Net: All-in-One Dehazing Network."* ICCV 2017.
- InsightFace: Deng, J., et al. *"ArcFace: Additive Angular Margin Loss for Deep Face Recognition."* CVPR 2019.
- Pretrained AOD-Net weights sourced from [MayankSingal/PyTorch-Image-Dehazing](https://github.com/MayankSingal/PyTorch-Image-Dehazing).
- This project extends the FAREP FYP (Fog-Adaptive Real-time Face Recognition Pipeline), Robotics Dept.
