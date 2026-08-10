
# RT-FAREP: Real-Time Fog-Adaptive Face Recognition

**A research-oriented extension of FAREP (FYP): a recognition-aware fine-tuning approach for face-preserving image dehazing, with a precisely quantified operational range and honest real-time performance analysis.**

> 🏙️ **Motivation:** In cities like Lahore and Delhi, dense winter smog severely degrades CCTV and surveillance face recognition accuracy. This project asks a concrete question: *does dehazing actually help recognition, and if so, under what conditions does it stop helping?*

---

## Pipeline Overview


[Foggy Image/Frame] → [AOD-Net Dehazing] → [RetinaFace Detection] → [ArcFace Recognition] → [Identity]



- **Dehazing:** [AOD-Net](https://arxiv.org/abs/1707.06543) (Li et al., ICCV 2017) — lightweight CNN (1,761 parameters) based on the atmospheric scattering model.
- **Face Detection + Recognition:** [InsightFace](https://github.com/deepinsight/insightface) `buffalo_l` pack — RetinaFace detector + ArcFace recognizer (512-d embeddings).

---

## Key Results

### 1. Baseline Dehazing Quality (SOTS-Outdoor, 500 image pairs)

| Metric | Value |
|---|---|
| Average PSNR | **19.63 dB** |
| Average SSIM | **0.8946** |

Consistent with the original AOD-Net paper's reported range (~19–20 dB PSNR on SOTS-outdoor), confirming a correctly reproduced baseline. Reproduced identically across two independent clean sessions (see Verification Notes below). Fog-density (β) vs. quality shows the expected negative correlation — performance degrades as fog gets denser (see `results/fog_density_correlation.png`).

### 2. Recognition-Aware Fine-Tuning (Core Contribution)

Generic dehazing models (including the pretrained AOD-Net baseline above) are optimized purely for visual quality (PSNR/SSIM) — they have no notion of whether their output preserves an identity's recognizability. We fine-tuned AOD-Net with a **combined loss** that adds a recognition-consistency term:



L = MSE(dehazed, clear) + λ · (1 − cosine_similarity(Embed(dehazed), Embed(clear)))


- **Training data:** 1,161 face images (10 LFW identities, 80/20 split) with fog synthetically applied on-the-fly (β ∈ [0.5, 3.0]).
- **Embedding model:** FaceNet (InceptionResnetV1, VGGFace2-pretrained), frozen, used only to compute the recognition-consistency signal during training. A *different* backbone (ArcFace/InsightFace) is used for evaluation below — this cross-backbone setup is a deliberate check that the fine-tuned dehazing generalizes, rather than overfitting to one recognizer's embedding space.
- **Result — Pretrained vs. Fine-tuned, evaluated with ArcFace:**

| β (fog density) | Pretrained similarity | Fine-tuned similarity | Improvement |
|---:|---:|---:|---:|
| 2.5 | 0.740 | 0.788 | +6.4% |
| 2.8 | 0.724 | 0.765 | +5.7% |
| 3.0 | 0.692 | 0.746 | +7.8% |
| 3.3 | 0.637 | 0.684 | +7.4% |
| 3.6 | 0.578 | 0.658 | +13.7% |
| 3.9 | 0.529 | 0.684 | +29.2% |
| 4.2 | *0 faces detected (fail)* | 0.561 | **operational range extended** |

**Findings:**
- **Average recognition-confidence improvement: +11.7%** across all fog levels where both models detected a face.
- **The improvement grows with fog density** (6.4% → 29.2%) — fine-tuning helps disproportionately more exactly where it matters most: the dense-fog regime where generic dehazing starts to fail.
- **Operational range extended from β≈3.9 to β≈4.2** — at β=4.2, the pretrained model fails to detect a face at all, while the fine-tuned model still produces a usable match.

**Caveat:** This comparison uses a single identity (George W. Bush, LFW) as a controlled case study — see Limitations.

Full data: `results/pretrained_operational_range.csv`, `results/finetuned_operational_range.csv`, `results/pretrained_vs_finetuned_merged.csv`
Visualization: `results/pretrained_vs_finetuned_comparison.png`, `results/training_curves.png`

### 3. Does Dehazing Improve Face Recognition Under Fog? (Baseline Study)

Using LFW face images with synthetically applied fog (atmospheric scattering model, β swept from 0 to 6.0), we compared **raw recognition** vs. **recognition after AOD-Net dehazing**:

| Fog Density (β) | Raw Detection | Dehazed Detection | Verdict |
|---|:---:|:---:|---|
| 0.5 – 2.5 | ✅ Works | ✅ Works | Dehazing gives marginal improvement |
| **3.0 – 3.3** | ❌ **Fails** | ✅ **Recovers** | **Dehazing's effective operating zone** |
| ≥ 3.6 | ❌ Fails | ❌ Fails | Beyond recovery |

**Finding:** AOD-Net dehazing extends the effective operating range of face recognition from β ≈ 2.5 up to β ≈ 3.3 — a critical zone where raw recognition completely fails but recovers after dehazing. Beyond β ≈ 3.6, transmission drops below a recoverable threshold (mean pixel intensity plateaus near the atmospheric light value, A = 0.9) and single-image dehazing cannot reconstruct lost scene information. This is a fundamental limitation of the physical imaging model, not a model-specific failure — it applies to any single-image dehazing method.

Full data: `results/fog_recognition_experiment.csv`, `results/fog_recognition_zoom.csv`
Visualization: `results/fog_vs_recognition_robustness.png`, `results/extreme_fog_example.png`

### 4. Real-Time Performance (Verified, Locked)

Reproduced across two independent clean Kaggle sessions (fresh session, no mid-notebook restarts, `torchvision` upgrade cell removed — see Known Environment Issue below). Timing uses a 5-frame warmup (excluded from measurement) and `time.perf_counter()`.

| Metric | Value |
|---|---|
| Detection-only FPS | **2.36** |
| Dehaze+Detect FPS | **2.17** |
| GPU | Tesla T4 |
| Environment | torch 2.10.0+cu128, cuDNN 91002 |

**Hardware note:** This is a mixed-hardware pipeline, not pure CPU or pure GPU. AOD-Net (dehazing) runs on the Tesla T4 GPU (`cuda:0`). Face detection/recognition (InsightFace/RetinaFace/ArcFace via ONNX Runtime) runs on **CPU** (`CPUExecutionProvider`) — verified by explicitly printing the active provider before benchmarking. `onnxruntime-gpu`'s CUDA provider was not engaged in this environment; this is the dominant cost in the pipeline, since detection/recognition currently outweighs the lightweight (1,761-parameter) dehazing step. Enabling `CUDAExecutionProvider` for the InsightFace models is expected to substantially close the real-time gap and is flagged as immediate future work.

**Reproducibility note:** PSNR/SSIM were identical (19.63 dB / 0.8946) across both verification runs, confirming deterministic pretrained-model behavior. The FPS numbers above superseded an earlier, unverified measurement (2.02 / 1.22) that was skewed by cold-start latency on the first timed frame — fixed by introducing a warmup period.

**Known environment issue:** An earlier notebook version included `pip install --upgrade torchvision`, which upgraded `torch` out of sync with the pre-installed cuDNN build and caused a `CUDNN_STATUS_SUBLIBRARY_VERSION_MISMATCH` crash in a FaceNet sanity-check cell. This line has been removed from the pipeline. Do not re-add a torchvision upgrade without also verifying cuDNN compatibility in a fresh session first.

---

## Repository Structure


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
└── models/
    ├── aod_net.pth                     # Pretrained AOD-Net weights (baseline)
    └── aod_net_finetuned.pth           # Recognition-aware fine-tuned AOD-Net


## Datasets Used

- **[SOTS-Outdoor (RESIDE)](https://www.kaggle.com/datasets/balraj98/synthetic-objective-testing-set-sots-reside)** — 500 hazy/clear image pairs, used for dehazing baseline evaluation.
- **[LFW (Labeled Faces in the Wild)](https://www.kaggle.com/datasets/atulanandjha/lfwpeople)** — used with synthetically applied fog for face recognition robustness testing.


## How to Reproduce

1. Open `notebooks/rt_farep_pipeline.ipynb` in a Python environment with GPU access (Kaggle recommended — this project was verified on Kaggle's Tesla T4 instances).
2. Install dependencies: `torch`, `insightface`, `onnxruntime-gpu`, `opencv-python`, `scikit-image`, `pandas`. **Do not** run `pip install --upgrade torchvision` — see Known Environment Issue above.
3. Run cells sequentially, top to bottom, in a single session without intermediate restarts — dataset paths assume Kaggle's `/kaggle/input/` structure; adjust paths if running locally.
4. Pretrained AOD-Net weights are fetched automatically from the source repository.


## Limitations & Future Work

- **Dense fog ceiling remains, but is pushed further:** Even the fine-tuned model eventually fails at extreme fog density (transmission → 0), a physics-based limitation no single-image method can fully overcome. Fine-tuning extended the usable range (β≈3.9 → β≈4.2) rather than eliminating the ceiling.
- **Small fine-tuning run:** 10 epochs on 1,161 images from 10 identities — sufficient to demonstrate the recognition-aware loss works, but a larger-scale run (more identities, more epochs, learning-rate scheduling) would likely yield further gains. The loss plateaued after ~2 epochs, consistent with AOD-Net's very low parameter count (1,761).
- **Real-time deployment:** Current FPS benchmarks reflect face detection/recognition running on CPU (ONNX Runtime CUDA provider not engaged in this environment) alongside GPU-based dehazing. Enabling the CUDA provider for InsightFace is expected to substantially improve FPS and is the top-priority next step.
- **Single-subject evaluation case study:** The β-sweep comparison uses one identity (George W. Bush, LFW) for controlled, interpretable analysis. Scaling to a full multi-identity verification protocol (e.g., LFW pairs) is a natural next step.
- **FiLM-conditioned dehazing** (fog-level-aware, as in the original FAREP FYP design) is a natural extension of this fine-tuning approach — conditioning the network explicitly on estimated fog density rather than training across a fixed β range.


## Acknowledgments

- AOD-Net: Li, B., et al. *"AOD-Net: All-in-One Dehazing Network."* ICCV 2017.
- InsightFace: Deng, J., et al. *"ArcFace: Additive Angular Margin Loss for Deep Face Recognition."* CVPR 2019.
- Pretrained AOD-Net weights sourced from [MayankSingal/PyTorch-Image-Dehazing](https://github.com/MayankSingal/PyTorch-Image-Dehazing).
- This project extends the FAREP FYP (Fog-Adaptive Real-time Face Recognition Pipeline), Robotics Dept.
