
# RT-FAREP: Real-Time Fog-Adaptive Face Recognition

**A research-oriented extension of FAREP (FYP): a recognition-aware fine-tuning approach for face-preserving image dehazing, validated at scale with a standard 6,000-pair LFW verification protocol.**

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

### 2. Large-Scale Recognition Verification (LFW, 6,000 pairs, standard 10-fold protocol)

**V1 of this project evaluated recognition robustness on a single identity as a controlled pilot case study.** That result was promising but statistically limited. V2 replaces it with the standard LFW "View 2" verification protocol: **6,000 image pairs (3,000 genuine, 3,000 impostor) across 10 cross-validation folds**, drawn from the full LFW dataset rather than one person — evaluated at six fog densities (β = 0.5 to 4.0) under three conditions: raw (no dehazing), pretrained AOD-Net, and the recognition-aware fine-tuned AOD-Net.

**Primary metric — Operational Reliability** (the fraction of pairs where both faces were successfully detected and embedded; unlike TAR, this metric stays statistically meaningful even as fog becomes extreme, since it doesn't collapse onto a shrinking, biased survivor subset):

| Fog Density (β) | Raw | Pretrained AOD-Net | Fine-tuned AOD-Net |
|---:|---:|---:|---:|
| 0.5 | 100.00% | 100.00% | 98.33% |
| 1.5 | 100.00% | 100.00% | 100.00% |
| 2.5 | 94.65% | 99.93% | 100.00% |
| 3.0 | 21.42% | 98.30% | **99.97%** |
| 3.5 | 0.02% | 55.92% | **99.77%** |
| 4.0 | 0.00% | 0.25% | **87.33%** |

**Finding:** At moderate fog (β ≤ 2.5), all three conditions perform comparably — dehazing gives marginal benefit, consistent with V1. The gap opens sharply from β ≥ 3.0: raw detection collapses almost completely (21% at β=3.0, ~0% beyond), and even the **pretrained** dehazing model starts failing at β ≥ 3.5 (44% miss rate at β=3.5, 99.75% miss rate at β=4.0). The **fine-tuned model remains reliable far into this regime** — still detecting and matching 87.3% of pairs at β=4.0, where the pretrained baseline has effectively stopped working (only ~15 of 6,000 pairs survived detection at that density).

**Secondary metric — TAR @ FAR = 0.1%** (reported only where the surviving sample is large enough to be meaningful; see caveats):

| β | Raw | Pretrained | Fine-tuned |
|---:|---:|---:|---:|
| 0.5 | 97.63% [97.04, 98.22] | 97.87% [97.14, 98.60] | 97.10% [96.30, 97.91] |
| 1.5 | 97.83% [97.44, 98.23] | 97.93% [97.40, 98.47] | 98.07% [97.46, 98.67] |
| 2.5 | 96.84% [95.69, 97.99] | 98.23% [97.75, 98.71] | 98.27% [97.88, 98.65] |
| 3.0 | 96.94%¹ | 96.99% [96.01, 97.97] | 98.40% [97.98, 98.82] |
| 3.5 | undefined² | 92.83%³ [90.27, 95.39] | 97.73% [96.91, 98.55] |
| 4.0 | undefined² | not reported⁴ | 91.70% [88.71, 94.69] |

*95% CI from 10-fold cross-validation in brackets.*

¹ Computed on only the 21.4% of pairs that survived detection at this density — likely an easier, non-representative subset; not directly comparable to lower-fog rows.
² 0/10 folds had any valid pairs (99.98–100% detection failure) — TAR is undefined, not zero.
³ Computed on the 55.9% surviving subset; treat as directionally informative, not a precise estimate.
⁴ **Deliberately excluded.** The raw output reported TAR = 1.0 for pretrained at β=4.0, but only 4 of 10 folds had any surviving pairs at all (~15 pairs out of 6,000 total). This is a small-sample artifact, not a genuine result, and should not be cited as "pretrained achieves perfect recognition at β=4.0."

**Takeaway for the recognition-aware loss:** the fine-tuning objective was never intended to improve TAR under mild fog (and doesn't — the two models are statistically indistinguishable for β ≤ 2.5, with fine-tuned even marginally lower at β=0.5). Its actual, verified effect is **extending the fog density at which recognition remains operational at all** — which is the more practically relevant claim for a surveillance deployment context.

Full data: `results/v2_tar_far_table.csv`
Plot PNG: `results/v2_results_plots.png`
Notebook: `notebooks/rt-farep-v2.ipynb`

<details>
<summary><strong>V1 pilot study (single-identity, superseded by the above)</strong></summary>

The original pilot used one identity (George W. Bush, LFW) for an interpretable, controlled case study before scaling up:

| β (fog density) | Pretrained similarity | Fine-tuned similarity | Improvement |
|---:|---:|---:|---:|
| 2.5 | 0.740 | 0.788 | +6.4% |
| 3.0 | 0.692 | 0.746 | +7.8% |
| 3.6 | 0.578 | 0.658 | +13.7% |
| 3.9 | 0.529 | 0.684 | +29.2% |
| 4.2 | *0 faces detected (fail)* | 0.561 | operational range extended |

This motivated the larger LFW-scale study above, which confirms the same qualitative pattern (growing benefit at high fog density) on a statistically robust sample.

</details>

### 3. Real-Time Performance

| Configuration | FPS | GPU | Environment |
|---|:---:|---|---|
| Detection-only | 2.36 | Tesla T4 | torch 2.10.0+cu128, cuDNN 91002 |
| Dehaze+Detection | 2.17 | Tesla T4 | torch 2.10.0+cu128, cuDNN 91002 |

Reproduced across two independent clean Kaggle sessions with a 5-frame warmup (excluded from timing) and `time.perf_counter()`. **These numbers were measured before the environment fix below** — InsightFace was running on `CPUExecutionProvider` at the time (verified by explicitly printing the active provider), with only AOD-Net on GPU. **A re-benchmark under the corrected, fully-GPU environment (see below) is still pending** and is expected to improve these numbers substantially, since detection/recognition — not the lightweight 1,761-parameter dehazing step — was the dominant cost.

---

## Environment Notes

**Resolved: InsightFace CPU fallback.** V1 documented `onnxruntime-gpu` silently falling back to CPU due to a CUDA/cuDNN version mismatch. This is fixed in V2 by pinning the ONNX Runtime GPU build and its CUDA 12 dependencies explicitly, rather than letting pip resolve them automatically:

```bash
pip uninstall -y onnxruntime onnxruntime-gpu -q
pip install insightface scikit-learn scikit-image opencv-python pandas matplotlib tqdm scipy -q
pip uninstall -y onnxruntime -q
pip install onnxruntime-gpu==1.19.2 -q
pip install nvidia-cublas-cu12 nvidia-cudnn-cu12 nvidia-curand-cu12 nvidia-cufft-cu12 -q
```

```python
import os
site_packages = '/usr/local/lib/python3.12/dist-packages'
cuda_lib_paths = [f'{site_packages}/nvidia/{p}/lib' for p in ['cublas', 'cudnn', 'curand', 'cufft']]
os.environ['LD_LIBRARY_PATH'] = ':'.join(cuda_lib_paths) + ':' + os.environ.get('LD_LIBRARY_PATH', '')
```

Verified working — `ort.get_available_providers()` returns `['TensorrtExecutionProvider', 'CUDAExecutionProvider', 'CPUExecutionProvider']`, and the InsightFace model-load log confirms `CUDAExecutionProvider` is actually applied to both the detection and recognition sub-models, not just listed as available.

**Still open:** `pip install --upgrade torchvision` (documented in V1 as the root cause of a `CUDNN_STATUS_SUBLIBRARY_VERSION_MISMATCH` crash) remains removed from the pipeline. Do not re-add it without re-verifying cuDNN compatibility in a fresh session.

---

## Repository Structure

```text
rt-farep/
├── README.md
├── notebooks/
│   ├── rt_farep_pipeline.ipynb            # V1: dehazing baseline, fine-tuning, single-identity pilot
│   └── rt-farep-v2.ipynb                  # V2: 6,000-pair LFW verification benchmark
├── results/
│   ├── aodnet_baseline_results.csv        # Per-image PSNR/SSIM (500 images)
│   ├── v2_tar_far_table.csv               # 6,000-pair LFW verification results (TAR, AUC, CI, failure rate)
│   ├── fog_density_correlation.png
│   ├── best_case.png
│   ├── worst_case.png
│   ├── fog_recognition_experiment.csv
│   ├── fog_vs_recognition_robustness.png
│   ├── extreme_fog_example.png
│   └── video_pipeline_results_optimized.csv
└── models/
    ├── aod_net.pth                        # Pretrained AOD-Net weights (baseline)
    └── aod_net_finetuned.pth              # Recognition-aware fine-tuned AOD-Net
```
## Datasets Used

- **[SOTS-Outdoor (RESIDE)](https://www.kaggle.com/datasets/balraj98/synthetic-objective-testing-set-sots-reside)** — 500 hazy/clear image pairs, used for dehazing baseline evaluation.
- **[LFW (Labeled Faces in the Wild)](https://www.kaggle.com/datasets/jessicali9530/lfw-dataset)** — standard 6,000-pair "View 2" verification protocol (`pairs.csv`), used for the large-scale recognition benchmark with synthetically applied fog.
- **[LFW-People](https://www.kaggle.com/datasets/atulanandjha/lfwpeople)** — used in V1 for the single-identity pilot study.

---

## How to Reproduce

1. Open `notebooks/RT_FAREP_V2.ipynb` for the main verification benchmark, or `notebooks/rt_farep_pipeline.ipynb` for the original dehazing baseline and fine-tuning run — both in a GPU-enabled environment (Kaggle recommended; verified on Tesla T4).
2. Run the ONNX Runtime install sequence exactly as shown in **Environment Notes** above — installing `onnxruntime-gpu` without pinning the version has previously caused a silent CPU fallback.
3. **Do not** run `pip install --upgrade torchvision` — see Environment Notes.
4. Run cells sequentially, top to bottom, in a single session without intermediate restarts. Dataset paths assume Kaggle's `/kaggle/input/` structure; adjust if running locally.
5. The V2 notebook checkpoints embeddings to disk (`embeddings_checkpoint.pkl`) after each fog-density condition, so an interrupted Kaggle session can resume rather than restart from β=0.5.

---

## Limitations & Future Work

- **Extreme-fog statistics are inherently sample-limited.** At β ≥ 3.5, the raw and (at β=4.0) pretrained conditions detect too few pairs for a statistically meaningful TAR — this is a property of the recognition failure itself, not a benchmarking gap, and is reported transparently above (operational reliability) rather than papered over with an unreliable TAR figure.
- **FPS re-benchmark pending.** The CUDA/CPU-fallback fix (see Environment Notes) was applied after the original FPS numbers were measured; those numbers still reflect the old CPU-bound InsightFace path and should be treated as a lower bound, not the current expected performance.
- **Small fine-tuning run:** 10 epochs on 1,161 images from 10 identities. The loss plateaued after ~2 epochs, consistent with AOD-Net's very low parameter count (1,761); a larger-scale fine-tuning run (more identities, more epochs, LR scheduling) would likely yield further gains, particularly at β ≥ 3.0 where the benefit is concentrated.
- **FiLM-conditioned dehazing** (fog-level-aware, as in the original FAREP FYP design) remains a natural extension — conditioning the network explicitly on estimated fog density rather than training across a fixed β range.

---

## Acknowledgments

- AOD-Net: Li, B., et al. *"AOD-Net: All-in-One Dehazing Network."* ICCV 2017.
- InsightFace: Deng, J., et al. *"ArcFace: Additive Angular Margin Loss for Deep Face Recognition."* CVPR 2019.
- Pretrained AOD-Net weights sourced from [MayankSingal/PyTorch-Image-Dehazing](https://github.com/MayankSingal/PyTorch-Image-Dehazing).
- This project extends the FAREP FYP (Fog-Adaptive Real-time Face Recognition Pipeline), Robotics Dept.
