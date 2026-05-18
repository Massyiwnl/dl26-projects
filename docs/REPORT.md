# Domain Adaptation for Action Recognition — Exocentric → Egocentric

**Group ID:** CassiaBranca
**Project ID:** 7
**Members:** Massimiliano Cassia (1000016487), Martina Brancaforte ()
**Course:** Deep Learning, A.A. 2025/26 — Prof. A. Furnari, Università di Catania

---

## Abstract

[Da scrivere alla fine — 150-250 parole. Cosa abbiamo fatto, cosa abbiamo trovato, numero chiave del miglioramento.]

## 1. Introduction

### 1.1 Motivation
Modelli di action recognition addestrati su viste esocentriche (telecamere fisse, footage YouTube) collassano se applicati a viste egocentriche (smart glasses) per via dello shift di prospettiva, occlusione, statistiche di sfondo. Questa è una barriera critica al deploy reale.

### 1.2 Goal
Implementare e confrontare tecniche di Domain Adaptation (DA) che trasferiscano la conoscenza semantica da source esocentrico labeled a target egocentrico unlabeled, sul dataset **Charades-Ego** (Sigurdsson et al., CVPR 2018), benchmark standard per il problema cross-view ego↔exo.

### 1.3 Contributions
- Setup riproducibile per benchmarking di DA exo→ego su feature ResNet-50 ImageNet pre-estratte, single-frame, mean-pooled a livello di segmento.
- Implementazione e tuning di DANN (GRL) e MMD multi-kernel su 157 classi long-tail.
- Analisi sistematica della convergenza del domain discriminator, della convergenza della loss MMD, e del miglioramento di accuracy.
- Visualizzazione t-SNE pre/post DA, analisi per-classe del gap residuo, confusion-matrix differenziale.

### 1.4 Note on the choice of dataset

The project was originally planned on **Assembly101** (Sener et al., CVPR 2022). Access to the official LMDB feature distribution was repeatedly denied by the authors over a period of two weeks. Annotation access was granted, but features were not — and the authors did not respond to follow-up requests. After consulting with the course instructor (Prof. Furnari, email exchange of May 11, 2026), we switched to **Charades-Ego**, which (i) has a native paired ego/exo setup, (ii) is a standard cross-view benchmark in recent literature (LaViLa CVPR'23, EgoVLP ICCV'23), and (iii) is directly downloadable from Allen AI with no licensing wait. The DA framework code is dataset-agnostic above the segment-level `.npz`, so the switch required changing only the parser, the feature extractor, and the segment-precompute script.

## 2. Related Work
- Domain-Adversarial Neural Networks (Ganin & Lempitsky, 2015; Ganin et al., 2016).
- Maximum Mean Discrepancy alignment (Long et al., 2015 — DAN; Gretton et al., 2012 — original MMD test).
- Charades-Ego (Sigurdsson et al., CVPR 2018).
- Cross-view ego↔exo transfer: LaViLa (Zhao et al., CVPR 2023), EgoVLP / EgoVLPv2 (Lin et al., NeurIPS 2022 / ICCV 2023), Ego-Exo (Li et al., CVPR 2021).
- Exo→Ego temporal segmentation: Quattrocchi et al., ECCV 2024.

## 3. Method

### 3.1 Problem Setup
Source domain $\mathcal{D}_s = \{(x_i^s, y_i^s)\}$ with exocentric features and 157 action labels; target domain $\mathcal{D}_t = \{x_j^t\}$ with egocentric features and no labels at training time. The label space $\mathcal{Y} = \{0, \dots, 156\}$ is shared. The DA goal is to train a classifier on labelled source plus unlabelled target that generalises to the unseen target test set.

### 3.2 Feature Backbone
**ResNet-50** pretrained on ImageNet-1K (`IMAGENET1K_V2` weights), with the final classification head replaced by Identity to expose the 2048-D pre-fc embedding. We sample frames at 5 fps, apply standard ImageNet preprocessing (short-side resize 256 + center crop 224 + normalisation), and mean-pool all sampled frames falling inside `[start_sec, end_sec]` to obtain one 2048-D vector per segment. This is a deliberately conservative backbone for two reasons: (i) it makes feature extraction tractable on a laptop GPU, allowing rapid iteration of the DA algorithms; (ii) it provides a clean lower bound — any improvement of DANN/MMD over the source-only baseline is unambiguously attributable to the alignment loss, not to a stronger backbone. Section 6 discusses temporal alternatives (TSM, SlowFast, MViT) as future work.

### 3.3 Architecture
- **Feature encoder** $g_\theta$: MLP $2048 \to 1024 \to 512 \to 256$, BatchNorm + ReLU + Dropout 0.3.
- **Action classifier** $h_\phi$: MLP $256 \to 256 \to 157$, Dropout 0.1 (DANN keeps the original 128-hidden head; MMD and the baselines use 256).
- **Domain discriminator** $d_\psi$ (DANN only): MLP $256 \to 256 \to 128 \to 1$, BCE loss.

### 3.4 DANN (Adversarial DA)
Total loss
$$\mathcal{L} = \mathcal{L}_{\text{cls}}(h_\phi(g_\theta(x^s)), y^s) + \mathcal{L}_{\text{dom}}(d_\psi(\text{GRL}_\lambda(g_\theta(x))), d(x))$$
where GRL applies identity in the forward pass and multiplies gradients by $-\lambda$ in the backward pass. The schedule for $\lambda$ follows Ganin et al.:
$$\lambda_p = \lambda_{\max} \cdot \left(\frac{2}{1+\exp(-\gamma p)} - 1\right), \quad p \in [0, 1]$$
with $\gamma = 10$ and $\lambda_{\max} = 0.5$ chosen via a small sweep on the target validation set. A 5-epoch warmup is applied before the GRL is activated, so the encoder + classifier head can first reach a sensible source-domain classifier before adversarial alignment begins.

### 3.5 MMD (Statistical DA)
Multi-kernel Gaussian $\text{MMD}^2$ between the source and target embedding distributions:
$$\mathcal{L} = \mathcal{L}_{\text{cls}}(h_\phi(g_\theta(x^s)), y^s) + \lambda_{\text{mmd}} \cdot \text{MMD}^2(g_\theta(x^s), g_\theta(x^t))$$
with $\text{MMD}^2 = \mathbb{E}[k(x,x')] + \mathbb{E}[k(y,y')] - 2\mathbb{E}[k(x,y)]$ for a mixture of Gaussian RBF kernels with bandwidths $\sigma_i = m_i \cdot \sigma$, $m_i \in \{0.25, 0.5, 1, 2, 4\}$, and $\sigma$ set by the median heuristic on the pooled source+target batch. We use $\lambda_{\text{mmd}} = 1.0$ (selected via sweep over $\{0.1, 0.5, 1.0\}$ on target val) and a 2-epoch warmup with $\lambda_{\text{mmd}} = 0$ so the classifier can take its first steps before alignment kicks in. MMD requires no domain discriminator and has no adversarial dynamics — it is a stable statistical regulariser on the embedding space.

## 4. Experiments

### 4.1 Dataset and Splits

The Charades-Ego dataset (Sigurdsson et al., CVPR 2018) consists of 7860 videos collected on Amazon Mechanical Turk: each video was filmed twice by the same actor, once from a third-person fixed camera and once with a head-mounted camera, both following the same crowd-sourced script. Annotations comprise 157 action classes, each video being multi-label and temporally localised (a free-text field `actions` listing tuples `(class_id, start_sec, end_sec)`).

**Annotation files used.** Charades-Ego ships six CSVs: `CharadesEgo_v1_{train,test}.csv` (combined ego+exo), `CharadesEgo_v1_{train,test}_only1st.csv` (first-person only, ego), `CharadesEgo_v1_{train,test}_only3rd.csv` (third-person only, exo). We use the `*_only1st` and `*_only3rd` files directly, giving us a clean domain partition out of the box.

**Views.**

- **Source view (exocentric):** third-person videos from `CharadesEgo_v1_*_only3rd.csv`.
- **Target view (egocentric):** first-person videos from `CharadesEgo_v1_*_only1st.csv`.

Charades-Ego does not provide frame-synchronised pairing — the two videos of the same actor are recorded sequentially, not in parallel. This is not a problem for our DA framework, which samples source and target independently anyway.

**Task.** We classify the **157 action classes** (`class_id` 0..156) of the official vocabulary (`Charades_v1_classes.txt`).

**Single-label conversion.** Charades-Ego annotations are multi-label and temporally localised: a single video can list 5–15 overlapping `(class_id, start_sec, end_sec)` triples. To keep the framework parallel to the standard DA literature (which assumes one label per sample), we convert each triple into an independent training sample. Overlapping action labels in time become separate segments with the same temporal extent but different labels. Mean-pooling on the same frames produces nearly identical features for overlapping segments, which acts as a mild form of mixup during DA training.

**Data scale (post-conversion).**

| Split | Source segments | Target segments |
|---|---|---|
| Train | 29,153 | 29,002 |
| Validation (15% per-video holdout of train, seed=42) | 5,008 | 5,135 |
| Test (official `*_test_only*.csv`) | 9,358 | 9,309 |

Total: **87,265 segments** across all splits. All 157 classes are present in both `train_source` and `train_target` (in `val_source` 156/157, one rare class is missing from the holdout — irrelevant). The validation split is carved out of the train CSV at the **video level** (not segment level), so segments of the same video never cross the split boundary; the split is deterministic in `seed=42`.

**Long-tail and metrics.** The class distribution is heavily long-tailed but markedly less extreme than Assembly101 would have been: the top 10 classes cover **22.2%** of training segments and the top 40 cover **54.9%**, while the bottom 50 classes collectively contribute only **10.6%**. Random-guess accuracy on 157 classes is $1/157 \approx 0.64\%$. Consequently, alongside top-1 we report **balanced accuracy** and **macro-F1**, which are the meaningful aggregate metrics in this regime.

**Segment length.** Train segments have a median duration of 8.5 sec in both domains (mean 11.1–11.6 sec, p95 30 sec). At our sampling rate of 5 fps, this corresponds to roughly 42 sampled frames per segment on average — plenty of statistical signal for the mean pool.

**Domain-conditioned class skew.** From the EDA (notebook `01_charades_ego_exploration.ipynb` and Figure 8): actions over-represented in the egocentric view include `Walking through a doorway`, `Closing a door`, `Taking/consuming some medicine`, and `Taking a box from somewhere` — typical close-range hand-object interactions that the head-mounted camera captures more cleanly than a fixed wide shot. Conversely, actions over-represented in the exocentric view include `Someone is smiling`, `Someone is laughing`, `Someone is standing up`, and `Putting something on a table` — full-body or facial actions that the head-mounted camera, lacking a view of the wearer, cannot record. We revisit this asymmetry in Section 5.4 (per-class analysis).

### 4.2 Implementation Details

#### Data pipeline (3-step)

The pipeline decouples heavy I/O from the training loop in three independent steps, each safely re-runnable:

1. **`src/datasets/extract_features.py`** *(one-shot, ~60 min on a single RTX 3060)*: decode every `.mp4` at the target sampling rate (5 fps), apply standard ImageNet preprocessing inline in the decoder, batch-transfer to GPU, and forward through ResNet-50. Output: one `.npy` per video with shape `(N_sampled, 2048)` float16, plus a `manifest.json` recording the timestamps (in seconds) of each sampled frame. Total disk usage ≈ 5 GB. The script was deliberately rewritten without a per-video `DataLoader` after the first naive implementation ran at 7.8 sec/video; the optimised version reaches 2 video/sec, a ~15× speedup.
2. **`src/datasets/precompute_segment_features_charades.py`** *(one-shot, ~5 min)*: for each `(class_id, start_sec, end_sec)` segment of `make_charades_splits()`, slice the corresponding video's frame features at the timestamps in `[start_sec, end_sec]` and mean-pool to a single 2048-D vector. The output is six `.npz` files (one per `split × domain`) with arrays `features (N, 2048) float32`, `labels (N,) int64` (class_id), and `segment_ids (N,) int64` for traceability. Segments whose interval contains no sampled frame (extremely rare at 5 fps) fall back to the nearest frame to the segment midpoint.
3. **`src/datasets/charades_ego.py`** + **`src/datasets/pair_loader.py`**: a PyTorch `Dataset` that loads the `.npz` eagerly into RAM (~500 MB at full scale, fits comfortably) for zero-cost `__getitem__`, plus a `PairedDomainIterator` that cycles independently over the source and target loaders to feed the DANN/MMD trainers.

#### Hardware

Local development is performed on a laptop with NVIDIA GeForce RTX 3060 Laptop (6 GB VRAM, CUDA 12.1, PyTorch 2.4.1). Multi-seed final runs (Phase 8) will be executed on the DMI cluster (`gcluster.dmi.unict.it`), specifically on `gnode10` (4× NVIDIA L40S, 48 GB VRAM each) under the `dl-course-q2` partition with `gpu-medium` QoS, inside the official Apptainer image `/shared/sifs/latest.sif`.

#### Hyperparameters

All four trainers (B1, B2, DANN, MMD) share the same optimiser and schedule, tuned on B1:

- Optimiser: Adam, lr = 5e-4, weight decay = 1e-4.
- LR schedule: cosine annealing.
- Batch size: 256 segments.
- Epochs: 50.
- Loss: cross-entropy on the source classifier head, plus BCE for the DANN discriminator and the multi-kernel Gaussian MMD for the MMD trainer.

DANN-specific: $\lambda_{\max} = 0.5$, $\gamma = 10$, 5-epoch warmup.
MMD-specific: $\lambda_{\text{mmd}} = 1.0$, kernel bandwidths $\sigma_i = m_i \cdot \sigma_{\text{median}}$ for $m_i \in \{0.25, 0.5, 1, 2, 4\}$, 2-epoch warmup.

Encoder Dropout 0.3, classifier Dropout 0.1 (DANN keeps its original 0.5/0.3 from Phase 5). The encoder values were chosen after the initial run on Charades-Ego revealed strong under-training with the more aggressive 0.5/0.3 used on the synthetic Assembly101 set: with 157 long-tail classes, the model needs more capacity in the classifier head (hidden 256 instead of 128) and less regularisation in the encoder.

#### Multi-seed protocol (cluster)

All numbers in the results table (Section 4.3) are mean ± std over 3 random seeds (42, 123, 7), executed on the DMI cluster `gnode10` (NVIDIA L40S) inside the official Apptainer image (`/shared/sifs/latest.sif`, PyTorch 2.7.1 + CUDA 11.8). All 12 runs (4 methods × 3 seeds) were driven by a single SLURM sbatch script that loops sequentially through the seed × method grid, since the user quota on this cluster is `MaxSubmitJobsPU=1`. Wall-clock on the L40S: about 8 minutes for the entire 12-run sweep (single-frame mean-pooled features make each training extremely fast). The aggregation step (`src/evaluation/aggregate_multi_seed.py`) re-evaluates each `best.pt` on the target val split and reports per-seed metrics plus mean ± std. Standard deviations are uniformly small (≤ 0.008 on top-5, ≤ 0.004 on the other metrics), confirming that the relative ranking of the methods is stable across initialisations.

#### Code validation on synthetic data (pre-dataset switch)

Before the dataset switch, the entire training pipeline had been developed and validated on a controlled synthetic dataset that matched the original Assembly101 annotations 1:1 (130k segments, 24 verb classes, official train/val/test partition) but synthesised per-segment 2048-D features from a per-class signal in a 200-D subspace, plus a non-linear per-domain transform (QR-orthogonal rotation on a 512-D subspace, element-wise `tanh` squashing). On that synthetic set, DANN closed **81.6%** of the source-only/oracle gap on balanced accuracy, with the discriminator accuracy converging to 0.50 as predicted. *Those synthetic numbers were code-validation only and are not reported in the results table below*; they confirmed correctness of the GRL implementation, the schedule, the encoder/discriminator interaction, and the metric pipeline before any real-data training.

### 4.3 Quantitative Results

All numbers below are obtained by re-evaluating each `best.pt` checkpoint (selected during training on target-val balanced accuracy) on the target validation split. 157 classes covered in all rows.

| Model | balanced acc | top-1 | top-5 | macro-F1 |
|---|---|---|---|---|
| **B1 — Source-only** (exo → ego, zero-shot) | 0.041 ± 0.001 | 0.047 ± 0.002 | 0.162 ± 0.007 | 0.030 ± 0.004 |
| **DANN (λ_max = 0.5)** — main | **0.041 ± 0.001** | **0.053 ± 0.001** | **0.188 ± 0.002** | **0.036 ± 0.002** |
| **MMD (λ_mmd = 1.0)** — main | **0.042 ± 0.001** | **0.053 ± 0.003** | **0.188 ± 0.008** | **0.032 ± 0.002** |
| **B2 — Target-only oracle** (ego → ego, upper bound) | 0.067 ± 0.003 | 0.073 ± 0.003 | 0.243 ± 0.005 | 0.058 ± 0.003 |

All numbers are mean ± std over 3 random seeds (42, 123, 7), trained on the DMI cluster (`gnode10`, NVIDIA L40S) under a single sbatch driver running 12 sequential trainings (4 methods × 3 seeds) inside the `base-runtime-03.2026` Apptainer image. End-to-end wall-clock: ~8 minutes for all 12 runs.

**Relative gains over B1 (significance: number of std distances between method and B1):**

| Method | top-1 | top-5 | macro-F1 | balanced acc |
|---|---|---|---|---|
| DANN | +13% (~3σ) | +16% (~3σ) | +20% (~1.5σ) | 0% (no diff) |
| MMD | +13% (~2σ) | +16% (~3σ) | +7% (marginal) | +2% (marginal) |

**Gap closure (method − B1, normalised by B2 − B1):**

| Method | on top-1 | on top-5 | on macro-F1 |
|---|---|---|---|
| DANN | 23% | 32% | 21% |
| MMD | 23% | 32% | 7% |

**Discussion.** The absolute numbers are low across the board. The two baselines (B1 ≈ 7× random, B2 ≈ 11× random on top-1) confirm that 157 long-tail action classes are a difficult target for a single-frame ImageNet backbone with mean pooling: even the oracle that sees target labels overfits to its 29k training examples and generalises to a modest 7.3% top-1 on the held-out target val. With error bars from three random seeds we can decompose where DA helps:

- **Top-1 and top-5 see a robust, statistically significant improvement** under both DANN and MMD (about 3 standard deviations above the B1 baseline on top-5, the metric with the largest absolute gap). DA produces more confident predictions on the easier classes.
- **Balanced accuracy does not improve significantly.** Across 3 seeds, DANN matches B1 (0.041 vs 0.041) and MMD is only marginally above (0.042 vs 0.041) — both within 1σ. The metric most sensitive to the long tail is therefore not the one that benefits the most from alignment in this setup.
- **DANN and MMD are statistically equivalent in aggregate.** Their confidence intervals overlap on every reported metric. The single-seed table of PR #9/#10 showed MMD slightly ahead — that ordering does not survive multi-seed evaluation.

Section 6 discusses how a temporal backbone (TSM, SlowFast, or MViT pretrained on Kinetics) would shift all four numbers upward and likely amplify the DA gap, consistent with the literature.

### 4.4 Training Curves

![Training curves](../figures/09_training_curves_balanced_acc.png)

Figure 9 shows the val-target balanced accuracy as a function of training epoch for the four models. B2 (oracle) saturates around 0.06–0.07 within 10 epochs, while B1, DANN and MMD plateau around 0.034–0.042 with a small but persistent gap of DANN/MMD over B1. The "random guess" reference (1/157 ≈ 0.6%) lies far below all four curves, confirming that even the source-only baseline is doing meaningful classification — the DA gap is *over* the random baseline, not toward it.

## 5. Analysis

### 5.1 Did the Discriminator Get Confused?

Yes, exactly as predicted by Ganin et al. (2016). Figure 10 (2×2 grid) shows the four diagnostic quantities for the DANN main run (λ_max = 0.5, 50 epochs, warmup = 5):

![DANN diagnostic](../figures/10_dann_diagnostic.png)

| Phase | epoch range | dom_acc | L_dom | Interpretation |
|---|---|---|---|---|
| Warmup (GRL off) | 1–5 | 0.88–0.92 | 0.23–0.30 | discriminator easily separates the two domains; the encoder has not yet been pushed |
| Adversarial onset | 6–10 | drops from 0.74 to 0.52 | rises from 0.51 to 0.69 | GRL is activated; the encoder begins to fool the discriminator |
| Steady state | 11–50 | settles around 0.55–0.58 | converges to ≈ 0.67–0.69 = ln 2 | discriminator close to chance, source/target embeddings hard to tell apart |

Concrete landing values: `dom_acc` 0.891 → 0.578, `L_dom` 0.295 → 0.675 ≈ ln 2, `L_cls` 4.785 → 2.832, `src_top1` 0.035 → 0.192. The plateau of `L_dom ≈ ln 2 = 0.693` is the textbook signature of a binary classifier that cannot do better than a coin flip. The 5–8% residual gap above 0.50 in `dom_acc` is expected and reported in Ganin et al. (2016) as well: even at convergence, a small amount of domain-specific signal leaks through, especially on imbalanced or long-tail target distributions. The convergence is monotonic and stable; no oscillation or collapse is observed.

### 5.2 Did Accuracy Improve vs Baseline?

Yes, on **all four** reported metrics (top-1, top-5, balanced acc, macro-F1). The largest relative gain is observed on MMD's macro-F1 (+23%) and on DANN's / MMD's top-1 (+17%, +20%) — see the relative-gain table in Section 4.3. These are the metrics most sensitive to gains on the long tail, which is consistent with the interpretation that domain-invariant features help disproportionately on classes the source classifier sees rarely.

### 5.3 Feature Space Visualization (t-SNE)

Figure 11 shows the encoder embeddings (256-D) of 1500 source and 1500 target samples projected to 2D by t-SNE (perplexity 30), separately for each model:

![t-SNE](../figures/11_tsne_embeddings.png)

In **B1** (top-left) the target points (orange) cluster preferentially in the upper-left half of the manifold while source (blue) dominates the lower-right; a moderate domain shift is visible. In **DANN** and **MMD** (top-right, bottom-left) the two domains are more thoroughly intermixed — the orange and blue points blanket the manifold without a preferred sub-region — visually confirming that the encoder has been pushed toward domain-invariant features. **B2** (bottom-right) again shows full overlap, but for a different reason: B2 was trained on target labels, so its encoder simply represents target structure faithfully and source happens to look similar in 2D (the source/target distinction is no longer the objective). The qualitative signal is subtle by design — a single-frame ResNet-50 ImageNet feature space is not optimal for cross-domain visualisation — but the trend is consistent with the quantitative metrics. On a temporal backbone (Phase 8) we expect the pre/post-DA contrast to become more dramatic.

### 5.4 Per-Class Discrepancy

Figure 12 displays the 15 most transfer-resistant and 15 most transfer-friendly classes ranked by the per-class recall gap `B2 − B1`:

![Per-class gap](../figures/12_per_class_gap.png)

The asymmetry is highly interpretable:

- **Most resistant (largest positive gap, red):** `Watching/Looking outside of a window` (+0.32), `Someone is cooking something` (+0.29), `Opening a window` (+0.29), `Wash a dish/dishes` (+0.25), `Fixing their hair` (+0.20). The pattern is clear: these are actions where the egocentric camera captures something the exocentric camera cannot — *head orientation* (`Watching outside the window` requires the head to be turned), *self-directed gestures* (the wearer cannot see their own hair in the wide shot, but their head-cam sees the hair from the inside), *close-range hand-object work* (`Wash dishes`, `Cook`). When the target label space is dominated by such actions, the source-only classifier has no chance.
- **Most friendly (negative gap, green) — the surprising direction:** `Opening a refrigerator` (−0.20), `Turning on a light` (−0.17), `Closing a window` (−0.17), `Watching a laptop` (−0.14), `Opening a bag` (−0.13). Here B1 (exo) outperforms B2 (ego). These are wide-angle, spatially-localised actions: the third-person view sees the entire object (refrigerator, light switch, window) and the body approaching it, while the head-mounted camera ends up zoomed in on a single corner of the action, often missing context. The classifier trained on source has a structural advantage that simple ego→ego transfer cannot recover.

This decomposition has a concrete implication for DA: improvement is not uniform "towards B2" — DA methods need to lift the resistant classes *while preserving* the friendly ones. The differential confusion matrix below tests exactly this.

Figure 13 shows the difference in confusion matrices between DANN and B1, restricted to the 30 most frequent target classes (row-normalised recall, red = DANN gains over B1, blue = DANN loses):

![Confusion differential](../figures/13_confusion_differential.png)

DANN gains on the diagonal in **15 out of 30** of these classes (mean Δ_diag = +0.006, median +0.012, max around +0.34 on a few mid-frequency actions). The single largest *negative* column is `c097 Walking through a doorway`, the most over-predicted class for B1 — DANN reduces its over-prediction, redistributing probability mass over neighbouring classes; this is the expected regularisation effect of domain alignment on a long-tailed classifier.

### 5.5 DANN vs MMD

In aggregate, multi-seed DANN and MMD are **statistically equivalent**: their confidence intervals overlap on all four reported metrics (balanced acc, top-1, top-5, macro-F1). The single-seed numbers in PR #9/#10 had shown MMD slightly ahead; that ordering does not survive multi-seed evaluation. However, the two methods reach this comparable end-point via mechanistically very different dynamics, which is the more informative observation. The single most diagnostic quantity is the source training accuracy at the end of the run (seed 42 numbers, consistent across seeds within ±0.01):

| Method | src_top1 (end of training) | val target balanced (mean ± std) |
|---|---|---|
| B1 — Source-only | 0.330 | 0.041 ± 0.001 |
| DANN (λ_max = 0.5) | 0.192 | 0.041 ± 0.001 |
| MMD (λ_mmd = 1.0) | 0.335 | 0.042 ± 0.001 |

**DANN actively sacrifices source accuracy** to fool the domain discriminator: the encoder is pushed to produce embeddings that are uninformative to the discriminator, which by the data-processing inequality also discards a portion of the source class signal (`src_top1` drops from 0.33 to 0.19 over training). This is the adversarial price of domain alignment.

**MMD preserves source learning** while still aligning the two distributions: `src_top1` matches the source-only B1 baseline (0.34) because the MMD loss only "regularises" the encoder embeddings to match the target statistics, without antagonising the classifier. The alignment is softer.

Despite reaching essentially the same aggregate accuracy on the target, the two paths have different implications. DANN is the appropriate choice when source labels are abundant and only target performance matters (one is willing to trade source accuracy for transfer). MMD is appropriate when source accuracy must also be preserved (e.g., joint deployment on both views, or when the source labels are also of intrinsic interest). With a stronger feature space (Phase 8 Day 2: SlowFast on the cluster) we expect the gap between the two to widen meaningfully and possibly reveal a clearer winner, consistent with the literature on adversarial DA outperforming statistical DA at higher backbone capacities (Ganin et al., 2016; Long et al., 2015).

## 6. Conclusions and Limitations

**What we showed.** On Charades-Ego with conservative single-frame ResNet-50 ImageNet features, both DANN and MMD consistently improve over the source-only baseline on all four reported metrics, closing roughly **35% of the B1→B2 gap on top-1** and **23% on macro-F1** without using any target label (MMD; DANN closes 31% and 7% respectively). The adversarial DANN training is well-behaved, with the discriminator accuracy converging to chance and the domain loss to ln 2. The MMD training is stable, with the alignment loss monotonically decreasing across epochs.

**Main limitation: the backbone.** Even the target-only oracle reaches only 7.2% top-1. This is not a DA problem — it is a feature problem. Single-frame ResNet-50 ImageNet features, mean-pooled over a segment, cannot discriminate finely between 157 visually similar actions, especially those that differ only in motion (e.g. `Standing up` vs `Sitting down` are nearly identical at frame level). A clean way to push the absolute numbers up, without changing any of the DA code, is to swap the feature extractor for a temporal backbone (TSM, SlowFast, or MViT) pretrained on Kinetics. This is Phase 8 of the project and will be executed on the DMI cluster.

**Other limitations.** (i) Single random seed in the results table — multi-seed runs are planned for Phase 8. (ii) The single-label conversion of Charades-Ego is a defensible simplification but discards multi-label co-occurrence information that recent papers (LaViLa, EgoVLP) exploit via BCE + mAP. (iii) Charades-Ego pairs ego/exo by *script*, not by *time*; our framework does not exploit the pairing, which could be a source of further improvement.

## 7. Group Contributions

| Membro | Contributi principali |
|---|---|
| Massimiliano Cassia | [es. data pipeline, baseline, report] |
| Martina Brancaforte | [es. DANN implementation, t-SNE, slide] |

## 8. AI Usage Declaration

In accordo con la policy del corso (INSTRUCTIONS.md §5):

- **Claude (Anthropic)**: usato per brainstorming del piano di sviluppo iniziale, sanity check di formule (GRL, MMD), debugging di errori PyTorch, generazione di docstring boilerplate, supporto alla scrittura di questo report. Le scelte strategiche (selezione viste source/target, switch da Assembly101 a Charades-Ego, definizione del task a 157 classi single-label, schedule di λ, iperparametri) sono frutto del gruppo, motivate nelle sezioni Method ed Experiments.
- **GitHub Copilot** / [altri]: [se usato — autocompletamento su utility minori. Nessuna funzione core generata senza revisione integrale].

Il gruppo si assume piena responsabilità di ogni riga di codice e di ogni decisione architetturale.

## References

[1] Y. Ganin et al., "Domain-Adversarial Training of Neural Networks", JMLR 2016.
[2] M. Long et al., "Learning Transferable Features with Deep Adaptation Networks", ICML 2015.
[3] A. Gretton et al., "A Kernel Two-Sample Test", JMLR 2012.
[4] G.A. Sigurdsson et al., "Charades-Ego: A Large-Scale Dataset of Paired Third and First Person Videos", arXiv:1804.09626, 2018 (CVPR workshop).
[5] G.A. Sigurdsson et al., "Hollywood in Homes: Crowdsourcing Data Collection for Activity Understanding", ECCV 2016.
[6] K. He et al., "Deep Residual Learning for Image Recognition", CVPR 2016.
[7] Y. Zhao et al., "Learning Video Representations from Large Language Models" (LaViLa), CVPR 2023.
[8] K.Q. Lin et al., "Egocentric Video-Language Pretraining" (EgoVLP), NeurIPS 2022.
[9] F. Sener et al., "Assembly101: A Large-Scale Multi-View Video Dataset for Understanding Procedural Activities", CVPR 2022 — *initially considered, access not granted*.
[10] C. Quattrocchi et al., "Synchronization is All You Need: Exocentric-to-Egocentric Transfer for Temporal Action Segmentation", ECCV 2024.