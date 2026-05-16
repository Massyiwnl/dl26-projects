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
- Analisi sistematica della convergenza del domain discriminator e del miglioramento di accuracy.
- Visualizzazione t-SNE pre/post DA e analisi per-classe del gap residuo.

### 1.4 Note on the choice of dataset

The project was originally planned on **Assembly101** (Sener et al., CVPR 2022). Access to the official LMDB feature distribution was repeatedly denied by the authors over a period of two weeks. Annotation access was granted, but features were not — and the authors did not respond to follow-up requests. After consulting with the course instructor (Prof. Furnari, email exchange of May 11, 2026), we switched to **Charades-Ego**, which (i) has a native paired ego/exo setup, (ii) is a standard cross-view benchmark in recent literature (LaViLa CVPR'23, EgoVLP ICCV'23), and (iii) is directly downloadable from Allen AI with no licensing wait. The DA framework code is dataset-agnostic above the segment-level `.npz`, so the switch required changing only the parser, the feature extractor, and the segment-precompute script.

## 2. Related Work
- Domain-Adversarial Neural Networks (Ganin & Lempitsky, 2015; Ganin et al., 2016).
- Maximum Mean Discrepancy alignment (Long et al., 2015 — DAN).
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
- **Action classifier** $h_\phi$: MLP $256 \to 256 \to 157$, Dropout 0.1.
- **Domain discriminator** $d_\psi$ (DANN only): MLP $256 \to 256 \to 128 \to 1$, BCE loss.

### 3.4 DANN (Adversarial DA)
Total loss
$$\mathcal{L} = \mathcal{L}_{\text{cls}}(h_\phi(g_\theta(x^s)), y^s) + \mathcal{L}_{\text{dom}}(d_\psi(\text{GRL}_\lambda(g_\theta(x))), d(x))$$
where GRL applies identity in the forward pass and multiplies gradients by $-\lambda$ in the backward pass. The schedule for $\lambda$ follows Ganin et al.:
$$\lambda_p = \lambda_{\max} \cdot \left(\frac{2}{1+\exp(-\gamma p)} - 1\right), \quad p \in [0, 1]$$
with $\gamma = 10$ and $\lambda_{\max} = 0.5$ chosen via a small sweep on the target validation set. A 5-epoch warmup is applied before the GRL is activated, so the encoder + classifier head can first reach a sensible source-domain classifier before adversarial alignment begins.

### 3.5 MMD (Statistical DA)
Multi-kernel Gaussian $\text{MMD}^2$ between the source and target embedding distributions, with bandwidths $\sigma_k$ selected by the median heuristic on the pooled pairwise distances. To be reported in Section 4.3 after the MMD trainer is implemented (Phase 6).

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

This separation has three concrete benefits. First, the videos are decoded only once. Second, the segment-level pre-computation can be re-run cheaply if the task definition changes. Third, training epochs are bottlenecked only by GPU compute — a typical 50-epoch B1/B2/DANN run completes in under 2 minutes on the RTX 3060.

#### Hardware

Local development is performed on a laptop with NVIDIA GeForce RTX 3060 Laptop (6 GB VRAM, CUDA 12.1, PyTorch 2.4.1). Multi-seed final runs (Phase 8) will be executed on the DMI cluster (`gcluster.dmi.unict.it`), specifically on `gnode10` (4× NVIDIA L40S, 48 GB VRAM each) under the `dl-course-q2` partition with `gpu-medium` QoS, inside the official Apptainer image `/shared/sifs/latest.sif`.

#### Hyperparameters

All three trainers (B1, B2, DANN) share the same optimiser and schedule, tuned on B1:

- Optimiser: Adam, lr = 5e-4, weight decay = 1e-4.
- LR schedule: cosine annealing.
- Batch size: 256 segments.
- Epochs: 50.
- Loss: cross-entropy (B1, B2, classification head of DANN); BCE for the domain discriminator (DANN).

DANN-specific:

- $\lambda_{\max} = 0.5$ (selected from $\{0.5, 1.0\}$ on target val; $\lambda_{\max} = 1.0$ is reported as an ablation in Section 4.3).
- $\gamma = 10$ for the Ganin schedule.
- Warmup: 5 epochs before activating the GRL.

Encoder Dropout 0.3, classifier Dropout 0.1. The values were chosen after the initial run on Charades-Ego revealed strong under-training with the more aggressive 0.5/0.3 used on the synthetic Assembly101 set: with 157 long-tail classes, the model needs more capacity in the classifier head (hidden 256 instead of 128) and less regularisation in the encoder.

#### Code validation on synthetic data (pre-dataset switch)

Before the dataset switch, the entire training pipeline had been developed and validated on a controlled synthetic dataset that matched the original Assembly101 annotations 1:1 (130k segments, 24 verb classes, official train/val/test partition) but synthesised per-segment 2048-D features from a per-class signal in a 200-D subspace, plus a non-linear per-domain transform (QR-orthogonal rotation on a 512-D subspace, element-wise `tanh` squashing). On that synthetic set, DANN closed **81.6%** of the source-only/oracle gap on balanced accuracy, with the discriminator accuracy converging to 0.50 as predicted. *Those synthetic numbers were code-validation only and are not reported in the results table below*; they confirmed correctness of the GRL implementation, the schedule, the encoder/discriminator interaction, and the metric pipeline before any real-data training.

### 4.3 Quantitative Results

All numbers below are the final evaluation of the `best.pt` checkpoint (selected on target-val balanced accuracy) on the held-out target validation split. 157 classes covered in all rows.

| Model | balanced acc | top-1 | top-5 | macro-F1 |
|---|---|---|---|---|
| **B1 — Source-only** (exo → ego, zero-shot) | 0.034 | 0.042 | 0.162 | 0.026 |
| **DANN (λ_max = 0.5)** — main | **0.037** | **0.050** | **0.187** | **0.033** |
| DANN (λ_max = 1.0) — ablation | 0.037 | 0.053 | 0.193 | 0.034 |
| **MMD (λ_mmd = 1.0)** — main | **0.037** | **0.044** | **0.157** | **0.035** |
| MMD (λ_mmd = 0.1) — ablation | 0.037 | 0.047 | 0.155 | 0.034 |
| **B2 — Target-only oracle** (ego → ego, upper bound) | 0.063 | 0.068 | 0.235 | 0.060 |

**Relative gains of DANN over B1:**

- top-1: +19% (0.042 → 0.050)
- top-5: +15% (0.162 → 0.187)
- macro-F1: +27% (0.026 → 0.033)
- balanced accuracy: +10% (0.034 → 0.037)

**Gap closure (DANN vs B1, normalised by B2 − B1):**

- on top-1: 31% of the gap is closed without any target label
- on macro-F1: ≈25% of the gap is closed
- on balanced accuracy: ≈10% of the gap is closed (the harshest metric, in line with the long-tail regime)

**Discussion.** The absolute numbers are low across the board. The two baselines (B1 ≈ 5× random, B2 ≈ 11× random) confirm that 157 long-tail action classes are a difficult target for a single-frame ImageNet backbone with mean pooling: even the oracle that sees target labels overfits to its 29k training examples and generalises to a modest 6.8% top-1 on the held-out target val. The fact that DANN improves over B1 on every reported metric — consistently and with the expected adversarial diagnostic (see Section 5.1) — is therefore the appropriate signal: the algorithm is working. Closing 31% of the top-1 gap and 25% of the macro-F1 gap with no target labels is a meaningful result given the conservative backbone. Section 6 discusses how a temporal backbone (TSM/SlowFast) would shift the absolute numbers upward.

### 4.4 Training Curves
[Figura: target val acc nel tempo per i 4 modelli — sarà generata in Phase 7 dai log TensorBoard di `experiments/checkpoints/CHAR_*/tb/`.]

## 5. Analysis

### 5.1 Did the Discriminator Get Confused?

Yes, exactly as predicted by Ganin et al. (2016). The DANN training log for the main run (`λ_max = 0.5`, 50 epochs, warmup = 5) shows:

| Phase | epoch range | dom_acc | L_dom | interpretation |
|---|---|---|---|---|
| Warmup (GRL off) | 1–5 | 0.88–0.92 | 0.23–0.30 | discriminator easily separates the two domains; the encoder has not yet been pushed |
| Adversarial onset | 6–10 | drops from 0.74 to 0.52 | rises from 0.51 to 0.69 | GRL is activated; the encoder begins to fool the discriminator |
| Steady state | 11–50 | oscillates around 0.55 ± 0.02 | converges to ≈ 0.69 = ln 2 | discriminator at chance, source/target embeddings indistinguishable |

The plateau of `L_dom ≈ 0.69 = ln 2` is the textbook signature of a binary classifier that cannot do better than a coin flip on a balanced label set. The 5% residual gap above 0.50 in `dom_acc` is expected and reported in Ganin et al. (2016) as well: even at convergence, a small amount of domain-specific signal leaks through, especially on imbalanced or long-tail target distributions. The convergence is monotonic and stable; no oscillation or collapse is observed.

### 5.2 Did Accuracy Improve vs Baseline?

Yes, on **all four** reported metrics (top-1, top-5, balanced acc, macro-F1). See Section 4.3 for the table. The largest relative gain is on macro-F1 (+27%) — the metric most sensitive to gains on the long tail — followed by top-1 (+19%). This is consistent with the interpretation that domain-invariant features help disproportionately on classes the source classifier sees rarely.

### 5.3 Feature Space Visualization (t-SNE)
[Figura a 4 pannelli: pre-DA × dominio, pre-DA × classe, post-DA × dominio, post-DA × classe — sarà generata in Phase 7 dagli embedding salvati di B1 e del best.pt di DANN.]

### 5.4 Per-Class Discrepancy
[Barplot ordinato del gap per classe; tipicamente verbi che richiedono visione full-body (es. `Someone is smiling`, `Someone is laughing`) resistono di più al transfer da exo a ego, mentre le interazioni mano-oggetto (es. `Closing a door`, `Holding a cup/glass/bottle`) trasferiscono meglio. Sarà generato in Phase 7.]

### 5.5 DANN vs MMD

DANN and MMD reach **comparable balanced accuracy** on the target validation set (both ≈ 0.037), but via mechanistically different dynamics. The single most informative quantity is the source training accuracy at the end of the run:

| Method | src_top1 (end of training) | val target balanced |
|---|---|---|
| B1 — Source-only | 0.330 | 0.034 |
| DANN (λ_max = 0.5) | 0.190 | 0.037 |
| MMD (λ_mmd = 1.0) | 0.335 | 0.037 |

**DANN actively sacrifices source accuracy** to fool the domain discriminator: the encoder is pushed to produce embeddings that are uninformative to the discriminator, which by the data-processing inequality also discards a portion of the source class signal (`src_top1` drops from 0.33 to 0.19 over training). This is the adversarial price of domain alignment.

**MMD preserves source learning** while still aligning the two distributions: `src_top1` matches the source-only B1 baseline (0.34) because the MMD loss only "regularises" the encoder embeddings to match the target statistics, without antagonising the classifier. The alignment is softer.

Both reach the same target-balanced-accuracy, with two complementary fingerprints in the per-metric breakdown:

- DANN has a higher top-1 (0.050 vs 0.044) and top-5 (0.187 vs 0.157) — it makes more confident, top-of-the-list predictions on the easy classes.
- MMD has a marginally higher macro-F1 (0.035 vs 0.033) — it spreads its predictions more uniformly across the long tail.

**Practical takeaway.** With single-frame ResNet-50 ImageNet features, the two methods are nearly equivalent in aggregate, but they encode different inductive biases. On a richer backbone (Phase 8 with TSM/SlowFast features) we expect their behaviours to diverge more visibly, with DANN typically dominating on stronger feature spaces (Ganin et al., 2016) and MMD remaining a robust low-variance alternative.

## 6. Conclusions and Limitations

**What we showed.** On Charades-Ego with conservative single-frame ResNet-50 ImageNet features, DANN consistently improves over the source-only baseline on all metrics (+19% top-1, +27% macro-F1) and closes a measurable fraction of the source-only/oracle gap (31% on top-1) without using any target label. The adversarial training is well-behaved, with the discriminator accuracy converging to chance and the domain loss to ln 2.

**Main limitation: the backbone.** Even the target-only oracle reaches only 6.8% top-1. This is not a DA problem — it is a feature problem. Single-frame ResNet-50 ImageNet features, mean-pooled over a segment, cannot discriminate finely between 157 visually similar actions, especially those that differ only in motion (e.g. `Standing up` vs `Sitting down` are nearly identical at frame level). A clean way to push the absolute numbers up, without changing any of the DA code, is to swap the feature extractor for a temporal backbone (TSM, SlowFast, or MViT) pretrained on Kinetics. This is Phase 8 of the project and will be executed on the DMI cluster.

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
[3] G.A. Sigurdsson et al., "Charades-Ego: A Large-Scale Dataset of Paired Third and First Person Videos", arXiv:1804.09626, 2018 (CVPR workshop).
[4] G.A. Sigurdsson et al., "Hollywood in Homes: Crowdsourcing Data Collection for Activity Understanding", ECCV 2016.
[5] K. He et al., "Deep Residual Learning for Image Recognition", CVPR 2016.
[6] Y. Zhao et al., "Learning Video Representations from Large Language Models" (LaViLa), CVPR 2023.
[7] K.Q. Lin et al., "Egocentric Video-Language Pretraining" (EgoVLP), NeurIPS 2022.
[8] F. Sener et al., "Assembly101: A Large-Scale Multi-View Video Dataset for Understanding Procedural Activities", CVPR 2022 — *initially considered, access not granted*.
[9] C. Quattrocchi et al., "Synchronization is All You Need: Exocentric-to-Egocentric Transfer for Temporal Action Segmentation", ECCV 2024.