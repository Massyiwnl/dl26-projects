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
Implementare e confrontare tecniche di Domain Adaptation (DA) che trasferiscano la conoscenza semantica da source esocentrico labeled a target egocentrico unlabeled, sul dataset Assembly101.

### 1.3 Contributions
- Setup riproducibile per benchmarking di DA exo→ego su feature TSM pre-estratte.
- Implementazione e tuning di DANN (GRL) e MMD multi-kernel.
- Analisi sistematica della convergenza del domain discriminator e del miglioramento di accuracy.
- Visualizzazione t-SNE pre/post DA e analisi per-classe del gap residuo.

## 2. Related Work
- Domain-Adversarial Neural Networks (Ganin & Lempitsky, 2015; Ganin et al., 2016).
- Maximum Mean Discrepancy alignment (Long et al., 2015 — DAN).
- Assembly101 (Sener et al., CVPR 2022).
- Exo→Ego transfer recente (Quattrocchi et al., ECCV 2024).

## 3. Method

### 3.1 Problem Setup
[Definizione formale: source $\mathcal{D}_s = \{(x_i^s, y_i^s)\}$, target $\mathcal{D}_t = \{x_j^t\}$, classi $\mathcal{Y}$, dominio comune $\mathcal{X}$.]

### 3.2 Feature Backbone
TSM (Temporal Shift Module) pretrained su EPIC-KITCHENS-100, feature 2048-D per segmento aggregate via mean pool.

### 3.3 Architecture
- Feature encoder $g_\theta$
- Action classifier $h_\phi$
- Domain discriminator $d_\psi$ (solo per DANN)

### 3.4 DANN (Adversarial DA)
[Loss totale, GRL forward/backward, schedule di λ secondo Ganin et al.: $\lambda_p = \frac{2}{1+\exp(-\gamma p)} - 1$]

### 3.5 MMD (Statistical DA)
[Multi-kernel Gaussian MMD², median heuristic per σ.]

## 4. Experiments

### 4.1 Dataset and Splits

The Assembly101 dataset (Sener et al., CVPR 2022) was accessed via the official Google Drive distribution maintained by the authors, in compliance with the CC BY-NC 4.0 license. Annotation access was granted on May 6, 2026; access to the main TSM-features Drive is being processed separately.

**Annotation files used.** We use the official `fine-grained-annotations`, which provide one row per `(segment, view)` pair with `start_frame`, `end_frame`, `verb_id`, `verb_cls`, and a flag `is_RGB` distinguishing fixed cameras (exocentric) from head-mounted ones (egocentric). The official train/validation/test splits are adopted as published. Subjects overlap across splits in the official protocol; we respect this protocol to remain comparable to published baselines on Assembly101.

**Views.** To align with the standard exocentric/egocentric DA setup in the literature, we restrict the data to two views:

- **Source view (exocentric):** `C10095_rgb` (camera ID `v1`), the frontal fixed RGB camera most commonly used as exocentric reference in Assembly101 papers.
- **Target view (egocentric):** `HMC_21176875_mono10bit` (camera ID `e1`), the head-mounted monochrome camera centrally aligned with the participant's hand-object workspace.

**Task.** We classify the **24 verb classes** of the fine-grained annotation (`verb_id`), rather than the 1,380 fine-grained `action_id`. This avoids the long-tail of the action space (1,380 classes are dominated by rare verb-noun combinations and would dilute the DA signal), while keeping a class set rich enough to expose interesting per-class viewpoint-resistance behaviour required by Extra Objective 3 of the track.

**Data scale (post-filtering).** Selecting only the source and target views above:

| Split | Source segments | Target segments |
|---|---|---|
| Train | 47,649 | 24,743 |
| Validation | 15,696 | 9,416 |
| Test | 21,934 | 11,307 |

All 24 verb classes are present in both domains. Of the 211 sequences with the source view in the train split, 102 also include the target view; the DA framework samples source and target independently, so the asymmetry is not a problem in practice.

**Long-tail and metrics.** The class distribution is heavily long-tailed: `pick up` and `put down` together cover 36% of training segments, while the rarest verb (`shake`) accounts for less than 0.4%. Consequently, alongside top-1 accuracy we report **balanced accuracy** and **macro-F1**, which are the meaningful aggregate metrics in this regime.

**Domain-conditioned class skew.** A useful preliminary observation from the EDA (see `notebooks/00_data_exploration.ipynb` and Figure 4): the egocentric target view captures `inspect` and the `attempt to *` verbs disproportionately more than the exocentric source view, while the source view captures completed actions (`pick up`, `put down`, `position`, `unscrew`, `remove`) more. This is consistent with head-mounted cameras catching close-range examinations more easily and fixed cameras catching wide-range manipulation gestures more easily. We will revisit this asymmetry in Section 5.4 (per-class analysis).

**Segment length.** Train segments have a median length of 28 frames @ 30 fps in the source view and 30 frames in the target view (≈1 second). Mean/p95 are 49.9/171 frames source, 54.5/187 frames target. Given the moderate length, segment-level features are obtained by **mean-pooling all the frame-level TSM embeddings within each segment**, with no temporal sub-sampling.

### 4.2 Implementation Details

#### Data pipeline (3-step)

The Assembly101 TSM features are distributed in LMDB format on the official Google Drive. Following the practice recommended by subsequent papers using this dataset — most notably **LTContext** (Bahrami et al., ICCV 2023), which states explicitly that *"loading from numpy can be faster, you can convert the .lmdb features to numpy"* — we adopt a 3-step pre-processing pipeline that decouples the heavy I/O from the training loop:

1. **`src/datasets/lmdb_to_npy.py`** *(one-shot, run on the cluster on real data)*: extract per-frame TSM features from the LMDB and save one `.npy` per `(sequence_id, view)` pair, restricted to the 2 chosen views. Output shape per file: `(N_frames, 2048)` float32.
2. **`src/datasets/precompute_segment_features.py`** *(one-shot)*: for each row of the official `train.csv`/`validation.csv`/`test.csv` filtered on the 2 views, slice the corresponding `.npy` over the segment frame range `[start_frame, end_frame]` and **mean-pool** to a single 2048-D vector. The output is a small set of `.npz` files (one per `split × domain`) with arrays `features (N, 2048) float32`, `labels (N,) int64` (verb_id), and `segment_ids (N,) int64` for traceability.
3. **`src/datasets/assembly101.py`** + **`src/datasets/pair_loader.py`**: a PyTorch `Dataset` that loads the `.npz` eagerly into RAM (~700 MB at full scale, fits comfortably) for zero-cost `__getitem__`, plus a `PairedDomainIterator` that cycles independently over the source and target loaders to feed the DANN/MMD trainers.

This separation has three concrete benefits. First, the LMDB is touched only once. Second, the segment-level pre-computation can be re-run cheaply if the task definition changes (e.g. switching from verb to action classification). Third, training epochs are bottlenecked only by GPU compute — a typical epoch is in the order of seconds on the L40S of the DMI cluster.

#### Synthetic dataset for local development

Since the real LMDB requires a separate access request to the main Assembly101 Drive (still pending at the time of this section), we developed and validated the entire pipeline on a synthetic dataset that mimics the expected output of step 1. The script `src/datasets/make_synthetic_segment_features.py` generates one segment-level vector per CSV row by sampling from a per-class Gaussian centred on `class_mean[verb_id]` plus a domain-specific bias `±d_vec` of norm 5. This produces 130k synthetic segments at full scale (the same numbers reported in Section 4.1) in less than 2 minutes, occupying ~1 GB of disk. Once Drive access is granted, switching to the real data will only require running steps 1 and 2 of the pipeline; downstream code (Datasets, loaders, trainers, evaluators) is unchanged.

#### Hardware

Local development is performed on a laptop with NVIDIA GeForce RTX 3060 (6 GB VRAM, CUDA 12.1, PyTorch 2.4.1). All final runs will be executed on the DMI cluster (`gcluster.dmi.unict.it`), specifically on `gnode10` (4 × NVIDIA L40S, 48 GB VRAM each) under the `dl-course-q2` partition with `gpu-medium` QoS (8 GB RAM, 5.6 GB VRAM, 6 h time limit), inside the official Apptainer image `/shared/sifs/latest.sif`.

#### Hyperparameters

To be filled in Section 4.3 once baselines and DANN/MMD have been trained.

#### Code validation on synthetic data (pre-Drive)

Pending access to the official Assembly101 TSM features Drive, the entire
training pipeline was validated on a controlled synthetic dataset that
matches the official annotations 1:1 (same 130k segments, same 24 verb
classes, same train/val/test partition) but synthesizes per-segment 2048-D
features from a per-class signal in a 200-D subspace, plus a non-linear
per-domain transform (QR-orthogonal rotation on a 512-D subspace, element-wise
`tanh` squashing, per-domain scale, and translation of norm 6). The
non-linearity is essential: a purely linear shift can be undone by the first
encoder layer, eliminating the DA signal we want to test.

On this synthetic set, baselines and DANN confirm the expected behaviour:

| Model | Train | Eval (target val) | top-1 | balanced acc | macro-F1 |
|---|---|---|---|---|---|
| B1 — Source-only | source labelled | target unseen | 0.663 | 0.683 | 0.527 |
| **DANN (GRL)** | source labelled + target unlabelled | target unseen | **0.994** | **0.941** | **0.953** |
| B2 — Target-only (oracle) | target labelled | target | 0.999 | 0.999 | 0.998 |

DANN closes **81.6%** of the source-only/oracle gap on balanced accuracy
(`+0.258` of the `+0.316` headroom), with the domain discriminator
accuracy converging to ~0.50 by epoch 6 — the expected signature of a
successful adversarial alignment as in Ganin et al. (2016). *These
synthetic numbers are for code validation only. Final results will be
reported on the real Assembly101 LMDB features.*

### 4.3 Quantitative Results
| Modello | Top-1 ↑ | Top-5 ↑ | Balanced Acc ↑ | Macro-F1 ↑ |
|---|---|---|---|---|
| B1 — Source-only | | | | |
| B2 — Target-only (oracle) | | | | |
| DANN (GRL) | | | | |
| MMD | | | | |

### 4.4 Training Curves
[Figura: target val acc nel tempo per i 4 modelli.]

## 5. Analysis

### 5.1 Did the Discriminator Get Confused?
[Curva domain discriminator accuracy vs step, mostrando convergenza a ~0.5. Risposta esplicita al minimum-objective 4 della track.]

### 5.2 Did Accuracy Improve vs Baseline?
[Tabella DANN vs B1 con delta percentuale; commento.]

### 5.3 Feature Space Visualization (t-SNE)
[Figura a 4 pannelli: pre-DA × dominio, pre-DA × classe, post-DA × dominio, post-DA × classe.]

### 5.4 Per-Class Discrepancy
[Barplot ordinato del gap per classe; quali verbi resistono al transfer e perché — discussione.]

### 5.5 DANN vs MMD
[Confronto, tradeoff, quando uno batte l'altro.]

## 6. Conclusions and Limitations
[Cosa abbiamo dimostrato, cosa è ancora aperto: long-tail, robustezza alla scelta delle viste, impatto del backbone TSM fisso.]

## 7. Group Contributions

| Membro | Contributi principali |
|---|---|
| Massimiliano Cassia | [es. data pipeline, baseline, report] |
| Martina Brancaforte | [es. DANN implementation, t-SNE, slide] |

## 8. AI Usage Declaration

In accordo con la policy del corso (INSTRUCTIONS.md §5):

- **Claude (Anthropic)**: usato per brainstorming del piano di sviluppo iniziale, sanity check di formule (GRL, MMD), debugging di errori PyTorch, generazione di docstring boilerplate. Le scelte strategiche (selezione viste source/target, coarse vs fine-grained, λ schedule, iperparametri) sono frutto del gruppo, motivate nelle sezioni Method ed Experiments.
- **GitHub Copilot** / [altri]: [se usato — autocompletamento su utility minori. Nessuna funzione core generata senza revisione integrale].

Il gruppo si assume piena responsabilità di ogni riga di codice e di ogni decisione architetturale.

## References

[1] Y. Ganin et al., "Domain-Adversarial Training of Neural Networks", JMLR 2016.  
[2] M. Long et al., "Learning Transferable Features with Deep Adaptation Networks", ICML 2015.  
[3] F. Sener et al., "Assembly101: A Large-Scale Multi-View Video Dataset for Understanding Procedural Activities", CVPR 2022.  
[4] J. Lin et al., "TSM: Temporal Shift Module for Efficient Video Understanding", ICCV 2019.  
[5] C. Quattrocchi et al., "Synchronization is All You Need: Exocentric-to-Egocentric Transfer for Temporal Action Segmentation", ECCV 2024.  
[Aggiungere paper effettivamente citati nel testo.]