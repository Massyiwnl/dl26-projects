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

The Assembly101 dataset (Sener et al., CVPR 2022) was accessed via the official Google Drive distribution maintained by the authors, in compliance with the CC BY-NC 4.0 license. Access was requested on **May 3, 2026** through the team member's personal Google account (the University of Catania uses Microsoft 365 for `@studium.unict.it` accounts, which are not natively Google-Workspace-compatible); the institutional email `cssmsm01b07i754z@studium.unict.it` was disclosed in the request as proof of academic affiliation.

To minimise download size and storage usage, only the **TSM features** (per-frame embeddings, 2048-D, extracted by a TSM-ResNet50 backbone pretrained on EPIC-KITCHENS-100, distributed in LMDB format) and the official **annotations** (from the public `assembly-101/assembly101-annotations` repository) were retrieved, restricted to two views:

- **Source view (exocentric)**: `C10095_rgb` (camera ID `v1` in the paper notation), the frontal fixed RGB camera most commonly used as exocentric reference in the literature.
- **Target view (egocentric)**: `HMC_21176875_mono10bit` (camera ID `e1`), the head-mounted monochrome camera centrally aligned with the participant's hand-object workspace.

The classification task targets **coarse verb classes** (~17 classes after mapping rare verbs), avoiding the long-tail of the 1380 fine-grained verb+noun classes which would obscure the DA signal. The split is performed at the **subject level**, ensuring no participant overlap between train/validation/test partitions, in line with the paper's evaluation protocol.

Initial Drive access lasts 14 days; renewal will be requested if the project requires it.

### 4.2 Implementation Details
- Hardware: cluster DMI UniCT, 1× GPU [modello].
- Hyperparameters: [tabella].
- Training time: [...].

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