# Domain Adaptation for Action Recognition — Exocentric → Egocentric

[![Report](https://img.shields.io/badge/Paper-REPORT.md-blue)](docs/REPORT.md)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

## 👥 Group and Project Information

- **Group ID**: CassiaBranca
- **Project ID**: 7 (Track 7 — Domain Adaptation Exocentric → Egocentric)
- **Membri**: Massimiliano Cassia (1000016487), Martina Brancaforte (1000015074)
- **Corso**: Deep Learning, A.A. 2025/26 — Prof. A. Furnari, Università di Catania

## 📝 Project Description

Implementazione di tecniche di Domain Adaptation (DA) non supervisionata per il transfer di modelli di action recognition da viste esocentriche (telecamere fisse, terza persona) a viste egocentriche (telecamere montate sulla testa, prima persona), sul dataset **Charades-Ego** (Sigurdsson et al., CVPR 2018). Confrontiamo due famiglie di approcci di feature alignment — **DANN** (adversarial via Gradient Reversal Layer, Ganin et al. 2016) e **MMD** (multi-kernel Gaussian, Long et al. 2015) — su feature ResNet-50 ImageNet pre-estratte single-frame mean-pooled per segmento, valutando la chiusura del gap fra baseline source-only e oracle target-supervised.

> 📖 **Report ufficiale**: per analisi teorica, esperimenti dettagliati, figure e contributi individuali → **[docs/REPORT.md](docs/REPORT.md)**.

## 🎯 Risultati principali

| Modello | balanced acc | top-1 | top-5 | macro-F1 |
|---|---|---|---|---|
| B1 — Source-only (zero-shot exo→ego) | 0.041 ± 0.001 | 0.047 ± 0.002 | 0.162 ± 0.007 | 0.030 ± 0.004 |
| **DANN** (λ_max = 0.5) | **0.041 ± 0.001** | **0.053 ± 0.001** | **0.188 ± 0.002** | **0.036 ± 0.002** |
| **MMD** (λ_mmd = 1.0) | **0.042 ± 0.001** | **0.053 ± 0.003** | **0.188 ± 0.008** | **0.032 ± 0.002** |
| B2 — Target-only oracle | 0.067 ± 0.003 | 0.073 ± 0.003 | 0.243 ± 0.005 | 0.058 ± 0.003 |

Media ± std su 3 seed (42, 123, 7), su `val_target` (5135 segmenti, 157 classi). Gap closure top-5: ~32% per entrambi DANN e MMD.

## 🛠 Riproducibilità tecnica

### 1. Setup ambiente

```bash
git clone https://github.com/Massyiwnl/dl26-projects.git
cd dl26-projects
conda env create -f environment.yml
conda activate dl-project
```

### 2. Setup dati (Charades-Ego)

Charades-Ego è scaricabile direttamente da Allen AI:

```bash
mkdir -p data/raw/charades-ego/CharadesEgo
cd data/raw/charades-ego

# Annotations (~3 MB)
wget https://prior.allenai.org/projects/data/charades-ego/CharadesEgo.zip
unzip CharadesEgo.zip

# Videos (~11 GB)
wget https://prior.allenai.org/projects/data/charades-ego/CharadesEgo_v1_480.tar
tar -xf CharadesEgo_v1_480.tar
```

Atteso in `data/raw/charades-ego/`:
- `CharadesEgo/` con i 6 CSV (`CharadesEgo_v1_{train,test}{,_only1st,_only3rd}.csv`) + `Charades_v1_classes.txt`
- `CharadesEgo_v1_480/` con 7860 `.mp4` (480p)

### 3. Estrazione feature (one-shot, ~60 min su RTX 3060)

ResNet-50 ImageNet-1K, frame sampling a 5 fps, mean-pool per segmento:

```bash
# Step 1: estrai feature frame-level (5 GB su disco)
python -m src.datasets.extract_features \
    --video-dir data/raw/charades-ego/CharadesEgo_v1_480 \
    --output-dir data/processed/charades-ego/frame_features \
    --fps 5

# Step 2: aggrega in feature per segmento (.npz, ~600 MB)
python -m src.datasets.precompute_segment_features_charades \
    --frame-dir data/processed/charades-ego/frame_features \
    --csv-dir data/raw/charades-ego/CharadesEgo \
    --output-dir data/processed/charades-ego/segment_features
```

Output: 6 file `.npz` (train/val/test × source/target).

### 4. Training (locale)

```bash
# B1 — baseline source-only
python -m src.training.trainer_supervised \
    --train-npz data/processed/charades-ego/segment_features/train_source.npz \
    --val-npz   data/processed/charades-ego/segment_features/val_target.npz \
    --output-dir experiments/checkpoints/B1_source_only \
    --epochs 50 --batch-size 256 --lr 5e-4 --weight-decay 1e-4 --seed 42

# B2 — oracle target-only
python -m src.training.trainer_supervised \
    --train-npz data/processed/charades-ego/segment_features/train_target.npz \
    --val-npz   data/processed/charades-ego/segment_features/val_target.npz \
    --output-dir experiments/checkpoints/B2_target_only \
    --epochs 50 --batch-size 256 --lr 5e-4 --weight-decay 1e-4 --seed 42

# DANN — adversarial DA
python -m src.training.trainer_dann \
    --train-source data/processed/charades-ego/segment_features/train_source.npz \
    --train-target data/processed/charades-ego/segment_features/train_target.npz \
    --val-target   data/processed/charades-ego/segment_features/val_target.npz \
    --output-dir experiments/checkpoints/DANN_lambda05 \
    --epochs 50 --batch-size 256 --lr 5e-4 --weight-decay 1e-4 \
    --lambda-max 0.5 --warmup-epochs 5 --seed 42

# MMD — statistical DA
python -m src.training.trainer_mmd \
    --train-source data/processed/charades-ego/segment_features/train_source.npz \
    --train-target data/processed/charades-ego/segment_features/train_target.npz \
    --val-target   data/processed/charades-ego/segment_features/val_target.npz \
    --output-dir experiments/checkpoints/MMD_lambda10 \
    --epochs 50 --batch-size 256 --lr 5e-4 --weight-decay 1e-4 \
    --lambda-mmd 1.0 --warmup-epochs 2 --seed 42
```

### 5. Training su cluster DMI (multi-seed, ~8 min totali per 12 run)

```bash
sbatch slurm/run_multi_seed.sbatch
```

Vedi `slurm/README.md` per dettagli completi.

### 6. Aggregazione multi-seed

```bash
python -m src.evaluation.aggregate_multi_seed
```

Stampa metriche per-seed, media ± std, e tabella markdown pronta per il REPORT.

### 7. Notebooks di analisi

- `notebooks/01_charades_ego_exploration.ipynb` — EDA su Charades-Ego (figure 5-8)
- `notebooks/02_analysis_charades.ipynb` — analisi post-training: training curves, t-SNE, per-class gap, confusion matrix differenziale (figure 9-13)

## 📁 Struttura della repo

```
dl26-projects/
├── src/
│   ├── datasets/      # parser Charades-Ego, feature extraction, segment precompute
│   ├── models/        # FeatureEncoder, ActionClassifier, GRL, DomainDiscriminator
│   ├── losses/        # multi-kernel Gaussian MMD
│   ├── training/      # trainer_supervised (B1/B2), trainer_dann, trainer_mmd
│   ├── evaluation/    # metriche aggregate, multi-seed aggregation
│   └── utils/         # checkpoint I/O, schedule λ, seed RNG
├── slurm/             # SLURM sbatch scripts per il cluster DMI
├── notebooks/         # EDA + analysis notebooks
├── figures/           # 13 figure auto-generate (PNG)
├── docs/REPORT.md     # Report ufficiale del progetto
├── experiments/       # checkpoints/, configs/, logs/ (gitignored)
└── data/              # raw/, processed/ (gitignored, ~12 GB)
```

## 📖 Reference

Il progetto è descritto in dettaglio in **[`docs/REPORT.md`](docs/REPORT.md)**. Per la dichiarazione dei contributi individuali e dell'uso di AI generative, vedere le sezioni 7-8 del REPORT.