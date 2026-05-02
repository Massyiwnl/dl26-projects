# Domain Adaptation for Action Recognition — Exocentric → Egocentric

[![Report](https://img.shields.io/badge/Paper-REPORT.md-blue)](docs/REPORT.md)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

## 👥 Group and Project Information

- **Group ID**: CassiaBranca
- **Project ID**: 7

## 📝 Project Description

Implementazione di tecniche di Domain Adaptation per il transfer di modelli di action recognition da viste esocentriche (telecamere fisse) a viste egocentriche (smart glasses) sul dataset Assembly101. Confrontiamo due approcci di feature alignment — adversarial (DANN/GRL) e statistical (MMD multi-kernel) — su feature TSM pre-estratte, valutando la chiusura del gap fra baseline source-only e oracle target-only.

> 📖 **Official Report**: per l'analisi teorica completa, gli esperimenti dettagliati e i contributi individuali, vedere [docs/REPORT.md](docs/REPORT.md).

## 🛠 Technical Reproducibility

### 1. Data and Environment Setup

**Prerequisiti:** conda/miniconda, accesso al cluster DMI (per training su GPU), credenziali Google per il download di Assembly101.

```bash
git clone https://github.com/Massyiwnl/dl26-projects.git
cd dl26-projects
conda env create -f environment.yml
conda activate dl-project
```

**Dataset:** Assembly101 fornisce feature TSM pre-estratte in formato LMDB. Seguire le istruzioni in [`scripts/download_data.sh`](scripts/download_data.sh) per scaricare:
- annotations dal repository ufficiale
- feature LMDB delle 2 viste usate (1 esocentrica + 1 egocentrica)

I dati attesi in `data/raw/` (gitignored). Lo script `scripts/precompute_features.sh` genera la cache `data/processed/` con feature aggregate per segmento.

### 2. Network Training

**Baseline source-only (zero-shot su target):**
```bash
python -m src.training.train --config experiments/configs/baseline_source_only.yaml
```

**Baseline target-only (oracle / upper bound):**
```bash
python -m src.training.train --config experiments/configs/baseline_target_only.yaml
```

**DANN (Adversarial DA con GRL):**
```bash
python -m src.training.train --config experiments/configs/dann.yaml
```

**MMD (extra):**
```bash
python -m src.training.train --config experiments/configs/mmd.yaml
```

Su cluster DMI, sottometti come job SLURM:
```bash
sbatch scripts/slurm/train_dann.slurm
```

### 3. Evaluation

```bash
python -m src.evaluation.evaluate --config experiments/configs/dann.yaml \
    --checkpoint experiments/checkpoints/dann/best.pt
```

Genera in `figures/`:
- confusion matrix per il modello specificato
- t-SNE pre/post DA
- per-class accuracy gap

---

*Per la dichiarazione dei contributi individuali e dell'uso di AI generative, vedere [`docs/REPORT.md`](docs/REPORT.md).*