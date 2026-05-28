# SLURM scripts per il cluster DMI

Script di sottomissione per il cluster `gcluster.dmi.unict.it`. Tutti i job girano sull'immagine Apptainer ufficiale `/shared/sifs/latest.sif` (PyTorch 2.7.1 + CUDA 11.8) e usano la partizione `dl-course-q2` (4× NVIDIA L40S sul nodo `gnode10`).

## Scripts disponibili

### `smoke_test_b1.sbatch`
Validazione end-to-end del setup: 1 training di B1 source-only a 5 epoche. Usalo dopo aver trasferito il repo + le segment features al cluster, per verificare che tutto giri prima di lanciare il sweep completo.

```bash
sbatch slurm/smoke_test_b1.sbatch
```

Tempo: ~30-60 secondi. Output in `logs/smoke_b1-<JOBID>.log`.

### `run_multi_seed.sbatch`
Esegue **12 training in sequenza** (4 metodi × 3 seed: 42, 123, 7) in un singolo job sbatch. È strutturato così perché la quota utente sul cluster `dl-course-q2` è `MaxSubmitJobsPU=1` (massimo un job in coda per utente, quindi job array di N task viene rifiutato come N submit). Il loop bash interno aggira il vincolo eseguendo i 12 training in serie sulla stessa GPU.

```bash
sbatch slurm/run_multi_seed.sbatch
```

Tempo: ~8 minuti sul nodo `gnode10` (L40S). Output: 12 cartelle `experiments/checkpoints/CLUSTER_{b1,b2,dann,mmd}_seed{42,123,7}/`. Log progressivo in `logs/multi_seed-<JOBID>.log`.

## Aggregazione risultati

Dopo che `run_multi_seed.sbatch` ha terminato, aggrega le 12 run con:

```bash
srun --partition=dl-course-q2 --gres=gpu:1 --time=00:10:00 \
     apptainer exec --nv /shared/sifs/latest.sif \
     python -m src.evaluation.aggregate_multi_seed
```

Stampa metriche per-seed, media ± std, e una tabella markdown pronta per il REPORT.

## Workflow completo dalla zero al risultato

```bash
# 1. Trasferimento repo (login node senza Internet)
tar --exclude=data --exclude=experiments -czf dl26-projects.tar.gz .
scp dl26-projects.tar.gz gcluster:~
ssh gcluster 'cd ~ && tar -xzf dl26-projects.tar.gz -C dl26-projects/'

# 2. Trasferimento segment features (~600 MB)
scp -r data/processed/charades-ego/segment_features gcluster:~/dl26-projects/data/processed/charades-ego/

# 3. Smoke test
ssh gcluster 'cd ~/dl26-projects && sbatch slurm/smoke_test_b1.sbatch'

# 4. Multi-seed sweep
ssh gcluster 'cd ~/dl26-projects && sbatch slurm/run_multi_seed.sbatch'

# 5. Aggregazione (~10 sec)
ssh gcluster 'cd ~/dl26-projects && srun ... python -m src.evaluation.aggregate_multi_seed'
```

## Note implementative

- **No `--qos`**: il nostro utente ha QoS `gpu-large/medium/xlarge` assegnate, ma non è necessario specificarle. SLURM assegna automaticamente quella di default.
- **No `--mail-user`**: nessuna notifica email impostata per evitare spam.
- **`set -e`**: lo script aborta se uno qualsiasi dei 12 training fallisce.
- **Loop bash interno**: scelta forzata dal vincolo `MaxSubmitJobsPU=1`. Su altri cluster un job array (`#SBATCH --array=0-2`) sarebbe più elegante.