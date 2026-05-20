# Reproducing the headline result

This guide walks through reproducing the headline number of this work — a
**validation phoneme error rate of 0.0887 on T15**, obtained by pretraining
the transformer with the AR-Binary (causal) objective on ~10 hours of
macaque Area 2 / S1 data (Chowdhury et al., 2020), then finetuning on T15.

The full pipeline has four stages: data preparation, pretraining, finetuning,
and evaluation.


## 0. Setup

```bash
# Clone and install in editable mode (Python ≥ 3.10)
git clone <repository-url> brain2text-ssl-nhp
cd brain2text-ssl-nhp
python -m venv .venv && source .venv/bin/activate
pip install -e .

# Point the configs at your local data and output directories
export DATA_DIR=/path/to/your/data
export CHECKPOINT_DIR=/path/to/your/checkpoints
export OUTPUT_DIR=/path/to/your/outputs
export PROJECT_ROOT=/path/to/your/runs
```

All configs reference these four environment variables via `${DATA_DIR}`,
`${CHECKPOINT_DIR}`, `${OUTPUT_DIR}`, and `${PROJECT_ROOT}`. OmegaConf
resolves them at load time from the process environment.


## 1. Data preparation

### 1.1 T15 (Brain-to-Text '25)

The T15 release is distributed already preprocessed (20 ms binned threshold
crossings + spike-band power, 512 features per bin, organized as per-session
HDF5). Place the release at `${DATA_DIR}/hdf5_data_final/`; each session
directory should contain `data_train.hdf5` and `data_val.hdf5` with groups
named `trial_<NNNN>`.

### 1.2 NHP somatosensory partition (Chowdhury et al., 2020)

Download the Chowdhury 2020 release (Han, Chips and Lando, three monkeys,
~10 hours of Area 2 / S1 recordings) and place the raw `.mat` files at
`${DATA_DIR}/monkey_data/chowdhury_2020/`. Then run:

```bash
python scripts/preprocess_nhp.py \
    --datasets chowdhury_2020 \
    --monkey-data-root ${DATA_DIR}/monkey_data \
    --output-dir ${DATA_DIR}/nhp_pretrain
```

The output is one subdirectory per subject under
`${DATA_DIR}/nhp_pretrain/chowdhury_2020_<Subject>/`, each containing
per-session HDF5 files of z-scored binned spike counts.

For the cross-dataset grid of the SSL study, the same script handles the
other NHP datasets via `--datasets 000070 000121 000128 000129 000688 000941 001201 odoherty_2017 ma_2023`.


## 2. SSL pretraining

Pretrain the transformer with the AR-Binary (causal) objective for 400
epochs on the somatosensory partition:

```bash
python scripts/pretrain.py \
    --config configs/ssl_study/ar_binary_causal_soma.yaml
```

By default the run saves a checkpoint every 20 epochs to
`${CHECKPOINT_DIR}/ssl_study/ar_binary_soma/`. The checkpoint to promote to
the full finetuning is `epoch_400.pt`.

### Optional: pretraining monitoring

The monitoring protocol (architecture.md §7) runs a short 30k-batch
finetuning of each saved checkpoint on T15 and reports the best monitoring
PER:

```bash
# Per-checkpoint finetuning configs are under configs/validations/pretrain_monitor/
for cfg in configs/validations/pretrain_monitor/ft_200k/*.yaml; do
    python scripts/finetune.py --config "$cfg" \
        --set model.ssl_checkpoint=${CHECKPOINT_DIR}/ssl_study/ar_binary_soma/epoch_${EPOCH}.pt
done
```

For the headline run, the best monitoring PER is reached at epoch 400.


## 3. Finetuning on T15

Promote the best monitoring checkpoint (epoch 400) to a full 400,000-batch
finetuning:

```bash
python scripts/finetune.py \
    --config configs/validations/multi_seed_t15/ft_seed10.yaml \
    --set model.ssl_checkpoint=${CHECKPOINT_DIR}/ssl_study/ar_binary_soma/epoch_400.pt \
    --set num_training_batches=400000
```

Final validation PER should reach **0.0887 ± seed noise** by the end of the
schedule. The multi-seed-FT250k mean across seeds 10/42/123 is
**0.0925 ± 0.0005** (the 250k horizon is shorter than 400k; the headline
0.0887 is the seed-10 number at 400k batches — see
[results.md §5](../docs/results.md) and §6.1).


## 4. Evaluation

### 4.1 Validation PER

The trainer reports per-epoch validation PER to the log and to W&B. The
"best" checkpoint is the one minimizing validation PER.

### 4.2 Cross-session CCA (within T15)

```bash
python scripts/evaluate_cca.py \
    --ckpt-no-ssl   ${CHECKPOINT_DIR}/tfs_baseline/best_model.pt \
    --ckpt-ssl      ${CHECKPOINT_DIR}/multi_seed_t15/seed10/best_model.pt \
    --base-config   configs/baselines/gru_defaults.yaml \
    --out-root      ${OUTPUT_DIR}/analysis/cca_within_t15
```

Outputs `cca_pairs.csv`, `summary.csv`, and the two figures (the boxplot
and the CC-decay plot). Expected mean top-4 CC: `ft_ssl ≈ 0.146` vs
`ft_no_ssl ≈ 0.123` — a +18.3 % relative gap.

### 4.3 Cross-participant CCA (T12 ↔ T15)

Requires a T12 finetuning checkpoint paired with the T15 one:

```bash
python scripts/evaluate_cca_cross_participant.py \
    --ckpt-no-ssl-t15  ${CHECKPOINT_DIR}/tfs_baseline/best_model.pt \
    --ckpt-no-ssl-t12  ${CHECKPOINT_DIR}/t12_no_ssl/best_model.pt \
    --ckpt-ssl-t15     ${CHECKPOINT_DIR}/multi_seed_t15/seed10/best_model.pt \
    --ckpt-ssl-t12     ${CHECKPOINT_DIR}/t12_ssl/best_model.pt \
    --out-root         ${OUTPUT_DIR}/analysis/cca_cross_participant
```

Expected mean top-4 CC: `ft_ssl ≈ 0.185` vs `ft_no_ssl ≈ 0.166` — a +11.2 %
relative gap.


## 5. Notes on reproducibility

- **Seed.** The headline 0.0887 is the seed-10 FT400k number. The
  multi-seed mean at FT250k is 0.0925 ± 0.0005. The exact number at FT400k
  varies by a few times 10⁻⁴ between seeds.
- **Data versions.** The Chowdhury 2020 release and the Brain-to-Text '25
  release are both static; both were downloaded once and not updated during
  the project. The NHP-side hours reported (~10 h for somatosensory) are
  after the session-quality filtering documented in
  [src/brain2text/data/preprocess_nhp.py](../scripts/preprocess_nhp.py)
  (minimum 30-second session duration; minimum 10 total spikes per channel
  scaled by session length).
- **Optimizer non-determinism.** AdamW with fused implementation on CUDA
  introduces small non-determinism even at fixed seed. The reported
  numbers were obtained on A100 / H100 hardware; reproductions on other
  hardware may shift by ~10⁻⁴ in PER.
