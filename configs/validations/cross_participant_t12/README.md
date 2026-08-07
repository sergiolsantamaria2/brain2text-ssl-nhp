# Cross-participant SSL generalization (T12 / Brain-to-Text '24)

**Goal:** validate that the winning SSL pretraining — AR-Binary (causal) on the
~10 h macaque somatosensory partition, epoch 400, the checkpoint behind the
PER 0.0887 headline on T15 — also helps when finetuning on a *different* human
participant: T12 from the Brain-to-Text '24 competition (Willett et al., 2023).

T12 is a different brain, a different implant (arrays in ventral 6v and area 44
rather than left precentral gyrus), different sessions and a different speech
corpus, but the same 20 ms binning and 512-feature TX + SBP structure — so the
same encoder and the same finetuning recipe apply without architectural changes.

**Design:** two config families × three seeds = 6 runs.

| Family                              | SSL checkpoint                                             | Seeds        |
|-------------------------------------|------------------------------------------------------------|--------------|
| [`ft_t12_ssl_ar_binary_soma/`](ft_t12_ssl_ar_binary_soma/) | `${CHECKPOINT_DIR}/ssl_study/ar_binary_soma/epoch_0400.pt` | 10, 42, 123 |
| [`ft_t12_no_ssl/`](ft_t12_no_ssl/) (control)               | none (random initialization)                                | 10, 42, 123 |

Apart from the encoder initialization and the random seed, the two families are
identical: the same finetuning recipe that produced the best T15 result
(`head_type: resffn`, `lr_max = 3e-4`, `weight_decay = 5e-4`, cosine schedule
over 200k batches, `white_noise_std = 0.8`, `constant_offset_std = 0.2`,
`log_transform: true`, `epsilon = 1e-8`).

This is the **seed-paired** comparison of the project: unlike the multi-seed T15
runs, which vary the seed of the SSL arm only, both families are run at three
seeds here, which is what establishes the SSL-versus-control gap as reproducible
rather than just the SSL arm's PER as stable.

## Data

The dataset block is loaded from [`dataset_t12.yaml`](dataset_t12.yaml), which
lists every T12 session that ends up with a non-empty `data_train.hdf5`. It is
generated from the Brain-to-Text '24 release by
[`scripts/preprocess_t12.py`](../../../scripts/preprocess_t12.py), which converts
the released `.mat` files into the same per-session HDF5 layout used for T15 and
z-scores each channel per recording block.

## Running

```bash
for seed in 10 42 123; do
    python scripts/finetune.py \
        --config configs/validations/cross_participant_t12/ft_t12_ssl_ar_binary_soma/ft_t12_ssl_ar_binary_soma_seed${seed}.yaml
    python scripts/finetune.py \
        --config configs/validations/cross_participant_t12/ft_t12_no_ssl/ft_t12_no_ssl_seed${seed}.yaml
done
```

## Result

Mean validation PER 0.20599 (SSL) against 0.21123 (control), a **2.5 % relative
improvement**, with no overlap between the two families at any seed. Per-seed
numbers in [docs/results.md §6.2](../../../docs/results.md).
