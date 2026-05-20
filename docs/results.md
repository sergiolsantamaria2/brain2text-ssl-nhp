# Results

This document collects the experimental results of the project. All numbers are
**validation phoneme error rate (PER)** unless otherwise specified, measured on
the official validation split of the corresponding dataset (T15 for
Brain-to-Text '25, T12 for Brain-to-Text '24).

A glossary of terms used throughout:

- **TFS** — Transformer From Scratch, the no-pretraining baseline.
- **Monitoring PER** — the validation PER after a short 30,000-batch
  finetuning of a given SSL checkpoint on T15. Used to select which
  SSL checkpoint to promote to a full 400k finetuning. See
  [architecture.md §7](architecture.md).
- **FT250k / FT400k** — full finetuning at 250,000 / 400,000 batches.


## 1. Brain-to-Text '25 competition: GRU baseline optimization (Chapter 5)

The encoder substrate for the entire thesis is the optimized version of the
official Brain-to-Text '25 GRU baseline. The starting point is the GRU
distributed by the competition organizers: 5 unidirectional layers of 768
hidden units, day-specific affine input, patch kernel 14 / stride 4, CTC loss
over 41 classes, default training schedule (linear LR decay, 100,000 batches).
With its default configuration it reaches **PER ≈ 0.110** on the T15
validation split.

The optimization explored configurations along the following axes; the
sweep configurations live under
[`configs/archive/earliest/01_brain2text25/`](../configs/archive/earliest/01_brain2text25/),
organized by dimension:

| Axis                          | Variants tested                                              | Winning choice                    |
|-------------------------------|--------------------------------------------------------------|-----------------------------------|
| Learning-rate schedule        | Linear decay (default), step decay, cosine annealing + warmup | Cosine + warmup                   |
| Speckled input masking        | Per-channel coordinated dropout at probabilities 0.0–0.4     | p = 0.2                           |
| Training-schedule length      | 100k (default) vs 120k batches                               | 120k (no overfitting signal)      |
| GRU depth / hidden width      | Sweep over layers ∈ {3, 5, 7} and units ∈ {512, 768, 1024}   | 5 layers × 768 units              |
| Inter-layer dropout           | Sweep over {0.2, 0.3, 0.4, 0.5}                              | 0.4                               |
| Input dropout                 | Sweep over {0.0, 0.1, 0.2}                                   | 0.2                               |
| Weight decay                  | Sweep                                                         | 1e-5                              |
| Patch (kernel, stride)        | Sweep                                                         | kernel 14, stride 4               |
| Batch size                    | Sweep                                                         | 32                                |
| Learning rate (max)           | Sweep                                                         | 4 × 10⁻³                          |
| Recurrent core                | GRU vs xLSTM (matched param count)                            | GRU (xLSTM did not match)         |
| Post-GRU head                 | None vs LayerNorm + Linear + activation + Dropout residual block | None (marginal gain, dropped)  |

The winning configuration brings T15 validation PER from **≈0.110 → 0.0978**,
an ~11 % relative improvement. The general empirical lesson — consistent
with what other competitors reported for Brain-to-Text '24 (Deng et al.,
2024) — is that the baseline GRU with the right schedule, the right
regularization, and the right amount of training is genuinely hard to
beat, and that progress on this task is unlikely to come from
architectural tweaks to the recurrence alone.

The canonical winning config:
[`configs/baselines/gru_baseline.yaml`](../configs/baselines/gru_baseline.yaml).
This is the encoder against which every later experiment in the thesis is
measured.


## 2. Matched baselines (GRU + Transformer-From-Scratch)

The two no-pretraining baselines are matched on T15 validation PER:

| Encoder                          | Validation PER | Source                       |
|----------------------------------|:--------------:|------------------------------|
| GRU baseline (BTT'25 + optimized) | **0.0978**     | Chapter 5 (Section 1 above)  |
| Transformer From Scratch (TFS)    | **0.097**      | Chapter 7, BIT recipe + log-transform + aug std 0.8 |

The Transformer-From-Scratch baseline (Chapter 7) is a from-scratch
reconstruction of the BIT specification (Zhang & He et al., 2025) on T15-only
data, with three additions needed to close the gap to the GRU: log-transform
on the input, the RNN-level augmentation values (white noise std 0.8,
constant offset std 0.2 — BIT's transformer-level values underfit), and an
AdamW + cosine schedule tuned by manual sweep within BIT's reported ranges.
Both baselines are reached on T15-only data with no pretraining. They are the
two yardsticks against which every later SSL configuration is compared.


## 3. SSL study on somatosensory (Chapter 8)

Ten SSL pretraining objectives were screened on the 10-hour Chowdhury et al.
(2020) Area 2 / S1 partition, with the monitoring protocol of
[architecture.md §7](architecture.md). The no-SSL transformer baseline in the
same 30k-batch monitoring regime sits at approximately **0.124**.

| Objective                                       | Target type          | Inductive bias                            | Monitoring PER | Promoted to grid? |
|-------------------------------------------------|----------------------|-------------------------------------------|:--------------:|:------------------:|
| Masked reconstruction 50 % (BIT-style)          | MSE / continuous     | Temporal masked spans                      | 0.124          | yes               |
| Masked reconstruction 15 % (BERT-style)         | MSE / continuous     | Light temporal masking                     | 0.126          |                   |
| Masked reconstruction 75 % (MAE-style)          | MSE / continuous     | Aggressive temporal masking                | 0.138          |                   |
| Causal next-step prediction (continuous)        | MSE / continuous     | Temporal autoregression                    | 0.140          |                   |
| Denoising autoencoder                           | MSE / continuous     | Gaussian noise reconstruction              | 0.132          |                   |
| Channel-mask MSE                                | MSE / continuous     | Cross-channel reconstruction               | 0.127          |                   |
| **AR-Binary (causal)**                          | **BCE / binary**     | **Temporal + cross-channel, binarized**    | **0.118**      | **yes**           |
| AR-Binary (bidirectional, dual loss)            | BCE / binary         | Cross-channel + relaxed temporal           | 0.116          | analysed as ablation |
| AR-Binary (bidirectional, hidden-only loss)     | BCE / binary         | Cross-channel only                         | 0.568          | analysed as ablation |
| Contrastive temporal (wav2vec 2.0 / SimCLR)     | InfoNCE              | Adjacent-patch positive pairs              | FT diverged    |                   |

Three observations:

1. **The six MSE-on-continuous-counts objectives cluster around the no-SSL
   baseline.** None of the masked-reconstruction variants (any ratio),
   the continuous next-step objective, the denoising autoencoder, or the
   channel-mask MSE produce a monitoring PER below 0.124.
2. **AR-Binary is the only family that improves on the baseline.** Both the
   causal variant (0.118) and the bidirectional + dual variant (0.116) land
   below 0.124. Removing the temporal half of the dual loss
   (bidirectional + hidden-only, 0.568) collapses the encoder — the dual
   structure is essential.
3. **The contrastive temporal objective optimises its InfoNCE loss
   efficiently but produces a checkpoint that the downstream finetuning
   cannot converge from.** This is the single sharpest demonstration in
   our experiments that pretraining loss does not predict downstream PER,
   and the empirical motivation for the monitoring protocol.

Configs: [configs/ssl_study/](../configs/ssl_study/).


## 4. Cross-dataset grid (Chapter 8)

The two objectives promoted to the grid — masked reconstruction (the BIT
recipe) and AR-Binary (causal) — were each pretrained on all four NHP
partitions of [chapter 4 of the thesis](#references):

| Partition         | Hours | Masked reconstruction | AR-Binary (causal) |
|-------------------|:-----:|:----------------------:|:------------------:|
| Control (no SSL)  | —     | 0.124                  | 0.124              |
| Reaching          | 117 h | 0.142                  | 0.129              |
| Fine motor        | 68 h  | 0.203                  | 0.437              |
| **Somatosensory** | **10 h** | 0.124               | **0.118**          |
| All NHP           | 200 h | 0.137                  | 0.138              |

Three patterns:

1. **Masked reconstruction never improves on the no-SSL control** on any of
   the four partitions, on its own.
2. **AR-Binary (causal) on somatosensory is the only cell that improves**
   on the baseline. Reaching is roughly neutral, fine motor collapses, all
   NHP matches the baseline.
3. **Motor cortex pretraining fails consistently across both objectives.**
   The six motor-cortex cells (three motor partitions × two objectives) all
   sit at or above the no-SSL baseline; five are strictly worse.

This pattern is the empirical anchor of the *motor-cortex-poison*
interpretation: motor-cortex data injects task-specific representations that
compete with attempted-speech motor decoding during finetuning; somatosensory
cortex (Area 2 of S1) provides cortical statistical structure without
committing the encoder to any specific motor task.

Configs: [configs/grid/](../configs/grid/).


## 5. Architectural ablations of AR-Binary (Chapter 8 §8.5)

Two architectural ablations isolate which components of AR-Binary carry the
transfer signal. Both hold the data, schedule, channel-mask ratio,
binarization and hyperparameters fixed, and vary only one architectural
component:

| Variant                                         | Attention      | Loss terms                  | Monitoring PER (soma) | FT400k PER (T15) |
|-------------------------------------------------|----------------|------------------------------|:---------------------:|:----------------:|
| **AR-Binary (causal, dual)** — headline         | causal         | `ar_visible + ar_hidden`     | **0.118**             | **0.0887**       |
| AR-Binary (bidirectional, dual) — ablation 4-ab2 | bidirectional | `ar_visible + ar_hidden`     | 0.116                 | 0.0892           |
| AR-Binary (bidirectional, hidden-only) — ablation 4-ab | bidirectional | `ar_hidden` only        | 0.568                 | —                |

The causal headline and the bidirectional-dual variant are **indistinguishable
on the somatosensory positive cell**, both at monitoring (0.118 vs 0.116) and
at full finetuning (0.0887 vs 0.0892, within seed noise). Removing
`ar_visible` (keeping only the cross-channel `ar_hidden`) collapses the
encoder to monitoring PER 0.568.

Extending the bidirectional-dual variant to the four-partition grid
preserves the motor-cortex-poison pattern (no motor-cortex cell improves
on the no-SSL baseline) but differs from the causal variant in the specific
failure modes:

| Partition         | AR-Binary (causal) | AR-Binary (bidirectional, dual) |
|-------------------|:------------------:|:-------------------------------:|
| Reaching          | 0.129              | 0.125                           |
| Fine motor        | 0.437              | 0.124                           |
| **Somatosensory** | **0.118**          | **0.116**                       |
| All NHP           | 0.138              | 0.764                           |

The bidirectional-dual variant **stabilises fine motor** (0.124 vs the
causal variant's 0.437) but **collapses on all NHP** (0.764 vs 0.138).
The two variants agree on the positive cell but encode the negative cells
differently.

We retain the **causal** variant as the headline configuration on the basis
of multi-seed robustness rather than mean PER: 1 of 3 finetuning seeds
diverged catastrophically for the bidirectional-dual variant on T15, while
0 of 3 causal seeds did. The dual loss is the essential architectural
component; the causal attention direction is dispensable in mean but
provides marginal robustness.

Configs: [configs/ablations/](../configs/ablations/).


## 6. Headline result (Chapter 8 §8.6)

The best monitoring checkpoint — AR-Binary (causal) on somatosensory at
pretraining epoch 400 — was promoted to a full 400,000-batch finetuning on
T15 with the same protocol used for the no-SSL transformer baseline of
[architecture.md §4](architecture.md). The model continues to improve well
past the 30k-batch monitoring window and plateaus at:

> **Validation PER = 0.0887 on T15**
> — a **9.3 % relative improvement** over the optimized GRU baseline of 0.0978,
> and an 8.6 % improvement over the matched TFS baseline of 0.097.

This is the only configuration of architecture, pretraining objective and
NHP dataset, across both the GRU experiments (Chapter 6) and the transformer
experiments (Chapter 8), that produces an improvement on the downstream
phoneme error rate.


## 7. Validations (Chapter 9)

Four independent validations test the result on four axes: reproducibility,
generalization, representational interpretation, and pipeline robustness.
All four are computed against the same pretrained checkpoint (epoch 400 of
AR-Binary on somatosensory).

### 7.1 Multi-seed reproducibility on T15 (§9.2)

Three random seeds, 250,000-batch finetuning, same configuration apart from
the seed:

| Seed       | Validation PER  |
|:----------:|:---------------:|
| 10         | 0.0918          |
| 42         | 0.0931          |
| 123        | 0.0925          |
| **Mean**   | **0.0925 ± 0.0005** |

Configs: [configs/validations/multi_seed_t15/](../configs/validations/multi_seed_t15/).

### 7.2 Cross-participant generalization to T12 (§9.3)

Three random seeds, 200,000-batch finetuning, two encoder initializations
(random init for the control, the AR-Binary somatosensory checkpoint for
the SSL family):

| Family                          | Seed 10  | Seed 42  | Seed 123 | Mean        | Std     |
|---------------------------------|:--------:|:--------:|:--------:|:-----------:|:-------:|
| AR-Binary (causal) somatosensory SSL | 0.20580 | 0.20456 | 0.20761 | **0.20599** | 0.00125 |
| No-SSL (control)                | 0.21071  | 0.21038  | 0.21261  | **0.21123** | 0.00098 |
| **Δ (SSL − control)**           |          |          |          | **−0.00524** |        |

The SSL family is below the no-SSL family for every seed individually
(no overlap). The multi-seed mean improvement on T12 is **2.5 % relative**,
in the same direction as the 9.3 % on T15 but smaller in magnitude.

Configs: [configs/validations/cross_participant_t12/](../configs/validations/cross_participant_t12/).

### 7.3 Representational analysis via CCA (§9.4)

CCA on encoder embeddings at the output of the final shared transformer
block, following the cross-session stability protocol of Gallego et al.
(2018). PCA(10) per session, then top-4 mean canonical correlation
across session pairs.

**Within T15 (820 session pairs):**

| Encoder        | Mean top-4 CC | Std     | Median  |
|----------------|:-------------:|:-------:|:-------:|
| random_init    | 0.1490        | 0.0462  | 0.1398  |
| ft_no_ssl      | 0.1232        | 0.0288  | 0.1193  |
| **ft_ssl**     | **0.1458**    | 0.0325  | 0.1413  |
| ft_ssl_shuf    | 0.0660        | 0.0135  | 0.0644  |
| raw_input      | 0.0932        | 0.0385  | 0.0907  |

**Cross-participant T12 ↔ T15 (984 session pairs):**

| Encoder        | Mean top-4 CC | Std     | Median  |
|----------------|:-------------:|:-------:|:-------:|
| random_init    | 0.1779        | 0.0286  | 0.1745  |
| ft_no_ssl      | 0.1663        | 0.0204  | 0.1635  |
| **ft_ssl**     | **0.1849**    | 0.0220  | 0.1832  |
| ft_ssl_shuf    | 0.0960        | 0.0112  | 0.0941  |
| raw_input      | 0.1343        | 0.0302  | 0.1326  |

In both analyses, the SSL-pretrained encoder achieves the highest mean top-4
canonical correlation of any trained encoder. The relative gap over the
no-SSL finetuned baseline is **+18.3 %** within T15 and **+11.2 %** across
T12 ↔ T15. The shuffled control collapses in both cases (0.066, 0.096),
confirming the structure is real.

The SSL-pretrained encoder produces representations that are
**session-invariant within a single participant and brain-invariant across
two different participants**, in the technical sense that linear projection
between two arbitrary sessions' encoder subspaces preserves more variance
than the same projection on the no-SSL alternative.

### 7.4 Pipeline robustness — FT → SSL → FT (§9.5)

Three-stage pipeline: (1) FT-initial of a random transformer on T15 for
50k batches, (2) SSL pretraining for 400 epochs on somatosensory starting
from the FT-initial weights, (3) FT-final on T15 for 250k batches. Compared
against the standard 1-2 pipeline (random init → SSL → FT 250k) at the
same seed:

| Pipeline                | Stages                              | Final PER (seed 10) | Δ vs 1-2     |
|-------------------------|-------------------------------------|:-------------------:|:------------:|
| **1-2 (standard SSL→FT)** | random init → SSL → FT 250k       | **0.0918**          | —            |
| 2-1-2 (FT-SSL-FT)         | random init → FT 50k → SSL → FT 250k | 0.0945            | +2.9 % (worse) |

Only one seed completed before submission; seeds 42 and 123 were cut by an
extended GPU-cluster maintenance window. The direction is the relevant
signal: the SSL stage does **not** benefit from prior exposure to the
downstream task. The somatosensory AR-Binary representation is a
**complementary regularization** rather than a refinement of pre-existing
phonetic structure.

Configs: [configs/validations/pipeline_2_1_2/](../configs/validations/pipeline_2_1_2/).


## 8. Summary table

| Validation                                          | Metric                           | Result            | Reading                                       |
|-----------------------------------------------------|----------------------------------|:-----------------:|-----------------------------------------------|
| **Headline (T15, FT400k)**                          | Validation PER vs GRU 0.0978     | **0.0887** (−9.3 %) | Single positive cell of the entire study      |
| Multi-seed reproducibility on T15                   | Final PER, mean of 3 seeds       | 0.0925 ± 0.0005   | Effect not seed-dependent                     |
| Cross-participant generalization to T12             | Δ PER vs no-SSL control          | −2.5 % (3 seeds, no overlap) | Same direction across participants |
| Cross-session representational stability (within T15) | CCA, `ft_ssl` vs `ft_no_ssl`   | +18.3 % relative  | SSL preserves cross-session alignment        |
| Cross-participant representational stability        | CCA, `ft_ssl` vs `ft_no_ssl`     | +11.2 % relative  | SSL preserves cross-brain alignment          |
| Pipeline robustness — FT→SSL→FT                     | Δ PER vs SSL→FT, seed 10         | +2.9 % worse (single seed) | SSL does not refine pre-existing phonemes |


## References

- Card, N. S. et al. (2024). NEJM.
- Chowdhury, R. H., Glaser, J. I., Miller, L. E. (2020). eLife.
- Gallego, J. A. et al. (2018). bioRxiv 447441v3.
- Willett, F. R. et al. (2023). Nature.
- Zhang, S., He, B. et al. (2025). BIT preprint.
