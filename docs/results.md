# Results

This document collects the experimental results of the project. All numbers are
**validation phoneme error rate (PER)** unless otherwise specified, measured on
the official validation split of the corresponding dataset (T15 for
Brain-to-Text '25, T12 for Brain-to-Text '24).

A glossary of terms used throughout:

- **TFS** — Transformer From Scratch, the no-pretraining baseline of Chapter 7.
- **Monitoring PER** — the validation PER after a short 30,000-batch
  finetuning of a given SSL checkpoint on T15. Used to rank SSL checkpoints
  on a cheap, downstream-grounded metric (see [architecture.md §7](architecture.md)).
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
| Training-schedule length      | 100k (default) vs longer runs                                | ~70k (the point where validation PER flattened) |
| GRU depth / hidden width      | Sweep over layers ∈ {3, 5, 7} and units ∈ {512, 768, 1024}   | 5 layers × 768 units              |
| Inter-layer dropout           | Sweep over {0.2, 0.3, 0.4, 0.5}                              | 0.4                               |
| Input dropout                 | Sweep over {0.0, 0.1, 0.2}                                   | 0.2                               |
| Weight decay                  | Sweep                                                         | 1e-5                              |
| Patch (kernel, stride)        | Sweep                                                         | kernel 14, stride 4               |
| Batch size                    | Sweep                                                         | 32                                |
| Learning rate (max)           | Sweep                                                         | 4 × 10⁻³                          |
| Recurrent core                | GRU vs xLSTM (matched param count)                            | GRU (xLSTM did not match)         |
| Post-GRU head                 | None vs LayerNorm + Linear + activation + Dropout residual block | None (marginal gain, dropped)  |

The winning configuration brings T15 validation PER from **≈0.110 → 0.0976**,
an ~11 % relative improvement. The general empirical lesson — consistent
with what other competitors reported for Brain-to-Text '24 (Willett et al.,
2024) — is that the baseline GRU with the right schedule, the right
regularization, and the right amount of training is genuinely hard to
beat, and that progress on this task is unlikely to come from
architectural tweaks to the recurrence alone.

The canonical winning config:
[`configs/baselines/gru_baseline.yaml`](../configs/baselines/gru_baseline.yaml).
This GRU is the encoder against which every later experiment in the thesis
is initially measured (the chapter-aligned reference). For the SSL headline
comparison of Section 5 below the relevant control is the **matched-budget
from-scratch transformer** of Section 2, not the GRU.


## 2. Matched baselines (GRU + Transformer-From-Scratch)

Two no-pretraining baselines anchor the rest of the analysis on T15:

| Encoder                          | Validation PER | Source                                                             |
|----------------------------------|:--------------:|--------------------------------------------------------------------|
| GRU baseline (BTT'25 + optimized) | **0.0976**     | Chapter 5 (Section 1 above), ~70k batches                          |
| Transformer From Scratch (TFS)    | **0.097**      | Chapter 7, BIT recipe + log-transform + aug 0.8, ~200k batches     |
| TFS at 400k-batch budget          | **0.0949**     | Same architecture, longer schedule, no-SSL control for §5 headline |

The Transformer-From-Scratch baseline (Chapter 7) is a from-scratch
reconstruction of the BIT specification (Zhang & He et al., 2025) on T15-only
data, with three additions needed to close the gap to the GRU: a log-transform
on the input, the RNN-level augmentation values (white noise std 0.8,
constant offset std 0.2 — BIT's transformer-level values underfit), and an
AdamW + cosine schedule tuned by manual sweep within BIT's reported ranges.

At ~200k batches the TFS reaches PER 0.097, matched to the GRU; **when
retrained under the longer 400,000-batch schedule used for the SSL headline,
the same architecture reaches PER 0.0949** — the no-SSL control against which
the SSL improvement is reported in Section 5.


## 3. Cross-dataset grid: region × objective (Chapter 8)

Three SSL objectives were evaluated across all four NHP partitions of the
corpus, each ranked by the monitoring protocol (30k-batch finetuning of every
saved checkpoint, best monitoring PER reported). The no-SSL control row is
the transformer of Chapter 7 finetuned from scratch for the same 30k batches
(~0.124 monitoring PER).

| Pretraining dataset | Hours | Masked reconstruction (BIT recipe) | AR-Binary (causal) | AR-Binary (bidirectional) |
|---------------------|:-----:|:----------------------------------:|:------------------:|:-------------------------:|
| Control (no SSL)    | —     | 0.124                              | 0.124              | 0.124                     |
| Reaching            | 117 h | 0.142                              | 0.129              | 0.125                     |
| Fine motor          | 68 h  | 0.203                              | 0.437              | 0.124                     |
| **Somatosensory**   | **10 h** | 0.124                           | **0.118**          | **0.116**                 |
| All NHP             | 200 h | 0.137                              | 0.138              | 0.764                     |

Three patterns:

1. **Masked reconstruction never improves on the no-SSL control** on any of
   the four partitions. This is BIT's own pretraining objective, and on its
   own — without the human pretraining data that BIT also uses — it does not
   help.
2. **Only somatosensory + AR-Binary improves on the baseline.** Both
   AR-Binary variants reach monitoring PER below 0.124 on somatosensory; on
   reaching they are roughly neutral, on fine motor and all NHP they collapse
   or are at the baseline. No other (region, objective) pair improves on the
   control.
3. **Motor cortex pretraining fails consistently across all three objectives.**
   The nine motor-cortex cells (three motor partitions × three objectives) all
   sit at or above the no-SSL baseline; five are strictly worse. This is the
   empirical anchor of the *motor-cortex-poison* hypothesis (NHP-specific):
   primate motor cortex carries task-specific dynamical structure that
   competes with attempted-speech motor decoding during finetuning;
   somatosensory cortex (Area 2 of S1) provides cortical statistical structure
   without committing the encoder to any particular motor task.

A separate **hidden-only check** — removing the temporal `ar_visible` term
from AR-Binary, keeping only the cross-channel `ar_hidden` term — collapses
the encoder to monitoring PER 0.568 on somatosensory, confirming that the dual
loss structure is essential and the cross-channel term alone is not
sufficient.

Configs: [`configs/ssl_study/`](../configs/ssl_study/) for the somatosensory
column, [`configs/grid/`](../configs/grid/) for the cross-dataset extensions,
[`configs/ablations/`](../configs/ablations/) for the bidirectional variants
and the hidden-only check.


## 4. Headline result (Chapter 8 §8.4)

Both somatosensory AR-Binary variants are statistically tied at the 30k-batch
monitoring horizon (0.118 causal, 0.116 bidirectional), so both were promoted
to a full 400,000-batch finetuning on T15. The same TFS architecture was
retrained from scratch at the same budget as the no-SSL control. The three
curves are shown side by side in the headline figure
([docs/figures/headline.svg](figures/headline.svg)):

| Configuration                            | Final PER  | Δ vs no-SSL TFS (400k) |
|------------------------------------------|:----------:|:----------------------:|
| no-SSL transformer (TFS, 400k batches)   | 0.0949     | —                      |
| AR-Binary (bidirectional) somatosensory  | 0.0892     | −6.0 % relative        |
| **AR-Binary (causal) somatosensory**     | **0.0887** | **−6.5 % relative**    |

Both AR-Binary variants finish well below the from-scratch baseline. The
bidirectional variant stays below it throughout the schedule; the causal
variant matches the baseline early before pulling clearly below it, and
reaches the lowest final PER.

> **Validation PER = 0.0887 on T15** — a **6.5 % relative improvement** over a
> from-scratch transformer trained at the same 400k-batch budget (0.0949).

The causal variant is adopted as the headline configuration because it reaches
the marginally lower final PER, but on the evidence of these two runs either
would serve equally well. The bidirectional variant differs from the no-SSL
baseline only in the self-supervised pretraining (same architecture, same
schedule), so the 6.0 % improvement it produces confirms the gain is not an
artifact of the causal attention mask.


## 5. Validations (Chapter 9)

Four independent validations test the result on four axes: reproducibility,
generalization, representational interpretation, and pipeline robustness.
All four are computed against the same pretrained checkpoint (epoch 400 of
AR-Binary causal on somatosensory).

### 5.1 Multi-seed reproducibility on T15 (§9.2)

Three random seeds, 250,000-batch finetuning, same configuration apart from
the seed:

| Seed       | Validation PER  |
|:----------:|:---------------:|
| 10         | 0.0918          |
| 42         | 0.0931          |
| 123        | 0.0925          |
| **Mean**   | **0.0925 ± 0.0005** |

Configs: [configs/validations/multi_seed_t15/](../configs/validations/multi_seed_t15/).

### 5.2 Cross-participant generalization to T12 (§9.3)

Three random seeds, 200,000-batch finetuning, two encoder initializations
(random init for the control, the AR-Binary somatosensory checkpoint for
the SSL family):

| Family                          | Seed 10  | Seed 42  | Seed 123 | Mean        | Std     |
|---------------------------------|:--------:|:--------:|:--------:|:-----------:|:-------:|
| No-SSL (control)                | 0.21071  | 0.21038  | 0.21261  | **0.21123** | 0.00098 |
| AR-Binary (causal) somatosensory SSL | 0.20580 | 0.20456 | 0.20761 | **0.20599** | 0.00125 |
| **Δ (SSL − control)**           |          |          |          | **−0.00524** |        |

The SSL family is below the no-SSL family for every seed individually
(no overlap). The multi-seed mean improvement on T12 is **2.5 % relative**,
in the same direction as on T15 but smaller — partly because T12 has fewer
training trials, partly because its cortical implant region differs from
T15's, and partly because the absolute baseline PER on T12 (0.211) is more
than twice T15's, leaving less room for a relative improvement.

Configs: [configs/validations/cross_participant_t12/](../configs/validations/cross_participant_t12/).

### 5.3 Representational analysis via CCA (§9.4)

CCA on encoder embeddings at the output of the final shared transformer
block, following the cross-session stability protocol of Gallego et al.
(2018). PCA(10) per session, then top-4 mean canonical correlation across
session pairs. Five encoder conditions:

- `random_init` — transformer architecture, never trained.
- `ft_no_ssl` — best no-SSL finetuned transformer (Chapter 7 TFS for T15).
- `ft_ssl` — best SSL-pretrained-then-finetuned (AR-Binary causal soma).
- `ft_ssl_shuf` — `ft_ssl` with one member of each pair temporally shuffled
  (sanity-check control).
- `raw_input` — PCA(10) on the 512-dim raw input, no encoder.

| Encoder        | Within-T15 (820 pairs) | Cross-participant T12↔T15 (984 pairs) |
|----------------|:----------------------:|:-------------------------------------:|
| random_init    | 0.1490                 | 0.1779                                |
| ft_no_ssl      | 0.1232                 | 0.1663                                |
| **ft_ssl**     | **0.1458**             | **0.1849**                            |
| ft_ssl_shuf    | 0.0660                 | 0.0960                                |
| raw_input      | 0.0932                 | 0.1343                                |

In both analyses, the SSL-pretrained encoder achieves the highest mean top-4
canonical correlation of any trained encoder. The relative gap over the
no-SSL finetuned baseline is **+18.3 %** within T15 and **+11.2 %** across
T12 ↔ T15. The shuffled control collapses in both cases (0.066, 0.096),
confirming the structure is real.

Reading: the no-SSL finetuning *erodes* the cross-session alignment that the
input already carries (raw_input 0.093, random_init 0.149, ft_no_ssl 0.123)
because the CTC loss on a single participant pushes each session toward its
own discriminative features. SSL pretraining gives the encoder a structure
strong enough that the finetuning loss cannot easily overwrite it — the
representation stays aligned across sessions and across two different brains.
The downstream PER improvement is a consequence of this representational
stability, not of any specific reconstruction quality at pretraining time.

### 5.4 Pipeline robustness — FT → SSL → FT (§9.5)

Three-stage pipeline: (1) FT-initial of a random transformer on T15 for
50k batches, (2) SSL pretraining for 400 epochs on somatosensory starting
from the FT-initial weights, (3) FT-final on T15 for 250k batches. Compared
against the standard 1-2 pipeline (random init → SSL → FT 250k) at the
same seed:

| Pipeline                | Stages                                     | Final PER (seed 10) | Δ vs 1-2     |
|-------------------------|--------------------------------------------|:-------------------:|:------------:|
| **1-2 (standard SSL→FT)** | random init → SSL → FT 250k              | **0.0918**          | —            |
| 2-1-2 (FT-SSL-FT)         | random init → FT 50k → SSL → FT 250k     | 0.0945              | +2.9 % (worse) |

The SSL stage does not benefit from prior exposure to the downstream task. If
anything, it is marginally worse off. This is consistent with the
motor-cortex-poison reading from the opposite side: the SSL is not refining
pre-existing phonetic structure, it is supplying a complementary cortical
regularization, and the standard 1-2 pipeline combines that with the
supervised CTC stage more cleanly than the 2-1-2 one.

Configs: [configs/validations/pipeline_2_1_2/](../configs/validations/pipeline_2_1_2/).


## 6. Summary table

| Validation                                          | Metric                           | Result                          | Reading                                       |
|-----------------------------------------------------|----------------------------------|:-------------------------------:|-----------------------------------------------|
| **Headline (T15, FT400k vs no-SSL TFS at same budget)** | Validation PER vs TFS 0.0949 | **0.0887** (−6.5 %)             | NHP-only pretraining (somatosensory + AR-Binary) improves the decoder |
| Multi-seed reproducibility on T15                   | Final PER, mean of 3 seeds       | 0.0925 ± 0.0005                 | Effect not seed-dependent                     |
| Cross-participant generalization to T12             | Δ PER vs no-SSL control          | −2.5 % (3 seeds, no overlap)    | Same direction across participants            |
| Cross-session representational stability (within T15) | CCA, `ft_ssl` vs `ft_no_ssl`   | +18.3 % relative                | SSL preserves cross-session alignment         |
| Cross-participant representational stability        | CCA, `ft_ssl` vs `ft_no_ssl`     | +11.2 % relative                | SSL preserves cross-brain alignment           |
| Pipeline robustness — FT→SSL→FT                     | Δ PER vs SSL→FT, seed 10         | +2.9 % worse (single seed)      | SSL is complementary regularization, not phonetic refinement |


## References

- Card, N. S. et al. (2024). NEJM.
- Chowdhury, R. H., Glaser, J. I., Miller, L. E. (2020). eLife.
- Gallego, J. A. et al. (2018). Nature Communications.
- Willett, F. R. et al. (2023). Nature.
- Willett, F. R. et al. (2024). *Brain-to-Text Benchmark '24: Lessons learned.* arXiv:2412.17227.
- Zhang, Y., He, L. et al. (2025). *Decoding inner speech with an end-to-end brain-to-text neural interface.* arXiv:2511.21740.
