# Architecture

This document describes the encoder architecture, the self-supervised pretraining
recipe, and the monitoring protocol used in this work. It is a technical companion
to the README; experimental results are summarized in [results.md](results.md).

The implementation is in [src/brain2text/](../src/brain2text/). All hyperparameters
referenced below appear in the YAML configurations under [configs/](../configs/).


## 1. The decoding task

The input is a sequence of 20 ms binned neural features
`x_1, ..., x_T ∈ R^512` recorded from four Utah arrays (256 threshold-crossing
channels + 256 spike-band-power channels). The output is a sequence of phonemes
`y_1, ..., y_L` over a 41-symbol vocabulary (40 ARPABET phonemes + CTC blank).
The mapping is supervised with **Connectionist Temporal Classification (CTC)**
(Graves et al., 2006); the input is substantially longer than the output and
the alignment between them is monotonic, which is exactly what CTC exploits.

The encoder produces one phoneme-logit vector per input bin; at evaluation time
the per-bin logits are greedy-decoded and collapsed into a phoneme sequence
through the CTC collapse function (merge consecutive duplicates, remove blanks).
The reported metric is the **phoneme error rate (PER)** — the edit distance
between the decoded sequence and the ground-truth phoneme sequence, normalized
by the ground-truth length.


## 2. Day-specific input layer

Intracortical recordings are not stationary across sessions: individual
electrodes drift, units appear and disappear, and the per-channel feature
distribution shifts on a timescale of days. Every encoder in this work
absorbs this drift through a **day-specific affine input layer**: a per-session
learnable matrix `W_d ∈ R^{512×512}` and bias `b_d ∈ R^{512}` applied to the
raw spike features at every 20 ms bin. The matrix is initialized to the
identity and the bias to zero, so at the start of training each session's
mapping is a no-op; during training, each `W_d` learns to compensate for that
session's drift.

This design is taken from the original Brain-to-Text releases (Willett et al.,
2023; Card et al., 2024) and is the standard solution for non-stationary
multi-session intracortical recordings. The shared part of the encoder
(the GRU stack, or the Transformer blocks) is then free to model
session-invariant structure.


## 3. The GRU baseline (Chapter 5)

The Brain-to-Text '25 competition baseline is a five-layer **unidirectional**
GRU operating on temporally patched spike features:

| Component                | Configuration                              |
|--------------------------|--------------------------------------------|
| Day-specific input layer | 512 → 512 affine + Softsign                |
| Temporal patching        | kernel 14 (280 ms), stride 4 (80 ms)       |
| Recurrent stack          | 5 unidirectional GRU layers, 768 hidden units, inter-layer dropout 0.4 |
| Output head              | Linear → 41 classes, CTC loss              |

Unidirectionality is deliberate: an online speech BCI must be causal, and at
this sequence length the empirical penalty for forbidding future context is
small. Training uses cosine learning-rate schedule with
`lr_max = 4e-3` and a short warmup, speckled input masking with per-channel
probability 0.2, input dropout 0.2, batch size 32, and 120,000 training
batches. The final validation PER of this configuration is **0.0978**, which
is the downstream substrate against which every later experiment in this work
is measured.

Implementation: [src/brain2text/models/gru.py](../src/brain2text/models/gru.py).
Config: [configs/baselines/gru_baseline.yaml](../configs/baselines/gru_baseline.yaml).


## 4. The Transformer encoder (Chapter 7)

The transformer encoder is a from-scratch reconstruction of the BIT
specification (Zhang & He et al., 2025) plus three additions that proved
necessary to match the GRU baseline. The architecture is summarized in the
table below; the additions are itemized after it.

| Component                | Configuration                                |
|--------------------------|----------------------------------------------|
| Day-specific input layer | 512 → 512 affine (Softsign disabled)         |
| Log transform on input   | `x ← log(1 + x)` applied before the day layer |
| Patch embedding          | non-overlapping, kernel 5 (100 ms), stride 5, projection to `d_model = 384` |
| Positional encoding      | rotary positional embeddings (RoPE; Su et al., 2024) |
| Transformer blocks       | 7 pre-norm blocks; bidirectional self-attention |
| Attention heads          | 6 heads, **per-head dimension 512**         |
| Feed-forward             | hidden dim `4 × 384 = 1536`, GELU            |
| Dropout                  | attention 0.4, FFN 0.2                       |
| Output head              | LayerNorm → Linear → 41 classes, CTC loss    |

The **per-head dimension of 512** is much larger than the conventional
`d_model / n_heads = 64` default; BIT identifies this as a critical
hyperparameter — pretraining does not work with the smaller value — and we
keep their setting.

Three additions on top of the BIT recipe were needed to close the gap to the
GRU baseline at PER = 0.0978:

1. **Log transform on the input.** TX counts at 20 ms bins are non-negative
   integers with a heavy right tail; the SBP values are similarly skewed.
   Replacing each input value `x` with `log(1 + x)` before the day-specific
   layer brings the distribution closer to symmetric and stabilises early
   training.
2. **Aggressive augmentation.** BIT specifies, for its T15 transformer,
   additive white noise std 0.2 and a per-trial constant offset std 0.05. At
   transformer-level augmentation the encoder underfits on T15-only data; we
   use the RNN-level values from BIT's own RNN baseline (white noise std 0.8,
   constant offset std 0.2) instead.
3. **Optimizer schedule.** AdamW with a cosine schedule (linear warmup, decay
   to a small minimum). A manual sweep within the same ranges BIT searched
   converged on `lr = 3e-4`, `weight_decay = 5e-4`, batch size 32, around
   180,000 training batches.

With these three additions the transformer reaches a validation PER of
**0.097** on T15 — within seed noise of the GRU baseline. This is the
**Transformer-from-Scratch (TFS) baseline**, the substrate the SSL experiments
of Chapter 8 are built on.

Implementation: [src/brain2text/models/transformer.py](../src/brain2text/models/transformer.py).
Config: [configs/baselines/transformer_from_scratch.yaml](../configs/baselines/transformer_from_scratch.yaml).


## 5. SSL pretraining: the AR-Binary objective (Chapter 8)

The headline pretraining objective of this work is **AR-Binary**: an
autoregressive next-step prediction task on binarized spike inputs with
30 % channel masking. The recipe is inspired by SpikeGPT (Zhu et al., 2023)
and applied to the same transformer architecture as the TFS baseline.

The input is first **binarized**: every TX value at every `(channel, bin)`
is replaced by the indicator `(x > 0)`. Then 30 % of the input channels are
randomly hidden — set to zero — for the entire trial. The transformer
processes this partially channel-masked binary sequence with **causal**
self-attention. From each output position, a subject-specific binary head
produces two simultaneous predictions:

- **`ar_visible`** — from the hidden state at position `t`, predict the
  binary spike pattern of the **visible** (unmasked) channels at position
  `t + 1`. A strictly temporal, autoregressive task; meaningful only because
  the causal attention prevents the model from looking at the future.
- **`ar_hidden`** — from the same hidden state, predict the binary spike
  pattern of the **hidden** (masked) channels at the **current** position
  `t`. A cross-channel task: given what the visible channels did, infer
  what the hidden channels did.

The total loss is the sum of the binary cross-entropy losses for these two
predictions, averaged over positions and across the relevant channel
subsets:

```
L = BCE(ar_visible_pred,  visible_targets_next) + BCE(ar_hidden_pred, hidden_targets_now)
```

The dual loss is deliberate. The two objectives push the encoder toward
two different kinds of structure: temporal regularity (visible channels'
future given their recent past) and cross-channel regularity (hidden
channels' present given visible channels' present). Pretraining ablations
in Chapter 8 show that **both terms are required**: removing `ar_visible`
(keeping only the cross-channel `ar_hidden`) collapses the encoder to
monitoring PER 0.568, while keeping both but switching from causal to
bidirectional attention recovers the headline within seed noise.
Binarization sidesteps the heavy-tailed distribution of raw spike counts
that complicates MSE-based objectives such as masked reconstruction.

Implementation: [src/brain2text/ssl/ar_binary_pretrain.py](../src/brain2text/ssl/ar_binary_pretrain.py).
Bidirectional dual variant: [ar_binary_bidir_pretrain.py](../src/brain2text/ssl/ar_binary_bidir_pretrain.py).
Hidden-only ablation: [ar_binary_hidden_only_pretrain.py](../src/brain2text/ssl/ar_binary_hidden_only_pretrain.py).
Headline config: [configs/ssl_study/ar_binary_causal_soma.yaml](../configs/ssl_study/ar_binary_causal_soma.yaml).


## 6. Pretrain → finetune pipeline

Three architectural elements distinguish the pretraining stage from the
finetuning stage:

- **Patch embedding** is **subject-specific**. Each NHP subject has its own
  learnable patch embedding that projects from its native channel count
  into the canonical 384-dimensional token. At finetuning time, the patch
  embedding is reinitialized for the target human dataset (512-channel T15
  or T12 input).
- **Output head** is task-specific. During pretraining, the head is the
  subject-specific binary head described in Section 5 (or the reversed
  patch embedding, for masked reconstruction). During finetuning, the head
  is a freshly initialized linear classifier over the 41 phoneme classes.
- **Only the seven shared transformer blocks and the final LayerNorm
  transfer** from pretraining to finetuning. The subject-specific patch
  embedding and the pretraining-specific output head are discarded.

The pretraining recipe used for the headline result trains for 400 epochs
on ~10 hours of macaque Area 2 (S1) data (Chowdhury et al., 2020), with
the AR-Binary (causal) objective at a 30 % channel mask ratio. The
finetuning recipe matches the TFS baseline of Section 4 verbatim — same
augmentation, same optimizer, same 180k–400k training batches — except
that the transformer blocks are initialized from the pretrained checkpoint
instead of from scratch.

Implementation of the training loop (used for both from-scratch training
and SSL finetuning): [src/brain2text/training/trainer.py](../src/brain2text/training/trainer.py).


## 7. Monitoring protocol

Two SSL pretraining runs with very similar final pretraining losses can
produce wildly different finetuning outcomes; the relative ranking of
pretraining objectives or datasets at pretraining time often inverts
after finetuning. Pretraining loss and pretraining reconstruction R² are
therefore not used as checkpoint-selection criteria.

The protocol used throughout this work is the following. Every fixed
number of pretraining epochs (typically every 20), the SSL run saves a
checkpoint. For each saved checkpoint a short **30,000-batch finetuning
job** is launched on T15, and the resulting validation PER is recorded.
This number is the **monitoring PER** of the checkpoint. The checkpoint
with the best monitoring PER is then promoted to a full 400,000-batch
finetuning, which produces the headline result.

The 30k-batch monitoring horizon is short enough to be cheap (about one
GPU-day per checkpoint) and long enough to discriminate clearly between
"this pretraining helps" and "this pretraining is at or below the no-SSL
baseline". In Chapter 8, the monitoring trajectory of the headline
AR-Binary (causal) somatosensory run swings between catastrophic
checkpoints (PER 0.4–0.6) and useful ones (PER below 0.124) at almost
every other saved epoch, while the pretraining loss decreases monotonically
throughout — only the monitoring PER identifies which checkpoint to
promote.

This protocol is itself a methodological contribution: across the ten
SSL objectives tested in Chapter 8, neither pretraining loss nor
reconstruction R² ranked objectives in the same order as the downstream
PER did. The most striking single counterexample is the contrastive
temporal objective, which optimises its InfoNCE pretraining loss without
difficulty but collapses the downstream finetuning entirely (monitoring
PER 1.000).

Implementation: configs under
[configs/validations/pretrain_monitor/](../configs/validations/pretrain_monitor/),
launched through `scripts/finetune.py`.


## 8. Representational analysis via CCA

The representational claim of the work — that AR-Binary (causal)
somatosensory pretraining injects a representation that is session- and
brain-invariant — is tested directly via **canonical correlation analysis
(CCA)** on encoder embeddings, following the cross-session stability
procedure of Gallego et al. (2018).

For each model condition, the encoder is run on every validation trial of
every session of the target dataset, and the hidden activations at the
output of the **final shared transformer block (after the final LayerNorm)**
are extracted. These are the activations that the pretraining contributes:
the patch embedding and the phoneme classifier are session-specific and
do not transfer. Each session's activations are then reduced to a
10-dimensional PCA subspace, and CCA finds the linear projections of
two session-level subspaces that maximize their alignment. The **mean of
the top-4 canonical correlations** summarizes how similar two sessions'
representations are after optimal linear transformation. Aggregated over
all session pairs (820 for the within-T15 analysis, 984 for the
cross-participant T12↔T15 analysis), this scalar is the representational
metric we report.

Two reference values complete the analysis:

- A **shuffled-temporal control** that destroys true cross-session
  alignment while preserving marginal statistics, and defines the floor
  of the metric.
- A **raw-input baseline** that applies PCA + CCA directly to the
  512-dimensional spike features, characterizing the alignment any
  encoder inherits "for free" from the input statistics.

Implementation: [src/brain2text/evaluation/cca.py](../src/brain2text/evaluation/cca.py).
CLI: `scripts/evaluate_cca.py` (within-T15) and
`scripts/evaluate_cca_cross_participant.py` (T12↔T15).


## References

- Card, N. S. et al. (2024). *An accurate and rapidly calibrating speech
  neuroprosthesis.* NEJM.
- Chowdhury, R. H., Glaser, J. I., Miller, L. E. (2020). *Area 2 of primary
  somatosensory cortex encodes kinematics of the whole arm.* eLife.
- Gallego, J. A. et al. (2018). *Long-term stability of cortical population
  dynamics underlying consistent behavior.* bioRxiv 447441v3.
- Graves, A., Fernández, S., Gomez, F., Schmidhuber, J. (2006). *Connectionist
  Temporal Classification.* ICML.
- Su, J. et al. (2024). *RoFormer: Enhanced transformer with rotary position
  embedding.* Neurocomputing.
- Vaswani, A. et al. (2017). *Attention is all you need.* NeurIPS.
- Willett, F. R. et al. (2023). *A high-performance speech neuroprosthesis.*
  Nature.
- Zhang, S., He, B. et al. (2025). *BIT: Brain-to-Text via pretraining on
  intracortical recordings.* Preprint.
- Zhu, R.-J. et al. (2023). *SpikeGPT: Generative pretrained language model
  with spiking neural networks.* Preprint.
