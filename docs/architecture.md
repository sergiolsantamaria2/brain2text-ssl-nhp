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
probability 0.2, input dropout 0.2, batch size 32, and approximately 70,000
training batches (the point at which the validation PER curve flattened). The
final validation PER of this configuration is **0.0976**, which is the
downstream substrate against which every later experiment in this work is
measured.

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
GRU baseline at PER = 0.0976:

1. **Log transform on the input.** TX counts at 20 ms bins are non-negative
   integers with a heavy right tail; the SBP values are similarly skewed.
   Replacing each input value `x` with `log(1 + x)` before the day-specific
   layer brings the distribution closer to symmetric and stabilises early
   training.
2. **Aggressive augmentation.** BIT specifies, for its T15 transformer,
   additive white noise std 0.2 and a per-trial constant offset std 0.05. At
   transformer-level augmentation the encoder underfits on T15-only data; we
   use the RNN-level values from BIT's own RNN baseline (white noise std 0.8,
   constant offset std 0.2) instead. Both are applied to the input *before* the
   log-transform. In addition, Gaussian smoothing across time of width 2.0 is
   applied to both the training and the validation inputs.
3. **Optimizer schedule.** AdamW with a cosine schedule (linear warmup, decay
   to a small minimum). A manual sweep within the same ranges BIT searched
   converged on `lr = 3e-4`, `weight_decay = 5e-4`, batch size 32, around
   200,000 training batches (the point at which the validation curve plateaus
   at the GRU baseline).

With these three additions the transformer reaches a validation PER of
**0.097** on T15 — within seed noise of the GRU baseline. This is the
**Transformer-from-Scratch (TFS) baseline**, the substrate the SSL experiments
of Chapter 8 are built on. When retrained under the longer 400,000-batch
schedule used as the no-SSL control for the headline comparison of Chapter 8,
the same architecture reaches a final PER of 0.0949.

The two encoders are at comparable scale, which matters for the architectural
argument of Section 3 above: the transformer holds ≈43 M shared parameters (the
seven blocks plus the patch embedding and final norm) and ≈12 M more across the
45 day-specific input layers, ~55 M in total; the GRU holds ≈32 M in its shared
recurrent stack and the same ≈12 M in its day-specific layers, ~44 M in total.
What distinguishes them for the SSL experiments is **where** those parameters
live — in shared self-attention recomputed at every pass, or in a persistent
recurrent state — not how many there are.

Implementation: [src/brain2text/models/transformer.py](../src/brain2text/models/transformer.py).
Config: [configs/baselines/transformer_from_scratch.yaml](../configs/baselines/transformer_from_scratch.yaml).


## 5. SSL pretraining: the AR-Binary objective (Chapter 8)

The objective that produces the headline result is **AR-Binary**: an
autoregressive prediction task on binarized spike inputs with 30 % channel
masking and a dual cross-channel + next-step loss. The recipe is inspired by
SpikeGPT (Zhu et al., 2023) and applied to the same transformer architecture
as the TFS baseline.

The input is first **binarized**: every TX value at every `(channel, bin)`
is replaced by the indicator `(x > 0)`. Then 30 % of the input channels are
randomly hidden — set to zero — for the entire trial. The transformer
processes this partially channel-masked binary sequence; from each output
position, a subject-specific binary head produces two simultaneous
predictions:

- **`ar_visible`** — from the hidden state at position `t`, predict the
  binary spike pattern of the **visible** (unmasked) channels at position
  `t + 1`. A strictly temporal, autoregressive task.
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
channels' present given visible channels' present). Binarization sidesteps
the heavy-tailed distribution of raw spike counts that complicates
MSE-based objectives such as masked reconstruction.

**Two variants are evaluated** in the cross-dataset grid:

- **AR-Binary (causal)** uses causal self-attention — each position attends
  only to itself and earlier positions — which makes the `ar_visible`
  next-step prediction non-trivial.
- **AR-Binary (bidirectional)** removes the causal mask but keeps the same
  dual loss intact. The `ar_visible` term is mathematically loosened (a
  bidirectional encoder can in principle read future positions), but the
  loss term is retained to preserve the dual structure.

Both variants are pretrained on all four NHP partitions and evaluated through
the same monitoring protocol (Section 7). On somatosensory at full
finetuning, the two variants reach essentially equal PER (causal 0.0887,
bidirectional 0.0892) and either would serve as the headline; we adopt the
causal variant because it reaches the marginally lower number. A separate
hidden-only check — keeping only the cross-channel `ar_hidden` term — confirms
that the dual structure is essential: removing the temporal term collapses the
downstream finetuning, so the temporal half of the loss is doing real work and
is not redundant with the cross-channel half.

Implementation: [src/brain2text/ssl/ar_binary_pretrain.py](../src/brain2text/ssl/ar_binary_pretrain.py).
Bidirectional variant: [ar_binary_bidir_pretrain.py](../src/brain2text/ssl/ar_binary_bidir_pretrain.py).
Hidden-only check: [ar_binary_hidden_only_pretrain.py](../src/brain2text/ssl/ar_binary_hidden_only_pretrain.py).
Headline config: [configs/ssl_study/ar_binary_causal_soma.yaml](../configs/ssl_study/ar_binary_causal_soma.yaml).


## 6. Pretrain → finetune pipeline

<picture>
  <source media="(prefers-color-scheme: dark)" srcset="figures/pipeline-dark.svg">
  <img alt="Two-column diagram. Left: AR-Binary self-supervised pretraining on macaque threshold crossings, binarized with 30 percent of channels hidden, through a subject-specific patch embedding into seven shared transformer blocks and a binary head predicting visible channels at the next step and hidden channels at the current step under BCE. Right: finetuning on human data with a day-specific affine layer, a reinitialized patch embedding, the same seven transformer blocks loaded with pretrained weights, and a reinitialized linear head trained with CTC. Only the seven blocks and the final LayerNorm transfer between the two stages." src="figures/pipeline-light.svg">
</picture>

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
augmentation, same optimizer, same 200k–400k training batches — except
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
baseline". The example trajectory in Chapter 6 of the thesis (Fig. 6.4) makes
the case: early in that run the monitoring PER swings between catastrophic
checkpoints, well above the no-SSL control, and useful ones at or below it,
stabilising only later in pretraining — while the pretraining loss for the
same run decreases monotonically throughout. The loss neither warns about the
catastrophic checkpoints nor indicates which epoch yields the best downstream
PER; only the monitoring PER does.

This protocol is itself a methodological contribution. It is introduced in
Chapter 6 of the thesis (the GRU SSL experiments) and reused throughout
Chapter 8: across the SSL objectives tested on the transformer, neither
pretraining loss nor reconstruction R² ranked objectives in the same order
as the downstream PER, so a monitoring metric grounded in downstream
performance is needed.

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

A raw `ρ̄_1:4` is not interpretable on its own, so two reference values complete
the analysis:

- A **shuffled floor.** One of the two matrices has its time axis **circularly
  shifted** before CCA. The shift breaks the correspondence between the two
  sessions while leaving each session's own statistics untouched — the same
  marginal distribution and, crucially, the same temporal autocorrelation, since
  a circular shift moves the trajectory in time rather than scrambling it. This
  makes it a *stricter* control than an i.i.d. permutation of timepoints, which
  would destroy the autocorrelation as well and return an artificially low
  number. What the floor measures is the alignment two genuinely unrelated
  sessions reach by chance at this dimensionality and sample size.
- A **raw-input baseline** that applies the same PCA-then-CCA procedure directly
  to the 512-dimensional spike features, with no encoder in between. Two
  sessions already share structure before any network touches them — the arrays
  sit in the same place and the features are z-scored per channel and day — and
  an untrained network passes much of that through intact. The baseline
  quantifies it, and turns the useful question into a directional one: does
  training move the encoder above this level, or below it?

Implementation: [src/brain2text/evaluation/cca.py](../src/brain2text/evaluation/cca.py).
CLI: `scripts/evaluate_cca.py` (within-T15) and
`scripts/evaluate_cca_cross_participant.py` (T12↔T15).


## References

- Card, N. S. et al. (2024). *An accurate and rapidly calibrating speech
  neuroprosthesis.* NEJM.
- Chowdhury, R. H., Glaser, J. I., Miller, L. E. (2020). *Area 2 of primary
  somatosensory cortex encodes kinematics of the whole arm.* eLife.
- Gallego, J. A. et al. (2018). *Cortical population activity within a preserved
  neural manifold underlies multiple motor behaviors.* Nature Communications 9, 4233.
- Gallego, J. A. et al. (2020). *Long-term stability of cortical population
  dynamics underlying consistent behavior.* Nature Neuroscience 23(2), 260–270.
- Graves, A., Fernández, S., Gomez, F., Schmidhuber, J. (2006). *Connectionist
  Temporal Classification.* ICML.
- Su, J. et al. (2024). *RoFormer: Enhanced transformer with rotary position
  embedding.* Neurocomputing.
- Vaswani, A. et al. (2017). *Attention is all you need.* NeurIPS.
- Willett, F. R. et al. (2023). *A high-performance speech neuroprosthesis.*
  Nature.
- Zhang, Y., He, L. et al. (2025). *Decoding inner speech with an end-to-end
  brain-to-text neural interface.* arXiv:2511.21740.
- Zhu, R.-J. et al. (2023). *SpikeGPT: Generative pre-trained language model
  with spiking neural networks.* arXiv:2302.13939.
