# brain2text-ssl-nhp

**Design and Study of Non-Human Primate Self-Supervised Pretraining Techniques for Human Speech Decoding**

[![Python](https://img.shields.io/badge/python-3.10%2B-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

Intracortical speech brain–computer interfaces (BCIs) can restore communication for
people with severe paralysis by decoding cortical activity into text, but their
accuracy is bounded by data scarcity: a human implant yields only a few thousand
labelled sentences, far too few for the large-scale pretraining that has driven
progress in adjacent fields such as automatic speech recognition. Recordings from
non-human primates are, by contrast, abundant and are produced with the same
Utah-array technology at the same cortical scale, raising the question this thesis
addresses: **whether non-human-primate data alone, used for self-supervised
pretraining, can improve a human speech decoder.**

The answer is yes, but narrowly — under one specific pairing of pretraining data and
pretraining objective, and only once the encoder is the right kind of network.

> Master's thesis, TU Wien Faculty of Informatics (Erasmus+ exchange), advised by
> Assist. Prof. Guillaume Bellec. Vienna, June 2026. Presented at ETSIT —
> Universidad Politécnica de Madrid.

---

## Headline result

Starting from the optimized Brain-to-Text '25 baseline, we rebuild the encoder as a
transformer and pretrain it on macaque recordings with a binarized autoregressive
objective before finetuning on human data. Pretraining on roughly ten hours of
macaque somatosensory cortex lowers the phoneme error rate on participant T15 to
**0.0887**, a **6.5 % relative improvement** over a from-scratch transformer trained
at the same 400k-batch budget.

| Configuration (400k batches)              | Validation PER | Relative improvement |
|-------------------------------------------|:--------------:|:--------------------:|
| Transformer, no pretraining (control)     | 0.0949         | —                    |
| AR-Binary (bidirectional), somatosensory  | 0.0892         | 6.0 %                |
| **AR-Binary (causal), somatosensory**     | **0.0887**     | **6.5 %**            |

All three are trained under the same schedule at the same budget; the only difference
is the self-supervised pretraining. That the bidirectional variant reaches an
equivalent value confirms the gain is not an artifact of the causal attention mask
used during pretraining. **No human data enters the pretraining stage.**

---

## Research questions

1. Can self-supervised pretraining on non-human-primate intracortical recordings,
   *alone*, improve the downstream phoneme error rate of a human speech BCI decoder?
2. If a transfer effect exists, what determines it — which cortical region recorded
   in the monkey, and which self-supervised objective?
3. If a positive combination of region and objective is found, does the effect
   generalize across seeds and across participants, and what is the representational
   signature behind it?

---

## Contributions

**Research contributions.**

- **A demonstration that non-human-primate data alone can improve a human speech
  decoder.** Pretraining a transformer encoder on roughly ten hours of macaque
  *somatosensory* cortex with a binarized autoregressive objective, and finetuning on
  participant T15, lowers the phoneme error rate to 0.0887 — a 6.5 % relative
  improvement over a from-scratch transformer trained at the same 400k-batch budget.
- **A map of what transfers across species.** Across a grid of cortical region against
  self-supervised objective, only somatosensory cortex paired with the autoregressive
  objective improves the decoder; every motor-cortex partition is neutral or harmful,
  and the masked-reconstruction objective used by prior work gives no benefit on
  non-human-primate data alone. This isolates **the cortical region, not corpus size**,
  as the variable that decides transfer.
- **A mechanistic account of the effect, backed by four independent validations.** The
  improvement reproduces across random seeds and transfers to a second participant; the
  somatosensory-pretrained encoder produces session- and brain-invariant
  representations that the no-SSL baseline does not (canonical correlation analysis);
  and a pipeline-order control shows the benefit behaves as a regularizing cortical
  prior rather than a refinement of pre-existing phonetic structure.

**Work and methodology.**

- **A strong, reproducible Brain-to-Text '25 baseline.** Starting from the official GRU
  baseline, a systematic optimization of optimizer, learning-rate schedule and
  data-augmentation choices lowers its validation PER on T15 from approximately 0.11 to
  0.0976, fixing the downstream reference point for the rest of the work.
- **A from-scratch transformer encoder matching that baseline.** The BIT transformer
  architecture is reconstructed from its description and tuned on the competition data
  alone, with no pretraining, until it reaches the GRU's PER — removing the encoder
  architecture as a confound for the pretraining experiments. This step was motivated
  by the negative result that the GRU rejects non-human-primate pretraining outright.

---

## Where this started: Brain-to-Text '25

The project began as a participation in the Brain-to-Text '25 Kaggle competition,
carried out in collaboration with another master's student under a shared supervisor.
The full decoding pipeline is the standard three-stage cascade — neural encoder,
*n*-gram language model, LLM re-scorer — and it splits naturally at the boundary
between the encoder, which has direct access to the brain signal, and the
language-model stages, which are a pure NLP problem on the encoder's output. This
split mirrored the division of work in our team: my colleague worked on the
language-model side; **I worked on the encoder side, up to the phoneme output**, and
that is what this repository contains. Phoneme error rate (PER) is the metric
reported throughout at the encoder boundary; word error rate is the metric the
competition scores at the output.

Optimizing the organizers' 5-layer GRU baseline — cosine schedule with warmup,
speckled input masking, training-schedule length and a hyperparameter sweep — brought
T15 validation PER from **≈0.11 to 0.0976**. That configuration is the encoder behind
our final competition submission, **which placed 54th of 463 teams**. The general
empirical lesson — consistent with what other competitors reported for
Brain-to-Text '24 — is that the baseline GRU with the right schedule, the right
regularization and the right amount of training is genuinely hard to beat, and that
progress on this task is unlikely to come from architectural tweaks to the recurrence
alone.

---

## Method in brief

The procedure, common to the whole study, is to pretrain the encoder on macaque data
only and then finetune it with CTC on the human participant data. Only the seven
shared transformer blocks and the final LayerNorm transfer between the two stages; the
subject-specific patch embedding and the pretraining-specific output head are
discarded. The finetuning recipe is held fixed and only the pretraining axis varies.

One methodological point matters here and recurs through every experimental chapter:
**the pretraining loss and the pretraining reconstruction R² do not predict downstream
PER.** Two pretraining runs with very similar final pretraining losses can produce
wildly different finetuning outcomes, and the relative ranking of objectives or
datasets at pretraining time often inverts after finetuning. Checkpoints are therefore
selected through a **monitoring protocol**: every saved checkpoint is briefly finetuned
on T15 for 30,000 batches and ranked by the validation PER it actually reaches.

Full architecture, objectives and protocols: **[docs/architecture.md](docs/architecture.md)**.

---

## What transfers: the cross-dataset grid

Three objectives were each pretrained on all four NHP partitions of the corpus and
evaluated through the monitoring protocol.

| Pretraining dataset | Hours | Masked reconstruction | AR-Binary (causal) | AR-Binary (bidirectional) |
|---------------------|:-----:|:---------------------:|:------------------:|:-------------------------:|
| Control (no SSL)    | —     | 0.124                 | 0.124              | 0.124                     |
| Reaching            | 117 h | 0.142                 | 0.129              | 0.125                     |
| Fine motor          | 68 h  | 0.203                 | 0.133              | 0.124                     |
| **Somatosensory**   | **10 h** | 0.124              | **0.118**          | **0.116**                 |
| All NHP             | 200 h | 0.137                 | 0.138              | 0.145                     |

Three patterns stand out.

- **Masked reconstruction never improves on the no-SSL control.** This is BIT's own
  pretraining objective, and on its own — without the human pretraining data that BIT
  also uses — it does not help.
- **Somatosensory is the only NHP dataset that improves on the baseline**, and only
  under the AR-Binary objectives. The two AR-Binary variants agree closely across the
  whole grid, so the effect does not depend on the attention direction used during
  pretraining.
- **No motor-cortex pretraining condition improves on the baseline under any
  objective**, and adding more motor data does not help — the largest partition at
  200 h stays at 0.137–0.145. Of the nine motor-cortex cells, eight are strictly worse
  than the control and one equals it.

The variable that decides the transfer is not, therefore, the volume of data, but the
cortical region of origin: the winning partition is at once the **smallest** (10 hours)
and the **only non-motor** one.

**Why.** Motor-cortex population activity is shaped by the specific movement being
performed, and the dynamics of one task are not interchangeable with those of another.
Pretraining on reaching or grasping therefore teaches the encoder reaching- or
grasping-specific structure, which finetuning must undo before it can fit attempted
speech — and that undoing is the headroom lost. We call this the *motor-cortex-poison*
hypothesis, stated specifically for NHP data. Somatosensory cortex (Area 2 of S1)
shares the same population-coding regime as motor cortex but processes sensory
feedback rather than generating motor commands, so it is not bound to any particular
motor task. This is why a 10-hour somatosensory dataset beats a 200-hour motor one:
what transfers is general cortical structure, and more motor data only means more
task-specific contamination.

AR-Binary matches this on the objective side. Its dual loss is essential — removing
the temporal term collapses the encoder — because the two components are
complementary: `ar_hidden` forces the encoder to learn which channels co-fire within a
moment of cortical activity, while `ar_visible` forces it to learn what comes next
across moments. Binarizing removes the dynamic-range problem of heavy-tailed spike
counts and gives the binary cross-entropy a clean 0/1 signal to model. **AR-Binary on
somatosensory works because both halves of the combination are general rather than
specific.**

The finding of this thesis is not that self-supervised pretraining helps in general —
most configurations do not — but that one specific pairing of region and objective
does.

---

## Validations

Four independent checks, all computed against the same pretrained checkpoint
(AR-Binary causal on somatosensory, epoch 400).

| Validation                          | Result                              | Reading                                              |
|-------------------------------------|:-----------------------------------:|------------------------------------------------------|
| Multi-seed on T15 (250k batches)    | 0.0925 ± 0.0005 over three seeds    | The pretrained encoder reaches a stable PER          |
| Cross-participant to T12 (3 seeds)  | 2.5 % relative, no seed overlap     | Same direction on a different brain and implant      |
| Representational analysis (CCA)     | +18.3 % within T15, +11.2 % T12↔T15 | The prior is session- *and* brain-invariant          |
| Pipeline robustness (FT→SSL→FT)     | +2.9 % worse than SSL→FT            | Complementary regularization, not phonetic refinement |

**One caveat should be stated explicitly**, as it is in the thesis (§9.2): the three
T15 runs vary the seed of the **SSL arm only**, so they establish that the pretrained
encoder reaches a stable PER rather than that the SSL-versus-control *gap* itself is
seed-robust. The seed-paired comparison against a no-SSL control, run at three seeds
in both families, is the **T12 experiment**, and it is that experiment which
establishes the gap as reproducible — the highest SSL run there is still below the
lowest control run.

The CCA result is the interesting one. A randomly initialized encoder scores
unexpectedly high (0.149 within T15), because a random projection does not distinguish
sessions and the cross-session alignment already present in the input passes through
an untrained network intact. No-SSL finetuning then *erodes* that alignment
(0.149 → 0.123): the supervised CTC loss on a single participant pushes each session's
representation toward whatever local features separate that session from the others.
SSL pretraining prevents this erosion (0.149 → 0.146): it gives the encoder a structure
strong enough that the finetuning loss cannot easily overwrite it. What the
somatosensory pretraining supplies is **structure the encoder keeps, not merely a
better point from which to descend the loss**.

Full tables, per-seed numbers and protocol: **[docs/results.md](docs/results.md)**.

---

## What did not work

The first round of pretraining experiments applied the same idea to the encoder we
already had — the optimized GRU from the competition. It does not work. The negative
result, however, is informative: it shows where the obstacle is, and it points toward
the architectural change needed to get past it.

| Pretraining condition   | Hours | Validation PER |
|-------------------------|:-----:|:--------------:|
| GRU control (no SSL)    | —     | **0.0976**     |
| Somatosensory           | 10 h  | 0.724          |
| Fine motor              | 68 h  | 0.718          |
| Reaching                | 117 h | 0.140          |
| All NHP                 | 200 h | 0.136          |

Two distinct failure modes are visible. The somatosensory and fine-motor conditions
fail to converge at all, with PER plateauing around 0.72 throughout finetuning. The
reaching and all-NHP conditions do converge, mirroring the shape of the no-SSL curve,
but settle at a worse minimum — around 0.14 instead of 0.0976, still ~40 % worse in
relative PER. **For the GRU, a committed recurrent prior is worse than no prior at
all.**

The cause is architectural. A recurrent network stores its temporal model in the
recurrent weights themselves, so pretraining writes the monkey dynamics into exactly
the part of the GRU that finetuning would otherwise be free to shape. A Transformer
holds no such persistent state: its temporal model lives in attention scores
recomputed at every pass, and its residual stream lets features learned earlier
survive later training rather than be overwritten.

---

## Relation to prior work

**BIT** (Zhang & He et al., 2025) is the most recent and most directly relevant
precedent — the only prior work whose downstream task is human speech decoding, and
the architecture reconstructed here. Its pretraining corpus combines six human
datasets with seven non-human-primate datasets, for approximately 367 hours. BIT does
not, however, isolate the contribution of those NHP recordings: the mixture is treated
as a single source and reported as a single number. That gap is the question this work
sets out to answer.

For context, BIT reports on T15: RNN baseline 9.64 %, BIT-TFS (from scratch) 8.87 %,
BIT-Human 7.61 %, BIT-All 7.12 %. **The 0.0887 here is not comparable to BIT-All.** It
is measured against this project's own matched-budget from-scratch control (0.0949),
on a 512-feature input where BIT uses only the 256 TX channels for T15, and with no
human data in pretraining at all. That it coincides numerically with BIT's
*from-scratch* figure is a coincidence, not a match to their pretrained result.

---

## Repository layout

- **[`src/brain2text/`](src/brain2text/)** — the library: `data/`, `models/`, `ssl/`,
  `training/`, `evaluation/`, `utils/`.
- **[`configs/`](configs/)** — YAML configurations grouped by experimental role:
  [`baselines/`](configs/baselines/), [`ssl_study/`](configs/ssl_study/),
  [`grid/`](configs/grid/), [`ablations/`](configs/ablations/),
  [`validations/`](configs/validations/). Superseded configurations are kept under
  `configs/archive/`.
- **[`scripts/`](scripts/)** — CLI entry points: `train.py`, `pretrain.py`,
  `finetune.py`, `evaluate_cca.py`, `evaluate_cca_cross_participant.py`,
  `preprocess_t12.py`, `preprocess_nhp.py`.
- **[`docs/`](docs/)** — [architecture.md](docs/architecture.md) (methods) and
  [results.md](docs/results.md) (the complete results panel).
- **[`examples/reproduce_headline.md`](examples/reproduce_headline.md)** — end-to-end
  recipe for the 0.0887 result.

This repository is a cleaned release of the thesis code. Datasets are **not** included
— all are public and obtained from their original sources (below). Every config
resolves its paths from the environment variables `DATA_DIR`, `CHECKPOINT_DIR`,
`OUTPUT_DIR` and `PROJECT_ROOT`.

---

## Reproducing the headline

Full walkthrough in **[examples/reproduce_headline.md](examples/reproduce_headline.md)**.
Short version:

```bash
pip install -e .
export DATA_DIR=/path/to/data CHECKPOINT_DIR=/path/to/checkpoints

# Preprocess the somatosensory partition (Chowdhury et al. 2020)
python scripts/preprocess_nhp.py \
    --datasets chowdhury_2020 \
    --monkey-data-root ${DATA_DIR}/monkey_data \
    --output-dir ${DATA_DIR}/nhp_pretrain

# Pretrain to epoch 400 — two stages sharing one checkpoint_dir.
# The second config resumes from epoch 200 rather than starting over.
python scripts/pretrain.py --config configs/ssl_study/ar_binary_causal_soma.yaml
python scripts/pretrain.py --config configs/ssl_study/ar_binary_causal_soma_extended.yaml

# Finetune on T15 at the 400k-batch budget
python scripts/finetune.py \
    --config configs/validations/multi_seed_t15/ft_seed10.yaml \
    --set num_training_batches=400000
```

Checkpoints are written zero-padded, as `epoch_0400.pt`. The finetuning configs already
point at `${CHECKPOINT_DIR}/ssl_study/ar_binary_soma/epoch_0400.pt`.

---

## Datasets

This work uses publicly available datasets only; none is redistributed here.

| Dataset | Content | Source |
|---------|---------|--------|
| **T15** | 10,948 attempted-speech sentences, 45 sessions spanning ~20 months, four Utah arrays (256 TX + 256 SBP) in the left precentral gyrus | Brain-to-Text '25 / Card et al. (2024) |
| **T12** | 8,800 training and 880 validation trials, 24 sessions, identical 512-feature structure, arrays in ventral 6v and area 44 | Brain-to-Text '24 / Willett et al. (2023) |
| **NHP corpus** | ~201 h across 22 macaque subjects and 586 recording sessions, from ten public Utah-array releases | see below |

All recordings come from rhesus macaques implanted with Utah arrays. Spike times are
converted into thresholded spike counts in 20-millisecond bins, matching the bin width
used for T15 and T12, and z-scored across days per channel. Unlike the human datasets,
spike-band power is not available for these recordings — only threshold crossings are
used.

The corpus combines Churchland et al. (2012), O'Doherty et al. (2017), Perich et al.
(2018), Even-Chen et al. (2019), **Chowdhury et al. (2020)** — the only non-motor
recording site in the corpus, and the partition that produces the headline — NLB
MC_Maze and MC_RTT (Pei et al., 2021), Ma et al. (2023), FALCON / Karpowicz et al.
(2024), and LINK / Temmar et al. (2025). Per-dataset sources, brain areas and hours are
in the `DATASET_REGISTRY` of [scripts/preprocess_nhp.py](scripts/preprocess_nhp.py).

Hours are reported after session-quality filtering, which drops sessions shorter than
about 30 seconds and sessions in which a large fraction of channels carry too few
spikes to be reliable. This is why some figures are smaller than those reported
elsewhere for the same release: the BIT paper reports 147 hours for Perich et al.
(2018) where we count 43, a discrepancy that reflects the more conservative
session-selection and channel-quality criteria we apply.

**On the animal data.** This project generates no new animal recordings: it reuses
public macaque datasets acquired earlier for other purposes, which corresponds
squarely to the *reduction* principle of the 3Rs. The central finding points the same
way, since ten well-chosen hours outperform two hundred — if what transfers is general
cortical structure rather than volume, the argument for continuing to grow primate
corpora weakens. The opposite reading also exists, and it would be dishonest to omit
it: demonstrating that primate recordings have value for human applications may
encourage generating more of them. A methodological project cannot resolve that
tension, but it should be made explicit, and it reinforces reuse of existing
recordings as the default over new acquisition.

---

## Limitations and future work

- **Broaden the cross-participant validation.** The transfer effect was confirmed on a
  single additional participant; confirming it on a third or fourth would settle
  whether the effect is a feature of the speech-BCI task broadly rather than of the two
  participants tested here.
- **Extend the somatosensory finding with more data.** The central result rests on a
  single non-motor source — the entire S1 component of the public corpus available to
  us, roughly 10 hours. Whether a non-motor recording *in general* supplies the
  transferable prior, which pretraining on other non-motor regions (visual, parietal,
  prefrontal) would settle, and how the effect scales with the amount of somatosensory
  data, are both open.
- **Decompose the AR-Binary recipe.** The objective combines binarization, channel
  masking, a dual loss and an attention direction; the experiments identify the dual
  loss as essential and the attention direction as marginal, but do not cleanly
  separate binarization from channel masking. A binarized masked reconstruction and a
  continuous-valued dual objective would isolate each.
- **Integrate with a language-model decoder.** This work stops at phoneme error rate,
  but the full pipeline ends at word error rate — the figure that matters for end-user
  communication, and the one that would make this result directly comparable to the
  state of the art on the same competition.

---

## References

- Card, N. S. et al. (2024). *An accurate and rapidly calibrating speech neuroprosthesis.* NEJM 391(7), 609–618.
- Chowdhury, R. H., Glaser, J. I., Miller, L. E. (2020). *Area 2 of primary somatosensory cortex encodes kinematics of the whole arm.* eLife 9, e48198.
- Gallego, J. A. et al. (2018). *Cortical population activity within a preserved neural manifold underlies multiple motor behaviors.* Nature Communications 9, 4233.
- Graves, A. et al. (2006). *Connectionist temporal classification: Labelling unsegmented sequence data with recurrent neural networks.* ICML.
- Su, J. et al. (2024). *RoFormer: Enhanced transformer with rotary position embedding.* Neurocomputing 568, 127063.
- Willett, F. R. et al. (2023). *A high-performance speech neuroprosthesis.* Nature 620(7976), 1031–1036.
- Willett, F. R. et al. (2024). *Brain-to-Text Benchmark '24: Lessons learned.* arXiv:2412.17227.
- Zhang, Y., He, L. et al. (2025). *Decoding inner speech with an end-to-end brain-to-text neural interface.* arXiv:2511.21740.
- Zhu, R.-J. et al. (2023). *SpikeGPT: Generative pre-trained language model with spiking neural networks.* arXiv:2302.13939.

## License

MIT — see [LICENSE](LICENSE). The datasets are governed by their own licenses.
