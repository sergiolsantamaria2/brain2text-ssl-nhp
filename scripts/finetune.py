#!/usr/bin/env python3
"""Finetune a phoneme decoder on T15 or T12 from an SSL pretraining checkpoint.

Functionally identical to scripts/train.py — the finetune vs from-scratch
distinction is encoded in the config (the `model.ssl_checkpoint` field points
to a pretrained checkpoint, and the schedule / aug parameters are tuned for
finetuning).

Examples:

    # Headline FT400k on T15 from the AR-Binary (causal) somatosensory checkpoint
    python scripts/finetune.py --config configs/validations/multi_seed_t15/ft_seed10.yaml

    # Cross-participant finetuning on T12
    python scripts/finetune.py \
        --config configs/validations/cross_participant_t12/ft_t12_ssl_ar_binary_soma/ft_t12_ssl_ar_binary_soma_seed10.yaml
"""

from brain2text.training.train import main

if __name__ == "__main__":
    main()
