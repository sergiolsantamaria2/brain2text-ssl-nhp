#!/usr/bin/env python3
"""Train a phoneme decoder (GRU or Transformer) from scratch with CTC loss.

Wraps brain2text.training.train. Examples:

    # GRU baseline (Chapter 5)
    python scripts/train.py --config configs/baselines/gru_baseline.yaml

    # Transformer from scratch (Chapter 7)
    python scripts/train.py --config configs/baselines/transformer_from_scratch.yaml
"""

from brain2text.training.train import main

if __name__ == "__main__":
    main()
