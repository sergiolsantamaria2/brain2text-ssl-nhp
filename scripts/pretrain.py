#!/usr/bin/env python3
"""Run an SSL pretraining job (Transformer encoder, NHP data).

The SSL objective is dispatched from the YAML's `ssl_objective` key. Supported
values: `masked`, `ar_binary`, `ar_binary_bidir`, `ar_binary_hidden_only`,
`contrastive_temporal`, `causal`.

Example (headline configuration, Chapter 8):

    python scripts/pretrain.py --config configs/ssl_study/ar_binary_causal_soma.yaml

For the GRU SSL pretraining of Chapter 6, use the dedicated launcher:

    python -m brain2text.ssl.gru_pretrain --config configs/.../ssl_gru_*.yaml
"""

from brain2text.ssl.masked_pretrain import main

if __name__ == "__main__":
    main()
