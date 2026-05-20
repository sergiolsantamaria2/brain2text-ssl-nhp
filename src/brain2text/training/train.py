from __future__ import annotations

from pathlib import Path
import argparse
from datetime import datetime

from omegaconf import OmegaConf

from .trainer import BrainToTextDecoder_Trainer


# OmegaConf resolver for timestamps in YAML: ${now:%Y-%m-%d_%H%M%S}
OmegaConf.register_new_resolver(
    "now",
    lambda fmt="%Y-%m-%d_%H%M%S": datetime.now().strftime(fmt),
    replace=True,
)


def _default_config_path() -> Path:
    # .../src/brain2text/training/train.py -> repo_root = parents[3]
    repo_root = Path(__file__).resolve().parents[3]
    return repo_root / "configs" / "baselines" / "gru_defaults.yaml"


def _flatten(items):
    out = []
    for x in items:
        if x is None:
            continue
        if isinstance(x, list):
            out.extend(x)
        else:
            out.append(x)
    return out


def _load_yaml(path: str):
    return OmegaConf.load(path)


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--config",
        action="append",
        default=None,
        help="Path(s) to YAML configs. May be passed multiple times; merged in order.",
    )
    parser.add_argument(
        "--override",
        action="append",
        default=[],
        help="(Legacy) Extra YAMLs, merged after --config.",
    )
    parser.add_argument(
        "--set",
        action="append",
        default=[],
        help="Dotlist overrides, e.g. model.n_units=512. May be repeated.",
    )
    parser.add_argument(
        "--print_config",
        action="store_true",
        help="Print the resolved configuration and exit.",
    )

    args = parser.parse_args()

    config_paths = args.config if args.config is not None else [str(_default_config_path())]
    override_paths = _flatten(args.override)
    dotlist = _flatten(args.set)

    # Merge YAMLs in order
    cfg = OmegaConf.create()
    for p in config_paths:
        cfg = OmegaConf.merge(cfg, _load_yaml(p))
    for p in override_paths:
        cfg = OmegaConf.merge(cfg, _load_yaml(p))
    if dotlist:
        cfg = OmegaConf.merge(cfg, OmegaConf.from_dotlist(dotlist))

    if args.print_config:
        print(OmegaConf.to_yaml(cfg, resolve=True))
        return

    trainer = BrainToTextDecoder_Trainer(cfg)
    mode = str(cfg.get("mode", "train")).lower()
    if mode == "train":
        trainer.train()
    else:
        # Additional commands (eval, infer) can be wired up here in the future.
        raise ValueError(f"Unsupported mode={mode}. Expected 'train'.")


if __name__ == "__main__":
    main()