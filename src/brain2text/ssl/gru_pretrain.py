#!/usr/bin/env python3
"""
GRU SSL Causal Pretraining — Training Script
===============================================

Trains the GRU with causal next-step prediction on NHP (+ optionally human)
neural data. The pretrained GRU weights transfer to the finetuning decoder.

Usage:
  python src/brain2text/ssl/gru_ssl_pretrain.py --config configs/gru_ssl/pretraining/all_nhp.yaml
  python src/brain2text/ssl/gru_ssl_pretrain.py --config configs/gru_ssl/pretraining/all_nhp.yaml --resume /path/to/ckpt.pt
"""

import argparse
import logging
import math
import os
import time
from pathlib import Path
from typing import Dict

import torch
import torch.nn as nn
from torch.cuda.amp import GradScaler, autocast

from brain2text.models.ssl_gru import (
    GRUSSLPretrainModel,
    build_gru_ssl_model,
    compute_r2_gru,
)
from brain2text.data.ssl_dataset import (
    SSLDataConfig,
    SSLPretrainDataset,
    create_ssl_dataloaders,
)

try:
    import yaml
    HAS_YAML = True
except ImportError:
    HAS_YAML = False

try:
    import wandb
    HAS_WANDB = True
except ImportError:
    HAS_WANDB = False


logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s %(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)


# ==============================================================================
# Config
# ==============================================================================

def load_config(config_path: str) -> dict:
    if not HAS_YAML:
        raise ImportError("PyYAML required: pip install pyyaml")
    with open(config_path) as f:
        return yaml.safe_load(f)


def build_data_config(cfg: dict) -> SSLDataConfig:
    dc = cfg.get("data", {})
    return SSLDataConfig(
        nhp_pretrain_dir=dc.get("nhp_pretrain_dir", SSLDataConfig.nhp_pretrain_dir),
        human_data_dir=dc.get("human_data_dir", SSLDataConfig.human_data_dir),
        include_human=dc.get("include_human", False),
        human_use_only_tx=dc.get("human_use_only_tx", True),
        window_bins=dc.get("window_bins", 500),
        patch_size=dc.get("patch_size", 1),  # Not used by GRU model (patching in model)
        min_session_bins=dc.get("min_session_bins", 100),
        batch_size=dc.get("batch_size", 64),
        n_batches_per_epoch=dc.get("n_batches_per_epoch", 1000),
        val_fraction=dc.get("val_fraction", 0.1),
        seed=dc.get("seed", 42),
        log_transform=dc.get("log_transform", False),
        white_noise_std=dc.get("white_noise_std", 0.8),
        constant_offset_std=dc.get("constant_offset_std", 0.2),
        gaussian_smooth_std=dc.get("gaussian_smooth_std", 2.0),
        gaussian_smooth_kernel_size=dc.get("gaussian_smooth_kernel_size", 11),
        num_workers=dc.get("num_workers", 4),
        pin_memory=dc.get("pin_memory", True),
        min_subject_hours=dc.get("min_subject_hours", 0.1),
    )


# ==============================================================================
# Trainer
# ==============================================================================

class GRUSSLTrainer:
    """GRU SSL Causal Pretraining Trainer."""

    def __init__(self, config: dict, device: torch.device):
        self.config = config
        self.device = device
        self.global_step = 0
        self.current_epoch = 0

        # ---- Data ----
        logger.info("Building dataloaders...")
        self.data_config = build_data_config(config)
        self.train_loader, self.val_loader = create_ssl_dataloaders(self.data_config)

        self.subject_channels = self._collect_subject_channels()
        logger.info(f"Subjects: {self.subject_channels}")

        # ---- Model ----
        logger.info("Building GRU SSL model...")
        mc = config.get("model", {})
        self.model = build_gru_ssl_model(
            subject_channels=self.subject_channels,
            gru_input_size=mc.get("gru_input_size", 7168),
            n_units=mc.get("n_units", 768),
            n_layers=mc.get("n_layers", 5),
            rnn_dropout=mc.get("rnn_dropout", 0.3),
            patch_size=mc.get("patch_size", 14),
            patch_stride=mc.get("patch_stride", 4),
            n_predict_steps=mc.get("n_predict_steps", 3),
        ).to(device)

        # ---- Optimizer ----
        tc = config.get("training", {})
        self.optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=tc.get("lr", 1e-3),
            weight_decay=tc.get("weight_decay", 0.1),
        )

        # ---- LR Scheduler (cosine with warmup) ----
        self.n_epochs = tc.get("n_epochs", 400)
        warmup_epochs = tc.get("warmup_epochs", 20)
        min_lr = tc.get("min_lr", 1e-6)
        base_lr = tc.get("lr", 1e-3)

        def lr_lambda(epoch):
            if epoch < warmup_epochs:
                return epoch / max(1, warmup_epochs)
            progress = (epoch - warmup_epochs) / max(1, self.n_epochs - warmup_epochs)
            return max(min_lr / base_lr, 0.5 * (1 + math.cos(math.pi * progress)))

        self.scheduler = torch.optim.lr_scheduler.LambdaLR(self.optimizer, lr_lambda)

        # ---- AMP ----
        self.use_amp = tc.get("use_amp", True) and device.type == "cuda"
        self.scaler = GradScaler(enabled=self.use_amp)
        self.grad_clip = tc.get("grad_clip", 10.0)

        # ---- Checkpointing ----
        self.checkpoint_dir = tc.get(
            "checkpoint_dir",
            "/gru_ssl_pretrain",
        )
        os.makedirs(self.checkpoint_dir, exist_ok=True)
        self.save_every = tc.get("save_every_epochs", 20)
        self.val_every = tc.get("val_every_epochs", 5)

        # ---- W&B ----
        self.use_wandb = False
        wc = config.get("wandb", {})
        if wc.get("enabled", False) and HAS_WANDB:
            try:
                wandb.init(
                    project=wc.get("project", "brain2text"),
                    group=wc.get("group", "gru_ssl"),
                    name=wc.get("run_name", "gru_ssl_v1"),
                    config=config,
                )
                self.use_wandb = True
                logger.info("W&B initialized")
            except Exception as e:
                logger.warning(f"W&B init failed: {e}")

        # ---- Compile ----
        if config.get("torch_compile", False) and hasattr(torch, "compile"):
            logger.info("Compiling model with torch.compile...")
            self.model = torch.compile(self.model)

        self.best_val_r2 = -float("inf")

    def _collect_subject_channels(self) -> Dict[str, int]:
        channels = {}
        for dataset in [self.train_loader.dataset, self.val_loader.dataset]:
            for sid, subj in dataset.subjects.items():
                channels[sid] = subj.n_channels
        return channels

    # ------------------------------------------------------------------
    # Train one epoch
    # ------------------------------------------------------------------
    def train_epoch(self) -> dict:
        self.model.train()
        total_loss = 0.0
        total_r2 = 0.0
        n_batches = 0
        epoch_start = time.time()

        for batch_idx, batch in enumerate(self.train_loader):
            neural_data = batch["neural_data"].to(self.device)
            subject_id = batch["subject_id"]

            # Dynamic subject registration
            key = self.model._sanitize_key(subject_id)
            if key not in self.model.input_projs:
                n_ch = batch["n_channels"]
                logger.info(f"Registering new subject: {subject_id} ({n_ch} ch)")
                self.model.register_subject(subject_id, n_ch)
                self.model.input_projs[key] = self.model.input_projs[key].to(self.device)
                self.model.pred_heads[key] = self.model.pred_heads[key].to(self.device)

            with autocast(enabled=self.use_amp):
                output = self.model(neural_data, subject_id)
                loss = output["loss"]

            self.optimizer.zero_grad(set_to_none=True)
            self.scaler.scale(loss).backward()

            if self.grad_clip > 0:
                self.scaler.unscale_(self.optimizer)
                nn.utils.clip_grad_norm_(self.model.parameters(), self.grad_clip)

            self.scaler.step(self.optimizer)
            self.scaler.update()

            total_loss += loss.item()
            with torch.no_grad():
                r2 = compute_r2_gru(
                    output["predictions"],
                    output["raw_patches"],
                    self.model.n_predict_steps,
                )
            total_r2 += r2
            n_batches += 1
            self.global_step += 1

            if (batch_idx + 1) % 100 == 0:
                avg_loss = total_loss / n_batches
                avg_r2 = total_r2 / n_batches
                lr = self.optimizer.param_groups[0]["lr"]
                elapsed = time.time() - epoch_start
                logger.info(
                    f"  [{batch_idx+1}/{len(self.train_loader)}] "
                    f"loss={avg_loss:.4f} R²={avg_r2:.4f} "
                    f"lr={lr:.2e} subject={subject_id} ({elapsed:.0f}s)"
                )
                if self.use_wandb:
                    wandb.log({
                        "train/loss": loss.item(),
                        "train/r2": r2,
                        "train/lr": lr,
                        "global_step": self.global_step,
                    })

        return {
            "train/loss_epoch": total_loss / max(1, n_batches),
            "train/r2_epoch": total_r2 / max(1, n_batches),
            "train/time_s": time.time() - epoch_start,
        }

    # ------------------------------------------------------------------
    # Validate
    # ------------------------------------------------------------------
    @torch.no_grad()
    def validate(self) -> dict:
        self.model.eval()
        total_loss = 0.0
        total_r2 = 0.0
        n_batches = 0
        subject_r2s: Dict[str, list] = {}

        for batch in self.val_loader:
            neural_data = batch["neural_data"].to(self.device)
            subject_id = batch["subject_id"]

            key = self.model._sanitize_key(subject_id)
            if key not in self.model.input_projs:
                continue

            with autocast(enabled=self.use_amp):
                output = self.model(neural_data, subject_id)

            total_loss += output["loss"].item()
            r2 = compute_r2_gru(
                output["predictions"],
                output["raw_patches"],
                self.model.n_predict_steps,
            )
            total_r2 += r2
            n_batches += 1

            if subject_id not in subject_r2s:
                subject_r2s[subject_id] = []
            subject_r2s[subject_id].append(r2)

        if n_batches == 0:
            return {"val/loss": float("nan"), "val/r2": float("nan")}

        metrics = {
            "val/loss": total_loss / n_batches,
            "val/r2": total_r2 / n_batches,
        }
        for sid, r2s in subject_r2s.items():
            metrics[f"val/r2_{sid}"] = sum(r2s) / len(r2s)

        return metrics

    # ------------------------------------------------------------------
    # Checkpointing
    # ------------------------------------------------------------------
    def save_checkpoint(self, epoch: int, is_best: bool = False):
        state = {
            "epoch": epoch,
            "global_step": self.global_step,
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "scheduler_state_dict": self.scheduler.state_dict(),
            "scaler_state_dict": self.scaler.state_dict(),
            "best_val_r2": self.best_val_r2,
            "config": self.config,
            "subject_channels": self.subject_channels,
            # Also save extracted GRU state for easy finetuning loading
            "gru_state": self.model.get_gru_state(),
        }

        path = os.path.join(self.checkpoint_dir, f"epoch_{epoch:04d}.pt")
        torch.save(state, path)
        logger.info(f"Saved checkpoint: {path}")

        if is_best:
            best_path = os.path.join(self.checkpoint_dir, "best.pt")
            torch.save(state, best_path)
            logger.info(f"Saved best model: {best_path}")

    def load_checkpoint(self, path: str):
        logger.info(f"Loading checkpoint: {path}")
        ckpt = torch.load(path, map_location=self.device, weights_only=False)

        # Register subjects from checkpoint
        for sid, n_ch in ckpt.get("subject_channels", {}).items():
            self.model.register_subject(sid, n_ch)

        self.model.load_state_dict(ckpt["model_state_dict"])
        self.optimizer.load_state_dict(ckpt["optimizer_state_dict"])
        self.scheduler.load_state_dict(ckpt["scheduler_state_dict"])
        if "scaler_state_dict" in ckpt:
            self.scaler.load_state_dict(ckpt["scaler_state_dict"])

        self.current_epoch = ckpt["epoch"] + 1
        self.global_step = ckpt["global_step"]
        self.best_val_r2 = ckpt.get("best_val_r2", -float("inf"))
        logger.info(f"Resumed from epoch {self.current_epoch}, step {self.global_step}")

    # ------------------------------------------------------------------
    # Main loop
    # ------------------------------------------------------------------
    def train(self):
        logger.info(f"Starting training for {self.n_epochs} epochs")
        logger.info(f"Device: {self.device}, AMP: {self.use_amp}")
        logger.info(f"Checkpoint dir: {self.checkpoint_dir}")

        for epoch in range(self.current_epoch, self.n_epochs):
            self.current_epoch = epoch
            logger.info(f"\n{'='*60}")
            logger.info(f"Epoch {epoch+1}/{self.n_epochs}")
            logger.info(f"{'='*60}")

            train_metrics = self.train_epoch()
            self.scheduler.step()

            logger.info(
                f"Epoch {epoch+1} train: "
                f"loss={train_metrics['train/loss_epoch']:.4f} "
                f"R²={train_metrics['train/r2_epoch']:.4f} "
                f"({train_metrics['train/time_s']:.0f}s)"
            )

            # Validate
            if (epoch + 1) % self.val_every == 0:
                val_metrics = self.validate()
                val_r2 = val_metrics["val/r2"]

                logger.info(
                    f"Epoch {epoch+1} val: "
                    f"loss={val_metrics['val/loss']:.4f} R²={val_r2:.4f}"
                )

                is_best = val_r2 > self.best_val_r2
                if is_best:
                    self.best_val_r2 = val_r2
                    logger.info(f"  *** New best R²: {val_r2:.4f} ***")

                if self.use_wandb:
                    wandb.log({
                        **train_metrics, **val_metrics,
                        "epoch": epoch + 1,
                    })

                if is_best:
                    self.save_checkpoint(epoch + 1, is_best=True)
            else:
                if self.use_wandb:
                    wandb.log({**train_metrics, "epoch": epoch + 1})

            # Periodic save
            if (epoch + 1) % self.save_every == 0:
                self.save_checkpoint(epoch + 1)

        self.save_checkpoint(self.n_epochs)
        logger.info(f"\nTraining complete! Best val R²: {self.best_val_r2:.4f}")

        if self.use_wandb:
            wandb.finish()


# ==============================================================================
# Entry point
# ==============================================================================

def main():
    parser = argparse.ArgumentParser(description="GRU SSL Causal Pretraining")
    parser.add_argument("--config", type=str, required=True, help="Path to YAML config")
    parser.add_argument("--resume", type=str, default=None, help="Checkpoint to resume from")
    parser.add_argument("--gpu", type=int, default=None, help="GPU index")
    args = parser.parse_args()

    config = load_config(args.config)

    gpu = args.gpu if args.gpu is not None else config.get("gpu", 0)
    if torch.cuda.is_available():
        device = torch.device(f"cuda:{gpu}")
        logger.info(f"Using GPU: {torch.cuda.get_device_name(device)}")
    else:
        device = torch.device("cpu")
        logger.warning("CUDA not available, using CPU")

    resume_path = args.resume or config.get("training", {}).get("resume_from", None)

    trainer = GRUSSLTrainer(config, device)

    if resume_path:
        trainer.load_checkpoint(resume_path)

    trainer.train()


if __name__ == "__main__":
    main()
