#!/usr/bin/env python3
"""
SSL Pretraining Training Script
=================================

Trains the BIT-style transformer encoder with masked reconstruction on
NHP + Human neural data.

Usage:
  python scripts/ssl_pretrain.py --config configs/ssl_pretrain.yaml
  python scripts/ssl_pretrain.py --config configs/ssl_pretrain.yaml --resume checkpoints/ssl_pretrain/epoch_100.pt

Features:
  - Mixed precision (AMP) training
  - Cosine LR scheduler with warmup
  - R² validation metric
  - W&B logging
  - Periodic checkpointing
  - Dynamic subject registration (new subjects appear automatically)
"""

import argparse
import logging
import math
import os
import sys
import time
from pathlib import Path
from typing import Dict, Optional

import torch
import torch.nn as nn
from torch.cuda.amp import GradScaler, autocast

# Project imports — adjust if running from different directory
from brain2text.models.ssl_transformer import (
    SSLTransformerEncoder,
    build_ssl_model,
    build_causal_ssl_model,
    compute_r2,
    compute_r2_causal,
)
from brain2text.ssl.ar_binary_pretrain import (
    build_ar_binary_ssl_model,
)
from brain2text.ssl.ar_binary_hidden_only_pretrain import (
    build_ar_binary_hidden_only_ssl_model,
)
from brain2text.ssl.ar_binary_bidir_pretrain import (
    build_ar_binary_bidir_ssl_model,
)
from brain2text.ssl.contrastive_pretrain import (
    build_contrastive_ssl_model,
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
# Configuration Loader
# ==============================================================================

def load_config(config_path: str) -> dict:
    """Load YAML config file."""
    if not HAS_YAML:
        raise ImportError("PyYAML required. Install with: pip install pyyaml")
    with open(config_path) as f:
        return yaml.safe_load(f)


def build_data_config(cfg: dict) -> SSLDataConfig:  # Map YAML config to typed Python dataclass
    """Build SSLDataConfig from the 'data' section of the config."""
    dc = cfg.get("data", {})
    return SSLDataConfig(
        nhp_pretrain_dir=dc.get("nhp_pretrain_dir", SSLDataConfig.nhp_pretrain_dir),
        human_data_dir=dc.get("human_data_dir", SSLDataConfig.human_data_dir),
        include_human=dc.get("include_human", True),
        human_use_only_tx=dc.get("human_use_only_tx", True),
        window_bins=dc.get("window_bins", 500),
        patch_size=dc.get("patch_size", 5),
        min_session_bins=dc.get("min_session_bins", 100),
        batch_size=dc.get("batch_size", 64),
        n_batches_per_epoch=dc.get("n_batches_per_epoch", 1000),
        val_fraction=dc.get("val_fraction", 0.1),
        seed=dc.get("seed", 42),
        log_transform=dc.get("log_transform", False),
        white_noise_std=dc.get("white_noise_std", 0.2),
        constant_offset_std=dc.get("constant_offset_std", 0.05),
        gaussian_smooth_std=dc.get("gaussian_smooth_std", 2.0),
        gaussian_smooth_kernel_size=dc.get("gaussian_smooth_kernel_size", 11),
        num_workers=dc.get("num_workers", 4),
        pin_memory=dc.get("pin_memory", True),
        min_subject_hours=dc.get("min_subject_hours", 0.1),
    )


# ==============================================================================
# Training Loop
# ==============================================================================

class SSLTrainer:
    """SSL Pretraining Trainer."""
    
    def __init__(
        self,
        config: dict,
        device: torch.device,
    ):
        self.config = config
        self.device = device
        self.global_step = 0
        self.current_epoch = 0
        
        # ---- Data ----
        logger.info("Building dataloaders...")
        self.data_config = build_data_config(config)
        self.train_loader, self.val_loader = create_ssl_dataloaders(self.data_config)
        
        # Collect subject → channel mapping from the datasets
        self.subject_channels = self._collect_subject_channels()
        logger.info(f"Subjects: {self.subject_channels}")
        
        # ---- Model ----
        logger.info("Building model...")
        mc = config.get("model", {})
        self.ssl_objective = config.get("ssl_objective", "masked")

        if self.ssl_objective == "ar_binary":
            self.model = build_ar_binary_ssl_model(
                subject_channels=self.subject_channels,
                embed_dim=mc.get("embed_dim", 384),
                n_heads=mc.get("n_heads", 6),
                head_dim=mc.get("head_dim", None),
                depth=mc.get("depth", 7),
                ff_dim=mc.get("ff_dim", None),
                patch_size=mc.get("patch_size", 5),
                channel_mask_ratio=mc.get("channel_mask_ratio", 0.3),
                dropout=mc.get("dropout", 0.2),
                attn_dropout=mc.get("attn_dropout", 0.4),
            ).to(device)
        elif self.ssl_objective == "ar_binary_hidden_only":
            self.model = build_ar_binary_hidden_only_ssl_model(
                subject_channels=self.subject_channels,
                embed_dim=mc.get("embed_dim", 384),
                n_heads=mc.get("n_heads", 6),
                head_dim=mc.get("head_dim", None),
                depth=mc.get("depth", 7),
                ff_dim=mc.get("ff_dim", None),
                patch_size=mc.get("patch_size", 5),
                channel_mask_ratio=mc.get("channel_mask_ratio", 0.3),
                dropout=mc.get("dropout", 0.2),
                attn_dropout=mc.get("attn_dropout", 0.4),
            ).to(device)
        elif self.ssl_objective == "ar_binary_bidir":
            self.model = build_ar_binary_bidir_ssl_model(
                subject_channels=self.subject_channels,
                embed_dim=mc.get("embed_dim", 384),
                n_heads=mc.get("n_heads", 6),
                head_dim=mc.get("head_dim", None),
                depth=mc.get("depth", 7),
                ff_dim=mc.get("ff_dim", None),
                patch_size=mc.get("patch_size", 5),
                channel_mask_ratio=mc.get("channel_mask_ratio", 0.3),
                dropout=mc.get("dropout", 0.2),
                attn_dropout=mc.get("attn_dropout", 0.4),
            ).to(device)
        elif self.ssl_objective == "contrastive_temporal":
            self.model = build_contrastive_ssl_model(
                subject_channels=self.subject_channels,
                embed_dim=mc.get("embed_dim", 384),
                n_heads=mc.get("n_heads", 6),
                head_dim=mc.get("head_dim", None),
                depth=mc.get("depth", 7),
                ff_dim=mc.get("ff_dim", None),
                patch_size=mc.get("patch_size", 5),
                proj_hidden_dim=mc.get("proj_hidden_dim", 256),
                proj_out_dim=mc.get("proj_out_dim", 128),
                tau_init=mc.get("tau_init", 0.1),
                tau_min=mc.get("tau_min", 0.01),
                tau_max=mc.get("tau_max", 1.0),
                dropout=mc.get("dropout", 0.2),
                attn_dropout=mc.get("attn_dropout", 0.4),
            ).to(device)
        elif self.ssl_objective == "causal":
            self.model = build_causal_ssl_model(
                subject_channels=self.subject_channels,
                embed_dim=mc.get("embed_dim", 384),
                n_heads=mc.get("n_heads", 6),
                head_dim=mc.get("head_dim", None),
                depth=mc.get("depth", 7),
                ff_dim=mc.get("ff_dim", None),
                patch_size=mc.get("patch_size", 5),
                n_predict_steps=mc.get("n_predict_steps", 3),
                dropout=mc.get("dropout", 0.2),
                attn_dropout=mc.get("attn_dropout", 0.4),
            ).to(device)
        else:
            self.model = build_ssl_model(
                subject_channels=self.subject_channels,
                embed_dim=mc.get("embed_dim", 384),
                n_heads=mc.get("n_heads", 6),
                head_dim=mc.get("head_dim", None),
                depth=mc.get("depth", 7),
                ff_dim=mc.get("ff_dim", None),
                patch_size=mc.get("patch_size", 5),
                mask_ratio=mc.get("mask_ratio", 0.5),
                max_mask_span=mc.get("max_mask_span", 15),
                denoising_noise_std=mc.get("denoising_noise_std", 0.0),
                channel_mask_ratio=mc.get("channel_mask_ratio", 0.0),
                dropout=mc.get("dropout", 0.2),
                attn_dropout=mc.get("attn_dropout", 0.4),
            ).to(device)
        
        # ---- Optimizer ----
        tc = config.get("training", {})
        self.optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=tc.get("lr", 5e-4),
            weight_decay=tc.get("weight_decay", 1e-5),
        )
        
        # ---- LR Scheduler ----
        self.n_epochs = tc.get("n_epochs", 400)
        warmup_epochs = tc.get("warmup_epochs", 10)
        min_lr = tc.get("min_lr", 1e-6)
        base_lr = tc.get("lr", 5e-4)
        
        # Cosine with warmup
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
        self.checkpoint_dir = tc.get("checkpoint_dir", "checkpoints/ssl_pretrain")
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
                    group=wc.get("group", "ssl_pretrain"),
                    name=wc.get("run_name", "ssl_pretrain_v1"),
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
        
        # ---- Best metric tracking ----
        self.best_val_r2 = -float("inf")
    
    def _collect_subject_channels(self) -> Dict[str, int]:
        """Collect subject_id → n_channels from datasets."""
        channels = {}
        for dataset in [self.train_loader.dataset, self.val_loader.dataset]:
            for sid, subj in dataset.subjects.items():
                channels[sid] = subj.n_channels
        return channels
    
    def train_epoch(self) -> dict:
        """Train for one epoch. Returns metrics dict."""
        self.model.train()
        total_loss = 0.0
        total_r2 = 0.0
        n_batches = 0
        
        epoch_start = time.time()
        
        for batch_idx, batch in enumerate(self.train_loader):
            neural_data = batch["neural_data"].to(self.device)  # (B, T, C)
            subject_id = batch["subject_id"]
            
            # Ensure subject is registered (handles late-appearing subjects)
            key = self.model._get_subject_key(subject_id)
            if key not in self.model.patch_embeds:
                n_ch = batch["n_channels"]
                logger.info(f"Registering new subject: {subject_id} ({n_ch} channels)")
                self.model.register_subject(subject_id, n_ch)
                # Move new params to device
                self.model.patch_embeds[key] = self.model.patch_embeds[key].to(self.device)
                if self.ssl_objective in ("ar_binary", "ar_binary_hidden_only", "ar_binary_bidir"):
                    self.model.output_heads[key] = self.model.output_heads[key].to(self.device)
                elif self.ssl_objective == "causal":
                    self.model.prediction_heads[key] = self.model.prediction_heads[key].to(self.device)
                elif self.ssl_objective == "contrastive_temporal":
                    pass  # projector is shared, no per-subject head to move
                else:
                    self.model.reversed_patch_embeds[key] = self.model.reversed_patch_embeds[key].to(self.device)
            
            # Forward pass with AMP
            # Drop device_type kwarg; we already use torch.cuda.amp.autocast
            with autocast(enabled=self.use_amp):
                output = self.model(neural_data, subject_id)
                loss = output["loss"]
            
            # Backward
            self.optimizer.zero_grad(set_to_none=True)
            self.scaler.scale(loss).backward()
            
            if self.grad_clip > 0:
                self.scaler.unscale_(self.optimizer)
                nn.utils.clip_grad_norm_(self.model.parameters(), self.grad_clip)
            
            self.scaler.step(self.optimizer)
            self.scaler.update()
            
            # Metrics
            total_loss += loss.item()
            with torch.no_grad():
                if self.ssl_objective in ("ar_binary", "ar_binary_hidden_only", "ar_binary_bidir"):
                    # Use accuracy as the "R²-equivalent" metric (higher is better)
                    r2 = output.get("accuracy", 0.0)
                elif self.ssl_objective == "contrastive_temporal":
                    # Use positive-identification accuracy as the proxy metric
                    r2 = output.get("accuracy", 0.0)
                elif self.ssl_objective == "causal":
                    B, T, C = neural_data.shape
                    n_patches = T // self.model.patch_size
                    raw_patches = neural_data.reshape(B, n_patches, self.model.patch_size * C)
                    r2 = compute_r2_causal(output["predictions"], raw_patches, self.model.n_predict_steps)
                else:
                    r2 = compute_r2(output["reconstructed"], neural_data, output["mask"])
            total_r2 += r2
            n_batches += 1
            self.global_step += 1
            
            # Periodic logging
            if (batch_idx + 1) % 100 == 0:
                avg_loss = total_loss / n_batches
                avg_r2 = total_r2 / n_batches
                lr = self.optimizer.param_groups[0]["lr"]
                elapsed = time.time() - epoch_start
                logger.info(
                    f"  [{batch_idx+1}/{len(self.train_loader)}] "
                    f"loss={avg_loss:.4f} R²={avg_r2:.4f} "
                    f"lr={lr:.2e} subject={subject_id} "
                    f"({elapsed:.0f}s)"
                )
                
                if self.use_wandb:
                    log_dict = {
                        "train/loss": loss.item(),
                        "train/r2": r2,
                        "train/lr": lr,
                        "train/subject": subject_id,
                        "global_step": self.global_step,
                    }
                    if self.ssl_objective in ("ar_binary", "ar_binary_bidir"):
                        log_dict["train/accuracy"] = output.get("accuracy", 0.0)
                        log_dict["train/ar_visible_loss"] = output.get("ar_visible_loss", torch.tensor(0.0)).item() if isinstance(output.get("ar_visible_loss"), torch.Tensor) else output.get("ar_visible_loss", 0.0)
                        log_dict["train/ar_hidden_loss"] = output.get("ar_hidden_loss", torch.tensor(0.0)).item() if isinstance(output.get("ar_hidden_loss"), torch.Tensor) else output.get("ar_hidden_loss", 0.0)
                        log_dict["train/spike_fraction"] = output.get("spike_fraction", 0.0)
                    elif self.ssl_objective == "ar_binary_hidden_only":
                        log_dict["train/accuracy"] = output.get("accuracy", 0.0)
                        log_dict["train/ar_hidden_loss"] = output.get("ar_hidden_loss", torch.tensor(0.0)).item() if isinstance(output.get("ar_hidden_loss"), torch.Tensor) else output.get("ar_hidden_loss", 0.0)
                        log_dict["train/spike_fraction"] = output.get("spike_fraction", 0.0)
                    elif self.ssl_objective == "contrastive_temporal":
                        log_dict["train/accuracy"] = output.get("accuracy", 0.0)
                        log_dict["train/tau"] = output.get("tau", float("nan"))
                    wandb.log(log_dict)
        
        metrics = {
            "train/loss_epoch": total_loss / max(1, n_batches),
            "train/r2_epoch": total_r2 / max(1, n_batches),
            "train/time_s": time.time() - epoch_start,
        }
        
        return metrics
    
    @torch.no_grad()
    def validate(self) -> dict:
        """Run validation. Returns metrics dict."""
        self.model.eval()
        total_loss = 0.0
        total_r2 = 0.0
        total_ar_visible_loss = 0.0
        total_ar_hidden_loss = 0.0
        n_ar_visible = 0
        n_ar_hidden = 0
        n_batches = 0
        subject_r2s: Dict[str, list] = {}

        for batch in self.val_loader:
            neural_data = batch["neural_data"].to(self.device)
            subject_id = batch["subject_id"]

            key = self.model._get_subject_key(subject_id)
            if key not in self.model.patch_embeds:
                continue  # Skip unregistered subjects in val

            # Drop device_type kwarg for compatibility
            with autocast(enabled=self.use_amp):
                # Use same mask_ratio for fair comparison
                output = self.model(neural_data, subject_id)

            total_loss += output["loss"].item()
            if self.ssl_objective in ("ar_binary", "ar_binary_hidden_only", "ar_binary_bidir"):
                r2 = output.get("accuracy", 0.0)
            elif self.ssl_objective == "contrastive_temporal":
                r2 = output.get("accuracy", 0.0)
            elif self.ssl_objective == "causal":
                B, T, C = neural_data.shape
                n_patches = T // self.model.patch_size
                raw_patches = neural_data.reshape(B, n_patches, self.model.patch_size * C)
                r2 = compute_r2_causal(output["predictions"], raw_patches, self.model.n_predict_steps)
            else:
                r2 = compute_r2(output["reconstructed"], neural_data, output["mask"])
            total_r2 += r2
            n_batches += 1

            # Track per-loss components for the AR-binary family (diagnostic).
            if "ar_visible_loss" in output:
                v = output["ar_visible_loss"]
                total_ar_visible_loss += v.item() if isinstance(v, torch.Tensor) else float(v)
                n_ar_visible += 1
            if "ar_hidden_loss" in output:
                h = output["ar_hidden_loss"]
                total_ar_hidden_loss += h.item() if isinstance(h, torch.Tensor) else float(h)
                n_ar_hidden += 1

            if subject_id not in subject_r2s:
                subject_r2s[subject_id] = []
            subject_r2s[subject_id].append(r2)

        if n_batches == 0:
            return {"val/loss": float("nan"), "val/r2": float("nan")}

        metrics = {
            "val/loss": total_loss / n_batches,
            "val/r2": total_r2 / n_batches,
        }

        if n_ar_visible > 0:
            metrics["val/ar_visible_loss"] = total_ar_visible_loss / n_ar_visible
        if n_ar_hidden > 0:
            metrics["val/ar_hidden_loss"] = total_ar_hidden_loss / n_ar_hidden

        # Per-subject R²
        for sid, r2s in subject_r2s.items():
            avg = sum(r2s) / len(r2s)
            metrics[f"val/r2_{sid}"] = avg

        return metrics
    
    def save_checkpoint(self, epoch: int, is_best: bool = False):
        """Save model checkpoint."""
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
        }
        
        path = os.path.join(self.checkpoint_dir, f"epoch_{epoch:04d}.pt")
        torch.save(state, path)
        logger.info(f"Saved checkpoint: {path}")
        
        if is_best:
            best_path = os.path.join(self.checkpoint_dir, "best.pt")
            torch.save(state, best_path)
            logger.info(f"Saved best model: {best_path}")
        
        # Also save shared encoder weights separately (for finetuning)
        encoder_path = os.path.join(self.checkpoint_dir, f"encoder_epoch_{epoch:04d}.pt")
        torch.save({
            "encoder_state": self.model.get_encoder_state(),
            "embed_dim": self.model.embed_dim,
            "epoch": epoch,
        }, encoder_path)
    
    def init_shared_from_ft(self, path: str):
        """Warm-start the shared transformer (blocks.* + final_norm.*) from a
        finetuning checkpoint. Mirror of transformer_decoder._load_ssl_weights.

        - Loads ONLY keys starting with 'blocks.' or 'final_norm.'.
        - Strips an optional 'module.' / '_orig_mod.' prefix from FT keys.
        - Leaves patch_embeds.* and output_heads.* at their fresh init.
        - Does NOT touch optimizer/scheduler/scaler/epoch/global_step.

        Used for the 2-1-2 pipeline (FT → SSL → FT) where the SSL stage starts
        from an encoder that already has phoneme structure rather than random.
        """
        from pathlib import Path

        if not Path(path).exists():
            raise FileNotFoundError(f"init_from_ft checkpoint not found: {path}")

        logger.info(f"Warm-starting SSL shared encoder from FT checkpoint: {path}")
        ckpt = torch.load(path, map_location="cpu", weights_only=False)
        if "model_state_dict" in ckpt:
            state = ckpt["model_state_dict"]
        elif "encoder_state" in ckpt:
            state = ckpt["encoder_state"]
        else:
            state = ckpt

        # Strip torch.compile / DDP prefixes
        def _strip(k: str) -> str:
            for p in ("module.", "_orig_mod."):
                if k.startswith(p):
                    return k[len(p):]
            return k

        my_state = self.model.state_dict()
        loaded = 0
        skipped_shape = 0
        for raw_key, param in state.items():
            key = _strip(raw_key)
            if not (key.startswith("blocks.") or key.startswith("final_norm.")):
                continue
            if key in my_state and my_state[key].shape == param.shape:
                my_state[key] = param
                loaded += 1
            else:
                skipped_shape += 1

        self.model.load_state_dict(my_state, strict=True)
        logger.info(
            f"  init_from_ft: {loaded} tensors loaded into shared encoder, "
            f"{skipped_shape} skipped (shape mismatch)"
        )
        if loaded == 0:
            raise RuntimeError(
                "init_from_ft: 0 tensors loaded — checkpoint has no blocks.* / final_norm.* "
                "keys. Wrong file? Wrong format?"
            )
        if skipped_shape > 0:
            raise RuntimeError(
                f"init_from_ft: {skipped_shape} tensors had shape mismatches. "
                "Check that FT and SSL share the same transformer architecture "
                "(embed_dim, n_heads, head_dim, depth, ff_dim)."
            )

    def load_checkpoint(self, path: str):
        """Resume training from checkpoint."""
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
        
        # ckpt["epoch"] = number of epochs already trained.
        # current_epoch is the 0-indexed loop counter for the NEXT epoch to run,
        # which equals the number of epochs already trained.
        self.current_epoch = ckpt["epoch"]
        self.global_step = ckpt["global_step"]
        self.best_val_r2 = ckpt.get("best_val_r2", -float("inf"))

        logger.info(f"Resumed from epoch {self.current_epoch}, step {self.global_step}")
    
    def train(self, stop_at_epoch: Optional[int] = None):
        """Full training loop.

        Args:
            stop_at_epoch: If set, stop training when current_epoch reaches this
                value (exclusive). Useful for staged training where pretraining
                is interleaved with monitoring/finetuning. Does NOT modify the
                cosine LR schedule, which still targets self.n_epochs.
        """
        last_epoch = stop_at_epoch if stop_at_epoch is not None else self.n_epochs
        logger.info(f"Starting training: epochs {self.current_epoch}→{last_epoch} (total schedule: {self.n_epochs})")
        logger.info(f"Device: {self.device}")
        logger.info(f"AMP: {self.use_amp}")
        logger.info(f"Checkpoint dir: {self.checkpoint_dir}")

        for epoch in range(self.current_epoch, last_epoch):
            self.current_epoch = epoch
            logger.info(f"\n{'='*60}")
            logger.info(f"Epoch {epoch+1}/{last_epoch} (schedule total: {self.n_epochs})")
            logger.info(f"{'='*60}")
            
            # Train
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
                    f"loss={val_metrics['val/loss']:.4f} "
                    f"R²={val_r2:.4f}"
                )
                
                is_best = val_r2 > self.best_val_r2
                if is_best:
                    self.best_val_r2 = val_r2
                    logger.info(f"  *** New best R²: {val_r2:.4f} ***")
                
                if self.use_wandb:
                    wandb.log({
                        **train_metrics,
                        **val_metrics,
                        "epoch": epoch + 1,
                    })
                
                # Save best
                if is_best:
                    self.save_checkpoint(epoch + 1, is_best=True)
            else:
                if self.use_wandb:
                    wandb.log({**train_metrics, "epoch": epoch + 1})
            
            # Periodic save
            if (epoch + 1) % self.save_every == 0:
                self.save_checkpoint(epoch + 1)

        # Always save the last epoch we ran (so staged training can resume / monitor it)
        if last_epoch > self.current_epoch:
            self.save_checkpoint(last_epoch)
        logger.info(f"\nTraining segment complete (epochs→{last_epoch}). Best val R²: {self.best_val_r2:.4f}")

        # Only finish wandb if we reached the full schedule
        if self.use_wandb and last_epoch >= self.n_epochs:
            wandb.finish()


# ==============================================================================
# Entry point
# ==============================================================================

def main():
    parser = argparse.ArgumentParser(description="SSL Pretraining for BIT-style Transformer")
    parser.add_argument("--config", type=str, required=True, help="Path to YAML config")
    parser.add_argument("--resume", type=str, default=None, help="Path to checkpoint to resume from")
    parser.add_argument(
        "--init_from_ft",
        type=str,
        default=None,
        help="Path to a finetuning checkpoint to warm-start the shared transformer "
             "(blocks.* + final_norm.*). Used by the 2-1-2 pipeline. Mutually "
             "exclusive with --resume in spirit; if both are given, --resume wins "
             "(meaningful only on a previously-started 2-1-2 SSL run).",
    )
    parser.add_argument("--gpu", type=int, default=None, help="GPU device index")
    parser.add_argument(
        "--stop_at_epoch",
        type=int,
        default=None,
        help="Stop training when current_epoch reaches this value (exclusive). "
             "Used for staged training: pretrain a chunk, monitor it, then resume. "
             "Does NOT modify the cosine LR schedule (still targets training.n_epochs).",
    )
    args = parser.parse_args()

    # Load config
    config = load_config(args.config)

    # Device
    gpu = args.gpu if args.gpu is not None else config.get("gpu", 0)
    if torch.cuda.is_available():
        device = torch.device(f"cuda:{gpu}")
        logger.info(f"Using GPU: {torch.cuda.get_device_name(device)}")
    else:
        device = torch.device("cpu")
        logger.warning("CUDA not available, using CPU")

    # Override resume from config if not on CLI
    resume_path = args.resume or config.get("training", {}).get("resume_from", None)
    init_from_ft = args.init_from_ft or config.get("training", {}).get("init_from_ft", None)

    # Build trainer
    trainer = SSLTrainer(config, device)

    # Warm-start from FT (only when not resuming a previous SSL run; resume wins).
    if init_from_ft and not resume_path:
        # Auto-detect: if a checkpoint already exists in the configured dir, the
        # SSL run has already started — skip warm-start, will resume below.
        existing = sorted(Path(trainer.checkpoint_dir).glob("epoch_*.pt"))
        if existing:
            logger.info(
                f"init_from_ft requested but {len(existing)} epoch checkpoints already exist "
                f"in {trainer.checkpoint_dir}; skipping FT warm-start (resume path)."
            )
        else:
            trainer.init_shared_from_ft(init_from_ft)

    # Resume if needed
    if resume_path:
        trainer.load_checkpoint(resume_path)

    # Train (optionally up to stop_at_epoch for staged pipelines)
    trainer.train(stop_at_epoch=args.stop_at_epoch)


if __name__ == "__main__":
    main()
