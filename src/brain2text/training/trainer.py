import torch 
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import LambdaLR
import random
import time
import os
import numpy as np
import math
import pathlib
import logging
import sys
import json
import pickle
from pathlib import Path

from brain2text.training.diphone_utils import DiphoneConverter, N_DIPHONE_CLASSES
try:
    import wandb
except ImportError:
    wandb = None

import editdistance

from brain2text.data.dataset import BrainToTextDataset, train_test_split_indicies
from brain2text.data.augmentations import gauss_smooth

from omegaconf import OmegaConf


torch.set_float32_matmul_precision('high')  # faster float32 matmuls on Ampere+
torch.backends.cudnn.deterministic = True   # reproducibility

if hasattr(torch, "_dynamo"):
    torch._dynamo.config.cache_size_limit = 64

from brain2text.models.gru import GRUDecoder
from brain2text.models.transformer import TransformerDecoder
from brain2text.models.gru_ssl_decoder import GRUWithSSLDecoder
class BrainToTextDecoder_Trainer:
    """
    This class will initialize and train a brain-to-text phoneme decoder
    
    Written by Nick Card and Zachery Fogg with reference to Stanford NPTL's decoding function
    """

    def __init__(self, args):
        '''
        args : dictionary of training arguments
        '''

        # Trainer fields
        self.args = args
        # --- Force output_dir from env (job-local) if provided ---
        job_out = os.environ.get("JOB_OUT_DIR", "")
        if job_out and args.get("mode", "") == "train":
            self.args["output_dir"] = job_out
            self.args["checkpoint_dir"] = os.path.join(job_out, "checkpoint")
        # --------------------------------------------------------
        self.logger = None 
        self.device = None
        self.model = None
        self.optimizer = None
        self.learning_rate_scheduler = None
        self.ctc_loss = None 

        self.best_val_PER = torch.inf
        self.best_val_loss = torch.inf


        self.train_dataset = None 
        self.val_dataset = None 
        self.train_loader = None 
        self.val_loader = None 

        self.transform_args = self.args['dataset']['data_transforms']

        # Create output directory
        if args['mode'] == 'train':
            os.makedirs(self.args["output_dir"], exist_ok=True)


        # Set up logging
        self.logger = logging.getLogger(__name__)
        for handler in self.logger.handlers[:]:  # make a copy of the list
            self.logger.removeHandler(handler)
        self.logger.setLevel(logging.INFO)
        formatter = logging.Formatter(fmt='%(asctime)s: %(message)s')        

        if args['mode']=='train':
            # During training, save logs to file in output directory
            fh = logging.FileHandler(str(pathlib.Path(self.args['output_dir'],'training_log')))
            fh.setFormatter(formatter)
            self.logger.addHandler(fh)

        # Always print logs to stdout
        sh = logging.StreamHandler(sys.stdout)
        sh.setFormatter(formatter)
        self.logger.addHandler(sh)

        # Configure device pytorch will use 
        if torch.cuda.is_available():
            gpu_num = self.args.get('gpu_number', 0)
            try:
                gpu_num = int(gpu_num)
            except ValueError:
                self.logger.warning(f"Invalid gpu_number value: {gpu_num}. Using 0 instead.")
                gpu_num = 0

            max_gpu_index = torch.cuda.device_count() - 1
            if gpu_num > max_gpu_index:
                self.logger.warning(f"Requested GPU {gpu_num} not available. Using GPU 0 instead.")
                gpu_num = 0

            try:
                self.device = torch.device(f"cuda:{gpu_num}")
                test_tensor = torch.tensor([1.0]).to(self.device)
                test_tensor = test_tensor * 2
            except Exception as e:
                self.logger.error(f"Error initializing CUDA device {gpu_num}: {str(e)}")
                self.logger.info("Falling back to CPU")
                self.device = torch.device("cpu")
        else:
            self.device = torch.device("cpu")

        self.logger.info(f'Using device: {self.device}')

        # Optional: Weights & Biases logging
        self.use_wandb = bool(self.args.get("wandb", {}).get("enabled", False))
        wandb_group = self.args["wandb"].get("group", None) or os.environ.get("WANDB_RUN_GROUP")
        wandb_job_type = self.args["wandb"].get("job_type", None) or os.environ.get("WANDB_JOB_TYPE")

        if self.use_wandb:
            if wandb is None:
                raise ImportError("wandb is enabled in rnn_args.yaml but wandb is not installed. Run: pip install wandb")
            wandb.init(
                project=self.args["wandb"]["project"],
                name=self.args["wandb"].get("run_name", None),
                tags=self.args["wandb"].get("tags", None),
                config=OmegaConf.to_container(self.args, resolve=True),
                group=wandb_group,
                job_type=wandb_job_type,
            )

        self.eval_cfg = self.args.get("eval", {})
        self._val_step_count = 0

        # Diphone support (DCoND)
        self.use_diphones = bool(self.args.get("use_diphones", False))
        self.diphone_converter = None
        if self.use_diphones:
            self.diphone_converter = DiphoneConverter()
            self.logger.info(
                f"Diphone mode ENABLED: n_classes changed from "
                f"{self.args['dataset']['n_classes']} to {self.diphone_converter.num_classes}"
            )
        # ------------------------------------------------


        # Set seed if provided 
        if self.args['seed'] != -1:
            np.random.seed(self.args['seed'])
            random.seed(self.args['seed'])
            torch.manual_seed(self.args['seed'])


        # Filter out sessions that have no training trials (missing/empty data_train.hdf5)
        try:
            import h5py
        except ImportError:
            h5py = None

        filtered_sessions = []
        filtered_val_probs = []

        dataset_dir = self.args["dataset"]["dataset_dir"]
        sessions = list(self.args["dataset"]["sessions"])
        val_probs = list(self.args["dataset"]["dataset_probability_val"])

        for s, p in zip(sessions, val_probs):
            train_fp = os.path.join(dataset_dir, s, "data_train.hdf5")
            if not os.path.exists(train_fp):
                self.logger.warning(f"Skipping session {s}: missing {train_fp}")
                continue
            if h5py is not None:
                try:
                    with h5py.File(train_fp, "r") as f:
                        if len(f.keys()) == 0:
                            self.logger.warning(f"Skipping session {s}: train file has 0 trials ({train_fp})")
                            continue
                except Exception as e:
                    self.logger.warning(f"Skipping session {s}: could not read {train_fp} ({e})")
                    continue

            filtered_sessions.append(s)
            filtered_val_probs.append(p)

        if len(filtered_sessions) == 0:
            raise RuntimeError("No valid training sessions found. Check dataset_dir and data files.")

        self.args["dataset"]["sessions"] = filtered_sessions
        self.args["dataset"]["dataset_probability_val"] = filtered_val_probs
        self.logger.info(f"Using {len(filtered_sessions)} sessions after filtering (from {len(sessions)}).")



        # Initialize the model (selectable via config)
        mcfg = self.args.get("model", {})
        decoder_type = str(mcfg.get("decoder_type", "gru")).lower()

        if decoder_type == "gru":
            DecoderCls = GRUDecoder
        elif decoder_type == "transformer":
            DecoderCls = TransformerDecoder
        elif decoder_type == "gru_ssl":
            DecoderCls = GRUWithSSLDecoder
        else:
            raise ValueError(f"Invalid model.decoder_type: {decoder_type}. Use 'gru', 'transformer', or 'gru_ssl'.")

        decoder_kwargs = dict(
            neural_dim=mcfg["n_input_features"],
            n_units=mcfg["n_units"],
            n_days=len(self.args["dataset"]["sessions"]),
            n_classes=self.diphone_converter.num_classes if self.use_diphones else self.args["dataset"]["n_classes"],
            rnn_dropout=mcfg["rnn_dropout"],
            input_dropout=mcfg["input_network"]["input_layer_dropout"],
            n_layers=mcfg["n_layers"],
            patch_size=mcfg["patch_size"],
            patch_stride=mcfg["patch_stride"],
        )

        if DecoderCls is TransformerDecoder:
            decoder_kwargs.update(dict(
                embed_dim=int(mcfg.get("embed_dim", 384)),
                n_heads=int(mcfg.get("n_heads", 6)),
                head_dim=mcfg.get("head_dim", None),
                ff_dim=mcfg.get("ff_dim", None),
                attn_dropout=float(mcfg.get("attn_dropout", 0.4)),
                ssl_checkpoint=mcfg.get("ssl_checkpoint", None),
                head_type=str(mcfg.get("head_type", "none")),
                head_num_blocks=int(mcfg.get("head_num_blocks", 0)),
                head_norm=str(mcfg.get("head_norm", "layernorm")),
                head_dropout=float(mcfg.get("head_dropout", 0.1)),
                head_activation=str(mcfg.get("head_activation", "gelu")),
                input_speckle_p=float(mcfg.get("input_speckle_p", 0.0)),
                input_speckle_mode=str(mcfg.get("input_speckle_mode", "feature")),
                time_mask_ratio=float(mcfg.get("time_mask_ratio", 0.0)),
                time_mask_max_span=int(mcfg.get("time_mask_max_span", 15)),
            ))

        if DecoderCls is GRUWithSSLDecoder:
            decoder_kwargs.update(dict(
                ssl_checkpoint=mcfg.get("ssl_checkpoint", None),
                ssl_embed_dim=int(mcfg.get("embed_dim", 384)),
                ssl_n_heads=int(mcfg.get("n_heads", 6)),
                ssl_n_layers=int(mcfg.get("ssl_n_layers", 7)),
                ssl_patch_size=int(mcfg.get("ssl_patch_size", 5)),
                head_type=str(mcfg.get("head_type", "none")),
                head_num_blocks=int(mcfg.get("head_num_blocks", 0)),
                head_norm=str(mcfg.get("head_norm", "none")),
                head_dropout=float(mcfg.get("head_dropout", 0.0)),
                head_activation=str(mcfg.get("head_activation", "gelu")),
                input_speckle_p=float(mcfg.get("input_speckle_p", 0.0)),
                input_speckle_mode=str(mcfg.get("input_speckle_mode", "feature")),
            ))

        # GRU-only knobs
        if DecoderCls is GRUDecoder:
            decoder_kwargs.update(dict(
                head_type=str(mcfg.get("head_type", "none")),
                head_num_blocks=int(mcfg.get("head_num_blocks", 0)),
                head_norm=str(mcfg.get("head_norm", "none")),
                head_dropout=float(mcfg.get("head_dropout", 0.0)),
                head_activation=str(mcfg.get("head_activation", "gelu")),
                input_speckle_p=float(mcfg.get("input_speckle_p", 0.0)),
                input_speckle_mode=str(mcfg.get("input_speckle_mode", "feature")),
                gru_ssl_checkpoint=mcfg.get("gru_ssl_checkpoint", None),
                gru_ssl_layernorm=bool(mcfg.get("gru_ssl_layernorm", False)),
                gru_ssl_skip_first_layer=bool(mcfg.get("gru_ssl_skip_first_layer", False)),
            ))

        # Make it robust: drop any kwargs not accepted by the selected decoder
        import inspect
        allowed = set(inspect.signature(DecoderCls.__init__).parameters.keys())
        allowed.discard("self")
        decoder_kwargs = {k: v for k, v in decoder_kwargs.items() if k in allowed}

        self.model = DecoderCls(**decoder_kwargs)

        # --- EMA (Exponential Moving Average) ---
        self.use_ema = bool(self.args.get("use_ema", False))
        if self.use_ema:
            from torch.optim.swa_utils import AveragedModel, get_ema_multi_avg_fn
            ema_decay = float(self.args.get("ema_decay", 0.999))
            self.ema_model = AveragedModel(self.model, multi_avg_fn=get_ema_multi_avg_fn(ema_decay))
            self.logger.info(f"EMA enabled (decay={ema_decay})")
        else:
            self.ema_model = None

        # Call torch.compile to speed up training (optional, controlled by config)
        self.logger.info("Using torch.compile (if available)")
        use_compile = bool(self.args.get("torch_compile", True))

        if use_compile and hasattr(torch, "compile"):
            try:
                self.model = torch.compile(self.model)
                self.logger.info("torch.compile enabled.")
            except Exception as e:
                self.logger.warning(f"torch.compile failed; falling back to eager. Reason: {e}")
        else:
            if not use_compile:
                self.logger.info("torch.compile disabled by config (torch_compile=false).")
            else:
                self.logger.info("torch.compile not available (torch<2.0). Skipping.")



        self.logger.info(f"Initialized RNN decoding model")

        self.logger.info(self.model)

        # Log how many parameters are in the model
        total_params = sum(p.numel() for p in self.model.parameters())
        self.logger.info(f"Model has {total_params:,} parameters")

        # Determine how many day-specific parameters are in the model
        day_params = 0
        for name, param in self.model.named_parameters():
            if 'day' in name:
                day_params += param.numel()
        
        self.logger.info(f"Model has {day_params:,} day-specific parameters | {((day_params / total_params) * 100):.2f}% of total parameters")

        # Create datasets and dataloaders
        train_file_paths = [os.path.join(self.args["dataset"]["dataset_dir"],s,'data_train.hdf5') for s in self.args['dataset']['sessions']]
        val_file_paths = [os.path.join(self.args["dataset"]["dataset_dir"],s,'data_val.hdf5') for s in self.args['dataset']['sessions']]

        # Ensure that there are no duplicate days
        if len(set(train_file_paths)) != len(train_file_paths):
            raise ValueError("There are duplicate sessions listed in the train dataset")
        if len(set(val_file_paths)) != len(val_file_paths):
            raise ValueError("There are duplicate sessions listed in the val dataset")

        # Split trials into train and test sets
        train_trials, _ = train_test_split_indicies(
            file_paths = train_file_paths, 
            test_percentage = 0,
            seed = self.args['dataset']['seed'],
            bad_trials_dict = None,
            )
        _, val_trials = train_test_split_indicies(
            file_paths = val_file_paths, 
            test_percentage = 1,
            seed = self.args['dataset']['seed'],
            bad_trials_dict = None,
            )
        
        # --- ensure output/checkpoint dirs exist (symlink-safe) ---
        self.args["output_dir"] = os.path.realpath(str(self.args["output_dir"]))
        if "checkpoint_dir" in self.args and self.args["checkpoint_dir"] is not None:
            self.args["checkpoint_dir"] = os.path.realpath(str(self.args["checkpoint_dir"]))
        else:
            self.args["checkpoint_dir"] = os.path.realpath(os.path.join(self.args["output_dir"], "checkpoint"))

        os.makedirs(self.args["output_dir"], exist_ok=True)
        os.makedirs(self.args["checkpoint_dir"], exist_ok=True)
        # ------------------------------------------


        # Save dictionaries to output directory to know which trials were train vs val 
        with open(os.path.join(self.args['output_dir'], 'train_val_trials.json'), 'w') as f: 
            json.dump({'train' : train_trials, 'val': val_trials}, f)

        # Determine if a only a subset of neural features should be used
        feature_subset = None
        if ('feature_subset' in self.args['dataset']) and self.args['dataset']['feature_subset'] != None: 
            feature_subset = self.args['dataset']['feature_subset']
            self.logger.info(f'Using only a subset of features: {feature_subset}')
            
        # train dataset and dataloader
        self.train_dataset = BrainToTextDataset(
            trial_indicies = train_trials,
            split = 'train',
            days_per_batch = self.args['dataset']['days_per_batch'],
            n_batches = self.args['num_training_batches'],
            batch_size = self.args['dataset']['batch_size'],
            must_include_days = None,
            random_seed = self.args['dataset']['seed'],
            feature_subset = feature_subset
            )
        self.train_loader = DataLoader(
            self.train_dataset,
            batch_size = None, # Dataset.__getitem__() already returns batches
            shuffle = self.args['dataset']['loader_shuffle'],
            num_workers = self.args['dataset']['num_dataloader_workers'],
            pin_memory = True 
        )

        # val dataset and dataloader
        self.val_dataset = BrainToTextDataset(
            trial_indicies = val_trials, 
            split = 'test',
            days_per_batch = None,
            n_batches = None,
            batch_size = self.args['dataset']['batch_size'],
            must_include_days = None,
            random_seed = self.args['dataset']['seed'],
            feature_subset = feature_subset   
            )
        self.val_loader = DataLoader(
            self.val_dataset,
            batch_size = None, # Dataset.__getitem__() already returns batches
            shuffle = False, 
            num_workers = 0,
            pin_memory = True 
        )

        self.logger.info("Successfully initialized datasets")

        # Create optimizer, learning rate scheduler, and loss
        self.optimizer = self.create_optimizer()

        if self.args['lr_scheduler_type'] == 'linear':
            self.learning_rate_scheduler = torch.optim.lr_scheduler.LinearLR(
                optimizer=self.optimizer,
                start_factor=1.0,
                end_factor=self.args['lr_min'] / self.args['lr_max'],
                total_iters=self.args['lr_decay_steps'],
            )

        elif self.args['lr_scheduler_type'] == 'cosine':
            self.learning_rate_scheduler = self.create_cosine_lr_scheduler(self.optimizer)

        elif self.args['lr_scheduler_type'] == 'cosine_stepdrop':
            self.learning_rate_scheduler = self.create_cosine_lr_scheduler(self.optimizer, use_stepdrop=True)

        else:
            raise ValueError(f"Invalid learning rate scheduler type: {self.args['lr_scheduler_type']}")

        
        # Blank index: 0 for phonemes, 1600 for diphones
        ctc_blank_idx = 1600 if self.use_diphones else 0
        self.ctc_loss = torch.nn.CTCLoss(blank=ctc_blank_idx, reduction='none', zero_infinity=True)

        # If a checkpoint is provided, then load from checkpoint
        if self.args['init_from_checkpoint']:
            self.load_model_checkpoint(self.args['init_checkpoint_path'])

        # Set rnn and/or input layers to not trainable if specified 
        for name, param in self.model.named_parameters():
            if (not self.args["model"]["rnn_trainable"]) and ("gru" in name):
                param.requires_grad = False

            elif (not self.args["model"]["input_network"]["input_trainable"]) and ("day_" in name):
                param.requires_grad = False


        # Send model to device
        self.model.to(self.device)
        if self.ema_model is not None:
            self.ema_model.to(self.device)

    def create_optimizer(self):
        '''
        Create the optimizer with special param groups

        Biases and day weights should not be decayed

        Day weights should have a separate learning rate

        When gru_ssl_checkpoint is set and lr_max_gru is configured,
        GRU params get a separate (lower) learning rate for transfer learning.

        When llrd_decay > 0 and decoder_type == "transformer", transformer
        blocks get layer-wise LR decay: block i has LR = lr_max * llrd_decay^(n_layers-1-i).
        Earlier blocks (closer to input) get smaller LRs to preserve pretrained
        features, later blocks are free to adapt.
        '''
        day_params = [p for name, p in self.model.named_parameters() if "day_" in name]

        # Check if discriminative LR is needed for pretrained GRU
        has_gru_ssl = bool(self.args.get("model", {}).get("gru_ssl_checkpoint"))
        use_discrim_lr = has_gru_ssl and ("lr_max_gru" in self.args)

        # Check if layer-wise LR decay (LLRD) is enabled for transformer
        llrd_decay = float(self.args.get("llrd_decay", 0.0))
        decoder_type = str(self.args.get("model", {}).get("decoder_type", "gru")).lower()
        use_llrd = (llrd_decay > 0.0) and (decoder_type == "transformer")

        def _is_gru(name):
            return name.startswith("gru.") or name == "h0"

        if use_discrim_lr:
            # Discriminative LR: separate GRU params into their own group
            self.logger.info(
                f"SSL transfer learning: GRU params get lr_max_gru={self.args['lr_max_gru']}"
            )

            gru_params = [
                p for name, p in self.model.named_parameters()
                if ("day_" not in name) and _is_gru(name)
            ]

            no_decay_params = [
                p for name, p in self.model.named_parameters()
                if ("day_" not in name) and (not _is_gru(name))
                and (("bias" in name) or ("norm" in name) or ("bn" in name))
            ]

            other_params = [
                p for name, p in self.model.named_parameters()
                if ("day_" not in name) and (not _is_gru(name))
                and ("bias" not in name) and ("norm" not in name) and ("bn" not in name)
            ]

            param_groups = [
                {'params': no_decay_params, 'weight_decay': 0, 'group_type': 'no_decay'},
                {'params': day_params, 'lr': self.args['lr_max_day'], 'weight_decay': self.args['weight_decay_day'], 'group_type': 'day_layer'},
                {'params': gru_params, 'lr': float(self.args['lr_max_gru']), 'group_type': 'gru_pretrained'},
                {'params': other_params, 'group_type': 'other'},
            ]

        elif use_llrd:
            # Layer-wise LR decay for transformer blocks
            lr_max = float(self.args["lr_max"])
            n_layers = len(self.model.blocks)

            # Collect per-layer block params (split into decay / no_decay)
            block_param_ids = set()
            block_decay_per_layer = [[] for _ in range(n_layers)]
            block_no_decay_per_layer = [[] for _ in range(n_layers)]
            for i, block in enumerate(self.model.blocks):
                for pname, p in block.named_parameters():
                    block_param_ids.add(id(p))
                    if ("bias" in pname) or ("norm" in pname) or ("bn" in pname):
                        block_no_decay_per_layer[i].append(p)
                    else:
                        block_decay_per_layer[i].append(p)

            # Non-block, non-day params (patch_embed, final_norm, head, out, etc.)
            non_block_no_decay = [
                p for name, p in self.model.named_parameters()
                if ("day_" not in name) and (id(p) not in block_param_ids)
                and (("bias" in name) or ("norm" in name) or ("bn" in name))
            ]
            non_block_other = [
                p for name, p in self.model.named_parameters()
                if ("day_" not in name) and (id(p) not in block_param_ids)
                and ("bias" not in name) and ("norm" not in name) and ("bn" not in name)
            ]

            param_groups = [
                {'params': non_block_no_decay, 'weight_decay': 0, 'group_type': 'no_decay'},
                {'params': day_params, 'lr': self.args['lr_max_day'],
                 'weight_decay': self.args['weight_decay_day'], 'group_type': 'day_layer'},
                {'params': non_block_other, 'group_type': 'other'},
            ]
            layer_lrs = []
            for i in range(n_layers):
                layer_lr = lr_max * (llrd_decay ** (n_layers - 1 - i))
                layer_lrs.append(layer_lr)
                param_groups.append({
                    'params': block_no_decay_per_layer[i],
                    'lr': layer_lr,
                    'weight_decay': 0,
                    'group_type': f'block_{i}_no_decay',
                })
                param_groups.append({
                    'params': block_decay_per_layer[i],
                    'lr': layer_lr,
                    'group_type': f'block_{i}',
                })

            self.logger.info(
                f"LLRD enabled: decay={llrd_decay}, n_layers={n_layers}, lr_max={lr_max:.2e}"
            )
            self.logger.info(
                f"  per-layer LRs (block_0 → block_{n_layers-1}): "
                f"{[f'{lr:.3e}' for lr in layer_lrs]}"
            )

        else:
            # Original behavior (no SSL or no discriminative LR)
            no_decay_params = [
                p for name, p in self.model.named_parameters()
                if ("day_" not in name) and (("bias" in name) or ("norm" in name) or ("bn" in name))
            ]

            other_params = [
                p for name, p in self.model.named_parameters()
                if ("day_" not in name) and ("bias" not in name) and ("norm" not in name) and ("bn" not in name)
            ]

            if len(day_params) != 0:
                param_groups = [
                        {'params' : no_decay_params, 'weight_decay' : 0, 'group_type' : 'no_decay'},
                        {'params' : day_params, 'lr' : self.args['lr_max_day'], 'weight_decay' : self.args['weight_decay_day'], 'group_type' : 'day_layer'},
                        {'params' : other_params, 'group_type' : 'other'}
                    ]
            else:
                param_groups = [
                        {'params' : no_decay_params, 'weight_decay' : 0, 'group_type' : 'no_decay'},
                        {'params' : other_params, 'group_type' : 'other'}
                    ]
            
        # AdamW: use fused implementation only when available in the local torch build
        adamw_kwargs = dict(
            lr=self.args["lr_max"],
            betas=(self.args["beta0"], self.args["beta1"]),
            eps=self.args["epsilon"],
            weight_decay=self.args["weight_decay"],
        )

        try:
            optim = torch.optim.AdamW(
                param_groups,
                fused=False,
                **adamw_kwargs,
            )
            self.logger.info("AdamW(fused=True) enabled.")
        except TypeError:
            optim = torch.optim.AdamW(
                param_groups,
                **adamw_kwargs,
            )
            self.logger.info("AdamW(fused) not available in this torch build. Using standard AdamW.")

        return optim 

    def create_cosine_lr_scheduler(self, optim, use_stepdrop: bool = False):

        lr_max = self.args['lr_max']
        lr_min = self.args['lr_min']
        lr_decay_steps = self.args['lr_decay_steps']

        lr_max_day =  self.args['lr_max_day']
        lr_min_day = self.args['lr_min_day']
        lr_decay_steps_day = self.args['lr_decay_steps_day']

        lr_warmup_steps = self.args['lr_warmup_steps']
        lr_warmup_steps_day = self.args['lr_warmup_steps_day']

        # Optional step-drop params (only used if use_stepdrop=True)
        stepdrop_step = int(self.args.get("lr_stepdrop_step", -1))
        stepdrop_factor = float(self.args.get("lr_stepdrop_factor", 1.0))
        if use_stepdrop and (stepdrop_step < 0 or stepdrop_factor <= 0):
            raise ValueError(f"Invalid stepdrop config: lr_stepdrop_step={stepdrop_step}, lr_stepdrop_factor={stepdrop_factor}")


        def lr_lambda(current_step, min_lr_ratio, decay_steps, warmup_steps):
            # Warmup
            if current_step < warmup_steps:
                base = float(current_step + 1) / float(max(1, warmup_steps))
            # Cosine decay
            elif current_step < decay_steps:
                progress = float(current_step - warmup_steps) / float(max(1, decay_steps - warmup_steps))
                cosine_decay = 0.5 * (1 + math.cos(math.pi * progress))
                base = max(min_lr_ratio, min_lr_ratio + (1 - min_lr_ratio) * cosine_decay)
            else:
                base = min_lr_ratio

            # Apply stepdrop (multiplicative) if enabled
            if use_stepdrop and current_step >= stepdrop_step:
                base = base * stepdrop_factor

            return base


        # Default lambda for standard (non-day, non-GRU-pretrained) groups.
        # Each group keeps its own initial LR (set in create_optimizer), so this
        # multiplicative factor scales them all proportionally — including LLRD
        # groups where each block has a different base LR.
        default_lambda = lambda step: lr_lambda(
            step, lr_min / lr_max, lr_decay_steps, lr_warmup_steps)
        day_lambda = lambda step: lr_lambda(
            step, lr_min_day / lr_max_day, lr_decay_steps_day, lr_warmup_steps_day)

        # GRU discriminative lambda (only if a gru_pretrained group exists)
        has_gru_group = any(
            pg.get('group_type') == 'gru_pretrained' for pg in optim.param_groups
        )
        if has_gru_group:
            _lr_max_gru = float(self.args["lr_max_gru"])
            _lr_min_gru = float(self.args.get("lr_min_gru", lr_min))
            _gru_unfreeze = int(self.args.get("gru_ssl_unfreeze_after", 0))
            _gru_warmup = int(self.args.get("lr_warmup_steps_gru", lr_warmup_steps))
            _min_ratio_gru = _lr_min_gru / _lr_max_gru

            self.logger.info(
                f"GRU LR schedule: lr_max_gru={_lr_max_gru}, unfreeze_after={_gru_unfreeze}, "
                f"warmup={_gru_warmup}"
            )

            gru_lambda = lambda step, _uf=_gru_unfreeze, _mr=_min_ratio_gru, _wm=_gru_warmup: (
                0.0 if step < _uf
                else lr_lambda(step - _uf, _mr, lr_decay_steps - _uf, _wm)
            )
        else:
            gru_lambda = None

        # Dispatch by group_type so LLRD (many block_* groups) works without
        # hardcoding the number of param groups.
        lr_lambdas = []
        for pg in optim.param_groups:
            gt = pg.get('group_type', 'other')
            if gt == 'day_layer':
                lr_lambdas.append(day_lambda)
            elif gt == 'gru_pretrained':
                lr_lambdas.append(gru_lambda)
            else:
                # no_decay, other, block_*, block_*_no_decay
                lr_lambdas.append(default_lambda)

        return LambdaLR(optim, lr_lambdas, -1)
        
    def load_model_checkpoint(self, load_path):
        ''' 
        Load a training checkpoint
        '''
        checkpoint = torch.load(load_path, weights_only = False) # checkpoint is just a dict

        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.learning_rate_scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        self.best_val_PER = checkpoint['val_PER']
        self.best_val_loss = checkpoint['val_loss'] if 'val_loss' in checkpoint.keys() else torch.inf


        self.model.to(self.device)
        
        # Send optimizer params back to GPU
        for state in self.optimizer.state.values():
            for k, v in state.items():
                if isinstance(v, torch.Tensor):
                    state[k] = v.to(self.device)

        self.logger.info("Loaded model from checkpoint: " + load_path)

    def save_model_checkpoint(self, save_path, PER, loss, model_state_dict=None):
        """Save a training checkpoint atomically.

        Uses legacy (non-zip) serialization to avoid inline_container.cc
        unexpected-pos errors on some HPC filesystems, writes to a temp file
        and atomically renames into place, and never crashes training if
        checkpoint saving fails.
        """
        checkpoint = {
            "model_state_dict": model_state_dict if model_state_dict is not None else self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "scheduler_state_dict": self.learning_rate_scheduler.state_dict(),
            "val_PER": PER,
            "val_loss": loss,
        }

        save_path = os.path.realpath(str(save_path))
        save_dir = os.path.dirname(save_path)
        os.makedirs(save_dir, exist_ok=True)

        tmp_save_path = save_path + ".tmp"

        try:
            # Legacy serializer avoids zip-container writer (inline_container.cc)
            torch.save(checkpoint, tmp_save_path, _use_new_zipfile_serialization=False)
            os.replace(tmp_save_path, save_path)  # atomic rename
            self.logger.info("Saved model to checkpoint: " + save_path)
        except Exception as e:
            self.logger.exception(f"Checkpoint save failed (continuing training): {save_path} ({e})")
            try:
                if os.path.exists(tmp_save_path):
                    os.remove(tmp_save_path)
            except OSError:
                pass
            return

        # Save the args file alongside the checkpoint (best-effort too)
        try:
            with open(os.path.join(self.args["checkpoint_dir"], "args.yaml"), "w") as f:
                OmegaConf.save(config=self.args, f=f)
        except Exception as e:
            self.logger.warning(f"Could not save args.yaml (continuing): {e}")

    def create_attention_mask(self, sequence_lengths):

        max_length = torch.max(sequence_lengths).item()

        batch_size = sequence_lengths.size(0)
        
        # Create a mask for valid key positions (columns)
        # Shape: [batch_size, max_length]
        key_mask = torch.arange(max_length, device=sequence_lengths.device).expand(batch_size, max_length)
        key_mask = key_mask < sequence_lengths.unsqueeze(1)
        
        # Expand key_mask to [batch_size, 1, 1, max_length]
        # This will be broadcast across all query positions
        key_mask = key_mask.unsqueeze(1).unsqueeze(1)
        
        # Create the attention mask of shape [batch_size, 1, max_length, max_length]
        # by broadcasting key_mask across all query positions
        attention_mask = key_mask.expand(batch_size, 1, max_length, max_length)
        
        # Convert boolean mask to float mask:
        # - True (valid key positions) -> 0.0 (no change to attention scores)
        # - False (padding key positions) -> -inf (will become 0 after softmax)
        attention_mask_float = torch.where(attention_mask, 
                                        True,
                                        False)
        
        return attention_mask_float

    def transform_data(self, features, n_time_steps, mode = 'train'):
        '''
        Apply various augmentations and smoothing to data
        Performing augmentations is much faster on GPU than CPU
        '''

        data_shape = features.shape
        batch_size = data_shape[0]
        channels = data_shape[-1]

        # Log-transform (applied before augmentation, in both train and val)
        if self.transform_args.get('log_transform', False):
            features = torch.sign(features) * torch.log1p(torch.abs(features))

        # We only apply these augmentations in training
        if mode == 'train':
            # add static gain noise 
            if self.transform_args['static_gain_std'] > 0:
                warp_mat = torch.tile(torch.unsqueeze(torch.eye(channels), dim = 0), (batch_size, 1, 1))
                warp_mat += torch.randn_like(warp_mat, device=self.device) * self.transform_args['static_gain_std']

                features = torch.matmul(features, warp_mat)

            # add white noise
            if self.transform_args['white_noise_std'] > 0:
                features += torch.randn(data_shape, device=self.device) * self.transform_args['white_noise_std']

            # add constant offset noise 
            if self.transform_args['constant_offset_std'] > 0:
                features += torch.randn((batch_size, 1, channels), device=self.device) * self.transform_args['constant_offset_std']

            # add random walk noise
            if self.transform_args['random_walk_std'] > 0:
                features += torch.cumsum(torch.randn(data_shape, device=self.device) * self.transform_args['random_walk_std'], dim =self.transform_args['random_walk_axis'])

            # randomly cutoff part of the data timecourse
            if self.transform_args['random_cut'] > 0:
                cut = np.random.randint(0, self.transform_args['random_cut'])
                features = features[:, cut:, :]
                n_time_steps = n_time_steps - cut

        # Apply Gaussian smoothing to data 
        # This is done in both training and validation
        if self.transform_args['smooth_data']:
            features = gauss_smooth(
                inputs = features, 
                device = self.device,
                smooth_kernel_std = self.transform_args['smooth_kernel_std'],
                smooth_kernel_size= self.transform_args['smooth_kernel_size'],
                )
            
        
        return features, n_time_steps

    def train(self):
        '''
        Train the model 
        '''

        # Set model to train mode (specificially to make sure dropout layers are engaged)
        self.model.train()

        # create vars to track performance
        train_losses = []
        val_losses = []
        val_PERs = []
        val_results = []

        val_steps_since_improvement = 0

        # training params 
        save_best_checkpoint = self.args.get('save_best_checkpoint', True)
        early_stopping = self.args.get('early_stopping', True)

        early_stopping_val_steps = self.args['early_stopping_val_steps']

        train_start_time = time.time()


        # train for specified number of batches
        for i, batch in enumerate(self.train_loader):
            
            self.model.train()
            self.optimizer.zero_grad()
            
            # Train step
            start_time = time.time() 

            # Move data to device
            features = batch['input_features'].to(self.device)
            labels = batch['seq_class_ids'].to(self.device)
            n_time_steps = batch['n_time_steps'].to(self.device)
            phone_seq_lens = batch['phone_seq_lens'].to(self.device)
            day_indicies = batch['day_indicies'].to(self.device)

            # Use autocast for efficiency
            with torch.autocast(device_type = "cuda", enabled = self.args['use_amp'], dtype = torch.bfloat16):

                # Apply augmentations to the data
                features, n_time_steps = self.transform_data(features, n_time_steps, 'train')

                ps = int(self.args["model"]["patch_size"])
                st = int(self.args["model"]["patch_stride"])

                if ps > 0:
                    if st <= 0:
                        raise ValueError(f"Invalid patch_stride={st} with patch_size={ps}")
                    adjusted_lens = torch.div((n_time_steps - ps), st, rounding_mode="floor") + 1
                    adjusted_lens = adjusted_lens.to(torch.int32)

                else:
                    adjusted_lens = n_time_steps.to(torch.int32)


                # Get phoneme predictions 
                logits = self.model(features, day_indicies)

                # Calculate CTC Loss
                # Convert labels to diphones if enabled
                if self.use_diphones:
                    diphone_labels, _ = self.diphone_converter.phonemes_to_diphones_batch(
                        labels, phone_seq_lens
                    )
                    ctc_targets = diphone_labels
                else:
                    ctc_targets = labels
                
                loss = self.ctc_loss(
                    log_probs = torch.permute(logits.log_softmax(2), [1, 0, 2]),
                    targets = ctc_targets,
                    input_lengths = adjusted_lens,
                    target_lengths = phone_seq_lens
                    )
                    
                loss = torch.mean(loss) # take mean loss over batches
            
                loss.backward()

                # Skip step if gradients are non-finite (NaN/Inf)
                if self.args["grad_norm_clip_value"] > 0:
                    try:
                        grad_norm = torch.nn.utils.clip_grad_norm_(
                            self.model.parameters(),
                            max_norm=self.args["grad_norm_clip_value"],
                            error_if_nonfinite=False,
                            foreach=True,   # solo si existe
                        )
                    except TypeError:
                        grad_norm = torch.nn.utils.clip_grad_norm_(
                            self.model.parameters(),
                            max_norm=self.args["grad_norm_clip_value"],
                            error_if_nonfinite=False,
                        )
                else:
                    grad_norm = torch.tensor(float("nan"), device=self.device)

                if not torch.isfinite(grad_norm):
                    self.logger.warning(f"Non-finite grad norm at step {i}: {grad_norm}. Skipping optimizer step.")
                    self.optimizer.zero_grad(set_to_none=True)
                    # optional: also skip scheduler step to keep LR schedule consistent
                    continue

                used_lrs = [pg["lr"] for pg in self.optimizer.param_groups]


                # Advance LR schedule for THIS optimizer step (so the current update uses the intended LR)
                self.learning_rate_scheduler.step()
                self.optimizer.step()

                # Update EMA model
                if self.ema_model is not None:
                    self.ema_model.update_parameters(self.model)

            # Log to wandb (per step)
            if self.use_wandb:
                lrs = self.learning_rate_scheduler.get_last_lr()  # one per param group

                log_dict = {
                    "train/loss": float(loss.detach().item()),
                    "train/grad_norm": float(grad_norm),
                }
                # Identify LRs by group_type (robust to LLRD / discriminative LR).
                for pg, lr_val in zip(self.optimizer.param_groups, lrs):
                    gt = pg.get("group_type", "other")
                    if gt == "other":
                        log_dict["lr/main"] = float(lr_val)
                    elif gt == "day_layer":
                        log_dict["lr/day"] = float(lr_val)
                    elif gt == "gru_pretrained":
                        log_dict["lr/gru"] = float(lr_val)
                    elif gt.startswith("block_") and not gt.endswith("_no_decay"):
                        # LLRD: log per-layer LR (decay group only to avoid dup)
                        log_dict[f"lr/{gt}"] = float(lr_val)
                # Fallback for the "main" key if no 'other' group exists (LLRD
                # still has 'other' for non-block params, so this is just safety).
                if "lr/main" not in log_dict:
                    log_dict["lr/main"] = float(lrs[-1])

                wandb.log(log_dict, step=i)

            
            # Save training metrics 
            train_step_duration = time.time() - start_time
            train_losses.append(loss.detach().item())

            # Incrementally log training progress
            if i % self.args['batches_per_train_log'] == 0:
                self.logger.info(f'Train batch {i}: ' +
                        f'loss: {(loss.detach().item()):.2f} ' +
                        f'grad norm: {grad_norm:.2f} '
                        f'time: {train_step_duration:.3f}')

            # Incrementally run a test step
            if i % self.args['batches_per_val_step'] == 0 or i == ((self.args['num_training_batches'] - 1)):
                self.logger.info(f"Running test after training batch: {i}")
                
                # Calculate metrics on val data
                start_time = time.time()
                val_metrics = self.validation(loader = self.val_loader, return_logits = self.args['save_val_logits'], return_data = self.args['save_val_data'])
                val_step_duration = time.time() - start_time

                # EMA validation (PER + loss).
                ema_val_metrics = None
                if self.ema_model is not None:
                    ema_start = time.time()
                    ema_val_metrics = self.validation(loader=self.val_loader, model=self.ema_model)
                    ema_dur = time.time() - ema_start
                    self.logger.info(
                        f'Val batch {i} (EMA): '
                        f'PER (avg): {ema_val_metrics["avg_PER"]:.4f} '
                        f'CTC Loss (avg): {ema_val_metrics["avg_loss"]:.4f} '
                        f'time: {ema_dur:.3f}'
                    )

                self.logger.info(
                    f'Val batch {i}: '
                    f'PER (avg): {val_metrics["avg_PER"]:.4f} '
                    f'CTC Loss (avg): {val_metrics["avg_loss"]:.4f} '
                    f'time: {val_step_duration:.3f}'
                )

                if self.args['log_individual_day_val_PER']:
                    for day in val_metrics['day_PERs'].keys():
                        self.logger.info(f"{self.args['dataset']['sessions'][day]} val PER: {val_metrics['day_PERs'][day]['total_edit_distance'] / val_metrics['day_PERs'][day]['total_seq_length']:0.4f}")

                # Save metrics
                val_PERs.append(val_metrics['avg_PER'])
                val_losses.append(val_metrics['avg_loss'])
                val_results.append(val_metrics)

                if self.use_wandb:
                    log_payload = {
                        "val/PER": float(val_metrics["avg_PER"]),
                        "val/loss": float(val_metrics["avg_loss"]),
                    }
                    if ema_val_metrics is not None:
                        log_payload["val/PER_ema"] = float(ema_val_metrics["avg_PER"])
                        log_payload["val/loss_ema"] = float(ema_val_metrics["avg_loss"])
                    wandb.log(log_payload, step=i)

                # Pick the model variant (normal or EMA) with the lower PER for checkpointing.
                best_per_this_step = float(val_metrics["avg_PER"])
                best_loss_this_step = float(val_metrics["avg_loss"])
                best_is_ema = False
                if ema_val_metrics is not None and ema_val_metrics["avg_PER"] < best_per_this_step:
                    best_per_this_step = float(ema_val_metrics["avg_PER"])
                    best_loss_this_step = float(ema_val_metrics["avg_loss"])
                    best_is_ema = True

                # PER as primary, loss as tie-break.
                new_best = False
                if best_per_this_step < self.best_val_PER:
                    tag = " (EMA)" if best_is_ema else ""
                    self.logger.info(f"New best val PER{tag} {self.best_val_PER:.4f} --> {best_per_this_step:.4f}")
                    self.best_val_PER = best_per_this_step
                    self.best_val_loss = best_loss_this_step
                    new_best = True
                elif best_per_this_step == self.best_val_PER and (best_loss_this_step < self.best_val_loss):
                    self.logger.info(f"New best val loss {self.best_val_loss:.4f} --> {best_loss_this_step:.4f}")
                    self.best_val_loss = best_loss_this_step
                    new_best = True

                if new_best:
                    if save_best_checkpoint:
                        if best_is_ema:
                            self.logger.info(f"Checkpointing EMA model (PER={best_per_this_step:.4f})")
                            ema_state = self.ema_model.module.state_dict()
                        else:
                            self.logger.info(f"Checkpointing model (PER={best_per_this_step:.4f})")
                            ema_state = None
                        self.save_model_checkpoint(
                            f'{self.args["checkpoint_dir"]}/best_checkpoint',
                            self.best_val_PER,
                            self.best_val_loss,
                            model_state_dict=ema_state,
                        )

                    if self.args.get("save_val_metrics", False):
                        ckpt_dir = os.path.realpath(self.args["checkpoint_dir"])
                        os.makedirs(ckpt_dir, exist_ok=True)
                        with open(os.path.join(ckpt_dir, "val_metrics.pkl"), "wb") as f:
                            pickle.dump(val_metrics, f)

                    val_steps_since_improvement = 0
                else:
                    val_steps_since_improvement += 1

                if bool(self.args.get("save_all_val_steps", False)):
                    self.save_model_checkpoint(
                        f'{self.args["checkpoint_dir"]}/checkpoint_batch_{i}',
                        val_metrics["avg_PER"],
                        val_metrics["avg_loss"],
                    )



                # Early stopping 
                if early_stopping and (val_steps_since_improvement >= early_stopping_val_steps):
                    self.logger.info(f'Overall validation PER has not improved in {early_stopping_val_steps} validation steps. Stopping training early at batch: {i}')
                    break
                
        # Log final training steps 
        training_duration = time.time() - train_start_time


        self.logger.info(f'Best avg val PER achieved: {self.best_val_PER:.5f}')
        self.logger.info(f'Total training time: {(training_duration / 60):.2f} minutes')

        # Save final model 
        if self.args['save_final_model']:
            self.save_model_checkpoint(
                f'{self.args["checkpoint_dir"]}/final_checkpoint_batch_{i}',
                val_PERs[-1],
                val_losses[-1],
            )


        train_stats = {}
        train_stats['train_losses'] = train_losses
        train_stats['val_losses'] = val_losses 
        train_stats['val_PERs'] = val_PERs
        train_stats['val_metrics'] = val_results

        if self.use_wandb:
            wandb.finish()



        return train_stats



    def validation(self, loader, return_logits=False, return_data=False, model=None):
        """Compute validation PER and CTC loss over the given loader."""
        model_to_use = model if model is not None else self.model
        model_to_use.eval()

        metrics = {}

        if return_logits:
            metrics['logits'] = []
            metrics['n_time_steps'] = []

        if return_data:
            metrics['input_features'] = []

        metrics['decoded_seqs'] = []
        metrics['true_seq'] = []
        metrics['phone_seq_lens'] = []
        metrics['transcription'] = []
        metrics['losses'] = []
        metrics['block_nums'] = []
        metrics['trial_nums'] = []
        metrics['day_indicies'] = []

        total_edit_distance = 0.0
        total_seq_length = 0.0

        if model is None:  # only increment for primary model validation
            self._val_step_count += 1

        # Calculate PER for each specific day
        day_per = {}
        for d in range(len(self.args['dataset']['sessions'])):
            if self.args['dataset']['dataset_probability_val'][d] == 1: 
                day_per[d] = {'total_edit_distance' : 0, 'total_seq_length' : 0}

        for i, batch in enumerate(loader):        

            features = batch['input_features'].to(self.device)
            labels = batch['seq_class_ids'].to(self.device)
            n_time_steps = batch['n_time_steps'].to(self.device)
            phone_seq_lens = batch['phone_seq_lens'].to(self.device)
            day_indicies = batch['day_indicies'].to(self.device)

            # Determine if we should perform validation on this batch
            day = day_indicies[0].item()
            if self.args['dataset']['dataset_probability_val'][day] == 0: 
                if self.args['log_val_skip_logs']:
                    self.logger.info(f"Skipping validation on day {day}")
                continue
            
            with torch.no_grad():

                with torch.autocast(device_type = "cuda", enabled = self.args['use_amp'], dtype = torch.bfloat16):
                    features, n_time_steps = self.transform_data(features, n_time_steps, 'val')

                    ps = int(self.args["model"]["patch_size"])
                    st = int(self.args["model"]["patch_stride"])

                    if ps > 0:
                        if st <= 0:
                            raise ValueError(f"Invalid patch_stride={st} with patch_size={ps}")
                        adjusted_lens = torch.div((n_time_steps - ps), st, rounding_mode="floor") + 1
                        adjusted_lens = adjusted_lens.to(torch.int32)

                    else:
                        adjusted_lens = n_time_steps.to(torch.int32)


                    logits = model_to_use(features, day_indicies)

                    # Convert labels to diphones if enabled
                    if self.use_diphones:
                        diphone_labels, _ = self.diphone_converter.phonemes_to_diphones_batch(
                            labels, phone_seq_lens
                        )
                        ctc_targets = diphone_labels
                    else:
                        ctc_targets = labels
                    
                    loss = self.ctc_loss(
                        torch.permute(logits.log_softmax(2), [1, 0, 2]),
                        ctc_targets,
                        adjusted_lens,
                        phone_seq_lens,
                    )
                    loss = torch.mean(loss)


                # Calculate PER per day and also avg over entire validation set
                batch_edit_distance = 0 
                decoded_seqs = []
                for iterIdx in range(logits.shape[0]):
                    # Marginalize diphones for PER calculation
                    if self.use_diphones:
                        logits_for_per = self.diphone_converter.marginalize(logits)
                    else:
                        logits_for_per = logits
                    decoded_seq = torch.argmax(logits_for_per[iterIdx, 0 : adjusted_lens[iterIdx], :].clone().detach(),dim=-1)
                    decoded_seq = torch.unique_consecutive(decoded_seq, dim=-1)
                    decoded_seq = decoded_seq.cpu().detach().numpy()
                    decoded_seq = np.array([i for i in decoded_seq if i != 0])

                    trueSeq = np.array(
                        labels[iterIdx][0 : phone_seq_lens[iterIdx]].cpu().detach()
                    )
            
                    batch_edit_distance += editdistance.eval(decoded_seq.tolist(), trueSeq.tolist())

                    decoded_seqs.append(decoded_seq)

            day = batch['day_indicies'][0].item()
                
            day_per[day]['total_edit_distance'] += batch_edit_distance
            day_per[day]['total_seq_length'] += torch.sum(phone_seq_lens).item()


            total_edit_distance += float(batch_edit_distance)
            total_seq_length += float(torch.sum(phone_seq_lens).item())


            # Record metrics
            if return_logits: 
                metrics['logits'].append(logits.cpu().float().numpy()) # Will be in bfloat16 if AMP is enabled, so need to set back to float32
                metrics['n_time_steps'].append(adjusted_lens.cpu().numpy())

            if return_data: 
                metrics['input_features'].append(batch['input_features'].cpu().numpy()) 

            metrics['decoded_seqs'].append(decoded_seqs)
            metrics['true_seq'].append(batch['seq_class_ids'].cpu().numpy())
            metrics['phone_seq_lens'].append(batch['phone_seq_lens'].cpu().numpy())
            metrics['transcription'].append(batch['transcriptions'].cpu().numpy())
            metrics['losses'].append(loss.detach().item())
            metrics['block_nums'].append(batch['block_nums'].numpy())
            metrics['trial_nums'].append(batch['trial_nums'].numpy())
            metrics['day_indicies'].append(batch['day_indicies'].cpu().numpy())

            # total_seq_length can be a tensor or a scalar depending on how it was accumulated
            if isinstance(total_seq_length, torch.Tensor):
                total_seq_length = total_seq_length.item()

        # --- finalize PER/loss ---
        metrics["day_PERs"] = day_per

        if total_seq_length == 0:
            metrics["avg_PER"] = float("nan")
        else:
            metrics["avg_PER"] = float(total_edit_distance / float(total_seq_length))

        metrics["avg_loss"] = float(np.mean(metrics["losses"])) if len(metrics["losses"]) > 0 else float("nan")

        return metrics