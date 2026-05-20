import torch 
from torch import nn

class GRUDecoder(nn.Module):
    '''
    Defines the GRU decoder

    This class combines day-specific input layers, a GRU, and an output classification layer
    '''
    def __init__(self,
                neural_dim,
                n_units,
                n_days,
                n_classes,
                rnn_dropout=0.0,
                input_dropout=0.0,
                n_layers=5,
                patch_size=0,
                patch_stride=0,
                # New: post-RNN head (training improvement)
                head_type: str = "none",          # "none" | "resffn"
                head_num_blocks: int = 0,         # e.g., 1 or 2
                head_norm: str = "none",          # "bn" | "layernorm" | "rmsnorm" | "none"
                head_dropout: float = 0.0,
                head_activation: str = "gelu",
                # New: speckled masking (coordinated dropout)
                input_speckle_p: float = 0.0,
                input_speckle_mode: str = "feature",
                # SSL pretrained GRU checkpoint
                gru_ssl_checkpoint: str = None,
                gru_ssl_layernorm: bool = False,
                gru_ssl_skip_first_layer: bool = False,
                ):

        '''
        neural_dim  (int)      - number of channels in a single timestep (e.g. 512)
        n_units     (int)      - number of hidden units in each recurrent layer - equal to the size of the hidden state
        n_days      (int)      - number of days in the dataset
        n_classes   (int)      - number of classes 
        rnn_dropout    (float) - percentage of units to droupout during training
        input_dropout (float)  - percentage of input units to dropout during training
        n_layers    (int)      - number of recurrent layers 
        patch_size  (int)      - the number of timesteps to concat on initial input layer - a value of 0 will disable this "input concat" step 
        patch_stride(int)      - the number of timesteps to stride over when concatenating initial input 
        '''
        super(GRUDecoder, self).__init__()
        
        self.neural_dim = neural_dim
        self.n_units = n_units
        self.n_classes = n_classes
        self.n_layers = n_layers 
        self.n_days = n_days

        self.rnn_dropout = rnn_dropout
        self.input_dropout = input_dropout
        
        self.patch_size = patch_size
        self.patch_stride = patch_stride
        self.head_type = str(head_type)
        self.head_num_blocks = int(head_num_blocks)
        self.head_norm = str(head_norm)
        self.head_dropout = float(head_dropout)
        self.head_activation = str(head_activation)

        self.input_speckle_p = float(input_speckle_p)
        self.input_speckle_mode = str(input_speckle_mode)


        # Parameters for the day-specific input layers
        self.day_layer_activation = nn.Softsign() # basically a shallower tanh 

       # Day-specific affine parameters (vectorized, compile-friendly)
        self.day_weights = nn.Parameter(
            torch.eye(self.neural_dim).unsqueeze(0).repeat(self.n_days, 1, 1)
        )  # (n_days, D, D)

        self.day_biases = nn.Parameter(
            torch.zeros(self.n_days, self.neural_dim)
        )  # (n_days, D)


        self.day_layer_dropout = nn.Dropout(input_dropout)
        
        self.input_size = self.neural_dim

        # If we are using "strided inputs", then the input size of the first recurrent layer will actually be in_size * patch_size
        if self.patch_size > 0:
            self.input_size *= self.patch_size

        # LayerNorm bridge for SSL transfer: normalizes input distribution
        # to match what the pretrained GRU expects (SSL used LayerNorm→Linear→GRU)
        if gru_ssl_layernorm and gru_ssl_checkpoint:
            self.pre_gru_norm = nn.LayerNorm(self.input_size)
        else:
            self.pre_gru_norm = nn.Identity()

        self.gru_ssl_skip_first_layer = gru_ssl_skip_first_layer

        self.gru = nn.GRU(
            input_size = self.input_size,
            hidden_size = self.n_units,
            num_layers = self.n_layers,
            dropout = self.rnn_dropout, 
            batch_first = True, # The first dim of our input is the batch dim
            bidirectional = False,
        )

        # Set recurrent units to have orthogonal param init and input layers to have xavier init
        for name, param in self.gru.named_parameters():
            if "weight_hh" in name:
                nn.init.orthogonal_(param)
            if "weight_ih" in name:
                nn.init.xavier_uniform_(param)

        # Optional post-GRU head
        ht = self.head_type.lower()
        if ht == "none" or self.head_num_blocks <= 0:
            self.head = nn.Identity()
        elif ht in ("resffn", "ffn"):
            self.head = nn.Sequential(*[
                ResidualFFNBlock(
                    d=self.n_units,
                    norm_type=self.head_norm,
                    dropout=self.head_dropout,
                    activation=self.head_activation,
                )
                for _ in range(self.head_num_blocks)
            ])
        else:
            raise ValueError(f"Unknown head_type={self.head_type}. Use: none, resffn.")

        # Prediciton head. Weight init to xavier
        self.out = nn.Linear(self.n_units, self.n_classes)
        nn.init.xavier_uniform_(self.out.weight)

        # Learnable initial hidden states
        self.h0 = nn.Parameter(nn.init.xavier_uniform_(torch.zeros(1, 1, self.n_units)))

        # Load pretrained GRU weights from SSL causal pretraining
        if gru_ssl_checkpoint:
            self._load_gru_ssl_weights(gru_ssl_checkpoint)

    def _load_gru_ssl_weights(self, checkpoint_path: str):
        """
        Load pretrained GRU + h0 weights from a GRU SSL causal pretraining checkpoint.

        The SSL model has identical GRU architecture (same input_size, hidden_size,
        n_layers) so weights transfer directly. Only gru.* and h0 are loaded;
        day-specific layers and output head keep their fresh initialization.
        """
        import logging
        from pathlib import Path
        _logger = logging.getLogger(__name__)

        ckpt_path = Path(checkpoint_path)
        if not ckpt_path.exists():
            _logger.warning(
                f"GRU SSL checkpoint not found: {checkpoint_path}. Training from scratch."
            )
            return

        _logger.info(f"Loading pretrained GRU weights from: {checkpoint_path}")
        ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=False)

        # Extract GRU state — prefer the pre-extracted gru_state key
        if "gru_state" in ckpt:
            gru_state = ckpt["gru_state"]
        elif "model_state_dict" in ckpt:
            gru_state = {
                k: v for k, v in ckpt["model_state_dict"].items()
                if k.startswith("gru.") or k == "h0"
            }
        else:
            gru_state = {
                k: v for k, v in ckpt.items()
                if k.startswith("gru.") or k == "h0"
            }

        if not gru_state:
            _logger.warning("No gru.* or h0 keys found in SSL checkpoint.")
            return

        # Selective layer loading: skip first GRU layer (weight_ih_l0, bias_ih_l0)
        # because it's the most input-distribution-dependent
        if self.gru_ssl_skip_first_layer:
            skip_keys = [k for k in gru_state if "_l0" in k and "_ih_" in k]
            for k in skip_keys:
                del gru_state[k]
            _logger.info(
                f"Skipping first-layer input keys (input-dependent): {skip_keys}"
            )

        # Load with strict=False (day layers, head, etc. are not in gru_state)
        missing, unexpected = self.load_state_dict(gru_state, strict=False)

        # Only warn about missing GRU keys (missing day/head keys are expected)
        missing_gru = [k for k in missing if k.startswith("gru.") or k == "h0"]
        if missing_gru:
            _logger.warning(f"Missing GRU keys from SSL checkpoint: {missing_gru}")

        _logger.info(f"Loaded {len(gru_state)} GRU tensors from SSL checkpoint.")

    def forward(self, x, day_idx, states = None, return_state = False):
        '''
        x        (tensor)  - batch of examples (trials) of shape: (batch_size, time_series_length, neural_dim)
        day_idx  (tensor)  - tensor which is a list of day indexs corresponding to the day of each example in the batch x. 
        '''

        # Apply day-specific layer to (hopefully) project neural data from the different days to the same latent space
        day_ids = day_idx.view(-1).long()  # (B,)

        day_weights = self.day_weights.index_select(0, day_ids)          # (B, D, D)
        day_biases  = self.day_biases.index_select(0, day_ids).unsqueeze(1)  # (B, 1, D)

        x = torch.einsum("btd,bdk->btk", x, day_weights) + day_biases
        x = self.day_layer_activation(x)


        # Apply dropout to the ouput of the day specific layer
        if self.input_dropout > 0:
            x = self.day_layer_dropout(x)

        # (Optionally) Perform input concat operation
        if self.patch_size > 0: 
  
            x = x.unsqueeze(1)                      # [batches, 1, timesteps, feature_dim]
            x = x.permute(0, 3, 1, 2)               # [batches, feature_dim, 1, timesteps]
            
            # Extract patches using unfold (sliding window)
            x_unfold = x.unfold(3, self.patch_size, self.patch_stride)  # [batches, feature_dim, 1, num_patches, patch_size]
            
            # Remove dummy height dimension and rearrange dimensions
            x_unfold = x_unfold.squeeze(2)           # [batches, feature_dum, num_patches, patch_size]
            x_unfold = x_unfold.permute(0, 2, 3, 1)  # [batches, num_patches, patch_size, feature_dim]

            # Flatten last two dimensions (patch_size and features)
            x = x_unfold.reshape(x.size(0), x_unfold.size(1), -1) 
        
        # LayerNorm bridge (active when SSL checkpoint + gru_ssl_layernorm)
        x = self.pre_gru_norm(x)

        # Determine initial hidden states
        if states is None:
            states = self.h0.expand(self.n_layers, x.shape[0], self.n_units).contiguous()

        # Speckled masking (training only)
        if self.training and self.input_speckle_p > 0:
            x = speckle_mask(x, self.input_speckle_p, self.input_speckle_mode)


        # Pass input through RNN
        output, hidden_states = self.gru(x, states)

        # Optional post-GRU head
        output = self.head(output)

        # Compute logits
        logits = self.out(output)

        
        if return_state:
            return logits, hidden_states
        
        return logits


class RMSNorm(nn.Module):
    def __init__(self, d: int, eps: float = 1e-8):
        super().__init__()
        self.eps = eps
        self.scale = nn.Parameter(torch.ones(d))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B,T,D)
        rms = x.pow(2).mean(dim=-1, keepdim=True).add(self.eps).sqrt()
        return (x / rms) * self.scale


class MyBatchNorm1d(nn.Module):
    def __init__(self, d, eps=1e-5, momentum=0.1):
        super().__init__()
        self.bn = nn.BatchNorm1d(d, eps=eps, momentum=momentum)

    def forward(self, x):
        if x.dim() == 3:  # (B,T,D)
            b, t, d = x.shape
            y = x.reshape(b * t, d)
            y = self.bn(y)
            return y.reshape(b, t, d)
        elif x.dim() == 2:  # (B,D)
            return self.bn(x)
        else:
            raise ValueError(f"MyBatchNorm1d expected 2D or 3D input, got shape={tuple(x.shape)}")

def build_time_norm(norm_type: str, d: int) -> nn.Module:
    norm_type = (norm_type or "none").lower()
    if norm_type == "bn":
        return MyBatchNorm1d(d)
    if norm_type == "layernorm":
        return nn.LayerNorm(d)
    if norm_type == "rmsnorm":
        return RMSNorm(d)
    if norm_type == "none":
        return nn.Identity()
    raise ValueError(f"Unknown norm_type={norm_type}. Use one of: bn, layernorm, rmsnorm, none.")

def get_activation(name: str) -> nn.Module:
    name = (name or "gelu").lower()
    if name == "gelu":
        return nn.GELU()
    if name == "relu":
        return nn.ReLU()
    if name == "silu":
        return nn.SiLU()
    raise ValueError(f"Unknown activation={name}. Use one of: gelu, relu, silu.")


def speckle_mask(x: torch.Tensor, p: float, mode: str) -> torch.Tensor:
    """
    Coordinated dropout / speckled masking.
    x: (B,T,D)
    mode:
      - 'feature': drop entire features across all timesteps (mask shape Bx1xD)
      - 'time':    drop entire timesteps across all features (mask shape BxTx1)
      - 'both':    elementwise (BxTxD)  (usually less stable; keep for ablation)
    """
    if p <= 0.0:
        return x
    mode = (mode or "feature").lower()
    B, T, D = x.shape
    if mode == "feature":
        mask = torch.rand(B, 1, D, device=x.device) < p
    elif mode == "time":
        mask = torch.rand(B, T, 1, device=x.device) < p
    elif mode == "both":
        mask = torch.rand(B, T, D, device=x.device) < p
    else:
        raise ValueError(f"Unknown speckle mode={mode}. Use: feature, time, both.")
    return x.masked_fill(mask, 0.0)


class ResidualFFNBlock(nn.Module):
    """
    Simple GPT-style MLP block without attention:
      x <- x + Dropout(Act(Linear(Norm(x))))
    Works on (B,T,D).
    """
    def __init__(self, d: int, norm_type: str, dropout: float, activation: str):
        super().__init__()
        self.norm = build_time_norm(norm_type, d)
        self.lin = nn.Linear(d, d)
        nn.init.xavier_uniform_(self.lin.weight)
        self.act = get_activation(activation)
        self.drop = nn.Dropout(p=float(dropout))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = self.lin(self.norm(x))
        y = self.act(y)
        y = self.drop(y)
        return x + y


