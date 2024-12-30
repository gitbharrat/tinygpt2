import math
import inspect
from dataclasses import dataclass

import torch
import torch.nn as nn
from torch.nn import functional as F


@dataclass
class GPTConfig:
    block_size: int = 1024  # Maximum Sequence Length
    vocab_size: int = 50257  # Vocabulary Size
    n_layer: int = 12  # Number of Layers
    n_head: int = 12  # Number of Attention Heads
    n_embd: int = 768  # Embedding Dimension
    dropout: float = 0.0  # Dropout Probability
    bias: bool = True  # Bias


class CausalSelfAttention(nn.Module):

    def __init__(self, config):
        super().__init__()

        assert config.n_embd % config.n_head == 0

        # Key, Querry, Value Pair
        self.c_attn = nn.Linear(config.n_embd, 3 * config.n_embd)
        # Output Projection
        self.c_proj = nn.Linear(config.n_embd, config.n_embd)
        # Regularization
        self.attn_dropout = nn.Dropout(config.dropout)
        self.resid_dropout = nn.Dropout(config.dropout)
        self.dropout = config.dropout
        self.n_head = config.n_head
        self.n_embd = config.n_embd

        # Called as bias in Hugging Face Library, but it is actually a mask.
        self.flash = hasattr(torch.nn.functional, "scaled_dot_product_attention")
        if not self.flash:
            self.register_buffer(
                "bias",
                torch.tril(
                    torch.ones(config.block_size, config.block_size).view(
                        1, 1, config.block_size, config.block_size
                    )
                ),
            )

    def forward(self, x):
        B, T, C = x.size()  # Batch Size, Sequence Length, Embedding Dimensionality

        # Calculate Querry, Key, Value for all the heads in the batch
        qkv = self.c_attn(x)
        q, k, v = qkv.split(self.n_embd, dim=2)

        k = k.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
        q = q.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
        v = v.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)

        if self.flash:
            y = torch.nn.functional.scaled_dot_product_attention(
                q,
                k,
                v,
                attn_mask=None,
                dropout_p=self.dropout if self.training else 0,
                is_causal=True,
            )
        else:
            att = (q @ k.transpose(-2, 1)) * (1.0 / math.sqrt(k.size(-1)))
            att = att.masked_fill(self.bias[:, :, :T, :T] == 0, float("-inf"))
            att = F.softmax(att, dim=-1)
            att = self.attn_dropout(att)
            y = att @ v

        # Concatination Operation
        y = y.transpose(1, 2).contiguous().view(B, T, C)

        y = self.resid_dropout(self.c_proj(y))
        return y


class MLP(nn.Module):

    def __init__(self, config):
        super().__init__()
        self.c_fc = nn.Linear(config.n_embd, 4 * config.n_embd, bias=config.bias)
        self.gelu = nn.GELU(approximate="tanh")
        self.c_proj = nn.Linear(4 * config.n_embd, config.n_embd, bias=config.bias)
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, x):
        x = self.c_fc(x)
        x = self.gelu(x)
        x = self.c_proj(x)
        x = self.dropout(x)
        return x


class Block(nn.Module):

    def __init__(self, config):
        super().__init__()
        self.ln_1 = nn.LayerNorm(config.n_embd, bias=config.bias)
        self.attn = CausalSelfAttention(config)
        self.ln_2 = nn.LayerNorm(config.n_embd, bias=config.bias)
        self.mlp = MLP(config)

    def forward(self, x):
        x = x + self.attn(self.ln_1(x))
        x = x + self.mlp(self.ln_2(x))
        return x


class GPT(nn.Module):

    def __init__(self, config):
        super().__init__()
        self.config = config
        assert config.vocab_size is not None
        assert config.block_size is not None
        self.transformer = nn.ModuleDict(
            dict(
                wte=nn.Embedding(config.vocab_size, config.n_embd),
                wpe=nn.Embedding(config.block_size, config.n_embd),
                drop=nn.Dropout(config.dropout),
                h=nn.ModuleList([Block(config) for _ in range(config.n_layer)]),
                ln_f=nn.LayerNorm(config.n_embd),
            )
        )

        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)

        self.transformer.wte.weight = self.lm_head.weight

        # Initialize all weights
        self.apply(self._init_weights)

        for pn, p in self.named_parameters():
            if pn.endswith("c_proj.weight"):
                torch.nn.init.normal_(
                    p, mean=0.0, std=0.02 / (math.sqrt(2 * config.n_layer))
                )

        # Report Parameters
        print("Number of Parameters: % .2fM" % (self.get_num_params() / 1e6))

    def get_num_params(self, non_embedding=True):
        n_params = sum([p.numel() for p in self.parameters()])
        if non_embedding:
            n_params -= self.transformer.wpe.weight.numel()
        return n_params

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, idx, targets=None):
        device = idx.device
        b, t = idx.size()
        assert (
            t <= self.config.block_size
        ), f"Sequence Length {t} should be less <= {self.config.block_size}"
        pos = torch.arange(0, t, dtype=torch.long, device=device)

        # Forward the GPT model
        tok_emb = self.transformer.wte(idx)
        pos_emb = self.transformer.wpe(pos)
        x = self.transformer.drop(tok_emb + pos_emb)  # Dropout
        for block in self.transformer.h:
            x = block(x)
        x = self.transformer.ln_f(x)

        if targets is not None:
            logits = self.lm_head(x)
            loss = F.cross_entropy(
                logits.view(-1, logits.size(-1)), targets.view(-1), ignore_index=-1
            )
        else:
            logits = self.lm_head(x[:, [-1], :])
            loss = None

        return logits, loss

    def crop_block(self, block_size):
        # Modify the block size, if required
        assert block_size <= self.config.block_size
        self.config.block_size = block_size
        self.transformer.wpe.weight = nn.Parameter(
            self.transformer.wpe.weight[:block_size]
        )
        for block in self.transformer.h:
            if hasattr(block.attn, "bias"):
                block.attn.bias = block.attn.bias[:, :, :block_size, :block_size]

    @classmethod
    def from_pretrained(cls, model_type, override_args=None):
        assert model_type in {"gpt2", "gpt2-medum", "gpt2-large", "gpt2-xl"}
        override_args = override_args or {}
        assert all(k == "dropout" for k in override_args)
        from transformers import GPT2LMHeadModel

        print("Loading weight from pretrained %s" % model_type)

        config_args = {
            "gpt2": dict(n_layer=12, n_head=12, n_embd=768),
            "gpt2-medium": dict(n_layer=24, n_head=16, n_embd=1024),
            "gpt2-large": dict(n_layer=36, n_head=20, n_embd=1280),
            "gpt2-xl": dict(n_layer=48, n_head=25, n_embd=1600),
        }[model_type]

        print("Forcing vocab_size=50257, block_size=1024 and bias=True")
        config_args["vocab_size"] = 50257
        config_args["block_size"] = 1024
        config_args["bias"] = True

        if "dropout" in override_args:
            print(f"Overriding Dropout Rate with {override_args['dropout']}")
            config_args["dropout"] = override_args["dropout"]

        # Create new GPT
        config = GPTConfig(**config_args)
        model = GPT(config)
        sd = model.state_dict()
        sd_keys = sd.keys()
        sd_keys = [
            k for k in sd_keys if not k.endswith(".attn.bias")
        ]  # Discard the mask

        # Init HF model
        model_hf = GPT2LMHeadModel.from_pretrained(model_type)
        sd_hf = model_hf.state_dict()

        # Copying the Parameters
        sd_keys_hf = sd_hf.keys()
        # Ignoring Mask Buffer in HF parameters
        sd_keys_hf = [k for k in sd_keys_hf if not k.endswith(".attn.masked_bias")]
        sd_keys_hf = [k for k in sd_keys_hf if not k.endswith(".attn.bias")]

        # Adjusting OpenAI Conv1D for Torch
        transposed = [
            "attn.c_attn.weight",
            "attn.c_proj.weight",
            "mlp.c_fc.weight",
            "mlp.c_proj.weight",
        ]

        assert len(sd_keys_hf) == len(
            sd_keys
        ), f"Mismatched Keys {len(sd_keys_hf) != len(sd_keys)}"

        for k in sd_keys_hf:
            if any(k.endswith(w) for w in transposed):
                assert sd_hf[k].shape[::-1] == sd[k].shape
                with torch.no_grad():
                    sd[k].copy_(sd_hf[k].t())
            else:
                assert sd_hf[k].shape == sd[k].shape, f"{sd_hf[k].shape},{sd[k].shape}"
                with torch.no_grad():
                    sd[k].copy_(sd_hf[k])

        return model

    def configure_optimizer(self, weight_decay, learning_rate, betas, device):

        # Filtering Gradient Params
        param_dict = {pn: p for pn, p in self.named_parameters()}
        param_dict = {pn: p for pn, p in param_dict.items() if p.requires_grad}

        # Optimizing Tensors using Weight Decay
        decay_params = [p for n, p in param_dict.items() if p.dim() >= 2]
        nodecay_params = [p for n, p in param_dict.items() if p.dim < 2]
        optim_groups = [
            {"params": decay_params, "weight_decay": weight_decay},
            {"params": nodecay_params, "weight_decay": 0.0},
        ]

        # Total Parameters
        num_decay_params = sum([p.numel() for p in decay_params])
        num_nodecay_params = sum([p.numel() for p in nodecay_params])
        print(
            f"Decayed Parameters Tensors: {len(decay_params)} with {num_decay_params:,} parameters"
        )
        print(
            f"Non-Decayed Parameters Tensors: {len(nodecay_params)} with {num_nodecay_params:,} parameters"
        )

        # AdamW in Fused
        fused_available = "fused" in inspect.signature(torch.optim.AdamW).parameters
        use_fused = fused_available and device == "cuda"
        extra_args = dict(fused=True) if use_fused else dict()
        optimizer = torch.optim.AdamW(
            optim_groups, learning_rate=learning_rate, betas=betas, **extra_args
        )
        print(f"Using Fused AdamW: {use_fused}")

        return optimizer

    def estimate_mfu(self, fwdbwd_per_iter, dt):
        # Estimating Model Flops Utilization
        N = self.get_num_params()
        cfg = self.config
        L, H, Q, T = cfg.n_layer, cfg.n_head, cfg.n_embd // cfg.n_head, cfg.block_size
        flops_per_token = 6 * N + 12 * L * H * Q * T
        flops_per_fwdbkwd = flops_per_token * T
        flops_per_iter = flops_per_fwdbkwd * fwdbwd_per_iter

        # Flops with respect to A100's
        flops_achieved = flops_per_iter * (1.0 / dt)
        flops_promised = 312e12
        mfu = flops_achieved / flops_promised

        return mfu

    @torch.no_grad()
    def generate(self, idx, max_new_tokens, temperature=1.0, top_k=None):
        for _ in range(max_new_tokens):
            # Truncating Sequence if Long
            idx_cond = (
                idx
                if idx.size(1) <= self.config.block_size
                else idx[:, -self.config.block_size :]
            )

            # Extract Logits Scaled by temperature
            logits, _ = self(idx_cond)
            logits = logits[:, -1, :] / temperature

            # Optional for top_k
            if top_k is not None:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:, [-1]]] = -float("Inf")

            # Apply Softmax for Normalized Probabilities
            probs = F.softmax(logits, dim=1)

            # Sample from the distribution
            idx_next = torch.multinomial(probs, num_samples=1)

            # Append Sampled Sequence to Runing Sequence
            idx = torch.cat((idx, idx_next), dim=1)

        return idx