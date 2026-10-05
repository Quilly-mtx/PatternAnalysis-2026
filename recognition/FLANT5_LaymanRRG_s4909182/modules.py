"""
Model components: FLAN-T5 backbone and LoRA adapter layers.

LoRA (Hu et al., ICLR 2022) freezes a pretrained weight W and learns a low-rank
update, so the layer computes  W x + (alpha / r) * B A x  with A of shape
(r, d_in) and B of shape (d_out, r). Only A and B receive gradients.
"""
import math

import torch
import torch.nn as nn


class LoRALinear(nn.Module):
    """A frozen nn.Linear with a trainable low-rank update added to its output."""

    def __init__(self, base, r=8, alpha=16, dropout=0.0):
        super().__init__()
        if r <= 0:
            raise ValueError("LoRA rank r must be positive")
        self.base = base
        self.r = r
        self.scaling = alpha / r
        self.dropout = nn.Dropout(dropout)
        self.merged = False

        for p in self.base.parameters():
            p.requires_grad = False

        self.lora_A = nn.Parameter(torch.empty(r, base.in_features))
        self.lora_B = nn.Parameter(torch.zeros(base.out_features, r))
        # A is random and B is zero, so B A = 0 and the layer starts identical to the pretrained one.
        nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))

    def delta_weight(self):
        """The low-rank update (alpha / r) * B A, same shape as the base weight."""
        return self.scaling * (self.lora_B @ self.lora_A)

    def forward(self, x):
        out = self.base(x)
        if self.merged:
            return out
        update = self.dropout(x) @ self.lora_A.t() @ self.lora_B.t()
        return out + self.scaling * update

    @torch.no_grad()
    def merge(self):
        """Fold the update into the base weight so inference costs the same as the original layer."""
        if not self.merged:
            self.base.weight += self.delta_weight()
            self.merged = True

    @torch.no_grad()
    def unmerge(self):
        """Undo merge(), restoring the pretrained base weight."""
        if self.merged:
            self.base.weight -= self.delta_weight()
            self.merged = False


def inject_lora(model, r=8, alpha=16, dropout=0.05, targets=("q", "v")):
    """
    Freeze every parameter of `model`, then wrap each nn.Linear whose attribute
    name is in `targets` with a LoRALinear. In T5 the attention projections are
    named q, k, v and o; the LoRA paper adapts query and value.

    Returns the number of layers wrapped.
    """
    for p in model.parameters():
        p.requires_grad = False

    wrapped = 0
    for parent in list(model.modules()):
        for name, child in list(parent.named_children()):
            if name in targets and isinstance(child, nn.Linear):
                setattr(parent, name, LoRALinear(child, r=r, alpha=alpha, dropout=dropout))
                wrapped += 1
    return wrapped


def set_lora_merged(model, merged):
    """Merge (True) or unmerge (False) every LoRA layer in the model."""
    for module in model.modules():
        if isinstance(module, LoRALinear):
            module.merge() if merged else module.unmerge()


def lora_state_dict(model):
    """Only the adapter tensors, so a LoRA checkpoint is a few MB instead of the full model."""
    return {k: v.detach().cpu() for k, v in model.state_dict().items() if "lora_" in k}


def count_parameters(model):
    """Return (total, trainable) parameter counts."""
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total, trainable


def build_model(name="google/flan-t5-base", mode="lora", r=8, alpha=16, dropout=0.05, return_attention=False):
    """
    Load a pretrained FLAN-T5 and prepare it for one of three settings:
      "zero_shot" - all weights frozen, used as the untuned baseline
      "lora"      - backbone frozen, LoRA adapters on attention q and v
      "full"      - every weight trainable

    return_attention switches to the slower attention implementation that can
    hand back its weights; it is only needed to plot attention maps.
    """
    # Imported here so the LoRA components above depend on PyTorch only.
    from transformers import AutoModelForSeq2SeqLM

    options = {"attn_implementation": "eager"} if return_attention else {}
    model = AutoModelForSeq2SeqLM.from_pretrained(name, **options)
    if mode == "lora":
        inject_lora(model, r=r, alpha=alpha, dropout=dropout)
    elif mode == "zero_shot":
        for p in model.parameters():
            p.requires_grad = False
    elif mode != "full":
        raise ValueError(f"unknown mode: {mode}")
    return model
