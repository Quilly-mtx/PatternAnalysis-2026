"""
Tests for the LoRA components in modules.py.

They check the properties the method relies on: the adapted layer starts out
identical to the pretrained one, only the low-rank matrices A and B are
trained, merging the update into the weight does not change the output, and
an adapter-only checkpoint reproduces the model. All checks use a tiny randomly
initialised T5, so they run offline in a few seconds:

    python test_lora.py
    python test_lora.py --pretrained google/flan-t5-base   # also check the real model's parameter count

Author: Weixiang Han (s4909182), COMP3710 Pattern Recognition Report, 2026.
AI assistance is disclosed in README.md (Artificial Intelligence Usage Disclosure).
"""
import argparse
import copy

import torch
from transformers import T5Config, T5ForConditionalGeneration

from modules import LoRALinear, count_parameters, inject_lora, lora_state_dict, set_lora_merged

R, ALPHA = 4, 8


def tiny_t5():
    """A small T5 with random weights: 2 encoder and 2 decoder layers, d_model 32."""
    config = T5Config(vocab_size=100, d_model=32, d_kv=8, d_ff=64, num_layers=2, num_heads=4,
                      decoder_start_token_id=0, pad_token_id=0)
    return T5ForConditionalGeneration(config).eval()


def test_layer_starts_identical():
    torch.manual_seed(0)
    base = torch.nn.Linear(32, 48, bias=False)
    reference = copy.deepcopy(base)
    layer = LoRALinear(base, r=R, alpha=ALPHA)
    x = torch.randn(4, 7, 32)
    # B is initialised to zero, so the update B A x is exactly zero.
    assert torch.equal(layer(x), reference(x))


def test_layer_output_matches_formula():
    torch.manual_seed(0)
    base = torch.nn.Linear(32, 48, bias=False)
    weight = base.weight.detach().clone()
    layer = LoRALinear(base, r=R, alpha=ALPHA)
    torch.nn.init.normal_(layer.lora_B)
    x = torch.randn(4, 7, 32)
    expected = x @ (weight + (ALPHA / R) * layer.lora_B @ layer.lora_A).t()
    assert torch.allclose(layer(x), expected, atol=1e-5)
    assert torch.linalg.matrix_rank(layer.delta_weight()) <= R


def test_merge_and_unmerge():
    torch.manual_seed(0)
    base = torch.nn.Linear(32, 48, bias=False)
    weight = base.weight.detach().clone()
    layer = LoRALinear(base, r=R, alpha=ALPHA)
    torch.nn.init.normal_(layer.lora_B)
    x = torch.randn(4, 7, 32)
    before = layer(x)
    layer.merge()
    assert torch.allclose(layer(x), before, atol=1e-5)
    layer.unmerge()
    assert torch.allclose(layer.base.weight, weight, atol=1e-6)


def test_only_a_and_b_get_gradients():
    torch.manual_seed(0)
    layer = LoRALinear(torch.nn.Linear(32, 48, bias=False), r=R, alpha=ALPHA)
    layer(torch.randn(4, 32)).pow(2).sum().backward()
    assert layer.base.weight.grad is None
    # At the first step B = 0, so the gradient reaching A (which is multiplied by B) is zero.
    assert layer.lora_B.grad.abs().sum() > 0
    assert layer.lora_A.grad.abs().sum() == 0


def test_inject_wraps_query_and_value_only():
    model = tiny_t5()
    wrapped = inject_lora(model, r=R, alpha=ALPHA, dropout=0.0)
    # Encoder self-attention, decoder self-attention and decoder cross-attention: 2 + 2 + 2 blocks, q and v each.
    assert wrapped == 12
    names = {name.split(".")[-1] for name, module in model.named_modules() if isinstance(module, LoRALinear)}
    assert names == {"q", "v"}
    total, trainable = count_parameters(model)
    assert trainable == wrapped * R * (32 + 32)
    trainable_names = {n.split(".")[-1] for n, p in model.named_parameters() if p.requires_grad}
    assert trainable_names == {"lora_A", "lora_B"}


def test_model_identical_at_init_and_after_merge():
    torch.manual_seed(0)
    plain = tiny_t5()
    adapted = copy.deepcopy(plain)
    inject_lora(adapted, r=R, alpha=ALPHA, dropout=0.0)
    ids = torch.randint(1, 100, (2, 9))
    with torch.no_grad():
        assert torch.equal(plain(input_ids=ids, labels=ids).logits, adapted(input_ids=ids, labels=ids).logits)
        for module in adapted.modules():
            if isinstance(module, LoRALinear):
                torch.nn.init.normal_(module.lora_B, std=0.02)
        unmerged = adapted(input_ids=ids, labels=ids).logits
        set_lora_merged(adapted, True)
        merged = adapted(input_ids=ids, labels=ids).logits
        set_lora_merged(adapted, False)
    assert torch.allclose(unmerged, merged, atol=1e-4)


def test_training_step_reduces_loss():
    torch.manual_seed(0)
    model = tiny_t5()
    inject_lora(model, r=R, alpha=ALPHA, dropout=0.0)
    model.train()
    ids = torch.randint(1, 100, (2, 9))
    optimizer = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=1e-2)
    first = None
    for _ in range(20):
        loss = model(input_ids=ids, labels=ids).loss
        first = loss.item() if first is None else first
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
    assert loss.item() < first


def test_adapter_checkpoint_round_trip():
    torch.manual_seed(0)
    model = tiny_t5()
    inject_lora(model, r=R, alpha=ALPHA, dropout=0.0)
    for module in model.modules():
        if isinstance(module, LoRALinear):
            torch.nn.init.normal_(module.lora_B, std=0.02)
    state = lora_state_dict(model)
    assert all("lora_" in key for key in state)

    fresh = copy.deepcopy(model)
    for module in fresh.modules():
        if isinstance(module, LoRALinear):
            torch.nn.init.zeros_(module.lora_B)
    result = fresh.load_state_dict(state, strict=False)
    assert not result.unexpected_keys
    ids = torch.randint(1, 100, (2, 9))
    with torch.no_grad():
        assert torch.allclose(model(input_ids=ids, labels=ids).logits, fresh(input_ids=ids, labels=ids).logits)


def check_pretrained(name):
    """Parameter count on a real checkpoint, e.g. 884,736 trainable for FLAN-T5-base at rank 8."""
    from modules import build_model
    model = build_model(name, "lora", r=8, alpha=16)
    total, trainable = count_parameters(model)
    layers = sum(isinstance(m, LoRALinear) for m in model.modules())
    print(f"{name}: {layers} layers adapted, {trainable:,} of {total:,} parameters trainable ({100 * trainable / total:.3f}%)")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run the LoRA tests")
    parser.add_argument("--pretrained", default=None, help="also report the parameter count of this checkpoint")
    args = parser.parse_args()

    tests = [obj for name, obj in sorted(globals().items()) if name.startswith("test_") and callable(obj)]
    for test in tests:
        test()
        print(f"passed  {test.__name__}")
    print(f"all {len(tests)} tests passed")
    if args.pretrained:
        check_pretrained(args.pretrained)
