"""
Shared helpers: seeding, device and autocast choice, ROUGE scoring, batched
generation, checkpoint I/O and training curves.
"""
import random
import re

import torch
from rouge_score import rouge_scorer

from dataset import MAX_SOURCE_LENGTH, MAX_TARGET_LENGTH, encode_reports
from modules import lora_state_dict

ROUGE_KEYS = ("rouge1", "rouge2", "rougeL", "rougeLsum")


def set_seed(seed):
    """Seed Python and PyTorch so runs are repeatable."""
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def pick_device():
    """CUDA if present, then Apple MPS, then CPU."""
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def autocast(device, enabled=True):
    """
    bfloat16 mixed precision on CUDA, full precision elsewhere.

    float16 is avoided on purpose: T5 activations overflow to NaN under fp16.
    """
    use = enabled and device.type == "cuda" and torch.cuda.is_bf16_supported()
    return torch.autocast(device_type="cuda" if use else "cpu", dtype=torch.bfloat16, enabled=use)


def _one_sentence_per_line(text):
    """rougeLsum is computed sentence by sentence and expects sentences separated by newlines."""
    return "\n".join(s.strip() for s in re.split(r"(?<=[.!?])\s+", text.strip()) if s.strip())


def rouge_per_example(predictions, references):
    """ROUGE F-measures (x100) for every prediction / reference pair, as a list of dicts."""
    scorer = rouge_scorer.RougeScorer(list(ROUGE_KEYS), use_stemmer=True)
    rows = []
    for prediction, reference in zip(predictions, references):
        scores = scorer.score(_one_sentence_per_line(reference), _one_sentence_per_line(prediction))
        rows.append({key: 100 * scores[key].fmeasure for key in ROUGE_KEYS})
    return rows


def mean_rouge(rows):
    """Average a list of per-example ROUGE dicts."""
    return {key: sum(row[key] for row in rows) / max(len(rows), 1) for key in ROUGE_KEYS}


@torch.no_grad()
def generate_summaries(model, tokenizer, reports, device, batch_size=64, num_beams=4,
                       max_new_tokens=MAX_TARGET_LENGTH, amp=True):
    """
    Beam-search a summary for every report.

    Reports are processed longest first so each batch holds similar lengths
    (less padding, and an out-of-memory error would show up immediately), then
    the outputs are put back in the original order.

    Returns (summaries, number of generated tokens).
    """
    was_training = model.training
    model.eval()
    order = sorted(range(len(reports)), key=lambda i: len(reports[i]), reverse=True)
    summaries, n_tokens = [None] * len(reports), 0
    for start in range(0, len(order), batch_size):
        indices = order[start:start + batch_size]
        inputs = encode_reports(tokenizer, [reports[i] for i in indices], MAX_SOURCE_LENGTH).to(device)
        with autocast(device, amp):
            output = model.generate(**inputs, num_beams=num_beams, max_new_tokens=max_new_tokens)
        n_tokens += int((output != tokenizer.pad_token_id).sum())
        for i, text in zip(indices, tokenizer.batch_decode(output, skip_special_tokens=True)):
            summaries[i] = text
    model.train(was_training)
    return summaries, n_tokens


def save_checkpoint(model, mode, path):
    """LoRA runs store only the adapter tensors; full fine-tuning stores the whole model."""
    state = lora_state_dict(model) if mode == "lora" else model.state_dict()
    torch.save(state, path)


def load_checkpoint(model, mode, path):
    """Inverse of save_checkpoint. The model must already be built in the same mode."""
    state = torch.load(path, map_location="cpu")
    result = model.load_state_dict(state, strict=(mode != "lora"))
    if result.unexpected_keys:
        raise RuntimeError(f"checkpoint has keys the model does not: {result.unexpected_keys[:3]}")


def plot_history(train_curve, history, path):
    """Save loss and dev ROUGE curves. train_curve is [(step, loss)], history is the list of dev records."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, (left, right) = plt.subplots(1, 2, figsize=(11, 4))
    left.plot(*zip(*train_curve), label="train (running mean)", alpha=0.8)
    steps = [record["step"] for record in history]
    left.plot(steps, [record["dev_loss"] for record in history], "o-", label="dev")
    left.set(xlabel="optimiser step", ylabel="cross-entropy loss", title="Loss")
    left.legend()
    for key in ROUGE_KEYS:
        right.plot(steps, [record[key] for record in history], "o-", label=key)
    right.set(xlabel="optimiser step", ylabel="F-measure x 100", title="Dev ROUGE")
    right.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
