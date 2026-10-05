"""
Example usage of the trained model: load weights, generate and display summaries.

    python predict.py --mode lora --checkpoint runs/lora_r8_s0/best.pt --data_dir data
    python predict.py --mode lora --checkpoint runs/lora_r8_s0/best.pt --text "No pleural effusion."
    python predict.py --mode zero_shot --data_dir data

For each example the script prints the report, the reference summary (when
there is one), the generated summary and its ROUGE-Lsum, and saves a heatmap of
the decoder's cross-attention: which words of the report the model was reading
while it wrote each word of the summary.
"""
import argparse
import os
import textwrap

import numpy as np
import torch
from transformers import AutoTokenizer

from dataset import PROMPT, SOURCE_COL, TARGET_COL, encode_reports, make_splits
from modules import build_model
from utils import generate_summaries, load_checkpoint, pick_device, rouge_per_example

# Longer texts make the heatmap unreadable, so they are printed but not plotted.
MAX_HEATMAP_WORDS = 80


def parse_args():
    parser = argparse.ArgumentParser(description="Generate layperson summaries with a trained model")
    parser.add_argument("--mode", required=True, choices=["zero_shot", "lora", "full"])
    parser.add_argument("--checkpoint", default=None, help="best.pt written by train.py (not needed for zero_shot)")
    parser.add_argument("--model", default="google/flan-t5-base")
    parser.add_argument("--rank", type=int, default=8, help="must match the LoRA checkpoint")
    parser.add_argument("--alpha", type=float, default=16, help="must match the LoRA checkpoint")
    parser.add_argument("--data_dir", default=None, help="folder with train.parquet and validation.parquet")
    parser.add_argument("--text", default=None, help="summarise this report instead of sampling test reports")
    parser.add_argument("--num_examples", type=int, default=5)
    parser.add_argument("--subset", default="unseen", choices=["all", "seen", "unseen"],
                        help="sample test reports whose text was / was not seen in training")
    parser.add_argument("--compare_zero_shot", action="store_true", help="also show the untuned model's output")
    parser.add_argument("--num_beams", type=int, default=4)
    parser.add_argument("--out_dir", default="predict_outputs")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", default=None, help="cuda, mps or cpu (default: best available)")
    args = parser.parse_args()
    if args.mode != "zero_shot" and not args.checkpoint:
        parser.error("--checkpoint is required for lora and full modes")
    return args


def group_pieces(tokens):
    """Join SentencePiece tokens into words. Returns (words, index of the word each token belongs to)."""
    words, owner = [], []
    for token in tokens:
        if token.startswith("▁") or not words:
            words.append(token.lstrip("▁"))
        else:
            words[-1] += token
        owner.append(len(words) - 1)
    return words, owner


@torch.no_grad()
def cross_attention(model, tokenizer, report, summary, device):
    """
    Word-level cross-attention for one (report, generated summary) pair.

    The summary is fed back through the decoder and the attention of the last
    decoder layer is averaged over heads. Attention is summed over the pieces
    of each report word and averaged over the pieces of each summary word.
    Returns (matrix [summary words x report words], report words, summary words).
    """
    inputs = encode_reports(tokenizer, [report]).to(device)
    labels = tokenizer([summary], return_tensors="pt").input_ids.to(device)
    output = model(**inputs, labels=labels, output_attentions=True)
    attention = output.cross_attentions[-1][0].mean(dim=0).float().cpu().numpy()

    # Drop the task prompt at the start of the input and the end-of-sequence token on both sides.
    n_prompt = len(tokenizer(PROMPT).input_ids) - 1
    attention = attention[:-1, n_prompt:-1]
    source_words, source_owner = group_pieces(tokenizer.convert_ids_to_tokens(inputs.input_ids[0])[n_prompt:-1])
    target_words, target_owner = group_pieces(tokenizer.convert_ids_to_tokens(labels[0])[:-1])

    source_owner, target_owner = np.array(source_owner), np.array(target_owner)
    by_source = np.stack([attention[:, source_owner == w].sum(axis=1) for w in range(len(source_words))], axis=1)
    by_word = np.stack([by_source[target_owner == w].mean(axis=0) for w in range(len(target_words))], axis=0)
    return by_word, source_words, target_words


def plot_attention(weights, source_words, target_words, path):
    """Save the cross-attention matrix as a heatmap with one row per summary word."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(0.3 * len(source_words) + 2.5, 0.24 * len(target_words) + 2.5))
    image = ax.imshow(weights, aspect="auto", cmap="viridis")
    ax.set_xticks(range(len(source_words)), source_words, rotation=90, fontsize=7)
    ax.set_yticks(range(len(target_words)), target_words, fontsize=7)
    ax.set(xlabel="radiology report (input)", ylabel="generated summary",
           title="Cross-attention: last decoder layer, mean over heads")
    fig.colorbar(image, ax=ax, fraction=0.03, label="attention weight")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def show(label, text):
    """Print a labelled, wrapped block of text."""
    print(textwrap.fill(text.strip(), width=100, initial_indent=f"{label:11s}", subsequent_indent=" " * 11))


def main():
    args = parse_args()
    device = torch.device(args.device) if args.device else pick_device()
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    model = build_model(args.model, args.mode, r=args.rank, alpha=args.alpha, dropout=0.0, return_attention=True)
    if args.mode != "zero_shot":
        load_checkpoint(model, args.mode, args.checkpoint)
    model.to(device).eval()

    if args.text:
        reports, references, tags = [args.text], [None], ["custom input"]
    else:
        test = make_splits(args.data_dir)["test"]
        if args.subset != "all":
            test = test[test["seen_in_train"] == (args.subset == "seen")]
        sample = test.sample(min(args.num_examples, len(test)), random_state=args.seed)
        reports, references = sample[SOURCE_COL].tolist(), sample[TARGET_COL].tolist()
        tags = [f"{source}, {'seen' if seen else 'unseen'} in training"
                for source, seen in zip(sample["source"], sample["seen_in_train"])]

    predictions, _ = generate_summaries(model, tokenizer, reports, device, num_beams=args.num_beams)
    baseline = None
    if args.compare_zero_shot and args.mode != "zero_shot":
        untuned = build_model(args.model, "zero_shot").to(device)
        baseline, _ = generate_summaries(untuned, tokenizer, reports, device, num_beams=args.num_beams)
        del untuned

    os.makedirs(args.out_dir, exist_ok=True)
    for i, (report, reference, prediction) in enumerate(zip(reports, references, predictions)):
        print(f"\n=== example {i + 1} ({tags[i]}) " + "=" * 40)
        show("REPORT", report)
        if reference:
            show("REFERENCE", reference)
        if baseline:
            show("ZERO-SHOT", baseline[i])
        show(args.mode.upper(), prediction)
        if reference:
            print(f"{'ROUGE-Lsum':11s}{rouge_per_example([prediction], [reference])[0]['rougeLsum']:.1f}")

        if not prediction.strip():
            continue
        weights, source_words, target_words = cross_attention(model, tokenizer, report, prediction, device)
        if len(source_words) <= MAX_HEATMAP_WORDS and len(target_words) <= MAX_HEATMAP_WORDS:
            path = os.path.join(args.out_dir, f"attention_{i + 1}.png")
            plot_attention(weights, source_words, target_words, path)
            print(f"{'HEATMAP':11s}{path}")


if __name__ == "__main__":
    main()
