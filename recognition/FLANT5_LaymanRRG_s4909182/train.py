"""
Training, validation, testing and checkpoint saving.

Examples:
    python train.py --mode zero_shot --data_dir data --out_dir runs/zero_shot
    python train.py --mode lora      --data_dir data --out_dir runs/lora_r8
    python train.py --mode full      --data_dir data --out_dir runs/full

Every run ends by scoring the held-out test split and writing predictions,
metrics, curves and resource usage to --out_dir. The checkpoint that is tested
is the one with the best ROUGE-Lsum on the dev split.
"""
import argparse
import json
import os
import time

import torch
import transformers
from transformers import AutoTokenizer

from dataset import SOURCE_COL, TARGET_COL, make_loader, make_splits
from modules import build_model, count_parameters
from utils import (ROUGE_KEYS, autocast, generate_summaries, load_checkpoint, mean_rouge, pick_device,
                   plot_history, rouge_per_example, save_checkpoint, set_seed)

# LoRA trains a small number of freshly initialised weights and tolerates a larger step size.
DEFAULT_LR = {"lora": 1e-3, "full": 1e-4}


def parse_args():
    parser = argparse.ArgumentParser(description="Fine-tune FLAN-T5 for layperson radiology summaries")
    parser.add_argument("--mode", required=True, choices=["zero_shot", "lora", "full"])
    parser.add_argument("--out_dir", required=True)
    parser.add_argument("--data_dir", default=None, help="folder with train.parquet and validation.parquet")
    parser.add_argument("--model", default="google/flan-t5-base")
    parser.add_argument("--epochs", type=int, default=2)
    parser.add_argument("--max_steps", type=int, default=0, help="stop after this many steps (0 = use --epochs)")
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--eval_batch_size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=None, help="default: 1e-3 for lora, 1e-4 for full")
    parser.add_argument("--weight_decay", type=float, default=0.0)
    parser.add_argument("--warmup_ratio", type=float, default=0.05)
    parser.add_argument("--rank", type=int, default=8)
    parser.add_argument("--alpha", type=float, default=16)
    parser.add_argument("--dropout", type=float, default=0.05)
    parser.add_argument("--num_beams", type=int, default=4)
    parser.add_argument("--eval_every", type=int, default=0, help="dev evaluation interval in steps (0 = each epoch)")
    parser.add_argument("--log_every", type=int, default=50)
    parser.add_argument("--dev_rouge_samples", type=int, default=500, help="dev reports generated per evaluation")
    parser.add_argument("--max_train_samples", type=int, default=0, help="subsample train, for smoke tests")
    parser.add_argument("--max_dev_samples", type=int, default=0, help="subsample dev, for smoke tests")
    parser.add_argument("--max_test_samples", type=int, default=0, help="subsample test, for smoke tests")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", default=None, help="cuda, mps or cpu (default: best available)")
    parser.add_argument("--no_amp", action="store_true", help="disable bfloat16 mixed precision")
    return parser.parse_args()


@torch.no_grad()
def dev_loss(model, loader, device, amp):
    """Teacher-forced cross-entropy on the dev split, averaged over target tokens."""
    model.eval()
    total, n_tokens = 0.0, 0
    for batch in loader:
        batch = {key: value.to(device) for key, value in batch.items()}
        with autocast(device, amp):
            loss = model(**batch).loss
        tokens = int((batch["labels"] != -100).sum())
        total += loss.item() * tokens
        n_tokens += tokens
    model.train()
    return total / n_tokens


def evaluate_dev(model, tokenizer, dev_loader, dev_sample, args, device):
    """Dev loss on the whole split plus ROUGE of generated summaries on a fixed sample of it."""
    record = {"dev_loss": dev_loss(model, dev_loader, device, not args.no_amp)}
    predictions, _ = generate_summaries(model, tokenizer, dev_sample[SOURCE_COL].tolist(), device,
                                        args.eval_batch_size, args.num_beams, amp=not args.no_amp)
    record.update(mean_rouge(rouge_per_example(predictions, dev_sample[TARGET_COL].tolist())))
    return record


def train(model, tokenizer, train_frame, dev_frame, args, device, checkpoint_path):
    """Optimise the trainable parameters and keep the checkpoint with the best dev ROUGE-Lsum."""
    generator = torch.Generator().manual_seed(args.seed)
    train_loader = make_loader(train_frame, tokenizer, args.batch_size, shuffle=True, generator=generator)
    dev_loader = make_loader(dev_frame, tokenizer, args.eval_batch_size)
    dev_sample = dev_frame.sample(min(args.dev_rouge_samples, len(dev_frame)), random_state=0)

    parameters = [p for p in model.parameters() if p.requires_grad]
    lr = args.lr if args.lr is not None else DEFAULT_LR[args.mode]
    optimizer = torch.optim.AdamW(parameters, lr=lr, weight_decay=args.weight_decay)

    total_steps = args.max_steps or len(train_loader) * args.epochs
    warmup_steps = max(1, int(args.warmup_ratio * total_steps))
    # Linear warm-up to the peak learning rate, then linear decay to zero.
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer,
        lambda step: (step + 1) / warmup_steps if step < warmup_steps
        else max(0.0, (total_steps - step) / max(1, total_steps - warmup_steps)),
    )
    eval_every = args.eval_every or len(train_loader)
    out_dir = os.path.dirname(checkpoint_path)

    history, train_curve, recent = [], [], []
    best, step, since_eval = -1.0, 0, []
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats()
    start = time.time()
    model.train()
    while step < total_steps:
        for batch in train_loader:
            batch = {key: value.to(device) for key, value in batch.items()}
            with autocast(device, not args.no_amp):
                loss = model(**batch).loss
            if not torch.isfinite(loss):
                raise RuntimeError(f"non-finite loss at step {step}")
            loss.backward()
            torch.nn.utils.clip_grad_norm_(parameters, 1.0)
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad(set_to_none=True)

            step += 1
            recent.append(loss.item())
            since_eval.append(loss.item())
            if step % args.log_every == 0:
                train_curve.append((step, sum(recent) / len(recent)))
                print(f"step {step}/{total_steps}  loss {train_curve[-1][1]:.4f}  "
                      f"lr {scheduler.get_last_lr()[0]:.2e}  {time.time() - start:.0f}s", flush=True)
                recent = []

            if step % eval_every == 0 or step == total_steps:
                record = {"step": step, "train_loss": sum(since_eval) / len(since_eval)}
                record.update(evaluate_dev(model, tokenizer, dev_loader, dev_sample, args, device))
                since_eval = []
                history.append(record)
                print("dev  " + "  ".join(f"{k} {v:.4f}" for k, v in record.items() if k != "step"), flush=True)
                if record["rougeLsum"] > best:
                    best = record["rougeLsum"]
                    save_checkpoint(model, args.mode, checkpoint_path)
                    print(f"saved best checkpoint (dev rougeLsum {best:.2f})", flush=True)
                with open(os.path.join(out_dir, "history.json"), "w") as f:
                    json.dump({"train_curve": train_curve, "dev": history}, f, indent=1)
                if train_curve:
                    plot_history(train_curve, history, os.path.join(out_dir, "curves.png"))
            if step >= total_steps:
                break

    return {
        "train_seconds": time.time() - start,
        "train_steps": step,
        "best_dev_rougeLsum": best,
        "peak_vram_train_mb": torch.cuda.max_memory_allocated() / 2**20 if device.type == "cuda" else None,
    }


def run_test(model, tokenizer, test_frame, args, device, out_dir):
    """
    Generate a summary for every held-out report and score it.

    ROUGE is reported for all reports, for reports whose text was / was not
    seen in training, and per source dataset. "copy_input" scores the untouched
    report against the reference: a system must beat it for ROUGE to mean anything.
    """
    reports, references = test_frame[SOURCE_COL].tolist(), test_frame[TARGET_COL].tolist()
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats()
    start = time.time()
    predictions, n_tokens = generate_summaries(model, tokenizer, reports, device, args.eval_batch_size,
                                               args.num_beams, amp=not args.no_amp)
    seconds = time.time() - start

    model_rows = rouge_per_example(predictions, references)
    copy_rows = rouge_per_example(reports, references)
    scored = test_frame[["source", "seen_in_train", SOURCE_COL, TARGET_COL]].copy()
    scored["prediction"] = predictions
    for key in ROUGE_KEYS:
        scored[key] = [row[key] for row in model_rows]
    scored.to_csv(os.path.join(out_dir, "test_predictions.csv"), index=False)

    groups = {"all": [True] * len(test_frame), "seen": test_frame["seen_in_train"], "unseen": ~test_frame["seen_in_train"]}
    for source in sorted(test_frame["source"].unique()):
        groups[f"source={source}"] = test_frame["source"] == source
    metrics = {}
    for name, mask in groups.items():
        picked = [i for i, keep in enumerate(mask) if keep]
        metrics[name] = {
            "n": len(picked),
            "model": mean_rouge([model_rows[i] for i in picked]),
            "copy_input": mean_rouge([copy_rows[i] for i in picked]),
        }
        print(f"test {name:22s} n={len(picked):5d}  "
              + "  ".join(f"{k} {v:5.2f}" for k, v in metrics[name]["model"].items())
              + f"  | copy_input rougeLsum {metrics[name]['copy_input']['rougeLsum']:5.2f}", flush=True)

    resources = {
        "test_seconds": seconds,
        "seconds_per_report": seconds / len(reports),
        "generated_tokens_per_second": n_tokens / seconds,
        "peak_vram_test_mb": torch.cuda.max_memory_allocated() / 2**20 if device.type == "cuda" else None,
    }
    return metrics, resources


def main():
    args = parse_args()
    os.makedirs(args.out_dir, exist_ok=True)
    set_seed(args.seed)
    device = torch.device(args.device) if args.device else pick_device()

    splits = make_splits(args.data_dir)
    train_frame, dev_frame, test_frame = splits["train"], splits["dev"], splits["test"]
    if args.max_train_samples:
        train_frame = train_frame.sample(args.max_train_samples, random_state=args.seed)
    if args.max_dev_samples:
        dev_frame = dev_frame.sample(args.max_dev_samples, random_state=0)
    if args.max_test_samples:
        test_frame = test_frame.sample(args.max_test_samples, random_state=0).reset_index(drop=True)

    tokenizer = AutoTokenizer.from_pretrained(args.model)
    model = build_model(args.model, args.mode, r=args.rank, alpha=args.alpha, dropout=args.dropout).to(device)
    total, trainable = count_parameters(model)
    print(f"device {device}  mode {args.mode}  parameters {total:,}  trainable {trainable:,} "
          f"({100 * trainable / total:.3f}%)  train {len(train_frame):,}  dev {len(dev_frame):,}  "
          f"test {len(test_frame):,}", flush=True)

    config = dict(vars(args), device=str(device), torch=torch.__version__, transformers=transformers.__version__,
                  train_rows=len(train_frame), dev_rows=len(dev_frame), test_rows=len(test_frame))
    with open(os.path.join(args.out_dir, "config.json"), "w") as f:
        json.dump(config, f, indent=1)

    resources = {"total_parameters": total, "trainable_parameters": trainable}
    if args.mode != "zero_shot":
        checkpoint_path = os.path.join(args.out_dir, "best.pt")
        resources.update(train(model, tokenizer, train_frame, dev_frame, args, device, checkpoint_path))
        load_checkpoint(model, args.mode, checkpoint_path)

    metrics, test_resources = run_test(model, tokenizer, test_frame, args, device, args.out_dir)
    resources.update(test_resources)
    with open(os.path.join(args.out_dir, "test_metrics.json"), "w") as f:
        json.dump({"metrics": metrics, "resources": resources}, f, indent=1)
    print("resources " + json.dumps(resources), flush=True)


if __name__ == "__main__":
    main()
