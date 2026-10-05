"""
Data loading, tokenisation and leakage-free splitting for BioLaySumm LaymanRRG.

The official test split has no reference summaries, so the official validation
split is used as the held-out test set and a dev split is carved out of train
for model selection.
"""
import hashlib
import re

import pandas as pd
from torch.utils.data import DataLoader, Dataset

HF_REPO = "hf://datasets/BioLaySumm/BioLaySumm2025-LaymanRRG-opensource-track/data"
SOURCE_COL = "radiology_report"
TARGET_COL = "layman_report"

# FLAN-T5 is instruction-tuned, so the task is stated as a natural-language prefix.
PROMPT = "Rewrite this radiology report in plain language for a patient: "
# The prompt is 14 tokens; 256 + prompt and 256 each truncate about 1.3% of pairs.
MAX_SOURCE_LENGTH = 272
MAX_TARGET_LENGTH = 256


def normalise(text):
    """Lower-case, drop punctuation and collapse whitespace, for duplicate detection only."""
    text = re.sub(r"\s+", " ", text.lower())
    return re.sub(r"[^a-z0-9 ]", "", text).strip()


def _bucket(key, n_buckets=1000):
    """Deterministic bucket in [0, n_buckets) from a string (stable across runs and machines)."""
    return int(hashlib.md5(key.encode("utf-8")).hexdigest(), 16) % n_buckets


def load_raw(data_dir=None):
    """
    Read the official train / validation parquet files.

    data_dir: folder holding train.parquet and validation.parquet. If None the
    files are read straight from the Hugging Face hub.
    """
    if data_dir is None:
        paths = {s: f"{HF_REPO}/{s}-00000-of-00001.parquet" for s in ("train", "validation")}
    else:
        paths = {s: f"{data_dir}/{s}.parquet" for s in ("train", "validation")}
    return {s: pd.read_parquet(p) for s, p in paths.items()}


def make_splits(data_dir=None, dev_fraction=0.02):
    """
    Build train / dev / test DataFrames without leakage between them.

    1. test  = official validation split, untouched. A boolean column
       `seen_in_train` marks rows whose report text also occurs in the final
       train split, so metrics can be reported on seen and unseen inputs.
    2. Train rows that share an `images_path` (same imaging study) with the
       test split are removed.
    3. Train is de-duplicated on (report, summary) pairs.
    4. dev is split off train by hashing the normalised report text, so every
       copy of a report lands on the same side and dev inputs are never seen
       during training.
    """
    raw = load_raw(data_dir)
    train, test = raw["train"].copy(), raw["validation"].copy()
    for df in (train, test):
        df["source_key"] = df[SOURCE_COL].map(normalise)

    # Same imaging study must not sit on both sides of the train / test boundary.
    train = train[~train["images_path"].isin(set(test["images_path"]))]

    # Exact repeats of a (report, summary) pair add no information.
    train = train.assign(target_key=train[TARGET_COL].map(normalise))
    train = train.drop_duplicates(["source_key", "target_key"]).drop(columns="target_key")

    # Group-wise dev split: the bucket depends only on the report text.
    in_dev = train["source_key"].map(_bucket) < int(dev_fraction * 1000)
    dev, train = train[in_dev], train[~in_dev]

    test["seen_in_train"] = test["source_key"].isin(set(train["source_key"]))

    return {
        "train": train.reset_index(drop=True),
        "dev": dev.reset_index(drop=True),
        "test": test.reset_index(drop=True),
    }


class LaymanDataset(Dataset):
    """(report, summary) text pairs. Tokenisation happens per batch in Collator."""

    def __init__(self, frame):
        self.reports = frame[SOURCE_COL].tolist()
        self.summaries = frame[TARGET_COL].tolist()

    def __len__(self):
        return len(self.reports)

    def __getitem__(self, index):
        return self.reports[index], self.summaries[index]


def encode_reports(tokenizer, reports, max_source_length=MAX_SOURCE_LENGTH):
    """Prefix each report with the task prompt and tokenise, padded to the longest in the batch."""
    return tokenizer(
        [PROMPT + report for report in reports],
        max_length=max_source_length,
        truncation=True,
        padding=True,
        return_tensors="pt",
    )


class Collator:
    """
    Turn a list of (report, summary) pairs into model inputs.

    Padding is dynamic (to the longest sequence in the batch) because most
    reports are far shorter than the length limit. Padded label positions are
    set to -100 so the cross-entropy loss ignores them.
    """

    def __init__(self, tokenizer, max_source_length=MAX_SOURCE_LENGTH, max_target_length=MAX_TARGET_LENGTH):
        self.tokenizer = tokenizer
        self.max_source_length = max_source_length
        self.max_target_length = max_target_length

    def __call__(self, batch):
        reports, summaries = zip(*batch)
        inputs = encode_reports(self.tokenizer, reports, self.max_source_length)
        labels = self.tokenizer(
            list(summaries),
            max_length=self.max_target_length,
            truncation=True,
            padding=True,
            return_tensors="pt",
        ).input_ids
        labels[labels == self.tokenizer.pad_token_id] = -100
        return {"input_ids": inputs.input_ids, "attention_mask": inputs.attention_mask, "labels": labels}


def make_loader(frame, tokenizer, batch_size, shuffle=False, generator=None):
    """DataLoader over a split. Only the training loader shuffles and drops the last partial batch."""
    return DataLoader(
        LaymanDataset(frame),
        batch_size=batch_size,
        shuffle=shuffle,
        drop_last=shuffle,
        collate_fn=Collator(tokenizer),
        generator=generator,
    )


if __name__ == "__main__":
    import sys

    splits = make_splits(sys.argv[1] if len(sys.argv) > 1 else None)
    train, dev, test = splits["train"], splits["dev"], splits["test"]
    for name, df in splits.items():
        print(f"{name:5s} rows {len(df):7d}  unique reports {df['source_key'].nunique():6d}")
    print("dev reports also in train:", int(dev["source_key"].isin(set(train["source_key"])).sum()))
    print("train/test shared images_path:", len(set(train["images_path"]) & set(test["images_path"])))
    print("test rows seen in train:", int(test["seen_in_train"].sum()), "unseen:", int((~test["seen_in_train"]).sum()))
    print("dev by source:", dev["source"].value_counts().to_dict())
