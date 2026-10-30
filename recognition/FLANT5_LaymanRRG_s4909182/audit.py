"""
Clinical safety audit of generated summaries against the source radiology report.

For nine tracked findings the audit decides whether each one is stated as
present, uncertain or negated, or not mentioned, first in the source report and
then in a summary, and counts three kinds of disagreement:
  polarity flip - stated as present in the report and negated in the summary, or the reverse
  omission      - present in the report, not mentioned at all in the summary
  unsupported   - present in the summary, not mentioned at all in the report
It also reports the median Flesch-Kincaid grade (Kincaid et al., 1975) of each
text. Summaries are compared with the source report, never with the reference,
because the references are themselves machine-generated.

The plain-language phrases for each finding were mined from sentence-aligned
train pairs: the phrases most over-represented in summary sentences whose
report sentence names the finding. A phrase used for two findings, such as
"collapsed lung" for both pneumothorax and atelectasis, counts as a mention
but never as evidence of polarity. Negation is decided per clause with
hand-written cue lists, following the idea of NegEx (Chapman et al., 2001).

Running the audit on the references measures the checker itself: they were
checked for faithfulness when the dataset was built, so most flips found there
are checker errors.

    python audit.py runs/lora_r8_official/test_predictions.csv runs/full_official/test_predictions.csv

Author: Weixiang Han (s4909182), COMP3710 Pattern Recognition Report, 2026.
AI assistance is disclosed in README.md (Artificial Intelligence Usage Disclosure).
"""
import argparse
import os
import re

import pandas as pd
import textstat

FINDINGS = ("pneumothorax", "effusion", "consolidation", "atelectasis", "cardiomegaly",
            "pneumonia", "nodule", "fracture", "mass")

# Phrases that name exactly one finding: the technical term plus its plain-language versions.
SPECIFIC = {
    "pneumothorax": [r"pneumothora\w*", r"air (?:\w+ ){0,4}(?:around|outside|surrounding) (?:\w+ ){0,3}lungs?",
                     r"air leak\w*"],
    "effusion": [r"(?<!pericardial )(?<!joint )effusion\w*", r"pleural fluid",
                 r"fluid (?:\w+ ){0,4}(?:around|surrounding|outside) (?:\w+ ){0,3}lungs?"],
    "consolidation": [r"consolidat\w*", r"solid (?:\w+ )?(?:area|part|region|patch)\w*"],
    "atelectasis": [r"atelecta\w*", r"underinflated"],
    "cardiomegaly": [r"cardiomegal\w*", r"heart(?:'s)? (?:\w+ ){0,5}(?:enlarged|bigger|larger|increased)",
                     r"(?:enlarged|large|big) heart", r"enlargement of the heart", r"cardiac enlargement",
                     r"(?:cardiac|heart) (?:silhouette|shadow|size) (?:\w+ ){0,3}(?:enlarged|increased)",
                     r"(?:enlarged|increased) (?:cardiac|heart) (?:silhouette|shadow|size)",
                     r"increased cardiothoracic (?:index|ratio)"],
    "pneumonia": [r"pneumoni\w*"],
    "nodule": [r"nodul\w*", r"round spots?", r"bumps?"],
    "fracture": [r"fractur\w*", r"broken", r"cracked"],
    "mass": [r"mass(?:es)?", r"tumou?rs?"],
}
# Plain-language phrases too vague to carry polarity, either shared by two findings or used loosely.
# They count as a mention, so a finding described this way is not an omission.
SHARED = [
    (r"collaps\w*", ("pneumothorax", "atelectasis")),
    (r"infections?|inflam\w*", ("pneumonia", "consolidation")),
    (r"lumps?", ("nodule", "mass")),
    (r"fluid", ("effusion",)),
    (r"dens(?:e|er|ity)|hazy|haziness|cloud\w*|opaci\w*|thicken\w*", ("consolidation",)),
    (r"spots?", ("nodule",)),
    (r"growths?", ("mass",)),
]

SPECIFIC_RE = {f: [re.compile(rf"\b{p}\b", re.I) for p in patterns] for f, patterns in SPECIFIC.items()}
SHARED_RE = [(re.compile(rf"\b{p}\b", re.I), findings) for p, findings in SHARED]

SENTENCE = re.compile(r"(?<=[.!?])\s+")
# A new clause starts after these words, so a negation earlier in the sentence does not reach across them.
CLAUSE = re.compile(r"[;:]|\b(?:but|however|although|though|except|whereas|while)\b"
                    r"|,? and (?=(?:the|there|a|an|it|this|these)\b)"
                    # but not before a definition such as ", which is called a pneumothorax"
                    r"|,\s*(?=(?:which(?! (?:is|are) (?:called|known|termed|a term))|related|associated|due|secondary|consistent|compatible|in relation)\b)", re.I)
# Phrases that contain "not" but describe something present (pseudo-negations in NegEx terms).
PSEUDO = [(re.compile(r"(?:\bnot|n't) (?:fully |completely |properly )?(?:inflated|inflating|expanded|expanding|aerated)\b",
                      re.I), "underinflated"),
          (re.compile(r"(?:\bnot|n't) (?:functioning|working) (?:properly|normally)\b|\bnot normal\b", re.I), "abnormal"),
          (re.compile(r"\b(?:has|have|had)(?:n't| not) changed\b", re.I), "unchanged"),
          (re.compile(r"\bwithout (?:\w+ ){0,3}(?:contrast|dye)\b", re.I), "unenhanced"),
          # "not seen in the previous study" describes the earlier study, not this one.
          (re.compile(r"(?:\bnot\b|n't)(?= (?:\w+ ){0,3}(?:in|on|at|than) (?:the |a )?(?:previous|prior|last|earlier)\b)",
                      re.I), " newly")]
# Hedged statements, statements about change over time ("no change in the effusion", "resolution of"),
# and "no new nodules" are neither present nor negated, so they never count as a flip.
UNCERTAIN = re.compile(r"\b(?:possibl[ey]|probabl[ey]|likely|suggest\w*|suspect\w*|suspicious|questionable"
                       r"|cannot be (?:ruled out|excluded)|(?:to )?rule out|ruling out|to exclude|may|might|could"
                       r"|resol\w+|improv\w+|cleared|clearing|decreas\w+|better|disappear\w*|no new"
                       r"|non-?specific|not specific"
                       r"|(?:no|without) (?:\w+ )?(?:changes?|variations?|worsening|progression|differences?))\b", re.I)
POST_UNCERTAIN = re.compile(r"^\W*(?:\w+ ){0,3}?(?:(?:is|are|to be) )?(?:ruled out|to rule out|suspected"
                            r"|cannot be (?:ruled out|excluded)|not excluded|likely|possible|probable"
                            r"|resol\w+|improv\w+|cleared|clearing|decreas\w+|better)", re.I)
NEGATION = re.compile(r"\b(?:no|not|without|absence of|absent|free of|clear of|negative for|nor|neither"
                      r"|ruled out|excluded)\b|n't\b", re.I)
POST_NEGATION = re.compile(r"^\W*(?:\w+ ){0,5}?(?:(?:is|are|was|were|has|have|been) )?(?:not|no longer) (?:been )?"
                           r"(?:seen|identified|present|visible|visuali[sz]ed|evident|observed|noted|appreciated"
                           r"|detected|demonstrated)"
                           r"|^\W*(?:\w+ ){0,5}?(?:is|are|was|were|has|have)n't (?:been )?(?:seen|identified|present|visible)"
                           r"|^\W*(?:\w+ ){0,3}?(?:is|are) (?:all )?(?:normal|unremarkable)", re.I)
# "not seen in the previous study" is about the earlier study, so it does not negate the finding now.
EARLIER = re.compile(r"\b(?:previous|prior|before|earlier|last)\b", re.I)
# When a finding is mentioned more than once, the strongest statement wins. A vague phrase that is not
# negated ("shared") outranks a negated specific one, so "a lung infection ... no clear sign of pneumonia"
# is not counted as a flip.
ORDER = ("present", "uncertain", "shared", "negated", "shared negated")


def polarity(clause, match):
    """'present', 'uncertain' or 'negated' for one mention, from cues earlier in the clause or right after it."""
    before, after = clause[:match.end()], clause[match.end():]
    if UNCERTAIN.search(before) or POST_UNCERTAIN.search(after):
        return "uncertain"
    if NEGATION.search(before) or (POST_NEGATION.search(after) and not EARLIER.search(after)):
        return "negated"
    return "present"


def statuses(text):
    """Map each finding mentioned in `text` to its status; findings not mentioned are left out."""
    found = {}
    for pattern, replacement in PSEUDO:
        text = pattern.sub(replacement, str(text))
    for sentence in SENTENCE.split(text):
        for clause in CLAUSE.split(sentence):
            for finding, patterns in SPECIFIC_RE.items():
                for pattern in patterns:
                    for match in pattern.finditer(clause):
                        found.setdefault(finding, set()).add(polarity(clause, match))
            for pattern, findings in SHARED_RE:
                for match in pattern.finditer(clause):
                    status = "shared negated" if polarity(clause, match) == "negated" else "shared"
                    for finding in findings:
                        found.setdefault(finding, set()).add(status)
    return {finding: next(s for s in ORDER if s in seen) for finding, seen in found.items()}


def compare(report, summary):
    """Return (flips, omissions, unsupported findings, report statuses) for one pair."""
    source, output = statuses(report), statuses(summary)
    flips = [f for f in FINDINGS if {source.get(f), output.get(f)} == {"present", "negated"}]
    omissions = [f for f in FINDINGS if source.get(f) == "present" and f not in output]
    unsupported = [f for f in FINDINGS if output.get(f) == "present" and f not in source]
    return flips, omissions, unsupported, source


def audit(frame, column):
    """Per-row audit flags for the summaries in `column`."""
    rows = [compare(report, summary) for report, summary in zip(frame["radiology_report"], frame[column])]
    flags = frame[["source", "seen_in_train", "radiology_report", "layman_report"]].copy()
    if column != "layman_report":
        flags[column] = frame[column]
    if "rougeLsum" in frame and column != "layman_report":
        flags["rougeLsum"] = frame["rougeLsum"]
    flags["flips"] = [";".join(r[0]) for r in rows]
    flags["omissions"] = [";".join(r[1]) for r in rows]
    flags["unsupported"] = [";".join(r[2]) for r in rows]
    flags["n_present"] = [sum(s == "present" for s in r[3].values()) for r in rows]
    flags["tracked"] = [any(s in ("present", "negated") for s in r[3].values()) for r in rows]
    flags["fkgl"] = frame[column].fillna("").map(textstat.flesch_kincaid_grade)
    return flags


def summarise(flags, long_report):
    """Rates per 100: summaries with a flip (of those whose report states a finding), omitted findings
    (of those present in the report), unsupported findings (per summary); and the median grade."""
    count = lambda column: flags[column].str.split(";").map(lambda items: len([i for i in items if i])).sum()
    return {
        "n": len(flags),
        "flip": 100 * (flags["flips"] != "").sum() / max(1, flags["tracked"].sum()),
        "omission": 100 * count("omissions") / max(1, flags["n_present"].sum()),
        "unsupported": 100 * count("unsupported") / max(1, len(flags)),
        "fkgl": flags.loc[long_report, "fkgl"].median(),
    }


def per_finding(flags):
    """Counts of flips, omissions and unsupported mentions for each finding."""
    lines = []
    for column in ("flips", "omissions", "unsupported"):
        items = flags[column].str.split(";").explode()
        counts = items[items != ""].value_counts()
        lines.append(f"  {column:12s}" + "  ".join(f"{f} {counts.get(f, 0)}" for f in FINDINGS))
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(description="Audit generated summaries against the source radiology reports")
    parser.add_argument("predictions", nargs="+", help="test_predictions.csv files written by train.py")
    parser.add_argument("--out_dir", default="audit_outputs", help="per-row flags are saved here for case selection")
    parser.add_argument("--min_report_words", type=int, default=10,
                        help="readability is compared only on reports at least this many words long")
    parser.add_argument("--seen_from", default=None,
                        help="predictions file whose seen_in_train flags are used for every run (default: the first); "
                             "use a run trained on the split you report, since zero-shot runs carry the flags of "
                             "whichever split they were evaluated with")
    args = parser.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)

    frames = {os.path.basename(os.path.dirname(os.path.abspath(p))): pd.read_csv(p) for p in args.predictions}
    first = next(iter(frames.values()))
    seen = pd.read_csv(args.seen_from) if args.seen_from else first
    for name, frame in frames.items():
        if not frame["radiology_report"].equals(seen["radiology_report"]):
            raise ValueError(f"{name} does not list the same test reports in the same order")
        frame["seen_in_train"] = seen["seen_in_train"]
    print(f"seen / unseen flags from {args.seen_from or args.predictions[0]}: {seen['seen_in_train'].sum():,} seen")
    runs = [("report", first, "radiology_report"), ("reference", first, "layman_report")]
    runs += [(name, frame, "prediction") for name, frame in frames.items()]

    long_report = first["radiology_report"].str.split().str.len() >= args.min_report_words
    print(f"readability on {long_report.sum():,} reports with >= {args.min_report_words} words")
    print(f"{'':12s}{'summaries':>24s} {'subset':8s}{'n':>7s}{'flip/100':>10s}{'omit/100':>10s}"
          f"{'unsup/100':>11s}{'FKGL':>7s}")
    for name, frame, column in runs:
        flags = audit(frame, column)
        if name != "report":
            flags.to_csv(os.path.join(args.out_dir, f"{name}_audit.csv"), index=False)
        subsets = {"all": flags.index == flags.index, "seen": flags["seen_in_train"], "unseen": ~flags["seen_in_train"]}
        for subset, mask in subsets.items():
            s = summarise(flags[mask], long_report[mask])
            print(f"{'':12s}{name:>24s} {subset:8s}{s['n']:7,d}{s['flip']:10.2f}{s['omission']:10.2f}"
                  f"{s['unsupported']:11.2f}{s['fkgl']:7.1f}")
        if name != "report":
            print(per_finding(flags))


if __name__ == "__main__":
    main()
