"""Measure reader_a's per-label operating point against the corpus.

The problem this exists for, stated as the number that gives it away:
`reader_a` calls **LungOpacity positive on 40 of 40 studies**. That is not
the model believing every chest is abnormal -- it is the report threshold
(`CNN_PROB_THRESHOLD = 0.5`) landing exactly on the model's own
`op_threshs` entry, because torchxrayvision's `op_norm` maps each label's
operating point to 0.5. So "positive" currently means "above the operating
point torchxrayvision chose", for every label, with no reference to this
corpus at all. Every downstream number inherits that: the G1 false-positive
rate (0.808), the disagreement count, the kappa.

**Weak labels, and they are weak on purpose-stated grounds.** Open-i
indexes each study with MeSH major terms, and 1379 studies are indexed
`normal`. That gives, per label:

    positives = studies whose MeSH maps to that label
    negatives = studies indexed `normal`

Both come from the *report*, not from the pixels. A study called normal is
normal according to the radiologist who dictated it; a study indexed
`Cardiomegaly` had a radiologist write that word. This is a real signal and
a genuinely independent one (the CNN never sees the text), but it is not a
pixel-verified ground truth, and it is not the hand-checked G1 gold set.

**What this script does not do:** it does not pick thresholds that make a
gate green. It reports, per label, the ROC over the weak labels and the
threshold that achieves a stated target specificity, along with the
sensitivity that buys. Choosing a target is a clinical decision that
belongs in the open, not a knob tuned until `make eval` passes -- and for
the critical labels the asymmetry is the opposite of the default one
(missing a pneumothorax can kill), which is why `--critical-sensitivity`
is a separate, explicitly higher target.

Usage:
    PYTHONPATH=src .venv/bin/python scripts/calibrate_operating_points.py \\
        [--limit-normals 600] [--specificity 0.90] [--critical-sensitivity 0.95]

Writes /tmp/medscope_operating_points.json. Costs no API calls; it is CPU
time only (~0.5s per film on this machine).
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from medscope.config import Settings  # noqa: E402
from medscope.data.openi import load_studies  # noqa: E402
from medscope.ontology import CRITICAL_LABELS, canonical  # noqa: E402
from medscope.readers.cnn import CNNReader  # noqa: E402

OUT_PATH = Path("/tmp/medscope_operating_points.json")

#: Below this many weak positives, no operating point is emitted. A
#: threshold fitted to a handful of cases is a number that looks like
#: calibration and is really an accident of which studies got indexed.
MIN_POSITIVES = 25


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--root", type=Path, default=None)
    parser.add_argument(
        "--recompute-from",
        type=Path,
        default=None,
        help="Re-derive thresholds from a previous run's saved raw scores instead of "
        "re-scoring the corpus. Scoring 1500 studies costs ~16 minutes of CPU; changing "
        "a target should not.",
    )
    parser.add_argument(
        "--limit-normals", type=int, default=600, help="Cap on `normal` studies scored"
    )
    parser.add_argument(
        "--limit-positives", type=int, default=900, help="Cap on labelled-positive studies scored"
    )
    parser.add_argument(
        "--specificity",
        type=float,
        default=0.90,
        help="Target specificity on the `normal` population for ordinary labels",
    )
    parser.add_argument(
        "--critical-sensitivity",
        type=float,
        default=0.95,
        help="Target sensitivity for CRITICAL_LABELS -- the asymmetry runs the other way there",
    )
    return parser.parse_args()


def _weak_labels(study) -> set[str]:
    """Canonical labels this study's MeSH indexing asserts."""
    labels = set()
    for term in study.mesh:
        head = term.split("/")[0].strip()
        canon = canonical(head)
        if canon is not None:
            labels.add(canon)
    return labels


def _is_normal(study) -> bool:
    return bool(study.mesh) and all(t.strip().lower() == "normal" for t in study.mesh)


def _auc(pos: list[float], neg: list[float]) -> float | None:
    if not pos or not neg:
        return None
    wins = sum((p > n) + 0.5 * (p == n) for p in pos for n in neg)
    return wins / (len(pos) * len(neg))


def _threshold_at_specificity(neg: list[float], target: float) -> float:
    """Lowest score a study must beat to keep `target` of normals below it."""
    if not neg:
        return float("nan")
    ordered = sorted(neg)
    idx = min(len(ordered) - 1, int(round(target * len(ordered))))
    return ordered[idx]


def _threshold_at_sensitivity(pos: list[float], target: float) -> float:
    """Highest threshold that still catches `target` of the positives."""
    if not pos:
        return float("nan")
    ordered = sorted(pos)
    idx = max(0, int(round((1 - target) * len(ordered))) - 1)
    return ordered[idx]


def main() -> None:
    args = _parse_args()
    settings = Settings()

    if args.recompute_from:
        saved = json.loads(Path(args.recompute_from).read_text())
        scores_pos = {k: v["pos"] for k, v in saved["scores"].items()}
        scores_neg = {k: v["neg"] for k, v in saved["scores"].items()}
        _emit(args, scores_pos, scores_neg, saved["n_normal_studies"], saved["n_labelled_studies"])
        return

    studies = [s for s in load_studies(args.root or settings.openi_root) if s.image_paths]

    normals = [s for s in studies if _is_normal(s)][: args.limit_normals]
    positives = [s for s in studies if not _is_normal(s) and _weak_labels(s)][: args.limit_positives]
    if not normals or not positives:
        sys.exit("calibrate_operating_points: corpus has no usable weak labels")

    reader = CNNReader(settings)
    scores_neg: dict[str, list[float]] = defaultdict(list)
    scores_pos: dict[str, list[float]] = defaultdict(list)
    n_pos_studies: dict[str, int] = defaultdict(int)

    start = time.perf_counter()
    total = len(normals) + len(positives)
    for i, study in enumerate(normals + positives, 1):
        # Scores only: Grad-CAM costs a backward pass per positive label and
        # nothing here looks at a locus.
        result = reader.read_study(study.image_paths, localize=False)
        by_label = {f.label: f.prob for f in result.findings}
        if _is_normal(study):
            for label, prob in by_label.items():
                scores_neg[label].append(prob)
        else:
            for label in _weak_labels(study):
                if label in by_label:
                    scores_pos[label].append(by_label[label])
                    n_pos_studies[label] += 1
        if i % 25 == 0:
            print(f"  [{i}/{total}] {time.perf_counter() - start:.0f}s", flush=True)

    _emit(args, scores_pos, scores_neg, len(normals), len(positives))


def _emit(args, scores_pos, scores_neg, n_normals: int, n_labelled: int) -> None:
    rows = []
    for label in sorted(set(scores_pos) | set(scores_neg)):
        pos, neg = scores_pos.get(label, []), scores_neg.get(label, [])
        auc = _auc(pos, neg)
        critical = label in CRITICAL_LABELS
        row = {
            "label": label,
            "n_pos": len(pos),
            "n_neg": len(neg),
            "auc": auc,
            "critical": critical,
            "positive_rate_at_0.5": (
                sum(p >= 0.5 for p in neg) / len(neg) if neg else None
            ),
        }
        # The shipped number is the REPORT threshold, and a report
        # threshold is a specificity decision: it governs what a draft
        # asserts, not what raises an alarm. The critical channel has its
        # own, deliberately lower threshold (`Settings.critical_threshold`,
        # 0.3) and `critical.triage` never reads this table -- so a
        # critical finding that misses the report still alerts. Importing
        # the alert channel's sensitivity-first asymmetry into this layer
        # would put the safety logic in the wrong place and make every
        # report mention effusion on half the normal chests.
        threshold = _threshold_at_specificity(neg, args.specificity) if neg else None
        if threshold is not None:
            row["threshold"] = round(float(threshold), 4)
            row["specificity"] = round(sum(n < threshold for n in neg) / len(neg), 4)
            row["sensitivity"] = (
                round(sum(p >= threshold for p in pos) / len(pos), 4) if pos else None
            )
            if len(pos) >= MIN_POSITIVES:
                row["rule"] = f"specificity>={args.specificity}"
            elif pos:
                row["rule"] = (
                    f"specificity>={args.specificity} "
                    f"(sensitivity from only {len(pos)} weak positives -- below the "
                    f"{MIN_POSITIVES} floor, read it as an indication, not a measurement)"
                )
            else:
                row["rule"] = (
                    f"specificity>={args.specificity} (sensitivity UNMEASURED -- "
                    "no weak positives in this corpus)"
                )
        else:
            row["rule"] = "withheld: no negative population"

        # Recorded, not shipped: what a sensitivity-first rule would pick
        # for a critical label. That is the number a future calibration of
        # the *alert* threshold needs, and keeping it here means that work
        # does not have to re-score the corpus.
        if critical:
            if len(pos) >= MIN_POSITIVES:
                sens_thr = _threshold_at_sensitivity(pos, args.critical_sensitivity)
                row["alert_threshold_candidate"] = round(float(sens_thr), 4)
                row["alert_rule"] = f"sensitivity>={args.critical_sensitivity}"
                row["alert_specificity"] = round(sum(n < sens_thr for n in neg) / len(neg), 4)
            else:
                row["alert_rule"] = (
                    f"withheld (critical): {len(pos)} weak positives < {MIN_POSITIVES} -- "
                    "this corpus cannot calibrate this label's alert threshold"
                )

        rows.append(row)

    OUT_PATH.write_text(
        json.dumps(
            {
                "generated_at": datetime.now(timezone.utc).isoformat(),
                "n_normal_studies": n_normals,
                "n_labelled_studies": n_labelled,
                "target_specificity": args.specificity,
                "target_critical_sensitivity": args.critical_sensitivity,
                "min_positives": MIN_POSITIVES,
                "labels": rows,
                # Raw per-label score distributions, so a different rule can
                # be evaluated without re-scoring 1500 studies (16 minutes of
                # CPU). Rounded to 4 places: the scores live in a ~0.02-wide
                # band around 0.5 for some labels, so trimming further would
                # start moving thresholds.
                "scores": {
                    label: {
                        "pos": [round(p, 4) for p in scores_pos.get(label, [])],
                        "neg": [round(n, 4) for n in scores_neg.get(label, [])],
                    }
                    for label in sorted(set(scores_pos) | set(scores_neg))
                },
            },
            indent=2,
        )
    )

    print(f"\n=== reader_a operating points (weak labels: MeSH; negatives: {n_normals} `normal` studies) ===")
    print(f"{'label':<26}{'n+':>5}{'AUC':>7}{'@0.5 阳性率(阴性群)':>20}{'thr':>8}{'sens':>7}{'spec':>7}  rule")
    for row in rows:
        auc = f"{row['auc']:.3f}" if row["auc"] is not None else "  -  "
        fpr = f"{row['positive_rate_at_0.5']:.3f}" if row["positive_rate_at_0.5"] is not None else "  -  "
        thr = f"{row['threshold']:.3f}" if row.get("threshold") is not None else "   -  "
        sens = f"{row['sensitivity']:.3f}" if row.get("sensitivity") is not None else "  -  "
        spec = f"{row['specificity']:.3f}" if row.get("specificity") is not None else "  -  "
        print(f"{row['label']:<26}{row['n_pos']:>5}{auc:>7}{fpr:>20}{thr:>8}{sens:>7}{spec:>7}  {row['rule']}")
    print(f"\nfull output: {OUT_PATH}")
    print(
        "\nRead the '@0.5' column first: it is the share of report-normal studies the "
        "current threshold already calls positive. Any label near 1.0 there is not "
        "detecting anything -- it is asserting."
    )


if __name__ == "__main__":
    main()
