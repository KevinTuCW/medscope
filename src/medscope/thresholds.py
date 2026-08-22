"""Per-label report thresholds, measured against this corpus.

`Settings.cnn_prob_threshold` is one number for eighteen labels, and it is
0.5 -- which is exactly where torchxrayvision's `op_norm` puts each
label's own operating point. "Positive" has therefore meant "above the
operating point torchxrayvision picked", with no reference to Open-i at
all. Measured over 600 studies whose own reports index them `normal`,
that threshold calls **LungOpacity positive on 90.0% of them**, Emphysema
on 91.2%, Infiltration on 81.3%. A label firing on nine of ten normal
chests is not detecting anything; it is asserting.

`data/operating_points.json` records what each label's threshold buys,
derived by `scripts/calibrate_operating_points.py` from MeSH weak labels
(positives) and the `normal`-indexed population (negatives). This module
is the single place that file is read, so `readers/cnn.py` and `merge.py`
cannot drift into disagreeing about what counts as positive.

Three rules govern what lands there, and the third is the one to read
twice:

1. Enough weak positives (>= 25), non-critical: threshold at target
   specificity on the normal population; measured sensitivity reported.
2. Enough weak positives, critical: threshold at target *sensitivity* --
   the asymmetry runs the other way when missing a case can kill.
3. **No weak positives, non-critical**: threshold from the normal
   population alone, capping how often the label fires on normal studies.
   Its sensitivity is unmeasured and is recorded as such. This is a
   deliberate, bounded trade: it can only make a label quieter, never
   louder, and it never applies to a critical label.

**Critical labels without enough positives keep the uncalibrated
threshold.** Pneumothorax has 19 weak positives in this corpus, below the
floor, so nothing here touches it -- and that is why this calibration
does not move gate G1's false-positive rate. Tightening the most
safety-critical label on a specificity argument alone would be the exact
move this project keeps refusing.

The critical-alert channel is untouched regardless: `critical.triage`
thresholds on `Settings.critical_threshold`, not on anything here.
"""

from __future__ import annotations

import json
from functools import lru_cache
from pathlib import Path

from medscope.ontology import CRITICAL_LABELS

OPERATING_POINTS_PATH = Path("data/operating_points.json")


@lru_cache(maxsize=1)
def _table() -> dict[str, dict]:
    """{label -> row} from the committed measurement, or empty if absent.

    Missing file is not an error: the fallback is the old global
    threshold, so a checkout without the measurement behaves exactly as
    the project did before it existed.
    """
    if not OPERATING_POINTS_PATH.is_file():
        return {}
    payload = json.loads(OPERATING_POINTS_PATH.read_text())
    return {
        row["label"]: row
        for row in payload.get("labels", [])
        if row.get("threshold") is not None
    }


def report_threshold(label: str, default: float) -> float:
    """The threshold above which `label` counts as positive for the report.

    `default` (the caller's global `cnn_prob_threshold`) is returned for
    any label the measurement has no row for.

    **A critical label's threshold is never raised above the default.**
    The measurement would have moved Pneumothorax from 0.500 to 0.511 --
    a small number with an ugly derivation: the sensitivity behind it
    rests on 19 weak positives, below this corpus's own floor. Making a
    life-threatening finding *harder* to get into a report on that
    evidence is not a calibration, it is a guess in the dangerous
    direction. Lowering one is still allowed: that direction fails safe.
    """
    row = _table().get(label)
    if not row:
        return float(default)
    threshold = float(row["threshold"])
    if label in CRITICAL_LABELS:
        return min(threshold, float(default))
    return threshold


def is_calibrated(label: str) -> bool:
    return label in _table()


def calibration_note(label: str) -> str | None:
    """The rule behind a label's threshold, for the audit trail. None when
    the label runs on the uncalibrated default."""
    row = _table().get(label)
    if not row:
        return None
    note = row.get("rule")
    default_clamped = label in CRITICAL_LABELS and float(row["threshold"]) > 0.5
    if default_clamped:
        return f"{note} [clamped: critical labels are never raised above the default]"
    return note
