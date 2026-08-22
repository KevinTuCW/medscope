"""P4 scouting run: are cloud image-edit models usable for counterfactual films?

The idea being tested, not asserted: a counterfactual -- *the same film
with the finding removed* -- is a stronger explanation than a Grad-CAM
blob, because it can be measured. Feed reader_a the original and the
edit; if the edit really removed finding L, `prob(L)` should collapse
while the other labels barely move.

**The control is the entire experiment.** Any edit at all pushes a film
away from the distribution the CNN was trained on, and that alone drags
probabilities around. So every study gets two edits from the same model
in the same run:

    targeted  -- "remove <finding>, keep everything else identical"
    control   -- a cosmetic instruction naming no pathology

and the number that matters is not "did prob(L) drop" but **did prob(L)
drop more under the targeted edit than under the control, and more than
the other labels did**. Without that comparison a drop proves only that
the picture changed.

This is a scouting script, not a pipeline stage: no gate consumes it, and
`medscope` imports nothing from it. It writes
/tmp/medscope_counterfactual_probe.json and prints a table.

Usage:
    USE_REAL_VLM=true PYTHONPATH=src .venv/bin/python \\
        scripts/counterfactual_probe.py --limit 3 [--root PATH]

Costs real image-generation calls (2 per study). Refuses to run without a
key rather than inventing a result, on the same principle as
scripts/calibrate_vlm.py.
"""

from __future__ import annotations

import argparse
import base64
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import httpx  # noqa: E402

from medscope.config import Settings  # noqa: E402
from medscope.data.openi import load_studies  # noqa: E402
from medscope.readers.cnn import CNNReader  # noqa: E402
from medscope.views import primary_view  # noqa: E402

OUT_PATH = Path("/tmp/medscope_counterfactual_probe.json")
EDIT_MODEL = "Qwen/Qwen-Image-Edit"

CONTROL_PROMPT = (
    "Keep the chest radiograph exactly as it is. Do not add, remove or alter any "
    "anatomical structure. Only adjust the overall brightness very slightly."
)


def _targeted_prompt(label: str) -> str:
    return (
        f"This is a chest radiograph showing {label}. Produce the same radiograph of the "
        f"same patient with the {label} resolved and no other change: identical projection, "
        "identical ribs, spine, diaphragm and cardiac borders, identical exposure. "
        "Change nothing except the abnormality itself."
    )


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--root", type=Path, default=None, help="Dataset root")
    parser.add_argument("--limit", type=int, default=3, help="Studies to probe (2 edits each)")
    parser.add_argument(
        "--model", default=EDIT_MODEL, help=f"Image-edit model id (default {EDIT_MODEL})"
    )
    return parser.parse_args()


def _edit(client: httpx.Client, settings: Settings, model: str, image: Path, prompt: str) -> bytes:
    payload = {
        "model": model,
        "prompt": prompt,
        "image": "data:image/png;base64,"
        + base64.b64encode(image.read_bytes()).decode("ascii"),
    }
    response = client.post(
        f"{settings.vlm_base_url}/images/generations",
        json=payload,
        headers={"Authorization": f"Bearer {settings.vlm_api_key}"},
    )
    response.raise_for_status()
    url = response.json()["images"][0]["url"]
    return client.get(url).content


def main() -> None:
    args = _parse_args()
    settings = Settings()

    if not settings.use_real_vlm or not settings.vlm_api_key:
        print(
            "counterfactual_probe: no keyed image-edit backend configured "
            "(USE_REAL_VLM / VLM_API_KEY).\n"
            "There is no offline stand-in for this and there should not be one: a "
            "synthetic 'counterfactual' would answer the question by construction.",
            file=sys.stderr,
        )
        sys.exit(2)

    studies = load_studies(args.root or settings.openi_root)[: args.limit]
    if not studies:
        sys.exit("counterfactual_probe: no studies found")

    reader = CNNReader(settings)
    work = Path("/tmp/medscope_counterfactual")
    work.mkdir(exist_ok=True)
    rows: list[dict] = []

    with httpx.Client(timeout=300.0, trust_env=False) as client:
        for study in studies:
            film = primary_view(study.image_paths)
            base = {f.label: f.prob for f in reader.read(film).findings}
            # Target the label this film is most confident about -- a
            # counterfactual for something the reader never saw would
            # measure nothing.
            target = max(base, key=base.get)

            record: dict = {"study_id": study.study_id, "film": str(film), "target": target,
                            "base_prob": base[target], "edits": {}}

            for kind, prompt in (
                ("targeted", _targeted_prompt(target)),
                ("control", CONTROL_PROMPT),
            ):
                out = work / f"{study.study_id}_{kind}.png"
                try:
                    out.write_bytes(_edit(client, settings, args.model, film, prompt))
                except Exception as exc:
                    record["edits"][kind] = {"error": f"{type(exc).__name__}: {exc}"}
                    continue
                after = {f.label: f.prob for f in reader.read(out).findings}
                record["edits"][kind] = {
                    "path": str(out),
                    "target_delta": after.get(target, 0.0) - base[target],
                    "mean_other_delta": (
                        sum(after[k] - base[k] for k in base if k != target) / (len(base) - 1)
                    ),
                    "probs": after,
                }
            rows.append(record)
            print(f"[{study.study_id}] target={target} base={base[target]:.3f} "
                  + " ".join(
                      f"{k}:Δ{v.get('target_delta', float('nan')):+.3f}"
                      for k, v in record["edits"].items()
                  ), flush=True)

    OUT_PATH.write_text(
        json.dumps(
            {"generated_at": datetime.now(timezone.utc).isoformat(), "model": args.model,
             "n_studies": len(rows), "studies": rows},
            indent=2,
        )
    )

    print("\n=== counterfactual probe ===")
    print(f"{'study':>8} {'target':<16} {'base':>6} {'Δtarget(t)':>11} {'Δother(t)':>10} "
          f"{'Δtarget(c)':>11} {'Δother(c)':>10}")
    for row in rows:
        t = row["edits"].get("targeted", {})
        c = row["edits"].get("control", {})
        print(
            f"{row['study_id']:>8} {row['target']:<16} {row['base_prob']:>6.3f} "
            f"{t.get('target_delta', float('nan')):>11.3f} {t.get('mean_other_delta', float('nan')):>10.3f} "
            f"{c.get('target_delta', float('nan')):>11.3f} {c.get('mean_other_delta', float('nan')):>10.3f}"
        )
    print(
        "\nRead it this way: the targeted edit is only evidence of anything if "
        "Δtarget(t) is clearly more negative than both Δother(t) and Δtarget(c). "
        "If every column moves together, the model changed the picture, not the finding."
    )
    print(f"\nfull output: {OUT_PATH}")


if __name__ == "__main__":
    main()
