"""Task 2.3, Part A: decide whether reader_b (the VLM) stays a peer reader
or gets demoted to `Settings.reader_b_mode = "describer"`.

Runs double reading (reader_a + reader_b) over a sample of studies and
reports the two statistics the decision rule is built on:

    - mean Cohen's kappa between the two readers (medscope.merge.cohens_kappa,
      via merge_reads)
    - disagreement rate: mean disagreements per study, and the share of
      studies with at least one disagreement

Prints the decision per `Settings`:

    kappa >= kappa_floor (0.4) AND disagreement_rate <= disagreement_ceiling (0.4)
        -> keep reader_b as a peer  (reader_b_mode = "reader")
    otherwise
        -> demote to describer      (reader_b_mode = "describer")

*** The honesty requirement ***

Without a real VLM API key, the only reader_b available is
`OfflineVLMClient`, which derives its "findings" from each study's own
ground-truth report text -- it is scoring the report against itself, not
measuring a model. A kappa computed that way is meaninglessly high (close
to circular agreement) and must never be used to justify keeping reader_b
as a peer. This script detects that case and refuses to print a
peer/describer recommendation: it prints the numbers labeled plainly as a
plumbing check, not a calibration result, and exits with a non-zero status
so nothing downstream can mistake this run for a passing calibration.

Only when `Settings.use_real_vlm` is true (a real keyed VLM configured) does
this script print an actual recommendation.

Usage:
    PYTHONPATH=src .venv/bin/python scripts/calibrate_vlm.py [--root PATH] [--limit N]

Defaults to `Settings.openi_root` (the gitignored full dataset, populated by
scripts/fetch_openi.py); for a quick offline smoke test against the 3
studies committed to the repo, pass --root data/samples/studies.

Writes full per-study output to /tmp/medscope_vlm_calibration.json (not the
repo).
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from medscope.config import Settings  # noqa: E402
from medscope.data.openi import load_studies  # noqa: E402
from medscope.merge import merge_reads  # noqa: E402
from medscope.readers.cnn import CNNReader  # noqa: E402
from medscope.readers.vlm import OfflineVLMClient, build_vlm_client, read_b  # noqa: E402
from medscope.state import StudyState  # noqa: E402

OUT_PATH = Path("/tmp/medscope_vlm_calibration.json")


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument(
        "--root", type=Path, default=None, help="Dataset root (default: Settings().openi_root)"
    )
    parser.add_argument("--limit", type=int, default=None, help="Only read the first N studies")
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    settings = Settings()
    root = args.root or settings.openi_root

    studies = load_studies(root)
    if args.limit:
        studies = studies[: args.limit]

    if not studies:
        print(f"No studies found under {root}.")
        print("For a quick offline smoke test against the repo's sample slice, try:")
        print("  PYTHONPATH=src .venv/bin/python scripts/calibrate_vlm.py --root data/samples/studies")
        sys.exit(1)

    # `use_real_vlm=False` (the default, no API key configured) means the
    # only reader_b available is OfflineVLMClient -- see the honesty
    # requirement in the module docstring.
    is_offline = not settings.use_real_vlm
    reader_a = CNNReader(settings)

    print(f"Reading {len(studies)} study(ies) from {root}...")
    if is_offline:
        print("reader_b: OfflineVLMClient (no USE_REAL_VLM / API key configured)")
    else:
        print(f"reader_b: real VLM ({settings.vlm_model})")

    per_study: list[dict] = []
    for study in studies:
        image_path = study.image_paths[0]
        state = StudyState(
            study_id=study.study_id,
            image_path=str(image_path),
            indication=study.indication,
            # OpenI's XML has no field distinct from INDICATION/FINDINGS/
            # IMPRESSION that represents "clinical history" -- leaving this
            # blank rather than reusing FINDINGS/IMPRESSION text, which
            # would hand reader_b the ground-truth answer as "history".
            history_text="",
        )

        read_a_result = reader_a.read(image_path)

        if is_offline:
            # Bound directly to this study's ground-truth text -- see the
            # honesty requirement above. build_vlm_client(settings) would
            # hand back an OfflineVLMClient with no impression text at all
            # (it has no per-study knowledge to give it), so it's built
            # here instead, exactly as tests/test_vlm_client.py does.
            client = OfflineVLMClient(impression_text=study.impression_text)
        else:
            client = build_vlm_client(settings)

        read_b_result = read_b(state, client, settings)

        _findings, disagreements, kappa = merge_reads(
            read_a_result, read_b_result, settings.cnn_prob_threshold, mode="reader"
        )
        per_study.append(
            {
                "study_id": study.study_id,
                "kappa": kappa,
                "n_disagreements": len(disagreements),
            }
        )

    mean_kappa = statistics.mean(s["kappa"] for s in per_study)
    mean_disagreements = statistics.mean(s["n_disagreements"] for s in per_study)
    share_with_disagreement = sum(1 for s in per_study if s["n_disagreements"] > 0) / len(per_study)

    print(f"\nStudies read: {len(per_study)}")
    print(f"Mean Cohen's kappa: {mean_kappa:.3f}")
    print(f"Mean disagreements per study: {mean_disagreements:.3f}")
    print(f"Share of studies with >= 1 disagreement (disagreement_rate): {share_with_disagreement:.1%}")

    result = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "root": str(root),
        "n_studies": len(per_study),
        "mean_kappa": mean_kappa,
        "mean_disagreements_per_study": mean_disagreements,
        "disagreement_rate": share_with_disagreement,
        "kappa_floor": settings.kappa_floor,
        "disagreement_ceiling": settings.disagreement_ceiling,
        "per_study": per_study,
        "offline_plumbing_check_only": is_offline,
    }
    OUT_PATH.write_text(json.dumps(result, indent=2, ensure_ascii=False))
    print(f"\nWrote per-study detail to {OUT_PATH}")

    if is_offline:
        print(
            "\n"
            "*** THESE NUMBERS ARE A PLUMBING CHECK, NOT A CALIBRATION RESULT. ***\n"
            "reader_b ran against OfflineVLMClient, which derives its findings from\n"
            "each study's OWN ground-truth report text -- this scores the report\n"
            "against itself, not a real model reading the image. The kappa above is\n"
            "meaninglessly high and MUST NOT be used to decide reader_b_mode.\n"
            "\n"
            "Refusing to print a peer/describer recommendation.\n"
            "To get an actual calibration decision, configure a real VLM\n"
            "(USE_REAL_VLM=true, VLM_MODEL, VLM_API_KEY)\n"
            "and rerun this script."
        )
        sys.exit(2)

    keep_as_peer = mean_kappa >= settings.kappa_floor and share_with_disagreement <= settings.disagreement_ceiling
    decision = "reader" if keep_as_peer else "describer"
    print(f"\nDecision: reader_b_mode = {decision!r}")
    if keep_as_peer:
        print(
            f"  kappa {mean_kappa:.3f} >= floor {settings.kappa_floor} and "
            f"disagreement_rate {share_with_disagreement:.3f} <= ceiling {settings.disagreement_ceiling} "
            "-- keep reader_b as a peer."
        )
    else:
        reasons = []
        if mean_kappa < settings.kappa_floor:
            reasons.append(f"kappa {mean_kappa:.3f} < floor {settings.kappa_floor}")
        if share_with_disagreement > settings.disagreement_ceiling:
            reasons.append(f"disagreement_rate {share_with_disagreement:.3f} > ceiling {settings.disagreement_ceiling}")
        print("  " + "; ".join(reasons) + " -- demote reader_b to describer.")


if __name__ == "__main__":
    main()
