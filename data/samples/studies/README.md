# Sample slice — 3 OpenI studies

This directory is a tiny, committed slice of the OpenI / Indiana University
chest X-ray collection, kept small so the repo is clone-and-run and
`tests/test_openi.py` runs fully offline against it (no network, no
dependency on the gitignored `data/openi/` full dataset).

Layout matches what `scripts/fetch_openi.py` produces under
`Settings.openi_root`:

```
ecgen-radiology/<N>.xml   -- report XML (unmodified from the source archive)
images/CXR<N>_*.png       -- image(s) for that study
```

## Why keyword matching alone would have picked the wrong "positive"

Of OpenI's 3955 reports, 2545 mention "pneumothorax" or "pneumomediastinum"
by string search, but 82.9% of those are pure negations ("no pneumothorax",
"without pneumomediastinum") — i.e. normal studies. Picking a "positive"
sample by keyword presence alone would very likely mislabel a normal study
as positive. Each study below was chosen only after reading its actual
`IMPRESSION` (and `FINDINGS`) text and confirming the polarity by hand.

## The 3 studies

### 1. Genuinely positive — study 797

- **IMPRESSION**: "1. Cardiomegaly without lung infiltrates."
- **FINDINGS**: "The heart size is enlarged. Tortuous aorta. Otherwise the
  mediastinal contour is within normal limits. The lungs are free of any
  focal infiltrates. There are no nodules or masses. No visible
  pneumothorax. No visible pleural fluid. ..."
- **MeSH**: `Cardiomegaly`, `Aorta/tortuous`
- **Why this bucket**: the cardiomegaly finding is asserted, not negated —
  "the heart size is enlarged" — and MeSH independently corroborates it.
  "without lung infiltrates" in the impression negates a *different*
  finding (infiltrates), not cardiomegaly itself; the study is genuinely
  positive for cardiomegaly. `Cardiomegaly` is also a core label in
  TorchXRayVision's `densenet121-res224-all` pathology list, which is what
  `reader_a` (Task 1.5+) runs, so this sample gives that reader a real
  chance to fire on something instead of reading a projection it can't
  interpret (see the view-position note below — this replaced an earlier,
  wrong pick).

  This study's `COMPARISON`/`INDICATION` fields also contain the raw
  `XXXX` de-identification placeholder ("PA and lateral chest x-XXXX dated
  XXXX", "XXXX-year-old female with chest pain."), so it's a second fixture
  (alongside study 38) that exercises XXXX-normalization — see the section
  below. `indication` loads as `"female with chest pain."`.

### 2. Clearly negative / normal — study 38

- **IMPRESSION**: "No acute cardiopulmonary process."
- **FINDINGS**: "Lungs are clear. There is no pneumothorax or pleural
  effusion. The heart and mediastinum are within normal limits. Bony
  structures are intact."
- **MeSH**: `normal`
- **Why this bucket**: every clinical statement is an explicit negation
  ("no pneumothorax", "no pleural effusion") and the MeSH major term is
  literally `normal` — no ambiguity. This study's `INDICATION` field also
  contains the raw `XXXX` de-identification placeholder twice ("XXXX, XXXX
  and shortness of breath for 3 days"), which doubles as a fixture for
  XXXX-normalization — see the section below. `indication` loads as
  `"shortness of breath for 3 days"`.

### 3. Descriptively ambiguous — study 1187

- **IMPRESSION**: "Minimally increased air space opacities bilaterally,
  most prominent in the lung bases. Findings are nonspecific, but may
  represent subsegmental atelectasis versus mild interstitial edema or an
  atypical infectious process."
- **FINDINGS**: "Minimally increased XXXX airspace opacities bilaterally,
  most prominent in the lung bases. Heart size is within normal limits. No
  pneumothorax or pleural effusion. Osseous structures are grossly intact."
- **MeSH**: `Opacity/lung/base/bilateral`
- **Why this bucket**: there is a real abnormal finding (bilateral opacity),
  but the radiologist explicitly hedges the differential with "nonspecific",
  "may represent ... versus ... or" — three competing explanations, none
  committed to. This is neither a clean positive nor a clean negative, which
  is exactly the case a downstream report-draft evaluator needs to handle
  without over-claiming certainty.

## XXXX is not a uniform placeholder — normalization strategy

`XXXX` in this corpus stands in for at least three different things: a
whole redacted word in a list ("XXXX, XXXX and shortness of breath"), an
age embedded in a compound word ("XXXX-year-old"), and a name-shaped
fragment inside an ordinary word ("chest x-XXXX" for "chest x-ray" — their
scrubber flags "ray" as a person's name). A naive substring substitution
(`text.replace("XXXX", "")`) produces a different flavor of garbage for
each case — most visibly a dangling leading hyphen: `"-year-old female
with chest pain"`. That version shipped once; a review caught it because
the original test only asserted `"XXXX" not in text`, which that broken
output also satisfied.

The loader (`_normalize_xxxx` in `src/medscope/data/openi.py`) does **not**
try to reconstruct what was redacted (no guessing "XXXX-year-old" means "45
years old"). Instead it **drops the whole whitespace-delimited token** that
contains `XXXX` — this handles the standalone-word and embedded-in-a-word
cases with one rule, since a partially-redacted word (`x-` or `-year-old`)
isn't reconstructible either and shouldn't be emitted as a truncated
fragment. Remaining tokens are rejoined with single spaces (so a run of
dropped tokens can't leave a double space), a leading orphaned coordinating
conjunction ("and"/"or"/"but" — left over from a redacted list like the
study-38 example above) is stripped, and a field with nothing alphanumeric
left becomes `""` rather than stray punctuation. Example:
`"XXXX-year-old female with chest pain."` → `"female with chest pain."`.

## View position cannot be inferred from the filename — read before picking more samples

The first pick for the positive bucket was study 3399 ("Patchy left lower
lobe infiltrate ... consistent with pneumonitis"), which was rejected and
replaced with study 797 above. Reason: 3399's image
(`CXR3399_IM-1643-2001.png`) is a **lateral** chest projection, not frontal
(PA/AP). `reader_a` is TorchXRayVision `densenet121-res224-all`, trained
overwhelmingly on frontal views — its output on a lateral isn't meaningful,
and this sample is exactly what Task 1.5's CNN-reader test and Task 2.6's
end-to-end happy path exercise. A lateral "positive" would make those tests
pass or fail for reasons unrelated to the code under test.

The assumption going in was that the `-1001` / `-2001` suffix in the image
filename encodes frontal vs. lateral. **It does not.** 3399's lateral is
`-2001`, but study 2608's `-1001` image is also a lateral (confirmed by eye
— "LT LAT" is burned into the image corner). The filename carries no view
information, and the DICOM `ViewPosition` tag isn't available to us since
`NLMCXR_dcm.tgz` is unreachable from this network. **Do not use the
filename suffix to infer view when picking future samples.**

What does work: frontal chest radiographs are approximately left-right
symmetric (two lung fields flanking the spine); laterals are not. Scoring
each image by its correlation with its own horizontal mirror separates the
two classes cleanly on the 4 images checked so far:

| study | suffix | actual view (eye-confirmed) | mirror-symmetry score |
|---|---|---|---|
| 3399  | `-2001` | lateral  | 0.151  |
| 2608  | `-1001` | lateral  | -0.049 |
| 797   | `-1001` | frontal  | 0.851  |
| (other confirmed frontal) | — | frontal | 0.833 / 0.836 |

A threshold around 0.5 splits frontal from lateral. If more samples need
to be added later, compute this score (or eyeball the image) rather than
trusting the filename.

## A note on image counts

Only 1 image is available per study in this sample (the 108-image probe
slice at `/tmp/openi_probe/imgs2/` — a partial-range download — happened to
contain a single view for each of these 3 study IDs). The loader still
returns `image_paths` as a list to preserve multi-view studies where they
exist in the full dataset (frontal + lateral); Phase 1 only reads the first
image regardless.
