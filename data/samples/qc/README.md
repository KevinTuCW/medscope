# QC fixture — 1 lateral image

This directory holds fixtures for `tests/test_qc.py`, kept separate from
`data/samples/studies/` because Task 1.2's tests assert `load_studies`
returns exactly 3 studies — a 4th image there would break that contract.

## CXR3399_IM-1643-2001.png

A real OpenI chest X-ray, **lateral** projection (eye-confirmed — see
`data/samples/studies/README.md` for how the filename suffix (`-2001`)
was ruled out as a view-position signal). Copied from the probe slice at
`/tmp/openi_probe/imgs2/`.

Used by `medscope.qc.check_quality`'s lateral-view screen: scoring an
image by the correlation between it and its own horizontal mirror
(frontal chest X-rays are approximately left-right symmetric about the
spine; laterals are not). Measured mirror-symmetry score for this image:
**+0.151** — well below the `Settings.lateral_symmetry_threshold` default
of 0.5, and well below the ~0.83–0.85 scores measured for the 3 frontal
images in `data/samples/studies/images/`.
