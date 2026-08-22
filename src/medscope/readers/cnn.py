"""reader_a -- the discriminative CNN reader.

The only component in the pipeline that produces quantitative
probabilities and spatial localization. reader_b (the VLM) can describe
and reason about clinical context but gives neither a calibrated number
nor a box; gate G2 requires every report sentence to cite a Finding
carrying a source, a probability, and a location, so this module is what
makes that gate satisfiable at all.

Import discipline: `torch` and `torchxrayvision` are imported lazily
inside `_load_model`, never at module top level, so importing this module
(or anything that transitively imports `medscope.state`) never pays a
torch import. `_load_model` is the only place model weights are touched;
it raises a readable error if the `cv` extra isn't installed.

`_infer` and `_gradcam` are deliberately the only two functions that call
`_load_model`. Tests stub both of them directly (see
tests/test_cnn_reader.py) so the fast suite never loads real weights and
never touches the network -- `_preprocess` runs for real in those tests,
since it only needs Pillow/numpy/torchxrayvision's pure array utilities,
no model.

Label count: a note in the project spec calls this "14 pathologies" --
that's CheXNet-era shorthand and does not match `densenet121-res224-all`.
This module never hardcodes the label list or its length; `CNNReader.read`
zips `model.pathologies` against the output vector at runtime.
"""

from __future__ import annotations

import time
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np

from medscope.config import Settings
from medscope.ontology import canonical
from medscope.state import Finding, ReadResult
from medscope.utils import clamp
from medscope.thresholds import report_threshold
from medscope.views import order_views

if TYPE_CHECKING:
    import torch

_model_cache: dict[str, object] = {}


def _load_model():
    """Lazily load and cache the TorchXRayVision model for this process.

    Cached once per process (not re-checked against Settings on later
    calls) -- "lazy, cached per process" per spec, not "reloaded on every
    config change".
    """
    if "model" in _model_cache:
        return _model_cache["model"]

    try:
        import torchxrayvision as xrv
    except ImportError as exc:
        raise RuntimeError(
            "CNNReader requires the 'cv' extra (torch + torchxrayvision). "
            "Install with: pip install -e '.[cv]'"
        ) from exc

    settings = Settings()
    model = xrv.models.DenseNet(weights=settings.cnn_weights)
    model.eval()
    _model_cache["model"] = model
    return model


def _to_tensor(arr: np.ndarray) -> "torch.Tensor":
    """Convert a numpy array to a torch tensor without `torch.from_numpy`.

    `torch.from_numpy` (and the reverse, `Tensor.numpy()`) go through
    torch's numpy C-API array interface, which raises `RuntimeError:
    Numpy is not available` when torch was built against the numpy 1.x
    ABI but numpy 2.x is installed at runtime -- true in this environment
    (torch 2.2.2 / numpy 2.5.2). `frombuffer` sidesteps that codepath
    entirely by going through raw bytes instead.
    """
    import torch

    arr = np.ascontiguousarray(arr, dtype=np.float32)
    flat = torch.frombuffer(bytearray(arr.tobytes()), dtype=torch.float32)
    return flat.reshape(arr.shape)


def _preprocess(image_path: str | Path, *, center_crop: bool = False) -> "torch.Tensor":
    """Grayscale -> xrv normalize -> resize to 224 -> (1, 1, 224, 224).

    Uses torchxrayvision's own array utilities (which depend on
    scikit-image, not torch), so this never touches `_load_model` and never
    needs real weights.

    **No center crop by default, and that is a clinical decision.**
    `XRayCenterCrop` squares the image by trimming the long axis, which on
    Open-i's portrait films (512x624 and similar) removes roughly 56px from
    the top and bottom -- the lung apices, where a pneumothorax collects,
    and the costophrenic angles, where an effusion collects. Measured on
    the G1 gold set, dropping the crop moves pneumothorax AUC from 0.593 to
    0.659 and effusion AUC from 0.899 to 0.917: the model gets better at
    telling positives from negatives, not merely more willing to alarm.
    Squashing the aspect ratio instead is the lesser distortion when the
    two findings the gate exists for live in the parts being cut off.

    `center_crop=True` is kept as a parameter, not deleted, so the old
    behaviour stays reproducible for the ablation that justified this
    default.
    """
    import torchxrayvision as xrv

    from contextlib import closing

    from medscope.film import open_film

    with closing(open_film(image_path)) as film:
        arr = np.asarray(film.convert("L"), dtype=np.float32)
    arr = xrv.datasets.normalize(arr, 255)
    arr = arr[None, ...]  # (1, H, W) -- add channel dim
    if center_crop:
        arr = xrv.datasets.XRayCenterCrop()(arr)
    arr = xrv.datasets.XRayResizer(224)(arr)  # (1, 224, 224)
    tensor = _to_tensor(arr).unsqueeze(0)  # (1, 1, 224, 224)
    return tensor


def _infer(tensor: "torch.Tensor") -> dict[str, float]:
    """Single small seam between preprocessing and the model -- stubbed in
    unit tests so they never load real weights.

    NOTE on sigmoid: torchxrayvision's calibrated weight sets (including
    `densenet121-res224-all`) carry per-class `op_threshs` and already
    apply sigmoid + threshold normalization inside `forward()` -- the raw
    model output is already in [0, 1]. Applying `torch.sigmoid` again here
    would double-squash an already-bounded value toward 0.5 for every
    label. Sigmoid is applied here only for the (uncalibrated) case where
    the model has no `op_threshs` and no `apply_sigmoid`, matching
    torchxrayvision's own forward() convention.
    """
    import torch

    model = _load_model()
    with torch.no_grad():
        output = model(tensor)
        already_bounded = getattr(model, "op_threshs", None) is not None or getattr(
            model, "apply_sigmoid", False
        )
        if not already_bounded:
            output = torch.sigmoid(output)
    probs = output[0].tolist()
    return dict(zip(model.pathologies, probs))


def _gradcam(tensor: "torch.Tensor", label: str) -> dict | None:
    """Grad-CAM on the final conv block (`model.features`, the standard
    DenseNet121 feature extractor before global pooling): feature maps
    weighted by globally-averaged gradients of the target class logit,
    ReLU'd, normalized, then reduced to the peak region as
    {"cx", "cy", "r"} in normalized 0-1 coordinates.

    Returns None if `label` isn't one of this model's pathologies, or if
    the CAM has no positive activation anywhere (nothing to localize).
    """
    import torch
    import torch.nn.functional as F

    model = _load_model()
    if label not in model.pathologies:
        return None
    idx = model.pathologies.index(label)

    model.zero_grad(set_to_none=True)
    x = tensor.clone().requires_grad_(True)
    feats = model.features(x)  # (1, C, H, W) -- final conv block output
    feats.retain_grad()  # not a leaf; need its gradient after backward
    pooled = F.adaptive_avg_pool2d(F.relu(feats), (1, 1)).flatten(1)
    logit = model.classifier(pooled)[0, idx]
    logit.backward()

    grads = feats.grad[0]  # (C, H, W)
    activations = feats[0].detach()  # (C, H, W)
    weights = grads.mean(dim=(1, 2))  # global-average-pooled gradients, (C,)
    cam = F.relu((weights[:, None, None] * activations).sum(dim=0)).detach()  # (H, W)

    cam_max = cam.max()
    if float(cam_max) <= 0:
        return None
    cam = cam / cam_max

    # Kept as torch ops throughout (no numpy round-trip): `Tensor.numpy()`
    # hits the same broken numpy C-API bridge as `torch.from_numpy` (see
    # `_to_tensor`).
    h, w = cam.shape
    peak = int(torch.argmax(cam).item())
    peak_y, peak_x = divmod(peak, w)
    cx = (peak_x + 0.5) / w
    cy = (peak_y + 0.5) / h

    # Radius from the spread of the region at or above half the peak
    # activation -- a compact CAM gives a small radius, a diffuse one a
    # large one. The peak pixel itself is always >= 0.5 (cam is normalized
    # to a max of 1.0), so this mask is never empty.
    mask = cam >= 0.5
    nz = torch.nonzero(mask)
    ys, xs = nz[:, 0], nz[:, 1]
    span = max(int(ys.max()) - int(ys.min()), int(xs.max()) - int(xs.min())) + 1
    r = span / (2 * max(h, w))

    return {
        "cx": clamp(float(cx), 0.0, 1.0),
        "cy": clamp(float(cy), 0.0, 1.0),
        "r": clamp(float(r), 0.0, 1.0),
    }


class CNNReader:
    """reader_a. Runs TorchXRayVision's densenet121-res224-all (or
    whatever `Settings.cnn_weights` names) independently of reader_b.
    """

    def __init__(self, settings: Settings | None = None):
        self.settings = settings or Settings()

    def read(self, image_path: str | Path) -> ReadResult:
        """Read a single film. Thin wrapper over `read_study` so there is
        exactly one aggregation path to reason about.
        """
        return self.read_study([image_path])

    def read_study(
        self,
        image_paths: list[str | Path] | list[Path] | list[str],
        *,
        localize: bool = True,
    ) -> ReadResult:
        """Read **every** film of a study and report one finding per label.

        A radiologist reads a study, not a file. Reading only the first
        film cost the G1 gate five confirmed-positive cases -- among them
        study 1525, whose four films include the one showing a large
        hydropneumothorax -- and which film "first" meant depended on
        filesystem enumeration order.

        Aggregation is the max probability across films, attributed to the
        film that produced it. Max is the reading that matches what the
        gate is for: a pneumothorax visible on one projection and not
        another is still a pneumothorax. It is also the aggregation that
        can be *wrong* in the safe direction -- it can over-call, never
        under-call, relative to any single film.

        Measured on the G1 gold set against the previous behaviour, this
        plus the dropped center crop takes critical recall from 22/27 to
        27/27 while the false-positive rate stays at 21/26 -- the same
        operating point, reading more of the study. That distinction is
        the whole point: raising recall by lowering a threshold (or by
        ensembling in a model that alarms on everything) buys the same
        number at a worse FPR, and would be one more welded-shut green
        light of the kind this project keeps refusing to install.
        """
        start = time.perf_counter()
        ordered = order_views(list(image_paths), self.settings)
        if not ordered:
            return ReadResult(
                reader="a",
                findings=[],
                latency_ms=int((time.perf_counter() - start) * 1000),
                notes=["reader_a received no films"],
            )

        # (tensor, probs) per film, kept so Grad-CAM can be computed later
        # on *the film that won a label*, not on whichever one is handy.
        reads = []
        for ref in ordered:
            tensor = _preprocess(ref.path)
            reads.append((ref, tensor, _infer(tensor)))

        best: dict[str, tuple[float, int]] = {}
        for idx, (_ref, _tensor, probs) in enumerate(reads):
            for raw_label, prob in probs.items():
                prob = clamp(float(prob), 0.0, 1.0)
                if raw_label not in best or prob > best[raw_label][0]:
                    best[raw_label] = (prob, idx)

        findings: list[Finding] = []
        for raw_label, (prob, idx) in best.items():
            ref, tensor, _ = reads[idx]
            label = canonical(raw_label)
            locus = None
            if localize and prob >= report_threshold(
                label if label is not None else raw_label, self.settings.cnn_prob_threshold
            ):
                # Grad-CAM costs a backward pass each -- only run it for
                # labels the reader is actually calling positive.
                locus = _gradcam(tensor, raw_label)
            findings.append(
                Finding(
                    label=label if label is not None else raw_label,
                    prob=prob,
                    source="cnn",
                    locus=locus,
                    raw_label=raw_label,
                    image_ref=str(ref.path),
                )
            )

        notes = [
            "reader_a read {} film(s): {}".format(
                len(ordered),
                ", ".join(f"{r.path.name} ({r.view}, symmetry {r.symmetry:.3f})" for r in ordered),
            )
        ]
        latency_ms = int((time.perf_counter() - start) * 1000)
        return ReadResult(reader="a", findings=findings, latency_ms=latency_ms, notes=notes)
