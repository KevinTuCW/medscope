"""Runtime configuration for medscope, loaded from environment / .env.

Deliberately does NOT hold the critical-findings label list. Its single
source of truth is `medscope.ontology.CRITICAL_LABELS` (a later task) — two
copies would inevitably drift in spelling and silently disable gate G1
(critical-finding recall must be 1.0).
"""

from pathlib import Path
from typing import Literal

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_prefix="MEDSCOPE_",
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    openi_root: Path = Path("data/openi")
    cnn_weights: str = "densenet121-res224-all"
    cnn_prob_threshold: float = 0.5
    # Deliberately lower than cnn_prob_threshold: over-calling a critical
    # finding is acceptable, missing one is not (gate G1).
    critical_threshold: float = 0.3
    # medscope.merge: when both readers call a label positive but their
    # probabilities are at least this far apart, it's a "magnitude"
    # disagreement rather than an agreement -- e.g. 0.52 vs 0.9 is a
    # "barely positive" call next to a "strongly positive" one, clinically
    # worth a second look even though both readers agree on presence. 0.3
    # is chosen so two confident positives (0.85 vs 0.95) still agree,
    # while a bare-threshold call next to a confident one (0.52 vs 0.85+)
    # does not.
    magnitude_gap: float = 0.3
    reader_b_mode: Literal["reader", "describer"] = "reader"
    vlm_model: str = ""
    vlm_base_url: str = ""
    vlm_api_key: str = ""
    kappa_floor: float = 0.4
    disagreement_ceiling: float = 0.4
    max_llm_judgments: int = 12
    use_real_vlm: bool = False
    # QC's lateral-view screen (medscope.qc): mirror-symmetry score below
    # this is flagged "lateral_view". 0.5 sits in a wide empty band between
    # measured frontal scores (~0.83-0.85) and lateral scores (-0.05-0.15)
    # on real OpenI images -- see data/samples/qc/README.md.
    lateral_symmetry_threshold: float = 0.5

    langfuse_public_key: str = ""
    langfuse_secret_key: str = ""
    langfuse_host: str = ""

    @property
    def tracing_enabled(self) -> bool:
        return bool(self.langfuse_public_key) and bool(self.langfuse_secret_key)
