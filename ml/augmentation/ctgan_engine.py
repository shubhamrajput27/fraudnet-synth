"""CTGAN-based synthetic fraud-row generator for data-rich clients (CLAUDE.md: Augment Mode).

Fits per-client, on that client's own real fraud rows only — never sees legit rows or another
client's data, and the fitted model/output never leaves the client. CPU-only (`enable_gpu=False`)
per the project's hard CPU-only constraint.

`batch_size` must be a multiple of `pac` (verified empirically against the installed sdv==1.37.4
/ ctgan==0.12.1: a non-multiple raises an `AssertionError` inside ctgan, not a friendly error) —
see `_valid_batch_size`.
"""
from dataclasses import dataclass

import pandas as pd
from sdv.metadata import Metadata
from sdv.single_table import CTGANSynthesizer

FEATURE_COLUMNS = ["Time"] + [f"V{i}" for i in range(1, 29)] + ["Amount"]


@dataclass
class CTGANEngineConfig:
    epochs: int = 300
    max_batch_size: int = 500
    pac: int = 10
    seed: int = 42


def _valid_batch_size(n_rows: int, pac: int, max_batch_size: int) -> int:
    """Largest multiple of `pac` that's <= max_batch_size, falling back to `pac` for tiny data
    (ctgan tolerates a batch size larger than the dataset — verified empirically)."""
    candidate = min(max_batch_size, (n_rows // pac) * pac)
    return max(pac, candidate)


class CTGANEngine:
    """Fits one CTGANSynthesizer per instance, trained on a single client's real fraud rows."""

    def __init__(self, config: CTGANEngineConfig | None = None):
        self.config = config or CTGANEngineConfig()
        self._synthesizer: CTGANSynthesizer | None = None
        self._n_train_rows: int = 0

    def fit(self, fraud_rows: pd.DataFrame) -> None:
        if fraud_rows.empty:
            raise ValueError("CTGANEngine.fit called with zero fraud rows — nothing to learn from.")

        features = fraud_rows[FEATURE_COLUMNS].reset_index(drop=True)
        self._n_train_rows = len(features)
        batch_size = _valid_batch_size(len(features), self.config.pac, self.config.max_batch_size)

        metadata = Metadata.detect_from_dataframe(features, table_name="fraud_rows")
        synthesizer = CTGANSynthesizer(
            metadata,
            epochs=self.config.epochs,
            batch_size=batch_size,
            pac=self.config.pac,
            enable_gpu=False,
            verbose=False,
        )
        synthesizer.fit(features)
        self._synthesizer = synthesizer

    def generate(self, n_rows: int) -> pd.DataFrame:
        if self._synthesizer is None:
            raise RuntimeError("CTGANEngine.generate called before fit().")
        synthetic = self._synthesizer.sample(num_rows=n_rows)
        synthetic["Class"] = 1
        return synthetic[FEATURE_COLUMNS + ["Class"]]
