"""sentence-transformers-based mode-collapse detection (CLAUDE.md locked validation stack).

The data is purely numeric (anonymized PCA components), not text, so there's no natural language
task here — this was an explicit design call, confirmed with the user 2026-08-03 (Phase 3): each
synthetic row is serialized to a canonical text string and embedded, then pairwise cosine
similarity among a batch's own synthetic rows measures how repetitive/collapsed the batch is.
This is deliberately a *different* signal from novelty.py (which compares synthetic rows against
real rows) — diversity.py only looks within the synthetic batch itself.
"""
import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer

FEATURE_COLUMNS = ["Time"] + [f"V{i}" for i in range(1, 29)] + ["Amount"]
DEFAULT_MODEL_NAME = "all-MiniLM-L6-v2"

_model_cache: dict[str, SentenceTransformer] = {}


def _get_model(model_name: str) -> SentenceTransformer:
    if model_name not in _model_cache:
        _model_cache[model_name] = SentenceTransformer(model_name)
    return _model_cache[model_name]


def _row_to_text(row: pd.Series) -> str:
    return ", ".join(f"{col}={row[col]:.4f}" for col in FEATURE_COLUMNS)


def compute_mode_collapse_score(
    synthetic_rows: pd.DataFrame, model_name: str = DEFAULT_MODEL_NAME
) -> tuple[float, float]:
    """Returns (mean_nearest_neighbor_similarity, max_nearest_neighbor_similarity) among the
    batch's own rows. Higher = more repetitive/collapsed. A batch of 1 row returns (0.0, 0.0)
    (mode collapse is undefined for a single row)."""
    n = len(synthetic_rows)
    if n < 2:
        return 0.0, 0.0

    texts = [_row_to_text(row) for _, row in synthetic_rows.iterrows()]
    model = _get_model(model_name)
    embeddings = model.encode(texts, normalize_embeddings=True, show_progress_bar=False)

    similarity = embeddings @ embeddings.T
    np.fill_diagonal(similarity, -np.inf)  # exclude self-similarity
    nearest_neighbor_sim = similarity.max(axis=1)

    return float(nearest_neighbor_sim.mean()), float(nearest_neighbor_sim.max())
