"""ml/common/data.py: D4's per-client scaler discipline and the augmented-arm fallback behavior
(Phase 3 rejected 2 of 4 banks' synthetic batches — load_client_training_data must silently fall
back to real-only for those banks, not error)."""
from types import SimpleNamespace

import numpy as np

from ml.common.data import fit_scaler, load_client_training_data, to_arrays


def test_fit_scaler_normalizes_to_zero_mean_unit_variance(synthetic_shard):
    scaler = fit_scaler(synthetic_shard)
    X, _ = to_arrays(synthetic_shard, scaler)
    assert np.allclose(X.mean(axis=0), 0, atol=1e-6)
    assert np.allclose(X.std(axis=0), 1, atol=1e-6)


def test_to_arrays_preserves_row_count_and_labels(synthetic_shard):
    scaler = fit_scaler(synthetic_shard)
    X, y = to_arrays(synthetic_shard, scaler)
    assert len(X) == len(synthetic_shard)
    assert len(y) == len(synthetic_shard)
    assert set(np.unique(y)) <= {0.0, 1.0}
    assert y.sum() == synthetic_shard["Class"].sum()


def test_scaler_fit_on_one_set_transforms_another_using_the_same_stats(synthetic_shard, rng):
    # This is the D4/D13 pattern: a client's own scaler is reused to transform the shared holdout.
    scaler = fit_scaler(synthetic_shard)
    other = synthetic_shard.copy()
    other["Amount"] = other["Amount"] + 1000  # clearly different distribution
    X_other, _ = to_arrays(other, scaler)
    # Transformed with the ORIGINAL scaler's stats, so it should NOT be zero-mean anymore.
    assert not np.allclose(X_other[:, -1].mean(), 0, atol=1e-3)


def test_augmented_falls_back_to_real_when_validated_batch_missing(tmp_path, monkeypatch, synthetic_shard):
    shards_dir = tmp_path / "shards"
    validated_dir = tmp_path / "validated"
    shards_dir.mkdir()
    validated_dir.mkdir()
    synthetic_shard.to_csv(shards_dir / "bank_B.csv", index=False)
    # No bank_B_validated.csv written at all — mirrors a bank never even producing candidates.

    monkeypatch.setattr("ml.common.data.CONFIG", SimpleNamespace(shards_dir=shards_dir, validated_dir=validated_dir))
    result = load_client_training_data("B", augmented=True)
    assert len(result) == len(synthetic_shard)


def test_augmented_falls_back_to_real_when_validated_batch_is_empty(tmp_path, monkeypatch, synthetic_shard):
    # Mirrors Phase 3's actual finding: Banks B and C's synthetic batches were REJECTED, leaving
    # an empty (header-only) validated CSV on disk. This must not silently include zero real rows.
    shards_dir = tmp_path / "shards"
    validated_dir = tmp_path / "validated"
    shards_dir.mkdir()
    validated_dir.mkdir()
    synthetic_shard.to_csv(shards_dir / "bank_C.csv", index=False)
    synthetic_shard.iloc[0:0].to_csv(validated_dir / "bank_C_validated.csv", index=False)  # empty, header only

    monkeypatch.setattr("ml.common.data.CONFIG", SimpleNamespace(shards_dir=shards_dir, validated_dir=validated_dir))
    result = load_client_training_data("C", augmented=True)
    assert len(result) == len(synthetic_shard)


def test_augmented_concatenates_real_and_validated_when_present(tmp_path, monkeypatch, synthetic_shard):
    shards_dir = tmp_path / "shards"
    validated_dir = tmp_path / "validated"
    shards_dir.mkdir()
    validated_dir.mkdir()
    synthetic_shard.to_csv(shards_dir / "bank_A.csv", index=False)
    extra_rows = synthetic_shard.iloc[:5]
    extra_rows.to_csv(validated_dir / "bank_A_validated.csv", index=False)

    monkeypatch.setattr("ml.common.data.CONFIG", SimpleNamespace(shards_dir=shards_dir, validated_dir=validated_dir))
    result = load_client_training_data("A", augmented=True)
    assert len(result) == len(synthetic_shard) + 5
