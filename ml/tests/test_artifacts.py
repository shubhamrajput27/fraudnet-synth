"""ml/common/artifacts.py: model/scaler persistence for Phase 6's /predict — a save/load
round-trip that doesn't reproduce identical predictions would silently break the dashboard demo."""
import numpy as np
import torch
from sklearn.preprocessing import StandardScaler

from ml.common.artifacts import load_manifest, load_model, load_scaler, save_manifest, save_model, save_scaler
from ml.common.model import N_FEATURES, new_model


def test_model_save_load_round_trip_produces_identical_predictions(tmp_path, monkeypatch):
    monkeypatch.setattr("ml.common.artifacts.CONFIG", type("C", (), {"models_dir": tmp_path})())
    model = new_model(seed=42)
    model.eval()
    x = torch.randn(3, N_FEATURES)
    with torch.no_grad():
        expected = model(x)

    save_model("run_x", model)
    loaded = load_model("run_x")
    with torch.no_grad():
        actual = loaded(x)

    assert torch.allclose(expected, actual)


def test_model_save_load_per_bank_keeps_banks_separate(tmp_path, monkeypatch):
    monkeypatch.setattr("ml.common.artifacts.CONFIG", type("C", (), {"models_dir": tmp_path})())
    model_a = new_model(seed=1)
    model_b = new_model(seed=2)
    save_model("run_x", model_a, bank="A")
    save_model("run_x", model_b, bank="B")

    loaded_a = load_model("run_x", bank="A")
    loaded_b = load_model("run_x", bank="B")
    x = torch.randn(3, N_FEATURES)
    with torch.no_grad():
        assert not torch.allclose(loaded_a(x), loaded_b(x))


def test_scaler_save_load_round_trip(tmp_path, monkeypatch):
    monkeypatch.setattr("ml.common.artifacts.CONFIG", type("C", (), {"models_dir": tmp_path})())
    scaler = StandardScaler()
    scaler.fit(np.random.default_rng(0).normal(size=(50, N_FEATURES)))

    save_scaler("run_x", scaler, bank="A")
    loaded = load_scaler("run_x", bank="A")

    sample = np.random.default_rng(1).normal(size=(4, N_FEATURES))
    assert np.allclose(scaler.transform(sample), loaded.transform(sample))


def test_manifest_round_trip(tmp_path, monkeypatch):
    monkeypatch.setattr("ml.common.artifacts.CONFIG", type("C", (), {"models_dir": tmp_path})())
    save_manifest("run_x", "federated_augmented", banks=["A", "B", "C", "D"])
    manifest = load_manifest("run_x")
    assert manifest == {"run_id": "run_x", "arm": "federated_augmented", "banks": ["A", "B", "C", "D"]}


def test_load_manifest_returns_none_when_missing(tmp_path, monkeypatch):
    monkeypatch.setattr("ml.common.artifacts.CONFIG", type("C", (), {"models_dir": tmp_path})())
    assert load_manifest("nonexistent_run") is None
