"""Step 8B: check_my_data must ACCEPT a clean single-bank laptop and REFUSE every unsafe setup.
Builds tiny fake laptops in a temp folder; no real data needed."""
import json

import numpy as np
import pandas as pd
import pytest

from ml.data.load import FEATURE_COLUMNS, LABEL_COLUMN
from ml.federated.demo.bankfiles import EXPORT_FILES, MANIFEST, file_summary
from ml.federated.demo.check_my_data import check


def _laptop(root, bank="bank_b"):
    """A clean laptop folder holding only `bank`, with a valid manifest (like an unzipped export)."""
    rng = np.random.default_rng(0)
    folder = root / "data" / "clients" / bank
    folder.mkdir(parents=True)
    for f in EXPORT_FILES:
        df = pd.DataFrame(rng.normal(size=(20, len(FEATURE_COLUMNS))), columns=FEATURE_COLUMNS)
        df[LABEL_COLUMN] = 1 if f == "synthetic_validated.csv" else [1] * 2 + [0] * 18
        df.to_csv(folder / f, index=False)
    manifest = {"bank": bank, "files": {f: file_summary(folder / f) for f in EXPORT_FILES}}
    (folder / MANIFEST).write_text(json.dumps(manifest), encoding="utf-8")
    for other in ("bank_a", "bank_c", "bank_d"):          # Git always creates these, with .gitkeep
        (root / "data" / "clients" / other).mkdir(parents=True, exist_ok=True)
        (root / "data" / "clients" / other / ".gitkeep").touch()
    return folder


def test_clean_laptop_passes(tmp_path):
    _laptop(tmp_path)
    ok, lines = check("bank_b", root=tmp_path)
    assert ok, lines
    assert any("checksums OK" in line for line in lines)


@pytest.mark.parametrize("break_it, expect", [
    (lambda r, f: (r / "data/clients/bank_c/train.csv").write_text("x"), "bank_c: 1 data file"),
    (lambda r, f: (f / "train.csv").write_text("tampered"), "does not match its manifest"),
    (lambda r, f: (f / "synthetic_candidates.csv").write_text("x"), "outside the handover set"),
    (lambda r, f: ((r / "data/raw").mkdir(parents=True), (r / "data/raw/creditcard.csv").write_text("x")),
     "raw Kaggle file present"),
    (lambda r, f: ((r / "data/processed").mkdir(parents=True), (r / "data/processed/global_train.csv").write_text("x")),
     "global train/val split present"),
    (lambda r, f: ((r / "data/processed").mkdir(parents=True), (r / "data/processed/global_test.csv").write_text("x")),
     "global test set present on a client laptop"),
])
def test_unsafe_setups_are_refused(tmp_path, break_it, expect):
    folder = _laptop(tmp_path)
    break_it(tmp_path, folder)
    ok, lines = check("bank_b", root=tmp_path)
    assert not ok
    assert any(expect in line for line in lines), lines


def test_wrong_bank_name_is_refused(tmp_path):
    _laptop(tmp_path, bank="bank_b")
    # Someone starts the client as bank_c on Prachi's (bank_b) laptop.
    ok, lines = check("bank_c", root=tmp_path)
    assert not ok


def test_global_test_allowed_only_on_server_laptop(tmp_path):
    _laptop(tmp_path, bank="bank_a")
    (tmp_path / "data/processed").mkdir(parents=True)
    (tmp_path / "data/processed/global_test.csv").write_text("x")
    assert not check("bank_a", root=tmp_path)[0]
    assert check("bank_a", root=tmp_path, server_laptop=True)[0]
