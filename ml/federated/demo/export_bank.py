"""Step 8B: pack ONE bank's ready-made files into exports/<bank>.zip for the in-person handover.

Run from the MAIN working folder (the only place that holds every bank):
    python -m ml.federated.demo.export_bank --bank bank_b      # or --all

The zip's root folder is <bank>/, so on the teammate's laptop
    Expand-Archive exports\\bank_b.zip -DestinationPath data\\clients\\
puts the files in data/clients/bank_b/.

Contents: train.csv, val.csv, test.csv, synthetic_validated.csv and manifest.json ONLY.
Never synthetic_candidates.csv, llm_cache/, models, other banks, data/raw or data/processed.

Before writing anything, the export checks every row (by fingerprint) and stops with an
error, creating no zip, if:
  * the bank's train/val/test files differ from the Step 2 partition (checksums), or
  * any real row is not in the global train pool (the source of every bank shard), or
  * any real row also appears in another bank's files or in the global val/test sets, or
  * any synthetic row is not labelled fraud, or is identical to ANY real row.
"""
import argparse
import json
import sys
import zipfile
from datetime import datetime, timezone

import pandas as pd

from ml.data.load import FEATURE_COLUMNS, LABEL_COLUMN, PROJECT_ROOT, load_config
from ml.federated.demo.bankfiles import BANKS, EXPORT_FILES, MANIFEST, REAL_FILES, file_summary, fingerprints


class ExportCheckError(Exception):
    pass


def _bank_dir(bank):
    return PROJECT_ROOT / load_config("partition")["clients_dir"] / bank


def verify_bank(bank: str) -> dict:
    """All export-time checks. Returns the manifest dict, or raises ExportCheckError."""
    processed = PROJECT_ROOT / load_config("partition")["processed_dir"]
    step2 = json.loads((PROJECT_ROOT / load_config("partition")["results_dir"] / "manifest.json").read_text())

    for f in EXPORT_FILES:
        if not (_bank_dir(bank) / f).exists():
            raise ExportCheckError(f"{bank}/{f} is missing")

    # 1) The real files are exactly the ones the Step 2 partition produced.
    for f in REAL_FILES:
        key = f"data/clients/{bank}/{f}"
        if file_summary(_bank_dir(bank) / f)["sha256"] != step2[key]:
            raise ExportCheckError(f"{key} differs from the Step 2 partition manifest")

    real = pd.concat([pd.read_csv(_bank_dir(bank) / f) for f in REAL_FILES])
    fp_real = fingerprints(real)
    if len(fp_real) != len(real):
        raise ExportCheckError(f"{bank}: duplicate real rows found (fingerprints not unique)")

    # 2) Every real row comes from the global TRAIN pool...
    fp_global_train = fingerprints(pd.read_csv(processed / "global_train.csv"))
    if not fp_real <= fp_global_train:
        raise ExportCheckError(f"{bank}: {len(fp_real - fp_global_train)} real rows are not from the global train pool")
    # ...and none belongs to the global val/test sets or another bank.
    for name in ("global_val.csv", "global_test.csv"):
        n = len(fp_real & fingerprints(pd.read_csv(processed / name)))
        if n:
            raise ExportCheckError(f"{bank}: {n} real rows also appear in {name}")
    for other in BANKS:
        if other == bank:
            continue
        other_rows = pd.concat([pd.read_csv(_bank_dir(other) / f) for f in REAL_FILES])
        n = len(fp_real & fingerprints(other_rows))
        if n:
            raise ExportCheckError(f"{bank}: {n} real rows also appear in {other}")

    # 3) Synthetic rows: fraud only, and never a copy of any real row anywhere.
    synth = pd.read_csv(_bank_dir(bank) / "synthetic_validated.csv")
    if list(synth.columns) != FEATURE_COLUMNS + [LABEL_COLUMN] or not (synth[LABEL_COLUMN] == 1).all():
        raise ExportCheckError(f"{bank}: synthetic_validated.csv has wrong columns or non-fraud rows")
    all_real = fp_global_train | fingerprints(pd.read_csv(processed / "global_val.csv")) | \
        fingerprints(pd.read_csv(processed / "global_test.csv"))
    n = len(fingerprints(synth) & all_real)
    if n:
        raise ExportCheckError(f"{bank}: {n} synthetic rows are identical to real rows")

    return {"bank": bank, "created_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "files": {f: file_summary(_bank_dir(bank) / f) for f in EXPORT_FILES},
            "checks": "step2 checksums; rows only from global train; disjoint from other banks and "
                      "global val/test; synthetic = fraud only, no copies of real rows"}


def export(bank: str) -> str:
    manifest = verify_bank(bank)  # raises before any file is written
    out_dir = PROJECT_ROOT / load_config("demo")["export"]["out_dir"]
    out_dir.mkdir(exist_ok=True)
    zpath = out_dir / f"{bank}.zip"
    with zipfile.ZipFile(zpath, "w", zipfile.ZIP_DEFLATED) as z:
        for f in EXPORT_FILES:
            z.write(_bank_dir(bank) / f, arcname=f"{bank}/{f}")
        z.writestr(f"{bank}/{MANIFEST}", json.dumps(manifest, indent=2))
    files = manifest["files"]
    summary = ", ".join(f"{f} {v['rows']:,} rows ({v['fraud']} fraud)" for f, v in files.items())
    return f"{zpath.relative_to(PROJECT_ROOT)}: {summary} | {zpath.stat().st_size / 1e6:.1f} MB"


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--bank", choices=BANKS)
    p.add_argument("--all", action="store_true")
    args = p.parse_args()
    banks = BANKS if args.all else [args.bank]
    if banks == [None]:
        p.error("give --bank <bank> or --all")
    for b in banks:
        try:
            print("OK  " + export(b))
        except ExportCheckError as e:
            print(f"EXPORT REFUSED for {b}: {e}. No zip was created.")
            sys.exit(1)


if __name__ == "__main__":
    main()
