"""Step 8B: show (and enforce) that THIS laptop holds only its own bank's data.

    python -m ml.federated.demo.check_my_data --bank bank_b
    python -m ml.federated.demo.check_my_data --bank bank_a --server-laptop   # also allows the global test set

Checks (the client refuses to start unless all pass):
  1. data/clients/<bank>/manifest.json names the same bank as --bank;
  2. every file's SHA-256 checksum, row count and fraud count match the manifest,
     and the bank folder holds nothing else (no candidates, caches or models);
  3. no other bank's DATA FILES exist here (the folders themselves always exist, because Git
     tracks their .gitkeep, so we look for files, not folders);
  4. the raw Kaggle file and the global train/val split are absent. The global test set is
     allowed only on the server laptop (--server-laptop).
"""
import argparse
import json
import sys
from pathlib import Path

from ml.data.load import DATA_ROOT, load_config
from ml.federated.demo.bankfiles import ALLOWED_IN_BANK_FOLDER, BANKS, MANIFEST, data_files, file_summary, sha256


def check(bank: str, root: Path = DATA_ROOT, server_laptop: bool = False) -> tuple[bool, list[str]]:
    clients = root / load_config("partition")["clients_dir"]
    processed = root / load_config("partition")["processed_dir"]
    raw = root / load_config("data")["raw_path"]
    problems, lines = [], []

    # 1 + 2: own bank
    own = clients / bank
    mpath = own / MANIFEST
    if not mpath.exists():
        problems.append(f"{bank}: no {MANIFEST} (unzip your bank file into data/clients/ first)")
        manifest = None
    else:
        manifest = json.loads(mpath.read_text(encoding="utf-8"))
        if manifest.get("bank") != bank:
            problems.append(f"manifest says {manifest.get('bank')!r} but you started as {bank!r}")
    if manifest:
        rows = fraud = 0
        for f, expect in manifest["files"].items():
            p = own / f
            if not p.exists():
                problems.append(f"{bank}/{f} missing")
                continue
            # Checksum first: a tampered or corrupted file is reported, never parsed.
            if sha256(p) != expect["sha256"]:
                problems.append(f"{bank}/{f} does not match its manifest (checksum)")
                continue
            got = file_summary(p)
            if got != expect:
                problems.append(f"{bank}/{f} does not match its manifest (rows/fraud)")
            if f != "synthetic_validated.csv":
                rows, fraud = rows + got["rows"], fraud + got["fraud"]
        extra = [p.relative_to(own).as_posix() for p in data_files(own) if p.name not in ALLOWED_IN_BANK_FOLDER]
        if extra:
            problems.append(f"{bank} folder has files outside the handover set: {extra[:5]}")
        ok_sums = not any(bank in p and "manifest" in p for p in problems)
        lines.append(f"{bank}: {len(manifest['files'])} files + manifest, {rows:,} real rows ({fraud} fraud), "
                     f"checksums {'OK' if ok_sums else 'FAILED'}")

    # 3: other banks must be empty
    others = {b: data_files(clients / b) for b in BANKS if b != bank}
    for b, files in others.items():
        if files:
            problems.append(f"{b}: {len(files)} data file(s) present on this laptop")
    lines.append(" | ".join(f"{b}: {'empty' if not f else f'{len(f)} FILES'}" for b, f in others.items()))

    # 4: no raw file, no global train/val; global test only on the server laptop
    raw_present = raw.exists()
    if raw_present:
        problems.append("raw Kaggle file present")
    split = {n: (processed / n).exists() for n in ("global_train.csv", "global_val.csv", "global_test.csv")}
    if split["global_train.csv"] or split["global_val.csv"]:
        problems.append("global train/val split present")
    if split["global_test.csv"] and not server_laptop:
        problems.append("global test set present on a client laptop")
    lines.append(f"raw Kaggle file: {'PRESENT' if raw_present else 'absent'} | global train/val: "
                 f"{'PRESENT' if split['global_train.csv'] or split['global_val.csv'] else 'absent'} | "
                 f"global test: {'present (server laptop, allowed)' if split['global_test.csv'] and server_laptop else ('PRESENT' if split['global_test.csv'] else 'absent')}")
    return not problems, lines + [f"PROBLEM: {p}" for p in problems]


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--bank", required=True, choices=BANKS)
    p.add_argument("--server-laptop", action="store_true", help="this laptop also runs the server")
    args = p.parse_args()
    ok, lines = check(args.bank, server_laptop=args.server_laptop)
    print(f"Data folder: {DATA_ROOT}")
    for line in lines:
        print("  " + line)
    print("RESULT: OK - this laptop holds only " + args.bank if ok else "RESULT: FAILED - fix the problems above")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
