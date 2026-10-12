"""Step 13: build the report's results annexure from real run outputs (decision D-055).

    python -m ml.evaluation.annexure                      # uses results/integration/dashboard_job_id.txt
    python -m ml.evaluation.annexure --job <job id>

Writes docs/RESULTS_ANNEXURE.md and results/final_annexure/{tables,figures,provenance.json,consistency.json}.
Rules: no number is typed by hand. Tables are rendered from the CSV/JSON named under them, figures are
copied byte-for-byte, every caption number is computed here, and every source's SHA-256 is recorded.
"""
import argparse
import hashlib
import json
import shutil
import subprocess
from datetime import datetime, timezone

import numpy as np
import pandas as pd

from ml.data.load import PROJECT_ROOT
from ml.evaluation.six_arm_report import load_long

OUT = PROJECT_ROOT / "results" / "final_annexure"
DOC = PROJECT_ROOT / "docs" / "RESULTS_ANNEXURE.md"
METRICS = ["precision", "recall", "f1", "pr_auc", "roc_auc", "accuracy", "tp", "fp", "fn"]


def sha256(path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def rel(path) -> str:
    return path.relative_to(PROJECT_ROOT).as_posix()


def md_table(df: pd.DataFrame) -> str:
    cells = [[str(c) for c in df.columns]] + [[str(v) for v in row] for row in df.itertuples(index=False)]
    esc = [[c.replace("|", "\\|") for c in row] for row in cells]
    return "\n".join(["| " + " | ".join(esc[0]) + " |", "|" + "---|" * len(esc[0])]
                     + ["| " + " | ".join(r) + " |" for r in esc[1:]])


class Annexure:
    def __init__(self):
        self.parts, self.items = [], []
        self.n_tab = self.n_fig = 0
        for sub in ("tables", "figures"):
            d = OUT / sub
            if d.exists():
                shutil.rmtree(d)           # only our own generated output folder is rebuilt
            d.mkdir(parents=True)

    def section(self, title):
        self.parts.append(f"## {title}\n")

    def table(self, slug, title, df, sources, note=""):
        self.n_tab += 1
        tid = f"Table A{self.n_tab}"
        csv = OUT / "tables" / f"T{self.n_tab:02d}_{slug}.csv"
        df.to_csv(csv, index=False)
        src = ", ".join(f"`{rel(s)}`" for s in sources)
        self.parts.append(f"### {tid}. {title}\n\n{md_table(df)}\n\n*Source: {src}"
                          f"{(' (' + note + ')') if note else ''} · CSV: `{rel(csv)}`*\n")
        for s in sources:
            self.items.append({"id": tid, "file": rel(csv), "source": rel(s), "source_sha256": sha256(s)})

    def figure(self, slug, title, source, caption):
        self.n_fig += 1
        fid = f"Figure A{self.n_fig}"
        dest = OUT / "figures" / f"F{self.n_fig:02d}_{slug}.png"
        shutil.copyfile(source, dest)
        self.parts.append(f"### {fid}. {title}\n\n![{title}](../{rel(dest)})\n\n{caption}\n\n*Source: `{rel(source)}`*\n")
        self.items.append({"id": fid, "file": rel(dest), "source": rel(source), "source_sha256": sha256(source)})


def consistency(job_dir) -> dict:
    """Dashboard run vs command-line Step 9 run: same seeds/arms/models -> identical metrics?"""
    a = pd.read_csv(job_dir / "long_results.csv")
    b = pd.read_csv(PROJECT_ROOT / "results" / "six_arms" / "long_results.csv")
    keys = ["seed", "arm", "model", "eval"]
    m = a.merge(b, on=keys, suffixes=("_ui", "_cli"), how="outer", indicator=True, validate="one_to_one")
    both = m[m["_merge"] == "both"]
    diffs = {k: float((both[f"{k}_ui"] - both[f"{k}_cli"]).abs().max()) for k in METRICS}
    unmatched = int((m["_merge"] != "both").sum())
    return {"rows_dashboard": len(a), "rows_cli": len(b), "unmatched_rows": unmatched,
            "max_abs_difference_per_metric": diffs,
            "identical": unmatched == 0 and all(v == 0 for v in diffs.values())}


def build(job: str, integ=None) -> dict:
    R = PROJECT_ROOT / "results"
    job_dir = R / "api_jobs" / job
    jobinfo = json.loads((job_dir / "job.json").read_text(encoding="utf-8"))
    if jobinfo["status"] != "succeeded" or jobinfo["params"].get("arms") != "all":
        raise SystemExit(f"job {job} must be a succeeded run of all six arms (got {jobinfo['status']}, {jobinfo['params']})")
    seeds = jobinfo["params"]["seeds"]
    cons = consistency(job_dir)
    an = Annexure()

    # ---- A. data ----
    an.section("A. Data and non-IID partition")
    eda = json.loads((R / "eda" / "eda_summary.json").read_text(encoding="utf-8"))
    cc = eda["class_counts"]
    an.table("dataset_summary", "Dataset summary (ULB credit-card fraud)", pd.DataFrame([
        ("transactions", f"{eda['n_rows']:,}"), ("fraud / genuine", f"{cc['fraud_1']:,} / {cc['genuine_0']:,}"),
        ("fraud rate", f"{eda['fraud_percent']}%"), ("accuracy of 'always genuine'", f"{eda['always_genuine_accuracy_percent']}%"),
        ("missing values", eda["missing_values_total"]), ("exact duplicate rows (dropped, D-005)", eda["duplicate_rows"]["total"]),
        ("time span (hours)", eda["time_span_hours"])], columns=["quantity", "value"]), [R / "eda" / "eda_summary.json"])
    an.figure("class_balance", "Class balance", R / "eda" / "class_balance.png",
              f"Fraud is {eda['fraud_percent']:.2f}% of transactions, which is why accuracy is not a headline metric.")
    shards = pd.read_csv(R / "partition" / "shard_stats.csv")
    an.table("split_and_shards", "Global split and bank shards", shards, [R / "partition" / "shard_stats.csv"])
    sh = shards[shards["part"] == "shard"].set_index("set")
    an.figure("shard_stats", "Non-IID partition: fraud rows and fraud rate per bank", R / "partition" / "shard_stats.png",
              "Explicit quotas (D-007): fraud rows " + ", ".join(f"{b[-1].upper()} {int(sh.loc[b, 'fraud'])}" for b in sh.index)
              + ". Banks A and B use Augment mode (CTGAN); Banks C and D use Schema mode (LLM).")

    # ---- B. synthetic data ----
    an.section("B. Synthetic data: generation and validation")
    val = pd.read_csv(R / "validation" / "validation_summary.csv").set_index("bank")
    rows, srcs = [], [R / "validation" / "validation_summary.csv"]
    for b in val.index:
        gen_dir = "ctgan" if val.loc[b, "mode"].startswith("Augment") else "llm"
        gs = json.loads((R / gen_dir / f"{b}_summary.json").read_text(encoding="utf-8"))
        rep = json.loads((R / "validation" / f"{b}_report.json").read_text(encoding="utf-8"))
        srcs += [R / gen_dir / f"{b}_summary.json", R / "validation" / f"{b}_report.json"]
        pat = rep.get("report_only_patterned_decimals_pct", {}).get("candidates")
        rows.append({"bank": b, "mode": val.loc[b, "mode"], "real train fraud": gs["n_real_fraud"],
                     "candidates": int(val.loc[b, "candidates"]), "mean KS (lower = closer)": gs["mean_ks_stat"],
                     "mean |corr diff|": gs["mean_abs_corr_diff"], "validated": int(val.loc[b, "validated"]),
                     "pass rate %": val.loc[b, "pass_%"], "fidelity validated": val.loc[b, "fidelity_validated"],
                     "too close to real (privacy)": int(val.loc[b, "Too close to a real row"]),
                     "patterned decimals % (report-only)": pat if pat is not None else "n/a"})
    an.table("synthetic_and_gate", "Synthetic generation and validation gate, per bank", pd.DataFrame(rows), srcs)
    an.figure("bank_a_real_vs_synthetic", "Augment mode: Bank A real fraud vs CTGAN candidates",
              R / "ctgan" / "bank_a_real_vs_synthetic.png",
              "CTGAN keeps the broad shape of real fraud; spikes at the range edges come from SDV's min/max clamping.")
    an.figure("bank_c_real_vs_synthetic", "Schema mode: Bank C real fraud vs LLM candidates",
              R / "llm" / "bank_c_real_vs_synthetic.png",
              f"The LLM received only aggregate statistics (no real rows). Mean |correlation difference| "
              f"{rows[2]['mean |corr diff|']} vs {rows[0]['mean |corr diff|']} for CTGAN (Bank A).")
    too_close = int(val["Too close to a real row"].sum())
    an.figure("gate_outcomes", "Validation gate outcome per bank", R / "validation" / "gate_outcomes.png",
              f"Same rules for both modes; {too_close} synthetic rows were rejected as near-copies of a real row.")

    # ---- C. six-arm results (from the dashboard job) ----
    an.section(f"C. Six-arm results (full run triggered from the dashboard, job `{job}`, seeds {seeds})")
    an.table("six_arm_global", "Six-arm comparison on the global test set (95 frauds), mean ± std over seeds",
             pd.read_csv(job_dir / "six_arm_global_table.csv"), [job_dir / "six_arm_global_table.csv"],
             "headline: precision, recall, F1, PR-AUC; ROC-AUC and accuracy secondary")
    an.table("local_tests", "Each bank's local test set, all six arms",
             pd.read_csv(job_dir / "local_test_table.csv"), [job_dir / "local_test_table.csv"])
    comp = pd.read_csv(job_dir / "paired_comparisons.csv")
    g = comp[(comp["eval_set"] == "global test") & comp["metric"].isin(["f1", "pr_auc"])]
    pc = pd.DataFrame({"family": g["family"], "comparison": g["comparison"], "bank": g["bank"], "metric": g["metric"],
                       "mean diff": g["mean_diff"].map("{:+.4f}".format),
                       "95% CI": [f"[{lo:+.4f}, {hi:+.4f}]" for lo, hi in zip(g["ci_low"], g["ci_high"])],
                       "seeds up": [f"{k}/{n}" for k, n in zip(g["improved_in"], g["n"])], "verdict": g["verdict"]})
    n_all = int(comp["metric"].isin(["f1", "pr_auc"]).sum())
    an.table("paired_global", "Paired comparisons on the global test set (95% CI)", pc, [job_dir / "paired_comparisons.csv"],
             f"all {n_all} F1/PR-AUC comparisons incl. local tests are in the CSV")
    an.figure("six_arm_f1_prauc", "Fraud F1 and PR-AUC by arm (mean ± std)", job_dir / "six_arm_f1_prauc.png",
              "Blue = real data only, aqua = real + validated synthetic. Error bars = std over seeds.")
    aug = comp[(comp["family"] == "augmentation") & comp["metric"].isin(["f1", "pr_auc"])]
    an.figure("paired_deltas", "Paired differences with 95% confidence intervals", job_dir / "paired_deltas.png",
              f"Blue = the interval excludes 0 (a clear difference). Augmentation: "
              f"{int((aug['verdict'] == 'no clear difference').sum())} of {len(aug)} comparisons show no clear difference.")
    _, curves = load_long(job_dir, seeds)
    c3 = np.array(curves["arm3"]).mean(axis=0)
    an.figure("convergence_seeds", "Federated convergence (global PR-AUC per round, mean ± std)", job_dir / "convergence_seeds.png",
              f"Arm 3 mean global PR-AUC: round 1 {c3[0]:.4f}, round 10 {c3[9]:.4f}, round {len(c3)} {c3[-1]:.4f}. "
              "The y-axis is zoomed.")
    fed = json.loads((R / "federated" / "arm3_federated_real_seed42.json").read_text(encoding="utf-8"))
    d = [x for x in fed["per_bank_per_round"] if x["bank"] == "bank_d"]
    an.figure("per_bank_arm3", "Per-bank view of one federated run (Arm 3, seed 42)",
              R / "federated" / "convergence_per_bank_arm3_federated_real.png",
              f"The global model's PR-AUC on Bank D's own validation set: round 1 {d[0]['val_pr_auc']:.4f}, "
              f"round {len(d)} {d[-1]['val_pr_auc']:.4f}.")

    # ---- D. integration evidence ----
    an.section("D. Integration evidence")
    ct = pd.DataFrame([(k, v) for k, v in cons["max_abs_difference_per_metric"].items()]
                      + [("rows compared (dashboard / CLI / unmatched)",
                          f"{cons['rows_dashboard']} / {cons['rows_cli']} / {cons['unmatched_rows']}")],
                      columns=["metric", "max |difference|"])
    an.table("dashboard_vs_cli", "Dashboard run vs command-line run (Step 9): identical?", ct,
             [job_dir / "long_results.csv", R / "six_arms" / "long_results.csv"], f"identical = {cons['identical']}")
    integ = integ or R / "integration"
    an.figure("full_run_finished", "The full run in the dashboard, finished", integ / "04_full_run_finished.png",
              f"Started from the browser as admin (job {job}): all six arms, seeds {seeds}, live federated rounds.")
    an.figure("results_page_light", "Six-arm results page (dashboard, light theme)",
              R / "dashboard_demo" / "02b_six_arm_results_light.png",
              "Headline fraud-class metrics in bold; accuracy and ROC-AUC greyed as secondary.")
    pm = json.loads((PROJECT_ROOT / "models" / "predict_arm3_seed42.json").read_text(encoding="utf-8"))
    an.figure("predict_page", "Test-a-transaction page", R / "dashboard_demo" / "05_predict_fraud_example.png",
              f"The federated Arm 3 model (decision threshold logit {pm['threshold_logit']}) scores a real fraud "
              "example from the global test set.")

    commit = subprocess.run(["git", "rev-parse", "--short", "HEAD"], capture_output=True, text=True,
                            cwd=PROJECT_ROOT).stdout.strip()
    now = datetime.now(timezone.utc).isoformat(timespec="seconds")
    head = (f"# FraudNet-Synth: Results Annexure\n\nGenerated {now} by `python -m ml.evaluation.annexure --job {job}` "
            f"from real run outputs (git commit `{commit}`). **No number below was typed by hand:** every table is "
            "rendered from the CSV/JSON named under it, every figure is copied byte-for-byte from the run that produced it, "
            "and every number in a caption is computed by the generator. SHA-256 checksums of all sources are in "
            "`results/final_annexure/provenance.json`.\n\nThe six-arm tables come from the **full run triggered from the "
            f"dashboard** (job `{job}`, seeds {seeds}, {jobinfo['started']} → {jobinfo['finished']} UTC). "
            f"Table A{an.n_tab} shows whether it is identical to the command-line Step 9 run: **{cons['identical']}**.\n\n"
            f"{an.n_tab} tables and {an.n_fig} figures.\n")
    DOC.write_text(head + "\n" + "\n".join(an.parts), encoding="utf-8")
    prov = {"generated_utc": now, "git_commit": commit, "dashboard_job": job, "dashboard_job_params": jobinfo["params"],
            "dashboard_job_started": jobinfo["started"], "dashboard_job_finished": jobinfo["finished"], "items": an.items}
    (OUT / "provenance.json").write_text(json.dumps(prov, indent=2), encoding="utf-8")
    (OUT / "consistency.json").write_text(json.dumps(cons, indent=2), encoding="utf-8")
    return {"job": job, "tables": an.n_tab, "figures": an.n_fig, **cons}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--job", help="dashboard job id (default: results/integration/dashboard_job_id.txt)")
    p.add_argument("--integration-dir", help="folder with that job's dashboard screenshots (default: results/integration)")
    args = p.parse_args()
    job = args.job or (PROJECT_ROOT / "results" / "integration" / "dashboard_job_id.txt").read_text().strip()
    res = build(job, PROJECT_ROOT / args.integration_dir if args.integration_dir else None)
    print(f"Annexure for job {res['job']}: {res['tables']} tables, {res['figures']} figures -> {rel(DOC)}")
    print(f"Dashboard vs CLI: {res['rows_dashboard']} vs {res['rows_cli']} rows, unmatched {res['unmatched_rows']}, "
          f"max |diff| {max(res['max_abs_difference_per_metric'].values())}, identical = {res['identical']}")


if __name__ == "__main__":
    main()
