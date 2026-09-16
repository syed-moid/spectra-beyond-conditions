"""A14 item 2 - empirical vs Monte-Carlo conditional median of log M.

The replicate set gives 20 realizations per condition tuple, so the conditional
median of log M can be measured directly and compared to the A1 Monte-Carlo
estimate (N = 2000 draws) at the same tuples. Agreement validates the A1 oracle
construction empirically.
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(Path(__file__).resolve().parent))
from a1_oracle_floor import conditional_logM, config_hash, git_sha  # noqa: E402
import dataset_paths as DS

NPZ = DS.REPLICATE
OUTDIR = DS.rev("A14")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=20260905)
    ap.add_argument("--n-draws", type=int, default=2000)
    ap.add_argument("--outdir", type=Path, default=OUTDIR)
    args = ap.parse_args()
    args.outdir.mkdir(parents=True, exist_ok=True)

    z = np.load(NPZ, allow_pickle=False)
    tid = z["tuple_id"]
    logM = np.log(np.clip(z["M"].astype(float), 1e-9, None))
    T, c, E = z["T_K"].astype(float), z["c_pct"].astype(float), z["E_kVcm"].astype(float)
    rng = np.random.default_rng(args.seed)

    rows = []
    for t in np.unique(tid):
        m = tid == t
        lm = logM[m]
        mc, om0, gm0 = conditional_logM(float(T[m][0]), float(c[m][0]), float(E[m][0]),
                                        args.n_draws, rng)
        rows.append({
            "dataset_version": DS.VERSION, "tuple_id": int(t), "n_draws_empirical": int(m.sum()),
            "T_K": float(T[m][0]), "c_pct": float(c[m][0]), "E_kVcm": float(E[m][0]),
            "omega0_baseline_meV": float(om0), "Gamma0_baseline_meV": float(gm0),
            "emp_median_logM": float(np.median(lm)), "mc_median_logM": float(np.median(mc)),
            "emp_mean_logM": float(lm.mean()), "mc_mean_logM": float(mc.mean()),
            "emp_sd_logM": float(lm.std(ddof=1)), "mc_sd_logM": float(mc.std()),
            "emp_iqr_logM": float(np.subtract(*np.percentile(lm, [75, 25]))),
            "mc_iqr_logM": float(np.subtract(*np.percentile(mc, [75, 25]))),
        })
    df = pd.DataFrame(rows)
    df["median_diff"] = df.emp_median_logM - df.mc_median_logM
    df["sd_diff"] = df.emp_sd_logM - df.mc_sd_logM
    df.to_csv(args.outdir / "replicate_vs_mc_conditional.csv", index=False)

    # Expected sampling scatter of a 20-sample median around the true median.
    exp_se = float(np.mean(df.mc_sd_logM * 1.2533 / np.sqrt(20)))
    summary = {
        "task": "A14_data", "dataset_version": DS.VERSION, "n_tuples": int(len(df)),
        "n_draws_per_tuple": 20, "mc_n_draws": args.n_draws,
        "median_diff_mean": float(df.median_diff.mean()),
        "median_diff_sd": float(df.median_diff.std(ddof=1)),
        "median_diff_p5_p95": [float(df.median_diff.quantile(.05)),
                               float(df.median_diff.quantile(.95))],
        "expected_se_of_20_sample_median": exp_se,
        "corr_emp_vs_mc_median": float(np.corrcoef(df.emp_median_logM, df.mc_median_logM)[0, 1]),
        "sd_diff_mean": float(df.sd_diff.mean()),
        "sd_ratio_mean": float((df.emp_sd_logM / df.mc_sd_logM).mean()),
        "mean_mc_sd_logM": float(df.mc_sd_logM.mean()),
        "mean_emp_sd_logM": float(df.emp_sd_logM.mean()),
    }
    (args.outdir / "replicate_vs_mc_summary.json").write_text(json.dumps(summary, indent=2))
    for k, v in summary.items():
        print(f"  {k}: {v}")

    (args.outdir / "run_sidecar.json").write_text(json.dumps({
        "task": "A14_data", "script": str(Path(__file__).relative_to(ROOT)),
        "seed": args.seed, "n_draws": args.n_draws, "dataset_version": DS.VERSION,
        "dataset": str(NPZ.relative_to(ROOT)), "config_hash_sha256_16": config_hash(),
        "git_commit": git_sha(), "date_utc": datetime.now(timezone.utc).isoformat(),
        "numpy": np.__version__, "pandas": pd.__version__}, indent=2))


if __name__ == "__main__":
    main()
