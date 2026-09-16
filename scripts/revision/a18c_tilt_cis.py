"""A18b confidence intervals (round-9 decisions 2 and 3).

(2) Asymmetry: paired bootstrap over val spectra of
        dMAE(-beta) - dMAE(+beta)
    for each base model at |beta| = 0.10 and 0.20. Claimed only if the CI
    excludes zero AND the sign is consistent across >= 3 of the 4 models.

(3) MAE-vs-rho opposition: tuple bootstrap (1000 resamples over the 250 tuples)
    of d_rho at beta = +-0.20 for the CNN.
"""

from __future__ import annotations

import json, sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(Path(__file__).resolve().parent))
from a4_eval import load_a4_model  # noqa: E402
from replicate_io import block_key, load_replicate  # noqa: E402
from sbc.data.dataset import InsSpectraDataset  # noqa: E402
import dataset_paths as DS

NPZ = DS.FULL
A4E = DS.rev("A4_extended") / "checkpoints"
OUT = DS.rev("A18b")
SEEDS = (42, 43, 44)
N_BOOT = 1000


def tuple_rhos(tid, y, pred):
    out = []
    for t in np.unique(tid):
        m = tid == t
        r = spearmanr(y[m], pred[m]).statistic if np.ptp(pred[m]) > 0 else 0.0
        out.append(0.0 if not np.isfinite(r) else float(r))
    return np.array(out)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(20260906)
    ds = InsSpectraDataset(NPZ, "val", severity=1.0, as_torch=False)
    X = np.stack([ds.get_augmented(i, 1.0) for i in range(len(ds))]).astype(np.float32)
    y = np.log(np.clip(ds._M.astype(float), 1e-9, None))
    cond = {"T_K": ds._T_K.astype(float), "c_pct": ds._c_pct.astype(float),
            "E_kVcm": ds._E_kVcm.astype(float)}
    grid = ds._omega_grid

    clean, blocks = load_replicate()
    Xr, tid = blocks[block_key(1.0)], clean["tuple_id"]
    yr = np.log(np.clip(clean["M"].astype(float), 1e-9, None))
    condr = {"T_K": clean["T_K"].astype(float), "c_pct": clean["c_pct"].astype(float),
             "E_kVcm": clean["E_kVcm"].astype(float)}

    def tilt(b):
        return (1.0 + b * (grid / 15.0)).astype(np.float32)

    rows, rho_rows = [], []
    for arch in ("cnn", "5a", "5b", "fusion"):
        ms = [load_a4_model(arch, "v7", s, ckpt_dir=A4E) for s in SEEDS]
        err = {}
        for b in (0.0, -0.10, 0.10, -0.20, 0.20):
            p = np.mean([m.predict_logM({"spectrum": (X * tilt(b)).astype(np.float32), **cond})
                         for m in ms], axis=0)
            err[b] = np.abs(p - y)
        for mag in (0.10, 0.20):
            d = (err[-mag] - err[0.0]) - (err[mag] - err[0.0])   # = err[-mag] - err[+mag]
            idx = rng.integers(0, len(d), size=(N_BOOT, len(d)))
            bm = d[idx].mean(1)
            lo, hi = np.percentile(bm, [2.5, 97.5])
            rows.append({"dataset_version": DS.VERSION, "model": arch, "abs_beta": mag,
                         "dMAE_neg": float(err[-mag].mean() - err[0.0].mean()),
                         "dMAE_pos": float(err[mag].mean() - err[0.0].mean()),
                         "asym_neg_minus_pos": float(d.mean()),
                         "lo": float(lo), "hi": float(hi),
                         "excludes_zero": bool(hi < 0 or lo > 0)})
            print(f"  {arch:7s} |beta|={mag:.2f}  dMAE(-)={rows[-1]['dMAE_neg']:+.4f} "
                  f"dMAE(+)={rows[-1]['dMAE_pos']:+.4f}  asym={d.mean():+.4f} "
                  f"[{lo:+.4f},{hi:+.4f}] {'SIG' if rows[-1]['excludes_zero'] else 'ns'}")

        if arch == "cnn":
            base_r = tuple_rhos(tid, yr, np.mean(
                [m.predict_logM({"spectrum": Xr, **condr}) for m in ms], axis=0))
            for b in (-0.20, 0.20):
                pr = np.mean([m.predict_logM(
                    {"spectrum": (Xr * tilt(b)).astype(np.float32), **condr}) for m in ms], axis=0)
                d = tuple_rhos(tid, yr, pr) - base_r
                idx = rng.integers(0, len(d), size=(N_BOOT, len(d)))
                bm = d[idx].mean(1)
                lo, hi = np.percentile(bm, [2.5, 97.5])
                rho_rows.append({"dataset_version": DS.VERSION, "model": arch, "beta": b,
                                 "d_rho": float(d.mean()), "lo": float(lo), "hi": float(hi),
                                 "excludes_zero": bool(hi < 0 or lo > 0)})
                print(f"  cnn rho beta={b:+.2f}: d_rho {d.mean():+.4f} [{lo:+.4f},{hi:+.4f}] "
                      f"{'SIG' if rho_rows[-1]['excludes_zero'] else 'ns'}")

    pd.DataFrame(rows).to_csv(OUT / "tilt_asymmetry_cis.csv", index=False)
    pd.DataFrame(rho_rows).to_csv(OUT / "tilt_rho_cis.csv", index=False)
    df = pd.DataFrame(rows)
    for mag in (0.10, 0.20):
        sub = df[df.abs_beta == mag]
        n_sig = int(sub.excludes_zero.sum())
        signs = np.sign(sub.asym_neg_minus_pos)
        consistent = int(max((signs > 0).sum(), (signs < 0).sum()))
        print(f"\n|beta|={mag}: {n_sig}/4 models significant; largest consistent sign group "
              f"{consistent}/4 -> asymmetry claim "
              f"{'SUPPORTED' if n_sig >= 3 and consistent >= 3 else 'NOT SUPPORTED as a rule'}")
    (OUT / "run_sidecar_cis.json").write_text(json.dumps({
        "task": "A18c", "n_boot": N_BOOT, "seeds": list(SEEDS),
        "date_utc": datetime.now(timezone.utc).isoformat()}, indent=2))


if __name__ == "__main__":
    main()
