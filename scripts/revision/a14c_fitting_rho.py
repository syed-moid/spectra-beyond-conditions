"""A14 Part B (standalone, incremental): within-tuple rho for fitting methods.

Fitting methods produce a per-spectrum estimate and therefore have a non-trivial
within-condition ranking score, unlike metadata models which are exactly 0. This
is the fairest single comparison between fitting and learning. Evaluated on a
fixed 100-tuple subsample (2000 spectra) for cost; the tuple ids are recorded.

Each (method, severity) cell is appended to the CSV as soon as it completes.
"""

from __future__ import annotations

import json, sys, time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(Path(__file__).resolve().parent))
import torch  # noqa: E402
from a3_fitting_baselines import fit_matched  # noqa: E402
from a3b_extra_baselines import fit_single_windowed, fit_two_mode  # noqa: E402
from a4_eval import load_a4_model  # noqa: E402
from a17_arch_sweep import build_variant  # noqa: E402
from replicate_io import load_replicate  # noqa: E402
from sbc.data.augmentations import apply_aug_params, state_from_clean  # noqa: E402
from sbc.data.latent_perturbations import LatentDraw  # noqa: E402
from sbc.data.merit import merit  # noqa: E402
from sbc.models.spectral_transformer import TargetStats, unstandardize_outputs  # noqa: E402
import dataset_paths as DS

REV = ROOT / "results" / "revision"
OUT = DS.rev("A14") / "rho_fitting_vs_learning_subsample.csv"
SEEDS = (42, 43, 44)
FIT_S = (0.5, 1.0, 2.0, 4.0)   # override with --severities
N_TUP = 100
SPEC = ["cnn_kernel45", "tf_patch60", "cnn", "fusion", "5a", "5b"]


def rhos(tid, y, p):
    out = []
    for t in np.unique(tid):
        m = tid == t
        r = spearmanr(y[m], p[m]).statistic if np.ptp(p[m]) > 0 else 0.0
        out.append(0.0 if not np.isfinite(r) else float(r))
    return np.array(out)


def boot(v, rng, n=1000):
    i = rng.integers(0, len(v), size=(n, len(v))); m = v[i].mean(1)
    return float(v.mean()), float(np.percentile(m, 2.5)), float(np.percentile(m, 97.5))


def append(row):
    df = pd.DataFrame([row])
    df.to_csv(OUT, mode="a", header=not OUT.exists(), index=False)


def main():
    OUT.parent.mkdir(parents=True, exist_ok=True)
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--severities", type=float, nargs="+", default=list(FIT_S))
    args = ap.parse_args()
    sev_list = args.severities
    rng = np.random.default_rng(20260906)
    clean, _ = load_replicate(want_blocks=False)
    tid = clean["tuple_id"]
    yr = np.log(np.clip(clean["M"].astype(float), 1e-9, None))
    grid = clean["omega_grid"]
    lat = [LatentDraw.from_dict(json.loads(str(t))) for t in clean["latent_json"]]
    par = [json.loads(str(t)) for t in clean["aug_params_json"]]
    # fixed subsample, selected here with a recorded seed (the A14b sidecar was
    # never written because that run died before its final write)
    sub_rng = np.random.default_rng(20260906)
    sub_t = np.sort(sub_rng.choice(np.unique(tid), size=N_TUP, replace=False))
    (DS.rev("A14") / "run_sidecar_fitting_subsample.json").write_text(json.dumps(
        {"subsample_seed": 20260906, "n_tuples": int(N_TUP),
         "tuple_ids": [int(t) for t in sub_t]}, indent=2))
    idx = np.nonzero(np.isin(tid, sub_t))[0]
    T2 = clean["T_K"][idx].astype(float); c2 = clean["c_pct"][idx].astype(float)
    E2 = clean["E_kVcm"][idx].astype(float)
    y2, tid2 = yr[idx], tid[idx]
    cond2 = {"T_K": T2, "c_pct": c2, "E_kVcm": E2}

    models = {}
    for a in ("cnn", "fusion", "5a", "5b"):
        models[a] = [("a4", load_a4_model(a, "v7", s, ckpt_dir=DS.rev("A4_extended") / "checkpoints"))
                     for s in SEEDS]
    for n in ("cnn_kernel45", "tf_patch60"):
        lst = []
        for s in SEEDS:
            ck = torch.load(DS.tuned_ckpt(n, s),
                            map_location="cpu", weights_only=False)
            net = build_variant(n, s); net.load_state_dict(ck["state_dict"]); net.eval()
            lst.append(("var", (net, TargetStats.from_dict(ck["target_stats"]))))
        models[n] = lst

    @torch.no_grad()
    def plogm(e, X):
        kind, m = e
        if kind == "a4":
            return m.predict_logM({"spectrum": X, **cond2})
        net, st = m
        o = np.concatenate([net(torch.from_numpy(X[i:i+2048].astype(np.float32))).numpy()
                            for i in range(0, len(X), 2048)], 0)
        return np.log(np.clip(unstandardize_outputs(o, st)["M"], 1e-9, None))

    for s in sev_list:
        X = np.empty((len(idx), grid.size), dtype=np.float64)
        for k, i in enumerate(idx):
            st = state_from_clean(float(clean["T_K"][i]), float(clean["c_pct"][i]),
                                  float(clean["E_kVcm"][i]), omega_grid=grid, latent=lat[i])
            X[k] = apply_aug_params(st, par[i], s, noise_rng=None).spectrum
        Xf = X.astype(np.float32)
        for name in SPEC:
            p = np.mean([plogm(e, Xf) for e in models[name]], 0)
            rm, lo, hi = boot(rhos(tid2, y2, p), rng)
            append({"dataset_version": DS.VERSION, "kind": "learned", "model": name, "severity": s,
                    "n_tuples": N_TUP, "rho_mean": rm, "rho_lo": lo, "rho_hi": hi,
                    "estimate_ok_frac": 1.0})
        for meth, fn in (("dho_single_win", fit_single_windowed), ("dho_two_mode", fit_two_mode),
                         ("dho_matched", fit_matched)):
            t0 = time.time()
            lm = np.full(len(idx), np.nan)
            for k in range(len(idx)):
                r = fn(X[k], grid, float(T2[k]), float(c2[k]), float(E2[k]))
                if r["ok"] and np.isfinite(r["omega0"]) and r["Gamma"] > 0:
                    lm[k] = np.log(max(merit(r["omega0"], r["Gamma"], float(T2[k]), float(E2[k])),
                                       1e-9))
            good = np.isfinite(lm)
            lmf = np.where(good, lm, np.nanmedian(lm))
            rm, lo, hi = boot(rhos(tid2, y2, lmf), rng)
            append({"dataset_version": DS.VERSION, "kind": "fitted", "model": meth, "severity": s,
                    "n_tuples": N_TUP, "rho_mean": rm, "rho_lo": lo, "rho_hi": hi,
                    "estimate_ok_frac": float(good.mean())})
            print(f"  {meth} s={s}: rho {rm:.4f} [{lo:.4f},{hi:.4f}] ok={good.mean():.2f} "
                  f"({time.time()-t0:.0f}s)", flush=True)
    (DS.rev("A14") / "run_sidecar_fitting_rho.json").write_text(json.dumps({
        "task": "A14c", "n_tuples": N_TUP, "severities": list(FIT_S),
        "date_utc": datetime.now(timezone.utc).isoformat()}, indent=2))
    print("done")


if __name__ == "__main__":
    main()
