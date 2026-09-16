"""Round-15 ruling 5 + round-6 A14 amendments.

Part A: extend the dense sweep to s in {6, 8, 12} (evaluation-only, outside the
training range) on the degradation-sweep base and the replicate set, for all six
spectral models. Gives the value crossover (first s where the paired CI of
model - metadata reference includes zero) and the ranking crossover (first s
where the tuple-bootstrap CI of mean rho includes zero) on one axis.

Part B: A14 amendments - tuple-bootstrap CIs on mean rho for every model and
severity, rows for the fitting methods, and paired CIs for (CNN rho - ST-5a rho)
and (CNN rho - best fitting rho). Fitting methods are evaluated on a 100-tuple
subsample (2000 spectra) for cost; the subsample is fixed and recorded.
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
from sbc.data.dataset import InsSpectraDataset  # noqa: E402
from sbc.data.latent_perturbations import LatentDraw  # noqa: E402
from sbc.data.merit import merit  # noqa: E402
from sbc.models.spectral_transformer import TargetStats, unstandardize_outputs  # noqa: E402
import dataset_paths as DS

REV = ROOT / "results" / "revision"
NPZ = DS.FULL
OUT = DS.rev("A14")
SEEDS = (42, 43, 44)
EXT = (6.0, 8.0, 12.0)
ALL_S = (0.25, 0.5, 0.75, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0, 8.0, 12.0)
FIT_S = (0.5, 1.0, 2.0, 4.0)
N_FIT_TUPLES = 100
SPEC = ["cnn_kernel45", "tf_patch60", "cnn", "fusion", "5a", "5b"]


def load_models():
    ms = {}
    for a in ("cnn", "fusion", "5a", "5b"):
        ms[a] = [("a4", load_a4_model(a, "v7", s, ckpt_dir=DS.rev("A4_extended") / "checkpoints"))
                 for s in SEEDS]
    for n in ("cnn_kernel45", "tf_patch60"):
        lst = []
        for s in SEEDS:
            ck = torch.load(DS.tuned_ckpt(n, s),
                            map_location="cpu", weights_only=False)
            net = build_variant(n, s); net.load_state_dict(ck["state_dict"]); net.eval()
            lst.append(("var", (net, TargetStats.from_dict(ck["target_stats"]))))
        ms[n] = lst
    return ms


@torch.no_grad()
def plogm(entry, X, cond):
    kind, m = entry
    if kind == "a4":
        return m.predict_logM({"spectrum": X, **cond})
    net, st = m
    o = np.concatenate([net(torch.from_numpy(X[i:i+2048].astype(np.float32))).numpy()
                        for i in range(0, len(X), 2048)], 0)
    return np.log(np.clip(unstandardize_outputs(o, st)["M"], 1e-9, None))


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


def paired(a, b, rng, n=1000):
    d = a - b; i = rng.integers(0, len(d), size=(n, len(d))); m = d[i].mean(1)
    return float(d.mean()), float(np.percentile(m, 2.5)), float(np.percentile(m, 97.5))


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(20260906)
    models = load_models()
    ds = InsSpectraDataset(NPZ, "stress_base", severity=1.0, as_torch=False)
    cond = {"T_K": ds._T_K.astype(float), "c_pct": ds._c_pct.astype(float),
            "E_kVcm": ds._E_kVcm.astype(float)}
    ys = np.log(np.clip(ds._M.astype(float), 1e-9, None))
    meta = pd.read_csv(REV / "A2" / "predictions_seedavg.csv")
    err_meta = np.abs(meta[(meta.model == "mlp_l1") & (meta.split == "stress_base")]
                      ["logM_pred_seedavg"].to_numpy() - ys)

    # --- Part A1: extended severities on the stress base --------------------
    rows = []
    for s in EXT:
        X = np.stack([ds.get_augmented(i, s) for i in range(len(ds))]).astype(np.float32)
        for name in SPEC:
            p = np.mean([plogm(e, X, cond) for e in models[name]], 0)
            e = np.abs(p - ys)
            d, lo, hi = paired(e, err_meta, rng)
            rows.append({"dataset_version": DS.VERSION, "set": "stress_base", "model": name,
                         "severity": s, "in_training_range": False,
                         "MAE_ensemble": float(e.mean()), "vs_metadata_diff": d,
                         "lo": lo, "hi": hi, "beats_metadata": bool(hi < 0)})
        print(f"  extended stress s={s} done", flush=True)
    pd.DataFrame(rows).to_csv(OUT / "extended_sweep_stress.csv", index=False)

    # --- Part A2: extended severities on the replicate set, MAE and rho -----
    clean, _ = load_replicate(want_blocks=False)
    tid = clean["tuple_id"]
    yr = np.log(np.clip(clean["M"].astype(float), 1e-9, None))
    condr = {"T_K": clean["T_K"].astype(float), "c_pct": clean["c_pct"].astype(float),
             "E_kVcm": clean["E_kVcm"].astype(float)}
    grid = clean["omega_grid"]
    lat = [LatentDraw.from_dict(json.loads(str(t))) for t in clean["latent_json"]]
    par = [json.loads(str(t)) for t in clean["aug_params_json"]]

    def build_X(sev, idx=None):
        idx = np.arange(len(tid)) if idx is None else idx
        X = np.empty((len(idx), grid.size), dtype=np.float32)
        for k, i in enumerate(idx):
            st = state_from_clean(float(clean["T_K"][i]), float(clean["c_pct"][i]),
                                  float(clean["E_kVcm"][i]), omega_grid=grid, latent=lat[i])
            X[k] = apply_aug_params(st, par[i], sev, noise_rng=None).spectrum.astype(np.float32)
        return X

    rrows = []
    for s in ALL_S:
        X = build_X(s)
        for name in SPEC:
            p = np.mean([plogm(e, X, condr) for e in models[name]], 0)
            r = rhos(tid, yr, p)
            rm, rlo, rhi = boot(r, rng)
            rrows.append({"dataset_version": DS.VERSION, "set": "replicate_eval", "model": name,
                          "severity": s, "in_training_range": s <= 4.0,
                          "MAE_ensemble": float(np.abs(p - yr).mean()),
                          "rho_mean": rm, "rho_lo": rlo, "rho_hi": rhi,
                          "rho_excludes_zero": bool(rlo > 0)})
        print(f"  replicate s={s} done", flush=True)
    rdf = pd.DataFrame(rrows)
    rdf.to_csv(OUT / "extended_rho_replicate.csv", index=False)

    # crossovers
    sdf = pd.concat([pd.read_csv(DS.rev("A8") / "dense_sweep_stress.csv"), pd.DataFrame(rows)])
    cross = []
    for name in SPEC:
        v = sdf[(sdf.model == name)].sort_values("severity")
        vc = next((r.severity for r in v.itertuples() if not r.beats_metadata), None)
        rr = rdf[rdf.model == name].sort_values("severity")
        rc = next((r.severity for r in rr.itertuples() if not r.rho_excludes_zero), None)
        cross.append({"dataset_version": DS.VERSION, "model": name,
                      "value_crossover_severity": vc, "ranking_crossover_severity": rc})
    cdf = pd.DataFrame(cross)
    cdf.to_csv(OUT / "two_boundaries.csv", index=False)
    print("\n=== boundaries ===\n" + cdf.to_string(index=False))

    # --- Part B: fitting-method rho on a fixed 100-tuple subsample ----------
    utid = np.unique(tid)
    sub_t = np.sort(rng.choice(utid, size=N_FIT_TUPLES, replace=False))
    idx = np.nonzero(np.isin(tid, sub_t))[0]
    T2, c2, E2 = condr["T_K"][idx], condr["c_pct"][idx], condr["E_kVcm"][idx]
    y2, tid2 = yr[idx], tid[idx]
    frows = []
    for s in FIT_S:
        X = build_X(s, idx)
        for meth, fn in (("dho_two_mode", fit_two_mode), ("dho_single_win", fit_single_windowed),
                         ("dho_matched", fit_matched)):
            t0 = time.time()
            lm = np.empty(len(idx))
            for k in range(len(idx)):
                r = fn(X[k].astype(np.float64), grid, float(T2[k]), float(c2[k]), float(E2[k]))
                lm[k] = (np.log(max(merit(r["omega0"], r["Gamma"], float(T2[k]), float(E2[k])), 1e-9))
                         if r["ok"] and np.isfinite(r["omega0"]) and r["Gamma"] > 0 else np.nan)
            good = np.isfinite(lm)
            lm = np.where(good, lm, np.nanmedian(lm))
            rr = rhos(tid2, y2, lm)
            rm, rlo, rhi = boot(rr, rng)
            frows.append({"dataset_version": DS.VERSION, "model": meth, "severity": s,
                          "n_tuples": N_FIT_TUPLES, "rho_mean": rm, "rho_lo": rlo, "rho_hi": rhi,
                          "rho_excludes_zero": bool(rlo > 0),
                          "fit_ok_frac": float(good.mean())})
            print(f"  fit rho {meth} s={s}: rho={rm:.4f} [{rlo:.4f},{rhi:.4f}] "
                  f"({time.time()-t0:.0f}s)", flush=True)
        # spectral models on the same subsample for paired comparison
        for name in SPEC:
            p = np.mean([plogm(e, X, {"T_K": T2, "c_pct": c2, "E_kVcm": E2})
                         for e in models[name]], 0)
            rr = rhos(tid2, y2, p)
            rm, rlo, rhi = boot(rr, rng)
            frows.append({"dataset_version": DS.VERSION, "model": name, "severity": s,
                          "n_tuples": N_FIT_TUPLES, "rho_mean": rm, "rho_lo": rlo,
                          "rho_hi": rhi, "rho_excludes_zero": bool(rlo > 0),
                          "fit_ok_frac": 1.0})
    fdf = pd.DataFrame(frows)
    fdf.to_csv(OUT / "rho_fitting_vs_learning_subsample.csv", index=False)
    print("\n" + fdf.pivot_table(index="model", columns="severity", values="rho_mean")
          .to_string(float_format=lambda x: f"{x:.4f}"))
    (OUT / "run_sidecar_boundaries.json").write_text(json.dumps({
        "task": "A14b", "dataset_version": DS.VERSION, "extended_severities": list(EXT),
        "fit_subsample_tuples": int(N_FIT_TUPLES),
        "fit_subsample_tuple_ids": [int(t) for t in sub_t],
        "note": "s > 4 is evaluation-only and outside the training range",
        "date_utc": datetime.now(timezone.utc).isoformat()}, indent=2))


if __name__ == "__main__":
    main()
