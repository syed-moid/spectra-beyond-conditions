"""A6 (auxiliary heads) and A10 (hold-out table).

A6: every trained spectral model already has omega0 and log Gamma heads (loss
weight 0.1). They are evaluated against the realized parameters rather than
training new single-target models, per round-2 A6: new models are trained only
if the heads are clearly under-trained, defined as R^2 for Gamma below that of
the two-mode fit at s = 0.25.

A10: full hold-out table over every split, with the metadata oracle, the two
metadata models that fail in opposite directions, and every spectral model.
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
import torch  # noqa: E402
from a4_eval import load_a4_model  # noqa: E402
from a17_arch_sweep import build_variant  # noqa: E402
from sbc.data.dataset import InsSpectraDataset  # noqa: E402
from sbc.models.spectral_transformer import TargetStats, unstandardize_outputs  # noqa: E402
import dataset_paths as DS

REV = ROOT / "results" / "revision"
A4E = DS.rev("A4_extended") / "checkpoints"
DATA = ROOT / "data"
NPZ = DS.FULL
SEEDS = (42, 43, 44)
SEVS = (0.25, 0.5, 1.0, 2.0, 4.0)
BASE = ["cnn", "fusion", "5a", "5b"]
TUNED = ("cnn_kernel45", "tf_patch60")   # resolved per seed via DS.tuned_ckpt
SPLITS = [("val", NPZ, "in-distribution"),
          ("holdout_c", NPZ, "interior check (c = 1%)"),
          ("holdout_c_extrap", DS.HOLDOUT_C_EXTRAP, "c-extrapolation (2.5%)"),
          ("holdout_E", NPZ, "E-extrapolation (4 kV/cm)"),
          ("holdout_T", NPZ, "T-extrapolation (600 K)")]
MONOTONE = {"val": "-", "holdout_c": "-",
            "holdout_c_extrap": "yes: Gamma0 linear in c (0.6*c*T); omega0 linear (1-0.08c)",
            "holdout_E": "yes: both quadratic and monotone in E on [0,4]",
            "holdout_T": "NO: omega0(T) is non-monotone, minimum 3.4 meV near 415 K"}


def load_all():
    models = {}
    for a in BASE:
        models[a] = [("a4", load_a4_model(a, "v7", s, ckpt_dir=A4E)) for s in SEEDS]
    for name in TUNED:
        ms = []
        for s in SEEDS:
            c = torch.load(DS.tuned_ckpt(name, s), map_location="cpu", weights_only=False)
            net = build_variant(name, s); net.load_state_dict(c["state_dict"]); net.eval()
            ms.append(("var", (net, TargetStats.from_dict(c["target_stats"]))))
        models[name] = ms
    return models


@torch.no_grad()
def predict_all(entry, X, cond):
    kind, m = entry
    if kind == "a4":
        return m.predict_all({"spectrum": X, **cond})
    net, stats = m
    o = np.concatenate([net(torch.from_numpy(X[i:i+2048].astype(np.float32))).numpy()
                        for i in range(0, len(X), 2048)], 0)
    return unstandardize_outputs(o, stats)


def metrics(t, p):
    ss = float(np.sum((t - p) ** 2)); tot = float(np.sum((t - t.mean()) ** 2))
    return float(np.abs(t - p).mean()), 1.0 - ss / tot, float(spearmanr(t, p).statistic)


def paired(a, b, rng, n=1000):
    d = a - b; i = rng.integers(0, len(d), size=(n, len(d))); m = d[i].mean(1)
    return float(d.mean()), float(np.percentile(m, 2.5)), float(np.percentile(m, 97.5))


def main():
    rng = np.random.default_rng(20260906)
    models = load_all()

    # ---------------- A6 ----------------
    stress = InsSpectraDataset(NPZ, "stress_base", severity=1.0, as_torch=False)
    cond = {"T_K": stress._T_K.astype(float), "c_pct": stress._c_pct.astype(float),
            "E_kVcm": stress._E_kVcm.astype(float)}
    om_t, gm_t = stress._omega_Q.astype(float), stress._Gamma_Q.astype(float)
    logM_t = np.log(np.clip(stress._M.astype(float), 1e-9, None))
    a6, binned = [], []
    for sev in SEVS:
        X = np.stack([stress.get_augmented(i, sev) for i in range(len(stress))]).astype(np.float32)
        for name, ms in models.items():
            outs = [predict_all(e, X, cond) for e in ms]
            om = np.mean([o["omega_Q"] for o in outs], 0)
            gm = np.mean([o["Gamma_Q"] for o in outs], 0)
            lm = np.mean([np.log(np.clip(o["M"], 1e-9, None)) for o in outs], 0)
            mo, r2o, spo = metrics(om_t, om)
            mg, r2g, spg = metrics(gm_t, gm)
            mr, r2r, spr = metrics(gm_t / om_t, gm / om)
            a6.append({"dataset_version": DS.VERSION, "model": name, "severity": sev,
                       "MAE_omega0_meV": mo, "R2_omega0": r2o, "spearman_omega0": spo,
                       "MAE_Gamma_meV": mg, "R2_Gamma": r2g, "spearman_Gamma": spg,
                       "MAE_Gamma_over_omega0": mr, "R2_Gamma_over_omega0": r2r,
                       "MAE_logM": float(np.abs(lm - logM_t).mean())})
            if sev == 1.0:
                ge = np.abs(gm - gm_t)
                q = np.quantile(ge, np.linspace(0, 1, 6)); q[-1] += 1e-9
                b = np.digitize(ge, q[1:-1])
                for k in range(5):
                    m = b == k
                    binned.append({"dataset_version": DS.VERSION, "model": name, "severity": 1.0,
                                   "gamma_err_bin": k, "n": int(m.sum()),
                                   "gamma_err_lo": float(q[k]), "gamma_err_hi": float(q[k+1]),
                                   "median_gamma_err_meV": float(np.median(ge[m])),
                                   "MAE_logM_in_bin": float(np.abs(lm - logM_t)[m].mean())})
        print(f"  A6 severity {sev} done", flush=True)
    pd.DataFrame(a6).to_csv(REV / "A6_gate.csv", index=False) if False else None
    (DS.rev("A6")).mkdir(exist_ok=True)
    pd.DataFrame(a6).to_csv(DS.rev("A6") / "aux_head_metrics.csv", index=False)
    pd.DataFrame(binned).to_csv(DS.rev("A6") / "logM_error_by_gamma_error.csv", index=False)

    # under-training gate: two-mode fit R^2 for Gamma at s = 0.25
    ex = pd.read_csv(DS.rev("A3") / "fit_results_extra_variants.csv")
    d = ex[(ex.method == "dho_two_mode") & (ex.severity == 0.25)]
    ok = d.fit_ok.astype(bool).to_numpy()
    _, r2_fit, _ = metrics(d.Gamma_true.to_numpy()[ok], d.Gamma_fit.to_numpy()[ok])
    best_head = max(x["R2_Gamma"] for x in a6 if x["severity"] == 0.25)
    print(f"\nA6 gate: two-mode fit R2(Gamma) at s=0.25 = {r2_fit:.4f}; "
          f"best aux head R2 = {best_head:.4f} -> heads "
          f"{'ADEQUATE' if best_head >= r2_fit else 'UNDER-TRAINED, train single-target models'}")

    # ---------------- A10 ----------------
    a2 = pd.read_csv(REV / "A2" / "predictions_seedavg.csv")
    a2v8 = pd.read_csv(REV / "A2_v8" / "predictions_seedavg_v8_newsets.csv")
    rows = []
    for split, npz, label in SPLITS:
        ds = InsSpectraDataset(npz, split, severity=1.0, as_torch=False)
        X = np.stack([ds.get_augmented(i, 1.0) for i in range(len(ds))]).astype(np.float32)
        y = np.log(np.clip(ds._M.astype(float), 1e-9, None))
        cd = {"T_K": ds._T_K.astype(float), "c_pct": ds._c_pct.astype(float),
              "E_kVcm": ds._E_kVcm.astype(float)}
        src = a2v8 if split == "holdout_c_extrap" else a2
        # Resolve the oracle through the ACTIVE dataset version. These were pinned
        # to A1_v8 while every output row was stamped with DS.VERSION, so running
        # this under a later version wrote v8 oracle values under a v10 label.
        # The conditions are bitwise identical across v7-v10, so the numbers
        # happened to agree -- but that is luck, not design, and the label was wrong.
        if split == "holdout_c_extrap":
            cands = [REV / f"A1_{DS.VERSION}_holdout_c_extrap", REV / "A1_v8_holdout_c_extrap"]
        else:
            cands = [DS.rev("A1"), REV / "A1_v8"]
        for orc_dir in cands:
            f = orc_dir / "per_tuple_conditional_logM.csv"
            if f.exists() and split in set(pd.read_csv(f)["split"]):
                break
        else:
            raise FileNotFoundError(
                f"no A1 oracle for split {split!r} under {DS.VERSION}; looked in "
                + ", ".join(str(c.name) for c in cands))
        if orc_dir is not cands[0]:
            print(f"  WARNING {split}: falling back to {orc_dir.name} for the oracle; "
                  f"rows will still be labelled {DS.VERSION}", flush=True)
        a1 = pd.read_csv(orc_dir / "per_tuple_conditional_logM.csv")
        orc = a1[a1.split == split]["logM_cond_median"].to_numpy()
        errs = {"metadata oracle (bound)": np.abs(orc - y)}
        for mm in ("mlp_l1", "hgb"):
            p = src[(src.model == mm) & (src.split == split)]["logM_pred_seedavg"].to_numpy()
            errs[f"metadata {mm}"] = np.abs(p - y)
        for name, ms in models.items():
            p = np.mean([np.log(np.clip(predict_all(e, X, cd)["M"], 1e-9, None)) for e in ms], 0)
            errs[name] = np.abs(p - y)
        best_meta = min(("metadata mlp_l1", "metadata hgb"), key=lambda k: errs[k].mean())
        best_spec = min(list(models), key=lambda k: errs[k].mean())
        for k, e in errs.items():
            rows.append({"dataset_version": DS.VERSION, "split": split, "split_label": label,
                         "monotone_in_extrapolated_axis": MONOTONE[split],
                         "model": k, "N": len(y), "MAE_logM": float(e.mean())})
        d, lo, hi = paired(errs[best_spec], errs[best_meta], rng)
        rows.append({"dataset_version": DS.VERSION, "split": split, "split_label": label,
                     "monotone_in_extrapolated_axis": MONOTONE[split],
                     "model": f"PAIRED {best_spec} - {best_meta}", "N": len(y),
                     "MAE_logM": d, "ci_lo": lo, "ci_hi": hi})
        print(f"  A10 {split}: best spectral {best_spec} {errs[best_spec].mean():.4f} vs "
              f"best metadata {best_meta} {errs[best_meta].mean():.4f} -> "
              f"{d:+.4f} [{lo:+.4f},{hi:+.4f}]", flush=True)
    (DS.rev("A10")).mkdir(exist_ok=True)
    pd.DataFrame(rows).to_csv(DS.rev("A10") / "holdout_table.csv", index=False)
    for d_ in (DS.rev("A6"), DS.rev("A10")):
        (d_ / "run_sidecar.json").write_text(json.dumps({
            "dataset_version": DS.VERSION, "seeds": list(SEEDS),
            "date_utc": datetime.now(timezone.utc).isoformat()}, indent=2))
    print("\nwrote A6 and A10")


if __name__ == "__main__":
    main()
