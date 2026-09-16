"""Figure 2 underlying table: all three methods on one common set of cases.

Round 46 item 10. The three-way recoverability map claims to place fitting,
learning and conditions-only prediction on one axis, but the fitting results
supplied were parameter errors on 600 spectra while the learning curve was
log M error on 2000. This builds the table the caption promises:

  * one evaluation population -- the 600 spectra of `test_v10` the fits use;
  * one target -- log M;
  * one failure policy, stated per column (headline: failures excluded).

Fitted (omega0, Gamma) are pushed through the generator's own merit function to
give a fitted log M, so the fitting curve is on the same target as the others.

Evaluation only.
"""
from __future__ import annotations
import argparse, json, sys
from datetime import datetime, timezone
from pathlib import Path
import numpy as np, pandas as pd, torch
from scipy.stats import skewnorm

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(Path(__file__).resolve().parent))
import a4_train_v8 as A4
from a17_arch_sweep import build_variant
from sbc.data.dataset import InsSpectraDataset
from sbc.models.spectral_transformer import TargetStats, unstandardize_outputs
from sbc.models.nonlinear_conditions_mlp import normalize_conditions
from sbc.data.merit import merit
from sbc.data.latent_perturbations import (ALPHA_MU_LOG, ALPHA_SIGMA_LOG, BETA_XI2_HIGH,
    BETA_XI2_LOW, BETA_XI2_SKEW, GAMMA_FLOOR, _BETA_XI2_LOC, _BETA_XI2_SCALE)
from sbc.data.spectrum_generator import Gamma_Q, omega_Q
import dataset_paths as DS

MODELS = [("cnn_kernel45", "variant"), ("tf_patch60", "variant"), ("fusion", "base"),
          ("cnn", "base"), ("5a", "base"), ("5b", "base")]
LABEL = {"cnn_kernel45": "tuned CNN", "tf_patch60": "tuned transformer", "fusion": "fusion",
         "cnn": "1D CNN", "5a": "ST-5a", "5b": "ST-5b",
         "dho_matched": "matched-model fit",
         "dho_matched_prior": "prior-regularized forward-model fit",
         "dho_single_win": "windowed single-mode fit"}
DS_SEEDS, TR_SEEDS = (0, 1, 2), (42, 43, 44)
SEVS = (0.5, 1.0, 2.0, 4.0)


def load(model, kind, ds, sd, root):
    ck = torch.load(root / f"ds{ds}" / f"{model}_v7_seed{sd}.pt", map_location="cpu", weights_only=False)
    net = build_variant(model, sd) if kind == "variant" else A4.build_model(ck["arch"], sd)
    net.load_state_dict(ck["state_dict"]); net.eval()
    return net, TargetStats.from_dict(ck["target_stats"])


@torch.no_grad()
def pred(net, stats, X, cond=None, chunk=2048):
    o = []
    for i in range(0, len(X), chunk):
        xb = torch.from_numpy(X[i:i + chunk].astype(np.float32))
        o.append(net(xb).numpy() if cond is None else
                 net(xb, torch.from_numpy(cond[i:i + chunk].astype(np.float32))).numpy())
    return unstandardize_outputs(np.concatenate(o, 0), stats)


def merit_vec(om, gm, T, E):
    return np.array([merit(float(a), float(b), float(t), float(e))
                     for a, b, t, e in zip(om, gm, T, E)])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=600)
    ap.add_argument("--oracle-draws", type=int, default=2000)
    ap.add_argument("--seed", type=int, default=20260914)
    ap.add_argument("--outdir", type=Path, default=None)
    args = ap.parse_args()
    out = args.outdir or DS.rev("A23"); out.mkdir(parents=True, exist_ok=True)

    d = InsSpectraDataset(DS.TEST, "test", severity=1.0, as_torch=False)
    idx = np.linspace(0, len(d) - 1, min(args.n, len(d))).astype(int)   # identical to A23
    T, c, E = d._T_K[idx].astype(float), d._c_pct[idx].astype(float), d._E_kVcm[idx].astype(float)
    cond = normalize_conditions(T, c, E)
    y = np.log(np.clip(d._M[idx].astype(float), 1e-9, None))
    print(f"  common evaluation set: {len(idx)} spectra of test_v10")

    rows = []

    # ---- learning: every model, every severity, mean over the nine runs -------
    root = DS.rev("A21") / "checkpoints"
    for sev in SEVS:
        X = np.stack([d.get_augmented(int(i), sev) for i in idx]).astype(np.float32)
        for model, kind in MODELS:
            per = []
            for ds_ in DS_SEEDS:
                for sd in TR_SEEDS:
                    net, st = load(model, kind, ds_, sd, root)
                    P = pred(net, st, X, cond if model == "fusion" else None)
                    per.append(float(np.abs(np.log(np.clip(P["M"], 1e-9, None)) - y).mean()))
            rows.append({"family": "learning", "method": model, "label": LABEL[model],
                         "severity": sev, "n": len(idx), "MAE_logM": float(np.mean(per)),
                         "sd_over_runs": float(np.std(per, ddof=1)), "failure_rate": 0.0,
                         "policy": "n/a"})
        print(f"  learning s={sev} done", flush=True)

    # ---- fitting: log M from the fitted parameters, at s = 1 -----------------
    fits = pd.read_csv(out / "fit_results_test.csv")
    rng = np.random.default_rng(args.seed)
    om0, gm0 = omega_Q, Gamma_Q
    repl = {}
    for j, i in enumerate(idx):
        ob, gb = om0(float(T[j]), float(c[j]) / 100.0, float(E[j])), gm0(float(T[j]), float(c[j]) / 100.0, float(E[j]))
        ax = rng.lognormal(ALPHA_MU_LOG, ALPHA_SIGMA_LOG, args.oracle_draws) * rng.standard_normal(args.oracle_draws)
        raw = skewnorm.rvs(a=BETA_XI2_SKEW, size=args.oracle_draws, random_state=rng)
        bx2 = np.clip(raw * _BETA_XI2_SCALE + _BETA_XI2_LOC, BETA_XI2_LOW, BETA_XI2_HIGH)
        lm = np.log(np.clip(merit_vec(ob * (1.0 + bx2), gb * np.maximum(GAMMA_FLOOR, 1.0 + ax),
                                      np.full(args.oracle_draws, T[j]),
                                      np.full(args.oracle_draws, E[j])), 1e-9, None))
        repl[int(i)] = float(np.median(lm))
    oracle_pred = np.array([repl[int(i)] for i in idx])

    pos = {int(i): j for j, i in enumerate(idx)}
    for name, g in fits.groupby("method"):
        g = g.set_index("idx").loc[[int(i) for i in idx]]
        ok = g.ok.to_numpy().astype(bool)
        lm_fit = np.log(np.clip(merit_vec(g.omega0_fit.to_numpy(), g.Gamma_fit.to_numpy(),
                                          T, E), 1e-9, None))
        e_excl = np.abs(lm_fit[ok] - y[ok])
        lm_repl = np.where(ok, lm_fit, oracle_pred)
        e_repl = np.abs(lm_repl - y)
        rows.append({"family": "fitting", "method": name, "label": LABEL.get(name, name),
                     "severity": 1.0, "n": int(len(g)), "MAE_logM": float(e_excl.mean()),
                     "sd_over_runs": np.nan, "failure_rate": float(1 - ok.mean()),
                     "policy": "failures excluded (headline)"})
        rows.append({"family": "fitting", "method": name + "_replaced",
                     "label": LABEL.get(name, name) + " (failures replaced)",
                     "severity": 1.0, "n": int(len(g)), "MAE_logM": float(e_repl.mean()),
                     "sd_over_runs": np.nan, "failure_rate": float(1 - ok.mean()),
                     "policy": "failures replaced by the conditions-only median"})

    # ---- conditions-only reference on the same 600 --------------------------
    for sev in SEVS:
        rows.append({"family": "reference", "method": "conditions_only",
                     "label": "conditions only (no spectrum)", "severity": sev, "n": len(idx),
                     "MAE_logM": float(np.abs(oracle_pred - y).mean()),
                     "sd_over_runs": np.nan, "failure_rate": 0.0,
                     "policy": "n/a (never sees a spectrum, so flat in severity)"})

    t = pd.DataFrame(rows)
    t.to_csv(out / "figure2_data.csv", index=False)
    print("\n" + t[t.severity == 1.0][["family", "label", "n", "MAE_logM", "failure_rate", "policy"]]
          .round(4).to_string(index=False))
    (out / "run_sidecar_figure2.json").write_text(json.dumps({
        "task": "A31_figure2_data", "dataset_version": DS.VERSION,
        "script": str(Path(__file__).resolve().relative_to(ROOT)),
        "eval_set": str(DS.TEST.relative_to(ROOT)),
        "subset": "np.linspace(0, N-1, 600) of the test split; identical to A23",
        "target": "log M for every method",
        "fitting_logM": "generator merit() applied to the fitted (omega0, Gamma)",
        "failure_policies": ["failures excluded (headline)",
                             "failures replaced by the conditions-only conditional median"],
        "training": "none", "seed": args.seed,
        "date_utc": datetime.now(timezone.utc).isoformat()}, indent=2))
    print(f"  wrote {out}/figure2_data.csv")


if __name__ == "__main__":
    main()
