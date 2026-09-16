"""Network linewidth recovery on the exact spectra the fitting baselines used.

Round 45 §1: every table must be readable from the output it cites. The §3.6
comparison puts network rows beside fit rows, so both must come from one file
evaluated on one subset. This script evaluates the A21 checkpoints on the
600-spectrum subset A23 fits (`np.linspace(0, N-1, 600)` of the untouched test
set at s = 1) and writes the combined table.

A conditions-only row is included so the comparison has a floor: what the
measurement conditions alone imply about omega0 and Gamma, with no spectrum.

Evaluation only; no training.
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
from sbc.data.latent_perturbations import (ALPHA_MU_LOG, ALPHA_SIGMA_LOG, BETA_XI2_HIGH,
    BETA_XI2_LOW, BETA_XI2_SKEW, GAMMA_FLOOR, _BETA_XI2_LOC, _BETA_XI2_SCALE)
from sbc.data.spectrum_generator import Gamma_Q, omega_Q
import dataset_paths as DS

MODELS = [("cnn_kernel45","variant"),("tf_patch60","variant"),("fusion","base"),
          ("cnn","base"),("5a","base"),("5b","base")]
LABEL = {"cnn_kernel45":"tuned CNN (kernel 45)","tf_patch60":"tuned transformer (patch 60)",
         "fusion":"fusion (spectrum + conditions)","cnn":"CNN (baseline)",
         "5a":"ST-5a","5b":"ST-5b"}
DS_SEEDS, TR_SEEDS = (0,1,2), (42,43,44)


def load(model, kind, ds, sd, root):
    ck = torch.load(root/f"ds{ds}"/f"{model}_v7_seed{sd}.pt", map_location="cpu", weights_only=False)
    net = build_variant(model, sd) if kind=="variant" else A4.build_model(ck["arch"], sd)
    net.load_state_dict(ck["state_dict"]); net.eval()
    return net, TargetStats.from_dict(ck["target_stats"])


@torch.no_grad()
def pred(net, stats, X, cond=None, chunk=2048):
    o=[]
    for i in range(0,len(X),chunk):
        xb=torch.from_numpy(X[i:i+chunk].astype(np.float32))
        o.append(net(xb).numpy() if cond is None else
                 net(xb, torch.from_numpy(cond[i:i+chunk].astype(np.float32))).numpy())
    return unstandardize_outputs(np.concatenate(o,0), stats)


def conditional_params(T, c, E, n, rng):
    """Monte Carlo conditional median of (omega0, Gamma) given conditions only."""
    om0, gm0 = omega_Q(T, c/100.0, E), Gamma_Q(T, c/100.0, E)
    ax = rng.lognormal(ALPHA_MU_LOG, ALPHA_SIGMA_LOG, n)*rng.standard_normal(n)
    raw = skewnorm.rvs(a=BETA_XI2_SKEW, size=n, random_state=rng)
    bx2 = np.clip(raw*_BETA_XI2_SCALE+_BETA_XI2_LOC, BETA_XI2_LOW, BETA_XI2_HIGH)
    return float(np.median(om0*(1.0+bx2))), float(np.median(gm0*np.maximum(GAMMA_FLOOR,1.0+ax)))


def stats_row(om_p, gm_p, om_t, gm_t, over):
    e = np.abs(gm_p-gm_t)
    return {"MAE_omega0": float(np.abs(om_p-om_t).mean()), "MAE_Gamma": float(e.mean()),
            "median_abs_err_Gamma": float(np.median(e)),
            "MAE_Gamma_overdamped": float(e[over].mean())}


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=600)
    ap.add_argument("--severity", type=float, default=1.0)
    ap.add_argument("--oracle-draws", type=int, default=2000)
    ap.add_argument("--seed", type=int, default=20260913)
    ap.add_argument("--outdir", type=Path, default=None)
    args=ap.parse_args()
    out=args.outdir or DS.rev("A23"); out.mkdir(parents=True, exist_ok=True)
    root=DS.rev("A21")/"checkpoints"

    d=InsSpectraDataset(DS.TEST,"test",severity=args.severity,as_torch=False)
    idx=np.linspace(0,len(d)-1,min(args.n,len(d))).astype(int)      # identical to A23
    X=np.stack([d.get_augmented(int(i),args.severity) for i in idx]).astype(np.float32)
    T,c,E = d._T_K[idx].astype(float), d._c_pct[idx].astype(float), d._E_kVcm[idx].astype(float)
    cond=normalize_conditions(T,c,E)
    om_t=d._omega_Q[idx].astype(float); gm_t=d._Gamma_Q[idx].astype(float)
    over=gm_t>om_t
    print(f"  n={len(idx)}  overdamped {over.mean():.1%}  (Gamma > omega0)")

    rows=[]
    for model,kind in MODELS:
        for ds in DS_SEEDS:
            for sd in TR_SEEDS:
                net,st=load(model,kind,ds,sd,root)
                P=pred(net,st,X, cond if model=="fusion" else None)
                rows.append({"dataset_version":DS.VERSION,"method":model,"label":LABEL[model],
                             "family":"network","dataset_seed":ds,"seed":sd,"n":len(idx),
                             "failure_rate":0.0,
                             **stats_row(P["omega_Q"],P["Gamma_Q"],om_t,gm_t,over),
                             "n_overdamped":int(over.sum())})
        print(f"  {model:14s} 9 runs done",flush=True)
    runs=pd.DataFrame(rows); runs.to_csv(out/"network_runs_test.csv",index=False)

    rng=np.random.default_rng(args.seed)
    cm=np.array([conditional_params(float(T[i]),float(c[i]),float(E[i]),args.oracle_draws,rng)
                 for i in range(len(idx))])

    agg=(runs.groupby(["method","label"],as_index=False)
             .agg(n=("n","first"),failure_rate=("failure_rate","first"),
                  MAE_omega0=("MAE_omega0","mean"),MAE_Gamma=("MAE_Gamma","mean"),
                  sd_MAE_Gamma=("MAE_Gamma","std"),
                  median_abs_err_Gamma=("median_abs_err_Gamma","mean"),
                  MAE_Gamma_overdamped=("MAE_Gamma_overdamped","mean"),
                  n_overdamped=("n_overdamped","first")))
    agg["family"]="network"; agg["aggregation"]="mean over 9 runs (3 realizations x 3 training seeds)"

    fit=pd.read_csv(out/"fit_summary_test.csv")
    fl={"dho_matched":"matched-model fit",
        "dho_matched_prior":"prior-regularized forward-model fit",
        "dho_single_win":"windowed single-mode fit"}
    fit["label"]=fit.method.map(fl); fit["family"]="fit"; fit["sd_MAE_Gamma"]=np.nan
    fit["aggregation"]="single pass over the 600 spectra"

    oc=pd.DataFrame([{"method":"conditions_only","label":"conditions only (no spectrum)",
        "family":"reference","n":len(idx),"failure_rate":0.0,
        **stats_row(cm[:,0],cm[:,1],om_t,gm_t,over),"n_overdamped":int(over.sum()),
        "sd_MAE_Gamma":np.nan,
        "aggregation":f"Monte Carlo conditional median, {args.oracle_draws} draws per spectrum"}])

    cols=["family","method","label","n","failure_rate","MAE_omega0","MAE_Gamma","sd_MAE_Gamma",
          "median_abs_err_Gamma","MAE_Gamma_overdamped","n_overdamped","aggregation"]
    comb=pd.concat([agg,fit,oc],ignore_index=True)[cols].sort_values(
        ["family","MAE_Gamma"],key=lambda s: s.map({"network":0,"fit":1,"reference":2}) if s.name=="family" else s)
    comb.to_csv(out/"network_vs_fit_test.csv",index=False)
    print("\n"+comb.round(4).to_string(index=False))

    best=comb[comb.method=="cnn_kernel45"].iloc[0]
    ml=comb[comb.method=="dho_matched"].iloc[0]; pr=comb[comb.method=="dho_matched_prior"].iloc[0]
    gaps={"overall_gap_ml_to_tuned_cnn_meV":float(ml.MAE_Gamma-best.MAE_Gamma),
          "overall_closed_by_prior_meV":float(ml.MAE_Gamma-pr.MAE_Gamma),
          "overall_fraction_closed":float((ml.MAE_Gamma-pr.MAE_Gamma)/(ml.MAE_Gamma-best.MAE_Gamma)),
          "overdamped_gap_ml_to_tuned_cnn_meV":float(ml.MAE_Gamma_overdamped-best.MAE_Gamma_overdamped),
          "overdamped_closed_by_prior_meV":float(ml.MAE_Gamma_overdamped-pr.MAE_Gamma_overdamped),
          "overdamped_fraction_closed":float((ml.MAE_Gamma_overdamped-pr.MAE_Gamma_overdamped)
                                             /(ml.MAE_Gamma_overdamped-best.MAE_Gamma_overdamped))}
    print("\n  prior-closed fractions:",json.dumps({k:round(v,4) for k,v in gaps.items()},indent=2))
    (out/"prior_gap_fractions.json").write_text(json.dumps(gaps,indent=2))
    (out/"run_sidecar_network.json").write_text(json.dumps({
        "task":"A25_network_vs_fit","dataset_version":DS.VERSION,
        "script":str(Path(__file__).resolve().relative_to(ROOT)),
        "eval_set":str(DS.TEST.relative_to(ROOT)),
        "subset":"np.linspace(0, N-1, 600) of the test split; identical to A23",
        "severity":args.severity,"checkpoints":str(root.relative_to(ROOT)),
        "regime_definition":"overdamped = Gamma > omega0",
        "aggregation":"per-model mean over 9 runs (3 dataset realizations x 3 training seeds)",
        "training":"none; evaluation of existing A21 checkpoints only",
        "seed":args.seed,"date_utc":datetime.now(timezone.utc).isoformat()},indent=2))
    print(f"  wrote {out}")


if __name__ == "__main__":
    main()
