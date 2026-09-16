"""Hold-out generalization on v10, with both metadata baselines in every row.

Round 45 §1 and §3.3. The v9 hold-out table (A10_v9) carried one run per model
and only one metadata baseline. This evaluates the nine A21 v10 runs per model
on the v10 hold-out splits and reports, for every row, both references:

  conditions-only (conditional median)  the MAE-optimal predictor given (T, c, E);
                                        a Monte Carlo estimate of the
                                        conditions-only Bayes risk under absolute loss
  marginal (global median)              the no-information predictor

The A21 checkpoints were trained on `ablation_v10_both_seed{0,1,2}`, which are
independent draws from `full_dataset_phase1_v10`; exact-row overlap between the
training blocks and every split used here was verified to be zero.

Evaluation only; no training.
"""
from __future__ import annotations
import argparse, json, sys
from datetime import datetime, timezone
from pathlib import Path
import numpy as np, pandas as pd, torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(Path(__file__).resolve().parent))
import a4_train_v8 as A4
import a1_oracle_floor as A1
from a17_arch_sweep import build_variant
from sbc.data.dataset import InsSpectraDataset
from sbc.models.spectral_transformer import TargetStats, unstandardize_outputs
from sbc.models.nonlinear_conditions_mlp import normalize_conditions
from realization_stats import pairwise_ranking_accuracy
import dataset_paths as DS

MODELS = [("cnn_kernel45","variant"),("tf_patch60","variant"),("fusion","base"),
          ("cnn","base"),("5a","base"),("5b","base")]
DS_SEEDS, TR_SEEDS = (0,1,2), (42,43,44)
MONOTONE = {
 "val": "-",
 "holdout_c": "interior check (c = 1%); within the training range",
 "holdout_c_extrap": "c-extrapolation (2.5%); Gamma0 linear in c (0.6*c*T), omega0 linear (1-0.08c)",
 "holdout_E": "E-extrapolation (4 kV/cm); both quadratic and monotone in E on [0, 4]",
 "holdout_T": "T-extrapolation (600 K); NOT monotone -- omega0(T) has a minimum near 415 K",
}


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


def baselines(T, c, E, y, n_draws, rng):
    """conditions-only conditional median and the marginal median, on these cases."""
    med = np.array([np.median(A1.conditional_logM(float(T[i]), float(c[i]), float(E[i]),
                                                  n_draws, rng)[0]) for i in range(len(y))])
    return float(np.abs(med-y).mean()), float(np.abs(y-np.median(y)).mean()), med


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--severity", type=float, default=1.0)
    ap.add_argument("--n-draws", type=int, default=2000)
    ap.add_argument("--seed", type=int, default=20260913)
    ap.add_argument("--outdir", type=Path, default=None)
    args=ap.parse_args()
    out=args.outdir or DS.rev("A10"); out.mkdir(parents=True, exist_ok=True)
    root=DS.rev("A21")/"checkpoints"

    splits=[("val",DS.FULL),("holdout_c",DS.FULL),("holdout_E",DS.FULL),
            ("holdout_T",DS.FULL),("holdout_c_extrap",DS.HOLDOUT_C_EXTRAP)]
    rng=np.random.default_rng(args.seed)
    data={}
    for name,npz in splits:
        d=InsSpectraDataset(npz,name,severity=args.severity,as_torch=False)
        X=np.stack([d.get_augmented(i,args.severity) for i in range(len(d))]).astype(np.float32)
        y=np.log(np.clip(d._M.astype(float),1e-9,None))
        cond=normalize_conditions(d._T_K,d._c_pct,d._E_kVcm)
        mae_cond,mae_marg,med=baselines(d._T_K,d._c_pct,d._E_kVcm,y,args.n_draws,rng)
        # Pairwise accuracy here pools every case in the split, so it ranks spectra taken at
        # DIFFERENT conditions. Conditions alone already order those well, so the network
        # column is only readable against the conditions-only column beside it. This is a
        # different quantity from the within-tuple accuracy of the replicate test set.
        acc_cond=pairwise_ranking_accuracy(np.zeros(len(y),int),y,med)
        data[name]=dict(X=X,y=y,cond=cond,mae_cond=mae_cond,mae_marg=mae_marg,n=len(y),
                        acc_cond=acc_cond,src=str(npz.relative_to(ROOT)))
        print(f"  {name:18s} n={len(y):5d}  conditions-only {mae_cond:.4f} (pairwise {acc_cond:.4f})"
              f"  marginal {mae_marg:.4f}",flush=True)

    rows=[]
    for model,kind in MODELS:
        for ds in DS_SEEDS:
            for sd in TR_SEEDS:
                net,st=load(model,kind,ds,sd,root)
                for name,_ in splits:
                    g=data[name]
                    P=pred(net,st,g["X"], g["cond"] if model=="fusion" else None)
                    p=np.log(np.clip(P["M"],1e-9,None))
                    rows.append({"dataset_version":DS.VERSION,"model":model,"dataset_seed":ds,
                        "seed":sd,"split":name,"n":g["n"],"severity":args.severity,
                        "MAE_logM":float(np.abs(p-g["y"]).mean()),
                        "pairwise_accuracy":pairwise_ranking_accuracy(np.zeros(g["n"],int),g["y"],p),
                        "MAE_conditions_only":g["mae_cond"],"MAE_marginal":g["mae_marg"],
                        "pairwise_accuracy_conditions_only":g["acc_cond"],
                        "pairwise_accuracy_scope":"pooled across conditions within the split",
                        "source":g["src"],"structure":MONOTONE[name]})
        print(f"  {model:14s} 9 runs x {len(splits)} splits done",flush=True)
    runs=pd.DataFrame(rows); runs.to_csv(out/"holdout_runs_v10.csv",index=False)

    agg=(runs.groupby(["split","model"],as_index=False)
             .agg(n=("n","first"),MAE_logM=("MAE_logM","mean"),sd_MAE=("MAE_logM","std"),
                  pairwise_accuracy=("pairwise_accuracy","mean"),
                  MAE_conditions_only=("MAE_conditions_only","first"),
                  MAE_marginal=("MAE_marginal","first"),
                  pairwise_accuracy_conditions_only=("pairwise_accuracy_conditions_only","first"),
                  pairwise_accuracy_scope=("pairwise_accuracy_scope","first"),
                  source=("source","first"),structure=("structure","first")))
    agg["improvement_over_conditions_only"]=agg.MAE_conditions_only-agg.MAE_logM
    agg["aggregation"]="mean over 9 runs (3 realizations x 3 training seeds)"
    agg.to_csv(out/"holdout_table_v10.csv",index=False)
    print("\n"+agg[["split","model","n","MAE_logM","sd_MAE","pairwise_accuracy",
                    "pairwise_accuracy_conditions_only","MAE_conditions_only",
                    "MAE_marginal"]].round(4).to_string(index=False))

    (out/"run_sidecar_v10.json").write_text(json.dumps({
        "task":"A26_holdouts_v10","dataset_version":DS.VERSION,
        "script":str(Path(__file__).resolve().relative_to(ROOT)),
        "checkpoints":str(root.relative_to(ROOT)),
        "checkpoint_note":("A21 runs, trained on ablation_v10_both_seed{0,1,2}; these are "
                           "independent draws from full_dataset_phase1_v10 and exact-row overlap "
                           "with every split evaluated here is zero"),
        "splits":{k:data[k]["src"] for k in data},
        "baselines":["conditions-only conditional median (MC estimate of the conditions-only "
                     "Bayes risk under absolute loss)","marginal (global median)"],
        "pairwise_accuracy_scope":("pooled across conditions within each split; NOT the "
            "within-tuple accuracy of the replicate test set, and not comparable to it"),
        "n_draws":args.n_draws,"severity":args.severity,
        "aggregation":"mean over 9 runs (3 dataset realizations x 3 training seeds)",
        "training":"none; evaluation of existing A21 checkpoints only",
        "seed":args.seed,"date_utc":datetime.now(timezone.utc).isoformat()},indent=2))
    print(f"  wrote {out}")


if __name__ == "__main__":
    main()
