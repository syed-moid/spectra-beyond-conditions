"""Feature-set ablation on v10 (§4.1, first probe).

Draft v8_2 cited `A3_v10/feature_set_ablation.csv`, which did not exist: the
numbers came from A3_v9. This reproduces exactly the A3 ablation protocol --
same feature extractor, same three feature groups, same five seeds, same
HistGradientBoostingRegressor settings -- on the v10 dataset, so the paragraph
can be read from the file it cites.

The conditions-only reference on the same evaluation set is included in the
output so the comparison in the text is self-contained.

Gradient-boosted trees on hand features; no neural training.
"""
from __future__ import annotations
import argparse, json, sys, time
from datetime import datetime, timezone
from pathlib import Path
import numpy as np, pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(Path(__file__).resolve().parent))
from a3_fitting_baselines import (features, FEATURE_NAMES, SEEDS, TRAIN_SEVERITY_RANGE)
import a1_oracle_floor as A1
from sbc.data.dataset import InsSpectraDataset
import dataset_paths as DS

GROUPS = {
    "position_width_only": ["peak_pos_meV", "fwhm_meV"],
    "plus_intensity": ["peak_pos_meV", "fwhm_meV", "peak_height", "integ_soft_window",
                       "central_height", "raw_max", "raw_min"],
    "all_11": list(FEATURE_NAMES),
}


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--severity", type=float, default=1.0)
    ap.add_argument("--seeds", type=int, nargs="+", default=list(SEEDS))
    ap.add_argument("--n-draws", type=int, default=2000)
    ap.add_argument("--seed", type=int, default=20260905)
    ap.add_argument("--outdir", type=Path, default=None)
    args=ap.parse_args()
    out=args.outdir or DS.rev("A3"); out.mkdir(parents=True, exist_ok=True)

    train=InsSpectraDataset(DS.FULL,"train",severity=1.0,as_torch=False)
    stress=InsSpectraDataset(DS.FULL,"stress_base",severity=1.0,as_torch=False)
    grid=stress._omega_grid; n=len(stress)
    rng=np.random.default_rng(args.seed)
    lo,hi=TRAIN_SEVERITY_RANGE
    tr_sev=np.exp(rng.uniform(np.log(lo),np.log(hi),size=len(train)))
    t0=time.time()
    Xtr=np.array([features(train.get_augmented(i,severity=float(tr_sev[i])),grid)
                  for i in range(len(train))])
    ytr=np.log(np.clip(train._M.astype(float),1e-9,None))
    Xte=np.array([features(stress.get_augmented(i,severity=args.severity),grid) for i in range(n)])
    yte=np.log(np.clip(stress._M.astype(float),1e-9,None))
    print(f"  features: train {Xtr.shape}, stress_base {Xte.shape} in {time.time()-t0:.0f}s",flush=True)

    orng=np.random.default_rng(args.seed)
    cond=np.array([np.median(A1.conditional_logM(float(stress._T_K[i]),float(stress._c_pct[i]),
                                                 float(stress._E_kVcm[i]),args.n_draws,orng)[0])
                   for i in range(n)])
    mae_cond=float(np.abs(cond-yte).mean()); mae_marg=float(np.abs(yte-np.median(yte)).mean())
    print(f"  conditions-only {mae_cond:.4f}  marginal {mae_marg:.4f}")

    rows=[]
    for gname,cols in GROUPS.items():
        sel=[FEATURE_NAMES.index(c) for c in cols]; per=[]
        for sd in args.seeds:
            g=HistGradientBoostingRegressor(loss="absolute_error",random_state=sd,
                                            early_stopping=True,validation_fraction=0.1)
            g.fit(Xtr[:,sel],ytr); per.append(float(np.abs(g.predict(Xte[:,sel])-yte).mean()))
        rows.append({"dataset_version":DS.VERSION,"feature_set":gname,"n_features":len(cols),
                     "severity":args.severity,"N":n,"n_seeds":len(args.seeds),
                     "MAE_logM_mean":float(np.mean(per)),"MAE_logM_sd":float(np.std(per,ddof=1)),
                     "MAE_conditions_only":mae_cond,"MAE_marginal":mae_marg,
                     "features":";".join(cols)})
        print(f"  {gname:20s} ({len(cols):2d} feats): {np.mean(per):.4f} +/- {np.std(per,ddof=1):.4f}",flush=True)
    df=pd.DataFrame(rows); df.to_csv(out/"feature_set_ablation.csv",index=False)
    print("\n"+df.drop(columns=["features"]).round(4).to_string(index=False))
    (out/"run_sidecar_feature_ablation.json").write_text(json.dumps({
        "task":"A27_feature_ablation","dataset_version":DS.VERSION,
        "script":str(Path(__file__).resolve().relative_to(ROOT)),
        "eval_set":f"{DS.FULL.relative_to(ROOT)} split stress_base",
        "train_split":"train, severity drawn log-uniform on [0.25, 4] as in the network protocol",
        "estimator":"HistGradientBoostingRegressor(loss='absolute_error', early_stopping=True, validation_fraction=0.1)",
        "seeds":list(args.seeds),"severity":args.severity,
        "reference":"conditions-only conditional median, Monte Carlo, and the marginal median",
        "n_draws":args.n_draws,"training":"gradient-boosted trees on hand features; no neural training",
        "seed":args.seed,"date_utc":datetime.now(timezone.utc).isoformat()},indent=2))
    print(f"  wrote {out}")


if __name__ == "__main__":
    main()
