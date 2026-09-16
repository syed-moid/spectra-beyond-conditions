"""Evaluation-only recomputations for round 45 (no training).

(E1) kappa_slow = Gamma - sqrt(Gamma^2 - omega0^2) everywhere a kappa appears.
(E2) Shuffle nulls with repetition: 200 ordinary permutations and 200 derangements
     per tuple, reporting the mean and SD of each null so the observed offset is
     compared against a distribution rather than a single draw.
(E3) Recoverability boundaries on pairwise ranking accuracy (primary) with
     Spearman retained as secondary.
(E4) Parameter recovery with a conditions-only comparator: the Monte Carlo
     conditional median of omega0 and log Gamma given (T, c, E), so the R^2
     values have a loss-appropriate reference rather than standing alone.
"""
from __future__ import annotations
import argparse, json, sys
from datetime import datetime, timezone
from pathlib import Path
import numpy as np, pandas as pd, torch
from scipy.stats import spearmanr, skewnorm

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(Path(__file__).resolve().parent))
import a4_train_v8 as A4
from a17_arch_sweep import build_variant
from sbc.data.dataset import InsSpectraDataset
from sbc.models.spectral_transformer import TargetStats, unstandardize_outputs
from sbc.data.latent_perturbations import (ALPHA_MU_LOG, ALPHA_SIGMA_LOG, BETA_XI2_HIGH,
    BETA_XI2_LOW, BETA_XI2_SKEW, GAMMA_FLOOR, _BETA_XI2_LOC, _BETA_XI2_SCALE)
from sbc.data.spectrum_generator import Gamma_Q, omega_Q
from realization_stats import pairwise_ranking_accuracy
import dataset_paths as DS

MODELS = [("cnn_kernel45","variant"),("tf_patch60","variant"),("fusion","base"),
          ("cnn","base"),("5a","base"),("5b","base")]
DS_SEEDS, TR_SEEDS = (0,1,2), (42,43,44)
SEVS = (0.25,0.5,0.75,1.0,1.5,2.0,3.0,4.0,6.0,8.0,12.0)


def kappa_slow(om, gm):
    """Exact slow decay rate; NaN where the mode is not overdamped."""
    return np.where(gm > om, gm - np.sqrt(np.maximum(gm**2 - om**2, 0.0)), np.nan)


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
    """Monte Carlo conditional median of (omega0, log Gamma) given conditions."""
    om0, gm0 = omega_Q(T, c/100.0, E), Gamma_Q(T, c/100.0, E)
    ax = rng.lognormal(ALPHA_MU_LOG, ALPHA_SIGMA_LOG, n)*rng.standard_normal(n)
    raw = skewnorm.rvs(a=BETA_XI2_SKEW, size=n, random_state=rng)
    bx2 = np.clip(raw*_BETA_XI2_SCALE+_BETA_XI2_LOC, BETA_XI2_LOW, BETA_XI2_HIGH)
    om = om0*(1.0+bx2); gm = gm0*np.maximum(GAMMA_FLOOR, 1.0+ax)
    return float(np.median(om)), float(np.median(np.log(np.clip(gm,1e-9,None))))


def derangement(rng, n):
    while True:
        p = rng.permutation(n)
        if not np.any(p == np.arange(n)): return p


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--outdir", type=Path, default=None)
    ap.add_argument("--n-null", type=int, default=200)
    ap.add_argument("--oracle-draws", type=int, default=2000)
    ap.add_argument("--seed", type=int, default=20260913)
    args=ap.parse_args()
    out=args.outdir or DS.rev("A24"); out.mkdir(parents=True, exist_ok=True)
    root=DS.rev("A21")/"checkpoints"; rng=np.random.default_rng(args.seed)

    te=InsSpectraDataset(DS.TEST,"test",severity=1.0,as_torch=False)
    rp=InsSpectraDataset(DS.REPLICATE_TEST,"replicate_test",severity=1.0,as_torch=False)
    zt=np.load(DS.REPLICATE_TEST,allow_pickle=False)
    tid=zt["tuple_id"][zt["split"]=="replicate_test"]
    from sbc.models.nonlinear_conditions_mlp import normalize_conditions
    c_te=normalize_conditions(te._T_K,te._c_pct,te._E_kVcm)
    c_rp=normalize_conditions(rp._T_K,rp._c_pct,rp._E_kVcm)
    y_rp=np.log(np.clip(rp._M.astype(float),1e-9,None))
    om_t=te._omega_Q.astype(float); lg_t=np.log(np.clip(te._Gamma_Q.astype(float),1e-9,None))
    ks_t=kappa_slow(om_t, te._Gamma_Q.astype(float)); over=te._Gamma_Q.astype(float)>om_t
    X_rp={s: np.stack([rp.get_augmented(i,s) for i in range(len(rp))]).astype(np.float32) for s in SEVS}
    X_te1=np.stack([te.get_augmented(i,1.0) for i in range(len(te))]).astype(np.float32)
    print(f"  test n={len(om_t)} ({over.mean():.1%} overdamped)  replicate_test n={len(y_rp)}")

    # ---- (E4) conditions-only comparator for the parameter targets -----------
    r2=lambda p,t: float(1-((p-t)**2).sum()/((t-t.mean())**2).sum())
    orng=np.random.default_rng(args.seed)
    cm=np.array([conditional_params(float(te._T_K[i]),float(te._c_pct[i]),float(te._E_kVcm[i]),
                                    args.oracle_draws,orng) for i in range(len(om_t))])
    oracle={"model":"conditions-only (MC conditional median)","dataset_seed":-1,"seed":-1,
            "R2_omega0":r2(cm[:,0],om_t),"R2_logGamma":r2(cm[:,1],lg_t),
            "pair_acc_omega0":pairwise_ranking_accuracy(np.arange(len(om_t)),om_t,cm[:,0]),
            "MAE_omega0":float(np.abs(cm[:,0]-om_t).mean()),
            "MAE_logGamma":float(np.abs(cm[:,1]-lg_t).mean())}
    print(f"  conditions-only comparator: R2(w0) {oracle['R2_omega0']:.4f}  R2(logG) {oracle['R2_logGamma']:.4f}")

    # ---- (E2) nulls, shared across models -----------------------------------
    perms=[]; ders=[]
    for _ in range(args.n_null):
        pi=np.arange(len(y_rp)); di=np.arange(len(y_rp))
        for t in np.unique(tid):
            q=np.nonzero(tid==t)[0]
            pi[q]=q[rng.permutation(len(q))]; di[q]=q[derangement(rng,len(q))]
        perms.append(pi); ders.append(di)
    print(f"  built {args.n_null} permutations and {args.n_null} derangements")

    prows=[]; brows=[]
    for model,kind in MODELS:
        for ds in DS_SEEDS:
            for sd in TR_SEEDS:
                net,st=load(model,kind,ds,sd,root)
                ct = c_te if model=="fusion" else None
                cr = c_rp if model=="fusion" else None
                P=pred(net,st,X_te1,ct)
                ks_p=kappa_slow(P["omega_Q"],P["Gamma_Q"])
                m=over & np.isfinite(ks_p) & np.isfinite(ks_t)
                prows.append({"model":model,"dataset_seed":ds,"seed":sd,
                    "R2_omega0":r2(P["omega_Q"],om_t),
                    "R2_logGamma":r2(np.log(np.clip(P["Gamma_Q"],1e-9,None)),lg_t),
                    "pair_acc_omega0":pairwise_ranking_accuracy(np.arange(len(om_t)),om_t,P["omega_Q"]),
                    "MAE_omega0":float(np.abs(P["omega_Q"]-om_t).mean()),
                    "MAE_logGamma":float(np.abs(np.log(np.clip(P["Gamma_Q"],1e-9,None))-lg_t).mean()),
                    "MAE_kappa_slow_overdamped":float(np.abs(ks_p[m]-ks_t[m]).mean()),
                    "R2_kappa_slow_overdamped":r2(ks_p[m],ks_t[m]),"n_overdamped":int(m.sum())})
                # severity boundaries on replicate_test
                pr1=None
                for s in SEVS:
                    Pr=pred(net,st,X_rp[s],cr)
                    p=np.log(np.clip(Pr["M"],1e-9,None))
                    acc=pairwise_ranking_accuracy(tid,y_rp,p)
                    rho=float(np.mean([spearmanr(y_rp[tid==t],p[tid==t]).statistic for t in np.unique(tid)]))
                    row={"model":model,"dataset_seed":ds,"seed":sd,"severity":s,
                         "pairwise_accuracy":acc,"rho":rho,
                         "MAE_logM":float(np.abs(p-y_rp).mean()),
                         "in_training_range":bool(0.25<=s<=4.0)}
                    if s==1.0:
                        pr1=p
                        an=[pairwise_ranking_accuracy(tid,y_rp,p[i]) for i in perms]
                        ad=[pairwise_ranking_accuracy(tid,y_rp,p[i]) for i in ders]
                        rn=[float(np.mean([spearmanr(y_rp[tid==t],p[i][tid==t]).statistic
                                           for t in np.unique(tid)])) for i in perms[:50]]
                        rd=[float(np.mean([spearmanr(y_rp[tid==t],p[i][tid==t]).statistic
                                           for t in np.unique(tid)])) for i in ders[:50]]
                        row.update(null_perm_acc_mean=float(np.mean(an)),null_perm_acc_sd=float(np.std(an,ddof=1)),
                                   null_der_acc_mean=float(np.mean(ad)),null_der_acc_sd=float(np.std(ad,ddof=1)),
                                   null_perm_rho_mean=float(np.mean(rn)),null_perm_rho_sd=float(np.std(rn,ddof=1)),
                                   null_der_rho_mean=float(np.mean(rd)),null_der_rho_sd=float(np.std(rd,ddof=1)),
                                   der_rho_expected=float(-rho/(len(y_rp)/len(np.unique(tid))-1)))
                    brows.append(row)
                print(f"  {model:14s} ds{ds} s{sd} done",flush=True)
    pd.DataFrame(prows+[oracle]).to_csv(out/"parameter_recovery_v10.csv",index=False)
    pd.DataFrame(brows).to_csv(out/"ranking_by_severity_v10.csv",index=False)
    (out/"run_sidecar.json").write_text(json.dumps({
        "task":"A24_recompute_v10","dataset_version":DS.VERSION,
        "script":str(Path(__file__).resolve().relative_to(ROOT)),
        "eval_sets":[str(DS.TEST.relative_to(ROOT)),str(DS.REPLICATE_TEST.relative_to(ROOT))],
        "kappa":"kappa_slow = Gamma - sqrt(Gamma^2 - omega0^2), exact slow decay rate",
        "n_null_permutations":args.n_null,"n_null_derangements":args.n_null,
        "oracle_draws":args.oracle_draws,
        "in_training_range":"s in [0.25, 4]; s > 4 is out-of-training-range",
        "training":"none; evaluation of existing A21 checkpoints only",
        "seed":args.seed,"date_utc":datetime.now(timezone.utc).isoformat()},indent=2))
    print(f"\n  wrote {out}")


if __name__=="__main__":
    main()
