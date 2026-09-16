"""A6 additions (round 13): identifiability in kappa = omega0^2 / Gamma, and a
metadata-oracle reference for parameter recovery.

kappa is the relaxational combination that a heavily overdamped DHO lineshape
constrains directly; (omega0, Gamma) individually become weakly identified as
Gamma/omega0 grows. Comparing kappa recovery to Gamma recovery separates two
readings of the 9.4x overdamped advantage.

The metadata oracle for a parameter is the conditional median of its realized
value given (T, c, E), computed by redrawing latents exactly as in A1. It is
severity-independent, and is the reference any spectral estimate must beat for
the estimate to carry realization-level information.
"""

from __future__ import annotations

import json, sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import skewnorm

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(Path(__file__).resolve().parent))
import torch  # noqa: E402
from a4_eval import load_a4_model  # noqa: E402
from a17_arch_sweep import build_variant  # noqa: E402
from sbc.data.dataset import InsSpectraDataset  # noqa: E402
from sbc.data.latent_perturbations import (  # noqa: E402
    ALPHA_MU_LOG, ALPHA_SIGMA_LOG, BETA_XI2_HIGH, BETA_XI2_LOW, BETA_XI2_SKEW,
    GAMMA_FLOOR, _BETA_XI2_LOC, _BETA_XI2_SCALE,
)
from sbc.data.spectrum_generator import Gamma_Q, omega_Q  # noqa: E402
from sbc.models.spectral_transformer import TargetStats, unstandardize_outputs  # noqa: E402
import dataset_paths as DS

REV = ROOT / "results" / "revision"
A4E = DS.rev("A4_extended") / "checkpoints"
NPZ = DS.FULL
OUT = DS.rev("A6")
SEEDS = (42, 43, 44)
N_DRAW = 2000


def oracle_params(T, c, E, rng):
    om0, gm0 = omega_Q(T, c / 100.0, E), Gamma_Q(T, c / 100.0, E)
    a = rng.lognormal(ALPHA_MU_LOG, ALPHA_SIGMA_LOG, N_DRAW)
    x = rng.standard_normal(N_DRAW)
    b = np.clip(skewnorm.rvs(a=BETA_XI2_SKEW, size=N_DRAW, random_state=rng)
                * _BETA_XI2_SCALE + _BETA_XI2_LOC, BETA_XI2_LOW, BETA_XI2_HIGH)
    om = om0 * (1.0 + b)
    gm = gm0 * np.maximum(GAMMA_FLOOR, 1.0 + a * x)
    return (float(np.median(om)), float(np.median(gm)),
            float(np.median(gm / om)), float(np.median(om ** 2 / gm)))


@torch.no_grad()
def predict(nets, X):
    outs = []
    for kind, m in nets:
        if kind == "a4":
            outs.append(m.predict_all({"spectrum": X, "T_K": None, "c_pct": None, "E_kVcm": None})
                        if False else None)
    return outs


def metrics(t, p):
    ss = float(np.sum((t - p) ** 2)); tot = float(np.sum((t - t.mean()) ** 2))
    return float(np.abs(t - p).mean()), 1.0 - ss / tot


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    ds = InsSpectraDataset(NPZ, "stress_base", severity=1.0, as_torch=False)
    T, c, E = ds._T_K.astype(float), ds._c_pct.astype(float), ds._E_kVcm.astype(float)
    cond = {"T_K": T, "c_pct": c, "E_kVcm": E}
    om_t, gm_t = ds._omega_Q.astype(float), ds._Gamma_Q.astype(float)
    kap_t = om_t ** 2 / gm_t
    over = gm_t > om_t

    # metadata oracle for parameters (severity independent)
    rng = np.random.default_rng(20260906)
    orc = np.array([oracle_params(T[i], c[i], E[i], rng) for i in range(len(T))])
    orc_om, orc_gm, orc_ratio, orc_kap = orc[:, 0], orc[:, 1], orc[:, 2], orc[:, 3]

    # models
    models = {}
    for a in ("cnn",):
        models[a] = [("a4", load_a4_model(a, "v7", s, ckpt_dir=A4E)) for s in SEEDS]
    lst = []
    for s in SEEDS:
        ck = torch.load(DS.tuned_ckpt("cnn_kernel45", s),
                        map_location="cpu", weights_only=False)
        n = build_variant("cnn_kernel45", s); n.load_state_dict(ck["state_dict"]); n.eval()
        lst.append(("var", (n, TargetStats.from_dict(ck["target_stats"]))))
    models["cnn_kernel45"] = lst

    @torch.no_grad()
    def pred_all(name, X):
        outs = []
        for kind, m in models[name]:
            if kind == "a4":
                outs.append(m.predict_all({"spectrum": X, **cond}))
            else:
                net, st = m
                o = np.concatenate([net(torch.from_numpy(X[i:i+2048].astype(np.float32))).numpy()
                                    for i in range(0, len(X), 2048)], 0)
                outs.append(unstandardize_outputs(o, st))
        return {k: np.mean([o[k] for o in outs], 0) for k in ("omega_Q", "Gamma_Q")}

    fits = pd.concat([pd.read_csv(DS.rev("A3") / "fit_results_per_spectrum.csv"),
                      pd.read_csv(DS.rev("A3") / "fit_results_extra_variants.csv")])

    # ---- item 1: kappa and omega0 by regime at s = 1 -----------------------
    X1 = np.stack([ds.get_augmented(i, 1.0) for i in range(len(ds))]).astype(np.float32)
    est = {n: pred_all(n, X1) for n in models}
    for meth in ("dho_matched", "dho_two_mode"):
        d = fits[(fits.method == meth) & (fits.severity == 1.0)].sort_values("idx")
        est[meth] = {"omega_Q": d.omega0_fit.to_numpy(), "Gamma_Q": d.Gamma_fit.to_numpy(),
                     "ok": d.fit_ok.astype(bool).to_numpy()}
    est["metadata oracle"] = {"omega_Q": orc_om, "Gamma_Q": orc_gm}

    rows = []
    for label, mask in (("overdamped", over), ("underdamped", ~over), ("all", np.ones_like(over))):
        for name, e in est.items():
            ok = mask & e.get("ok", np.ones_like(mask))
            if ok.sum() < 5:
                continue
            kap = e["omega_Q"] ** 2 / np.clip(e["Gamma_Q"], 1e-9, None)
            mk, r2k = metrics(kap_t[ok], kap[ok])
            mo, r2o = metrics(om_t[ok], e["omega_Q"][ok])
            mg, r2g = metrics(gm_t[ok], e["Gamma_Q"][ok])
            rows.append({"dataset_version": DS.VERSION, "severity": 1.0, "regime": label,
                         "model": name, "n": int(ok.sum()),
                         "MAE_kappa_meV": mk, "R2_kappa": r2k,
                         "MAE_omega0_meV": mo, "R2_omega0": r2o,
                         "MAE_Gamma_meV": mg, "R2_Gamma": r2g})
    df = pd.DataFrame(rows)
    df.to_csv(OUT / "identifiability_kappa.csv", index=False)
    print(df[df.regime == "overdamped"][["model", "n", "MAE_kappa_meV", "R2_kappa",
                                         "MAE_omega0_meV", "MAE_Gamma_meV"]]
          .to_string(index=False, float_format=lambda x: f"{x:.4f}"))

    # ---- item 2: parameter-level crossing severity -------------------------
    prows = []
    mo_o, _ = metrics(om_t, orc_om); mg_o, _ = metrics(gm_t, orc_gm)
    mr_o, _ = metrics(gm_t / om_t, orc_ratio); mk_o, _ = metrics(kap_t, orc_kap)
    prows.append({"dataset_version": DS.VERSION, "model": "metadata oracle", "severity": "any",
                  "MAE_omega0_meV": mo_o, "MAE_Gamma_meV": mg_o,
                  "MAE_Gamma_over_omega0": mr_o, "MAE_kappa_meV": mk_o})
    for sev in (0.25, 0.5, 1.0, 2.0, 4.0):
        X = np.stack([ds.get_augmented(i, sev) for i in range(len(ds))]).astype(np.float32)
        for name in models:
            e = pred_all(name, X)
            kap = e["omega_Q"] ** 2 / np.clip(e["Gamma_Q"], 1e-9, None)
            prows.append({"dataset_version": DS.VERSION, "model": name, "severity": sev,
                          "MAE_omega0_meV": metrics(om_t, e["omega_Q"])[0],
                          "MAE_Gamma_meV": metrics(gm_t, e["Gamma_Q"])[0],
                          "MAE_Gamma_over_omega0": metrics(gm_t / om_t,
                                                           e["Gamma_Q"] / e["omega_Q"])[0],
                          "MAE_kappa_meV": metrics(kap_t, kap)[0]})
        print(f"  severity {sev} done", flush=True)
    pdf = pd.DataFrame(prows)
    pdf.to_csv(OUT / "parameter_recovery_vs_metadata_oracle.csv", index=False)
    print("\n" + pdf.to_string(index=False, float_format=lambda x: f"{x:.4f}"))

    # scatter additions
    sc = pd.read_csv(OUT / "predicted_vs_reference_scatter.csv")
    sc["kappa_true"] = sc.omega0_true ** 2 / sc.Gamma_true
    sc["kappa_pred"] = sc.omega0_pred ** 2 / sc.Gamma_pred.clip(lower=1e-9)
    sc.to_csv(OUT / "predicted_vs_reference_scatter.csv", index=False)
    pd.DataFrame([{"dataset_version": DS.VERSION, "reference": "metadata_oracle_Gamma",
                   "MAE_Gamma_meV": mg_o}]).to_csv(OUT / "figure1_reference_lines.csv", index=False)
    (OUT / "run_sidecar_identifiability.json").write_text(json.dumps({
        "task": "A6c", "dataset_version": DS.VERSION, "n_draws": N_DRAW, "seeds": list(SEEDS),
        "date_utc": datetime.now(timezone.utc).isoformat()}, indent=2))


if __name__ == "__main__":
    main()
