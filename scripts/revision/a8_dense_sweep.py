"""A8 - dense degradation sweep and one-factor-at-a-time decomposition.

Dense combined sweep: s in {0.25, 0.5, 0.75, 1, 1.5, 2, 3, 4} on the
degradation-sweep base and (round-6 addition 4) on the replicate set, so MAE and
within-tuple rho appear on identical axes.

One-factor sweeps isolate each degradation channel: only that channel is
applied, at five levels chosen to reproduce the MEDIAN physical quantity the
combined pipeline produces at s in {0.25, 0.5, 1, 2, 4} (from A0 section 6), so
the axes are in physical units and comparable to the combined sweep.

Crossover is defined as the first severity at which the paired CI of
(model - metadata reference) includes zero.
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
from replicate_io import block_key, load_replicate  # noqa: E402
from sbc.data.augmentations import apply_aug_params, state_from_clean  # noqa: E402
from sbc.data.dataset import InsSpectraDataset  # noqa: E402
from sbc.data.latent_perturbations import LatentDraw  # noqa: E402
from sbc.models.spectral_transformer import TargetStats, unstandardize_outputs  # noqa: E402
import dataset_paths as DS

REV = ROOT / "results" / "revision"
A4E = DS.rev("A4_extended") / "checkpoints"
NPZ = DS.FULL
OUT = DS.rev("A8")
SEEDS = (42, 43, 44)
DENSE = (0.25, 0.5, 0.75, 1.0, 1.5, 2.0, 3.0, 4.0)
FWHM_K = 2.0 * np.sqrt(2.0 * np.log(2.0))

# A0 section 6 medians at s = 0.25, 0.5, 1, 2, 4
LEVELS = {
    "resolution": [0.715, 1.012, 1.431, 2.023, 2.861],       # FWHM, meV
    "counts":     [37147, 18574, 9287, 4643, 2322],           # N at peak
    "background": [0.037, 0.074, 0.147, 0.294, 0.589],        # bg / peak
}
BASE = ["cnn", "fusion", "5a", "5b"]
TUNED = ("cnn_kernel45", "tf_patch60")   # resolved per seed via DS.tuned_ckpt
A16 = [("5a", "fixed1"), ("5a", "low"), ("fusion", "fixed1")]


def load_models():
    ms = {}
    for a in BASE:
        ms[a] = [("a4", load_a4_model(a, "v7", s, ckpt_dir=A4E)) for s in SEEDS]
    for a, p in A16:
        ms[f"{a}_{p}"] = [("a4", load_a4_model(a, p, s, ckpt_dir=A4E)) for s in SEEDS]
    for name in TUNED:
        lst = []
        for s in SEEDS:
            c = torch.load(DS.tuned_ckpt(name, s), map_location="cpu", weights_only=False)
            net = build_variant(name, s); net.load_state_dict(c["state_dict"]); net.eval()
            lst.append(("var", (net, TargetStats.from_dict(c["target_stats"]))))
        ms[name] = lst
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


SHAPED_SEED = 20260911
_SHAPE_CFG = {"linear_slope_max_per_meV": 0.05, "exp_tau_range_meV": (3.0, 15.0),
              "bg_types": ("linear", "poly2", "exp")}


def draw_shaped_backgrounds(levels, seed=SHAPED_SEED):
    """One non-flat background shape per magnitude level, drawn from the config ranges.

    The scripted `background` factor is flat: linear with slope 0 and
    intercept_frac = level. This draws a genuine shape at the same magnitude, so
    figS2 contrasts flat against shaped at matched background/peak. Every drawn
    parameter is returned and written to the CSV and the sidecar; the v8 table
    recorded only the level, which is why it could not be reproduced.
    """
    rng = np.random.default_rng(seed)
    out = {}
    for lv in levels:
        bg_type = str(rng.choice(_SHAPE_CFG["bg_types"]))
        if bg_type == "linear":
            m = _SHAPE_CFG["linear_slope_max_per_meV"]
            shape = {"slope_per_meV": float(rng.uniform(-m, m)),
                     "intercept_frac": float(lv)}
        elif bg_type == "poly2":
            shape = {"a0": float(lv),
                     "a1": float(rng.uniform(-lv, lv)) / 15.0,
                     "a2": float(rng.uniform(-lv, lv)) / 225.0}
        else:
            lo, hi = _SHAPE_CFG["exp_tau_range_meV"]
            shape = {"A_frac": float(lv), "tau_meV": float(rng.uniform(lo, hi))}
        out[float(lv)] = {"bg_type": bg_type, "bg_shape": shape}
    return out


def one_factor_params(kind, level, noise_seed, shaped=None):
    p = {"skew": 0.0, "cp_width_mult": None, "zero_offset_meV": 0.0,
         "res_sigma_const_meV": 0.0, "res_sigma_lin": 0.0, "res_eta": 0.3,
         "noise_amplitude": 0.0, "noise_seed": int(noise_seed),
         "bg_type": "linear", "bg_shape": {"slope_per_meV": 0.0, "intercept_frac": 0.0},
         "bg_magnitude_s1": 0.0}
    if kind == "resolution":
        p["res_sigma_const_meV"] = float(level) / FWHM_K
    elif kind == "counts":
        p["noise_amplitude"] = float(1.0 / np.sqrt(level))
    elif kind == "background":
        p["bg_shape"] = {"slope_per_meV": 0.0, "intercept_frac": float(level)}
    elif kind == "background_shaped":
        p["bg_type"] = shaped["bg_type"]
        p["bg_shape"] = dict(shaped["bg_shape"])
    return p


def paired(a, b, rng, n=1000):
    d = a - b; i = rng.integers(0, len(d), size=(n, len(d))); m = d[i].mean(1)
    return float(d.mean()), float(np.percentile(m, 2.5)), float(np.percentile(m, 97.5))


def tuple_rho(tid, y, p):
    out = []
    for t in np.unique(tid):
        m = tid == t
        r = spearmanr(y[m], p[m]).statistic if np.ptp(p[m]) > 0 else 0.0
        out.append(0.0 if not np.isfinite(r) else float(r))
    return float(np.mean(out))


def main():
    OUT.mkdir(exist_ok=True)
    rng = np.random.default_rng(20260906)
    models = load_models()
    ds = InsSpectraDataset(NPZ, "stress_base", severity=1.0, as_torch=False)
    cond = {"T_K": ds._T_K.astype(float), "c_pct": ds._c_pct.astype(float),
            "E_kVcm": ds._E_kVcm.astype(float)}
    y = np.log(np.clip(ds._M.astype(float), 1e-9, None))
    meta = pd.read_csv(REV / "A2" / "predictions_seedavg.csv")
    mp = meta[(meta.model == "mlp_l1") & (meta.split == "stress_base")]["logM_pred_seedavg"].to_numpy()
    err_meta = np.abs(mp - y)

    # ---- dense combined sweep, stress base -------------------------------
    rows = []
    for s in DENSE:
        X = np.stack([ds.get_augmented(i, s) for i in range(len(ds))]).astype(np.float32)
        for name, ms in models.items():
            per = [np.abs(plogm(e, X, cond) - y).mean() for e in ms]
            p = np.mean([plogm(e, X, cond) for e in ms], 0)
            e = np.abs(p - y)
            d, lo, hi = paired(e, err_meta, rng)
            rows.append({"dataset_version": DS.VERSION, "set": "stress_base", "model": name,
                         "severity": s, "MAE_mean": float(np.mean(per)),
                         "MAE_sd": float(np.std(per, ddof=1)), "MAE_ensemble": float(e.mean()),
                         "vs_metadata_diff": d, "lo": lo, "hi": hi,
                         "beats_metadata": bool(hi < 0)})
        print(f"  dense stress s={s} done", flush=True)
    # metadata row
    for s in DENSE:
        rows.append({"dataset_version": DS.VERSION, "set": "stress_base", "model": "metadata_mlp_l1",
                     "severity": s, "MAE_mean": float(err_meta.mean()), "MAE_sd": 0.0,
                     "MAE_ensemble": float(err_meta.mean()), "vs_metadata_diff": 0.0,
                     "lo": 0.0, "hi": 0.0, "beats_metadata": False})

    # ---- dense sweep on the replicate set (MAE and rho on identical axes) --
    clean, _ = load_replicate(want_blocks=False)
    tid = clean["tuple_id"]
    yr = np.log(np.clip(clean["M"].astype(float), 1e-9, None))
    condr = {"T_K": clean["T_K"].astype(float), "c_pct": clean["c_pct"].astype(float),
             "E_kVcm": clean["E_kVcm"].astype(float)}
    grid = clean["omega_grid"]
    lat = [LatentDraw.from_dict(json.loads(str(t))) for t in clean["latent_json"]]
    par = [json.loads(str(t)) for t in clean["aug_params_json"]]
    rrows = []
    for s in DENSE:
        Xr = np.empty((len(tid), grid.size), dtype=np.float32)
        for i in range(len(tid)):
            st = state_from_clean(float(clean["T_K"][i]), float(clean["c_pct"][i]),
                                  float(clean["E_kVcm"][i]), omega_grid=grid, latent=lat[i])
            Xr[i] = apply_aug_params(st, par[i], s, noise_rng=None).spectrum.astype(np.float32)
        for name in list(BASE) + list(TUNED):
            p = np.mean([plogm(e, Xr, condr) for e in models[name]], 0)
            rrows.append({"dataset_version": DS.VERSION, "set": "replicate_eval", "model": name,
                          "severity": s, "MAE_ensemble": float(np.abs(p - yr).mean()),
                          "rho": tuple_rho(tid, yr, p)})
        print(f"  dense replicate s={s} done", flush=True)

    pd.DataFrame(rows).to_csv(OUT / "dense_sweep_stress.csv", index=False)
    pd.DataFrame(rrows).to_csv(OUT / "dense_sweep_replicate.csv", index=False)

    # crossover per model
    cross = []
    df = pd.DataFrame(rows)
    for name in models:
        sub = df[(df.model == name)].sort_values("severity")
        first = next((r.severity for r in sub.itertuples() if not r.beats_metadata), None)
        cross.append({"dataset_version": DS.VERSION, "model": name,
                      "value_crossover_severity": first})
    pd.DataFrame(cross).to_csv(OUT / "value_crossover.csv", index=False)
    print("\ncrossovers:", {c["model"]: c["value_crossover_severity"] for c in cross})

    # ---- one-factor sweeps ------------------------------------------------
    lat_s = [LatentDraw.from_dict(json.loads(str(t))) for t in
             np.load(NPZ, allow_pickle=False)["latent_json"][np.load(NPZ, allow_pickle=False)["split"] == "stress_base"]]
    par_s = [json.loads(str(t)) for t in
             np.load(NPZ, allow_pickle=False)["aug_params_json"][np.load(NPZ, allow_pickle=False)["split"] == "stress_base"]]
    g = ds._omega_grid
    frows = []
    for kind, levels in LEVELS.items():
        for lv in levels:
            Xf = np.empty((len(ds), g.size), dtype=np.float32)
            for i in range(len(ds)):
                st = state_from_clean(float(ds._T_K[i]), float(ds._c_pct[i]), float(ds._E_kVcm[i]),
                                      omega_grid=g, latent=lat_s[i])
                pp = one_factor_params(kind, lv, par_s[i]["noise_seed"])
                Xf[i] = apply_aug_params(st, pp, 1.0, noise_rng=None).spectrum.astype(np.float32)
            for name in list(BASE) + list(TUNED):
                p = np.mean([plogm(e, Xf, cond) for e in models[name]], 0)
                e = np.abs(p - y)
                d, lo, hi = paired(e, err_meta, rng)
                frows.append({"dataset_version": DS.VERSION, "factor": kind, "level": lv,
                              "model": name, "MAE_ensemble": float(e.mean()),
                              "vs_metadata_diff": d, "lo": lo, "hi": hi,
                              "beats_metadata": bool(hi < 0)})
            print(f"  one-factor {kind}={lv} done", flush=True)
    pd.DataFrame(frows).to_csv(OUT / "one_factor_sweeps.csv", index=False)

    # ---- shaped-background sweep (figS2, round-37 decision 3) --------------
    shapes = draw_shaped_backgrounds(LEVELS["background"])
    brows = []
    for lv in LEVELS["background"]:
        sh = shapes[float(lv)]
        Xf = np.empty((len(ds), g.size), dtype=np.float32)
        for i in range(len(ds)):
            st = state_from_clean(float(ds._T_K[i]), float(ds._c_pct[i]), float(ds._E_kVcm[i]),
                                  omega_grid=g, latent=lat_s[i])
            pp = one_factor_params("background_shaped", lv, par_s[i]["noise_seed"], shaped=sh)
            Xf[i] = apply_aug_params(st, pp, 1.0, noise_rng=None).spectrum.astype(np.float32)
        for name in list(BASE) + list(TUNED):
            pr = np.mean([plogm(e, Xf, cond) for e in models[name]], 0)
            e = np.abs(pr - y)
            d, lo, hi = paired(e, err_meta, rng)
            brows.append({"dataset_version": DS.VERSION, "factor": "background_shaped",
                          "level": lv, "model": name, "MAE_ensemble": float(e.mean()),
                          "vs_metadata_diff": d, "lo": lo, "hi": hi,
                          "beats_metadata": bool(hi < 0),
                          "bg_type": sh["bg_type"],
                          **{f"bg_{k}": v for k, v in sh["bg_shape"].items()}})
        print(f"  one-factor background_shaped={lv} ({sh['bg_type']}) done", flush=True)
    pd.DataFrame(brows).to_csv(OUT / "one_factor_background_shaped.csv", index=False)
    (OUT / "run_sidecar.json").write_text(json.dumps({
        "task": "A8", "dataset_version": DS.VERSION, "dense_severities": list(DENSE),
        "one_factor_levels": LEVELS, "seeds": list(SEEDS),
        "shaped_background_seed": SHAPED_SEED,
        "shaped_background_draws": {str(k): v for k, v in shapes.items()},
        "crossover_definition": "first severity at which the paired CI of "
                                "(model - metadata reference) includes zero",
        "date_utc": datetime.now(timezone.utc).isoformat()}, indent=2))
    print("\nwrote", OUT)


if __name__ == "__main__":
    main()
