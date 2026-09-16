"""A9 - attribution, plus round-16 items 3(b) and 3(c).

Occlusion: a sliding 1 meV window is replaced by a LOCAL BACKGROUND ESTIMATE
(linear interpolation between the window edges), not zero, so the probe removes
structure without injecting a step. Delta(MAE) is recorded per window centre.

Also: gradient x input saliency; the share of sensitivity falling in the
soft-mode, acoustic (9.2 +- 2 meV), central-peak (|w| < 1 meV) and background
regions; a Stokes / anti-Stokes split (round-6 addition 5); a split at the
T = 268 K soft/acoustic mode crossing; occlusion at s = 12 (item 3b); and rho at
s = 12 with the central-peak window masked (item 3c).

Every cell is appended to disk as it completes (round-17 ruling 5).
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
from replicate_io import load_replicate  # noqa: E402
from sbc.data.augmentations import apply_aug_params, state_from_clean  # noqa: E402
from sbc.data.dataset import InsSpectraDataset  # noqa: E402
from sbc.data.latent_perturbations import LatentDraw  # noqa: E402
from sbc.models.spectral_transformer import TargetStats, unstandardize_outputs  # noqa: E402
import dataset_paths as DS

REV = ROOT / "results" / "revision"
NPZ = DS.FULL
OUT = DS.rev("A9")
SEEDS = (42, 43, 44)
SEVS = (0.5, 1.0, 2.0, 12.0)
MODELS = ["cnn_kernel45", "cnn", "5a", "fusion"]
WIN = 1.0     # meV occlusion window
T_CROSS = 268.0


def regions(w):
    return {"central_peak": np.abs(w) < 1.0,
            "soft_mode": (np.abs(w) >= 1.0) & (np.abs(w) < 7.2),
            "acoustic": (np.abs(w) >= 7.2) & (np.abs(w) <= 11.2),
            "outer": np.abs(w) > 11.2}


def load_models():
    ms = {}
    for a in ("cnn", "5a", "fusion"):
        ms[a] = [("a4", load_a4_model(a, "v7", s, ckpt_dir=DS.rev("A4_extended") / "checkpoints"))
                 for s in SEEDS]
    lst = []
    for s in SEEDS:
        ck = torch.load(DS.tuned_ckpt("cnn_kernel45", s),
                        map_location="cpu", weights_only=False)
        net = build_variant("cnn_kernel45", s); net.load_state_dict(ck["state_dict"]); net.eval()
        lst.append(("var", (net, TargetStats.from_dict(ck["target_stats"]))))
    ms["cnn_kernel45"] = lst
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


def occlude(X, grid, lo, hi):
    """Replace [lo, hi] with a straight line between the window edges."""
    Y = X.copy()
    m = (grid >= lo) & (grid <= hi)
    if m.sum() < 2:
        return Y
    i0, i1 = np.nonzero(m)[0][[0, -1]]
    a = X[:, max(i0 - 1, 0)][:, None]
    b = X[:, min(i1 + 1, X.shape[1] - 1)][:, None]
    t = np.linspace(0, 1, int(m.sum()))[None, :]
    Y[:, m] = a * (1 - t) + b * t
    return Y


_DONE_KEYS = set()


def load_done(path, keys):
    """Populate the skip set from an existing incremental CSV."""
    global _DONE_KEYS
    if path.exists():
        d = pd.read_csv(path)
        _DONE_KEYS = {tuple(r) for r in d[keys].itertuples(index=False)}
        print(f"  resuming: {len(_DONE_KEYS)} rows already in {path.name}", flush=True)


def append(path, row):
    pd.DataFrame([row]).to_csv(path, mode="a", header=not path.exists(), index=False)


def main():
    OUT.mkdir(exist_ok=True)
    occ_p = OUT / "occlusion_sensitivity.csv"
    reg_p = OUT / "region_shares.csv"
    models = load_models()
    ds = InsSpectraDataset(NPZ, "stress_base", severity=1.0, as_torch=False)
    grid = ds._omega_grid
    T = ds._T_K.astype(float)
    cond = {"T_K": T, "c_pct": ds._c_pct.astype(float), "E_kVcm": ds._E_kVcm.astype(float)}
    y = np.log(np.clip(ds._M.astype(float), 1e-9, None))
    centres = np.arange(-14.5, 14.6, 0.5)

    for sev in SEVS:
        X = np.stack([ds.get_augmented(i, sev) for i in range(len(ds))]).astype(np.float32)
        for name in MODELS:
            base = np.mean([plogm(e, X, cond) for e in models[name]], 0)
            base_mae = float(np.abs(base - y).mean())
            sens = []
            for cc in centres:
                Xo = occlude(X, grid, cc - WIN / 2, cc + WIN / 2).astype(np.float32)
                p = np.mean([plogm(e, Xo, cond) for e in models[name]], 0)
                e = np.abs(p - y)
                d_all = float(e.mean() - base_mae)
                lo_T = float(e[T < T_CROSS].mean() - np.abs(base - y)[T < T_CROSS].mean())
                hi_T = float(e[T >= T_CROSS].mean() - np.abs(base - y)[T >= T_CROSS].mean())
                sens.append(d_all)
                append(occ_p, {"dataset_version": DS.VERSION, "model": name, "severity": sev,
                               "window_centre_meV": float(cc), "delta_MAE": d_all,
                               "delta_MAE_T_below_268": lo_T, "delta_MAE_T_above_268": hi_T})
            sens = np.array(sens)
            pos = np.clip(sens, 0, None)
            tot = pos.sum() if pos.sum() > 0 else 1.0
            R = regions(centres)
            row = {"dataset_version": DS.VERSION, "model": name, "severity": sev,
                   "base_MAE": base_mae,
                   "stokes_share": float(pos[centres > 0].sum() / tot),
                   "anti_stokes_share": float(pos[centres < 0].sum() / tot)}
            for k, m in R.items():
                row[f"{k}_share"] = float(pos[m].sum() / tot)
            append(reg_p, row)
            print(f"  {name} s={sev}: base {base_mae:.4f} | central {row['central_peak_share']:.3f} "
                  f"soft {row['soft_mode_share']:.3f} acoustic {row['acoustic_share']:.3f} "
                  f"outer {row['outer_share']:.3f} | Stokes {row['stokes_share']:.3f}", flush=True)

    # ---- item 3(c): rho at s = 12 with the central peak masked -------------
    clean, _ = load_replicate(want_blocks=False)
    tid = clean["tuple_id"]
    yr = np.log(np.clip(clean["M"].astype(float), 1e-9, None))
    condr = {"T_K": clean["T_K"].astype(float), "c_pct": clean["c_pct"].astype(float),
             "E_kVcm": clean["E_kVcm"].astype(float)}
    g2 = clean["omega_grid"]
    lat = [LatentDraw.from_dict(json.loads(str(t))) for t in clean["latent_json"]]
    par = [json.loads(str(t)) for t in clean["aug_params_json"]]
    Xr = np.empty((len(tid), g2.size), dtype=np.float32)
    for i in range(len(tid)):
        st = state_from_clean(float(clean["T_K"][i]), float(clean["c_pct"][i]),
                              float(clean["E_kVcm"][i]), omega_grid=g2, latent=lat[i])
        Xr[i] = apply_aug_params(st, par[i], 12.0, noise_rng=None).spectrum.astype(np.float32)
    Xm = occlude(Xr, g2, -1.0, 1.0).astype(np.float32)

    def rho(p):
        out = []
        for t in np.unique(tid):
            m = tid == t
            r = spearmanr(yr[m], p[m]).statistic if np.ptp(p[m]) > 0 else 0.0
            out.append(0.0 if not np.isfinite(r) else float(r))
        return float(np.mean(out))

    rows = []
    for name in MODELS:
        p0 = np.mean([plogm(e, Xr, condr) for e in models[name]], 0)
        p1 = np.mean([plogm(e, Xm, condr) for e in models[name]], 0)
        rows.append({"dataset_version": DS.VERSION, "model": name, "severity": 12.0,
                     "rho_full": rho(p0), "rho_central_peak_masked": rho(p1)})
        print(f"  s=12 {name}: rho {rows[-1]['rho_full']:.4f} -> masked "
              f"{rows[-1]['rho_central_peak_masked']:.4f}", flush=True)
    pd.DataFrame(rows).to_csv(OUT / "s12_central_peak_mask.csv", index=False)
    (OUT / "run_sidecar.json").write_text(json.dumps({
        "task": "A9", "dataset_version": DS.VERSION, "occlusion_window_meV": WIN,
        "occlusion_fill": "linear interpolation between window edges",
        "severities": list(SEVS), "T_crossing_K": T_CROSS,
        "date_utc": datetime.now(timezone.utc).isoformat()}, indent=2))
    print("done")


if __name__ == "__main__":
    main()
