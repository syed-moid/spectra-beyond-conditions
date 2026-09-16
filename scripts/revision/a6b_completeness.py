"""A6 completeness (round-12 ruling 4).

(a) parameter-recovery table at s in {0.5, 2, 4};
(b) MAE Gamma split by overdamped (Gamma > omega0) vs underdamped at s = 1, for
    the tuned CNN and the matched-model fit - the regime v7 section 2.1 claims
    fitting fails in;
(c) predicted-vs-reference scatter CSVs for the A11 figure.
"""

from __future__ import annotations

import json, sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(Path(__file__).resolve().parent))
import torch  # noqa: E402
from a17_arch_sweep import build_variant  # noqa: E402
from sbc.data.dataset import InsSpectraDataset  # noqa: E402
from sbc.models.spectral_transformer import TargetStats, unstandardize_outputs  # noqa: E402
import dataset_paths as DS

REV = ROOT / "results" / "revision"
NPZ = DS.FULL
OUT = DS.rev("A6")
SEEDS = (42, 43, 44)


@torch.no_grad()
def predict(nets, X):
    outs = []
    for net, st in nets:
        o = np.concatenate([net(torch.from_numpy(X[i:i+2048].astype(np.float32))).numpy()
                            for i in range(0, len(X), 2048)], 0)
        outs.append(unstandardize_outputs(o, st))
    return {k: np.mean([o[k] for o in outs], 0) for k in ("M", "omega_Q", "Gamma_Q")}


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    ds = InsSpectraDataset(NPZ, "stress_base", severity=1.0, as_torch=False)
    om_t, gm_t = ds._omega_Q.astype(float), ds._Gamma_Q.astype(float)
    nets = []
    for s in SEEDS:
        c = torch.load(DS.tuned_ckpt("cnn_kernel45", s), map_location="cpu", weights_only=False)
        n = build_variant("cnn_kernel45", s); n.load_state_dict(c["state_dict"]); n.eval()
        nets.append((n, TargetStats.from_dict(c["target_stats"])))

    fits = pd.concat([pd.read_csv(DS.rev("A3") / "fit_results_per_spectrum.csv"),
                      pd.read_csv(DS.rev("A3") / "fit_results_extra_variants.csv")])

    # (b) overdamped split at s = 1
    over = gm_t > om_t
    X1 = np.stack([ds.get_augmented(i, 1.0) for i in range(len(ds))]).astype(np.float32)
    pr = predict(nets, X1)
    rows_b = []
    for label, mask in (("overdamped (Gamma > omega0)", over), ("underdamped", ~over)):
        rows_b.append({"dataset_version": DS.VERSION, "regime": label, "n": int(mask.sum()),
                       "model": "tuned 1D CNN",
                       "MAE_Gamma_meV": float(np.abs(pr["Gamma_Q"] - gm_t)[mask].mean()),
                       "MAE_omega0_meV": float(np.abs(pr["omega_Q"] - om_t)[mask].mean())})
        for meth in ("dho_matched", "dho_two_mode"):
            d = fits[(fits.method == meth) & (fits.severity == 1.0)].sort_values("idx")
            ok = d.fit_ok.astype(bool).to_numpy() & mask
            rows_b.append({"dataset_version": DS.VERSION, "regime": label, "n": int(ok.sum()),
                           "model": meth,
                           "MAE_Gamma_meV": float(np.abs(d.Gamma_fit.to_numpy() - gm_t)[ok].mean()),
                           "MAE_omega0_meV": float(np.abs(d.omega0_fit.to_numpy() - om_t)[ok].mean())})
    pd.DataFrame(rows_b).to_csv(OUT / "gamma_by_damping_regime.csv", index=False)
    print(pd.DataFrame(rows_b).to_string(index=False, float_format=lambda x: f"{x:.4f}"))

    # (a) other severities + (c) scatter CSVs
    rows_a, scat = [], []
    for sev in (0.5, 1.0, 2.0, 4.0):
        X = np.stack([ds.get_augmented(i, sev) for i in range(len(ds))]).astype(np.float32)
        p = predict(nets, X)
        rows_a.append({"dataset_version": DS.VERSION, "model": "cnn_kernel45", "severity": sev,
                       "MAE_omega0_meV": float(np.abs(p["omega_Q"] - om_t).mean()),
                       "MAE_Gamma_meV": float(np.abs(p["Gamma_Q"] - gm_t).mean())})
        if sev == 1.0:
            scat.append(pd.DataFrame({"dataset_version": DS.VERSION, "model": "tuned_cnn",
                                      "severity": sev, "omega0_true": om_t,
                                      "omega0_pred": p["omega_Q"], "Gamma_true": gm_t,
                                      "Gamma_pred": p["Gamma_Q"]}))
            for meth in ("dho_matched", "dho_two_mode"):
                d = fits[(fits.method == meth) & (fits.severity == 1.0)].sort_values("idx")
                scat.append(pd.DataFrame({"dataset_version": DS.VERSION, "model": meth, "severity": sev,
                                          "omega0_true": om_t, "omega0_pred": d.omega0_fit.to_numpy(),
                                          "Gamma_true": gm_t, "Gamma_pred": d.Gamma_fit.to_numpy(),
                                          "fit_ok": d.fit_ok.to_numpy()}))
    pd.DataFrame(rows_a).to_csv(OUT / "tuned_cnn_params_by_severity.csv", index=False)
    pd.concat(scat, ignore_index=True).to_csv(OUT / "predicted_vs_reference_scatter.csv", index=False)
    print("\n" + pd.DataFrame(rows_a).to_string(index=False, float_format=lambda x: f"{x:.4f}"))
    (OUT / "run_sidecar_completeness.json").write_text(json.dumps({
        "task": "A6b", "dataset_version": DS.VERSION, "seeds": list(SEEDS),
        "date_utc": datetime.now(timezone.utc).isoformat()}, indent=2))


if __name__ == "__main__":
    main()
