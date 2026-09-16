"""A17 paired comparisons: variants vs family base, and best-of-family."""
from __future__ import annotations
import json, sys
from pathlib import Path
import numpy as np, pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(Path(__file__).resolve().parent))
import torch  # noqa
from a17_arch_sweep import VARIANTS, build_variant, n_params  # noqa
from a4_eval import load_a4_model  # noqa
from sbc.data.dataset import InsSpectraDataset  # noqa
from sbc.models.spectral_transformer import TargetStats, unstandardize_outputs  # noqa
import dataset_paths as DS

A17 = DS.rev("A17")
A4E = DS.rev("A4_extended") / "checkpoints"
# The sweep's variants are split across two result directories: the
# architecture variants were trained into A17, the receptive-field/patch-width
# variants into A19. VARIANTS spans both, so checkpoints are resolved over both.
CKPT_DIRS = tuple(dict.fromkeys(
    [A17 / "checkpoints"] + [DS.rev(t) / "checkpoints" for t in DS.TUNED_CKPT_TASKS]))
NPZ = DS.FULL
SEEDS = (42, 43, 44)


def variant_ckpt(name, seed):
    for d in CKPT_DIRS:
        p = d / f"{name}_v7_seed{seed}.pt"
        if p.exists():
            return p
    raise FileNotFoundError(
        f"{name}_v7_seed{seed}.pt not found in " + " or ".join(str(d) for d in CKPT_DIRS))


def predict_variant(name, seeds, X):
    outs = []
    for s in seeds:
        ck = torch.load(variant_ckpt(name, s), map_location="cpu",
                        weights_only=False)
        net = build_variant(name, s); net.load_state_dict(ck["state_dict"]); net.eval()
        st = TargetStats.from_dict(ck["target_stats"])
        with torch.no_grad():
            o = net(torch.from_numpy(X)).numpy()
        outs.append(np.log(np.clip(unstandardize_outputs(o, st)["M"], 1e-9, None)))
    return np.mean(outs, axis=0)


def paired(a, b, rng, n_boot=1000):
    d = a - b; i = rng.integers(0, len(d), size=(n_boot, len(d))); m = d[i].mean(1)
    return float(d.mean()), float(np.percentile(m, 2.5)), float(np.percentile(m, 97.5))


def main():
    A17.mkdir(parents=True, exist_ok=True)
    ds = InsSpectraDataset(NPZ, "val", severity=1.0, as_torch=False)
    X = np.stack([ds.get_augmented(i, 1.0) for i in range(len(ds))]).astype(np.float32)
    y = np.log(np.clip(ds._M.astype(float), 1e-9, None))
    batch = {"spectrum": X, "T_K": ds._T_K.astype(float), "c_pct": ds._c_pct.astype(float),
             "E_kVcm": ds._E_kVcm.astype(float)}
    rng = np.random.default_rng(20260906)

    # Family bases. The sweep trains its own `cnn_base` and `tf_base` on the same
    # dataset, split and seeds as every other variant, so those are the right
    # comparators and the paired difference is within one dataset. Historically
    # the bases were taken from A4_extended instead; that directory exists only
    # for v9, and borrowing the A21 grid would compare across datasets
    # (A17 trains on full_dataset_phase1_vN, A21 on ablation_vN_both_seed0), which
    # would confound architecture with dataset realization. Fall back to A4_extended
    # only where the sweep has no base of its own.
    A4_BASE = {"cnn_base": "cnn", "tf_base": "5a"}

    def base_from_a4(name):
        return np.mean([np.log(np.clip(
            load_a4_model(A4_BASE[name], "v7", s, ckpt_dir=A4E).predict_M(batch), 1e-9, None))
            for s in SEEDS], axis=0)

    preds = {}
    for name in VARIANTS:
        if name in A4_BASE and not all(
                any((d / f"{name}_v7_seed{s}.pt").exists() for d in CKPT_DIRS) for s in SEEDS):
            print(f"  {name}: no base in the sweep directories, falling back to A4_extended",
                  flush=True)
            preds[name] = base_from_a4(name)
        else:
            preds[name] = predict_variant(name, SEEDS, X)
    err = {k: np.abs(v - y) for k, v in preds.items()}

    rows = []
    for name, (fam, _) in VARIANTS.items():
        base = "cnn_base" if fam == "cnn" else "tf_base"
        d, lo, hi = paired(err[name], err[base], rng)
        rows.append({"dataset_version": DS.VERSION, "family": fam, "variant": name,
                     "n_parameters": n_params(build_variant(name, 42)),
                     "MAE_logM_ensemble": float(err[name].mean()),
                     "vs_base_diff": d, "vs_base_lo": lo, "vs_base_hi": hi,
                     "significant_vs_base": bool(hi < 0 or lo > 0)})
    df = pd.DataFrame(rows).sort_values(["family", "MAE_logM_ensemble"])
    df.to_csv(A17 / "variant_paired_vs_base.csv", index=False)
    print(df.to_string(index=False, float_format=lambda x: f"{x:.4f}"))

    best_cnn = min([k for k, v in VARIANTS.items() if v[0] == "cnn"], key=lambda k: err[k].mean())
    best_tf = min([k for k, v in VARIANTS.items() if v[0] == "tf"], key=lambda k: err[k].mean())
    d, lo, hi = paired(err[best_cnn], err[best_tf], rng)
    fam = {"best_cnn": best_cnn, "best_tf": best_tf,
           "best_cnn_MAE": float(err[best_cnn].mean()), "best_tf_MAE": float(err[best_tf].mean()),
           "diff_cnn_minus_tf": d, "lo": lo, "hi": hi, "significant": bool(hi < 0 or lo > 0),
           "note": "best-of-family selected on the same val split used for the CI; "
                   "mildly optimistic for both families"}
    (A17 / "best_of_family.json").write_text(json.dumps(fam, indent=2))
    print(f"\nbest CNN = {best_cnn} ({fam['best_cnn_MAE']:.4f}); "
          f"best transformer = {best_tf} ({fam['best_tf_MAE']:.4f})")
    print(f"paired difference {d:+.4f} [{lo:+.4f}, {hi:+.4f}] "
          f"{'SIGNIFICANT' if fam['significant'] else 'not significant'}")


if __name__ == "__main__":
    main()
