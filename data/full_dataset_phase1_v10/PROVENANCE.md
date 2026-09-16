# full_dataset_phase1_v10 — provenance

## What changed from v9

The central-peak term was replaced. v9 added an even Lorentzian in ω; v10 adds the equilibrium
relaxational form carried by the same Bose factor as the oscillators:

    S_cp(w) = A(T) * [n(w)+1] * w*tau / (1 + (w*tau)^2),   tau = 1/(1.5 meV),
    A(T)    = I_c / (pi * k_B * T),   I_c = 600 * exp(-|T - T_C| / 30),  T_C = 395 K.

The value at w = 0 is unchanged (127.324 in generator units), so the peak height of the term is
the same; what changes is that the assembled spectrum — not only its oscillator part — now
satisfies detailed balance. The assembled-spectrum residual on S(+5)/S(-5) against exp(5/k_BT)
falls from 2.5e-05 (v9) to 9.0e-08 (v10).

Nothing else changed. `bose()`, the oscillator lineshape, the resolution kernel, the backgrounds
and the Poisson scaling are identical to v9.

## Verification against v9

Every stored array except the spectra is **bitwise identical** to v9:

    omega_grid, T_K, c_pct, E_kVcm, omega_Q, Gamma_Q, M, stratum, split,
    aug_params_json, latent_json, replay_test_indices, replay_test_severities,
    master_seed, augmentation_config_yaml, schema_version

so labels, latents, conditions and splits are unchanged, and the A1 metadata oracle reproduces
its v9 value exactly.

`spectra_clean` differs by a relative L2 of **8.0e-07** over the whole dataset. Per split:

| split | relative L2 (v9 -> v10) |
|---|---|
| train | 8.261e-07 |
| val | 7.913e-07 |
| stress_base | 8.397e-07 |
| holdout_c | 7.735e-07 |
| holdout_E | 8.723e-07 |
| **holdout_T** | **0.000e+00** |

Two checks confirm the change is exactly the central-peak term and nothing else.
`holdout_T` is at T = 600 K, which is 205 K from T_C, so the central peak is switched off there
by the |T - T_C| < 50 K gate — and that split is bitwise unchanged, as it must be. And the
difference across the other splits is concentrated at |w| ~ 1.5 meV, which is the central peak's
HWHM.

`replay_test_reference` differs, as it must: it stores replayed spectra.

## Generator hashes

SHA-256 of `manuscript_phase1/src/data/spectrum_generator.py`:

| version | SHA-256 (source file) |
|---|---|
| v7 | 306dc767... |
| v8 | 9e68bfed... |
| v9 | 4035f7a3... |
| v10 | 8cb82ecfe714cac2a1d37e41ca2928b4c10ff42bf676aec1b61866ffbe4398ab |

## Known defect in the stored metadata

The `generator_git_sha` field **inside this .npz is stale**: it reads
`997f2a0e1fe6431497da000c017528cc6a83b21d`, which is the repository HEAD at generation time and
is byte-for-byte the same value stored in the v9 file. It records the last commit, and the
generator change that distinguishes v9 from v10 was uncommitted when both datasets were written,
so **that field does not distinguish v9 from v10 and must not be used to identify the generator
version.** The source-file SHA-256 above is the identifier that does.

This is recorded rather than silently repaired: rewriting the field would change the file without
changing what produced it. Whoever cuts the release tag should confirm that the committed
generator hashes to `8cb82ecf...` before the tag is used to reproduce anything.

## How results depend on this file

Augmentation replay reconstructs each degraded spectrum from the generator source at run time, so
a checkpoint trained under one generator version does not reproduce its recorded score under
another. Reproducing any number requires checking out the pinned generator, not merely obtaining
this .npz.
