# v9 rerun driver

`review_and_modify/scripts/run_pipeline.py` runs round-32 step 3 in ordered stages,
in as many sessions as you like.

## Typical use

```bash
# plan a session before committing to it (no work is done)
python review_and_modify/scripts/run_pipeline.py --start 17:00 --until 08:00 --dry-run

# run it: start now, stop by 08:00, keep the Mac awake
python review_and_modify/scripts/run_pipeline.py --until 08:00 --caffeinate

# where did it get to
python review_and_modify/scripts/run_pipeline.py --status

# next evening: same command, it continues from where it stopped
python review_and_modify/scripts/run_pipeline.py --until 08:00 --caffeinate
```

`--until` takes `08:00` (next occurrence), `2026-09-10T08:00`, or a duration
like `14h`.

## How stopping works

At the deadline the running stage gets SIGTERM, then SIGKILL after 120 s, and
the driver exits. Nothing is left half-written in a way that a restart would
double-count, because of the two stage kinds:

- **resumable** — the script keeps its own per-cell record and skips what is
  already on disk (`a9_attribution`, the twelve `a7_*`, `a18b_tilt_train`).
  These are started even if the deadline is close: whatever they finish is kept.
- **atomic** — the script writes in one pass and does not deduplicate on
  re-entry. The driver deletes that stage's outputs before every attempt, and
  refuses to start one that does not fit in the remaining time. So an atomic
  stage is either absent or complete, never partial.

`a14c_fitting_rho` is the reason the wipe matters: it appends without checking
what is already there, so a restart without the wipe would silently double its
rows.

## If it dies (OOM, crash, power)

The driver stops at the failed stage rather than letting downstream stages read
a half-written input. Re-run the same command: it retries that stage (wiping
first if atomic) and carries on. Per-stage logs are in `review_and_modify/logs/`,
state in `review_and_modify/.pipeline_state.json`.

If the machine dies without the driver exiting, the lock file may be stale — the
driver detects that by checking the recorded pid and clears it.

## Other flags

| flag | effect |
|---|---|
| `--status` | progress table and remaining estimate |
| `--from STAGE` | skip ahead |
| `--only STAGE [...]` | run just these |
| `--reset STAGE [...]` | mark stages pending so they re-run |
| `--skip-blocked` | continue past a stage that has no runnable script |
| `--start HH:MM` | dry-run only: plan from a future start time |

## Known blocker

`a7_postprocess` has no script. The v8 files
(`oracle_per_setting.csv`, `improvement_over_oracle.csv`, `rho_per_setting.csv`,
`variance_decomposition.csv`, `cross_mode_correlation.csv`,
`mps_reproducibility_pairs.csv`, `training_runs_dedup.csv`) were produced ad hoc
and the code was not kept. `a12_digest` reads four of them, so the digest will
report them missing until the step is rebuilt. Everything before and after it
runs; pass `--skip-blocked` to continue.

## Estimates

`est` values are measured v8 durations scaled by 1.42x, the v9/v8 factor
observed on A4 (400 vs 283 s/run). The driver records actuals and prints the
ratio as each stage finishes, so the numbers sharpen as you go.
