# Benchmarking & Profiling PASCAL (Phase 0/1, then vectorization)

This replaces `PARALLELIZATION_GUIDE.md`, `QUICKSTART_PARALLEL.md` and
`CYTHON_OPTIMIZATION_GUIDE.md`, which were removed. Those documents were
written in a previous session, never actually run, and contained
illustrative/fabricated benchmark numbers (e.g. "Sequential: 245.32s,
Parallel: 41.18s, 5.96x speedup") rather than measurements. Everything below
was actually executed.

## What's here

- `benchmarks/scenario.py` — builds a self-contained synthetic `Pascal1D`
  scenario (in-memory `ConstantReader`-style environment, no real netCDF
  input data, no dependency on the missing `aux_funcs.py` the old testcase
  scripts relied on). Deterministic given a seed. Good for timing/scaling/
  profiling work; the biological parameter values (cmm thresholds etc.) are
  illustrative, not validated against field data.
- `tests/test_smoke.py` — two fast pytest checks:
  1. a tiny sequential run completes and produces output files
  2. with `stochastic=False`, two runs with identical inputs produce
     identical population trajectories (this determinism is the baseline
     Phase 3 will use to check a parallel implementation doesn't diverge)
- `benchmarks/run_local_benchmark.py` — CLI to run the sequential simulation
  at a configurable size, optionally under `cProfile`.

Run the smoke tests:
```bash
python -m pytest tests/ -v
```

Run a benchmark + profile:
```bash
python benchmarks/run_local_benchmark.py --n-super 200 --duration 0.5 \
    --profile --profile-out profile.stats
python view_profile.py profile.stats   # richer breakdown, still useful as-is
```

## A bug this surfaced (fixed)

`coupler.py::write_lifestats()` crashed with `KeyError: 'total_fecundity'`
on any run that ends before a single super-individual has died — the
death-related columns only exist in `individual_stats` once
`record_lifestats()` has run at least once. Any short benchmark/smoke-test
run hits this by construction, so it's fixed (guarded) now rather than
worked around with unrealistic settings.

## Real profiling results

Run: 200 super-individuals, 10,000 virtual individuals each, 0.5 sim-years
(730 six-hourly timesteps), single core, `pascal_modular` conda env.

```
wall_time_s=20.282
ms_per_timestep=27.784
final_population_size=770556  (started at 200 super-individuals seeded)
```

Top cost centers by cumulative time (`cProfile`, sorted by `cumtime`):

| function | cumtime | tottime | calls |
|---|---|---|---|
| `update_lifestage` (coupler, all individuals) | 9.11s (45%) | 0.05s | 730 |
| `log_spatial` (coupler) | 8.91s (44%) | 0.22s | 730 |
| `resolve_spatial` (data_logger) | 7.75s (38%) | **7.58s** | 730 |
| `stage_3_9` (individual) | 5.18s | 0.37s | 80,244 |
| `verticalmigration_dsc1`/`dsc2` | 3.7s combined | 2.2s | 124k |
| `numpy.asarray` | 2.11s | 2.11s | 33,755 |
| `active_supindividuals` (coupler) | 2.06s | 0.02s | 6,277 |

### This changes the plan

The previous guides' entire framing — "parallelize `update_lifestage`,
that's the embarrassingly-parallel bottleneck" — **is only half the
picture**. `log_spatial` costs almost exactly as much as `update_lifestage`
(44% vs 45%), and nearly all of that (7.58s of 7.75s — 38% of total runtime)
is `data_logger.py::resolve_spatial()`'s **self time**: a hand-rolled,
purely-serial nested Python loop (`for i, this_cxyz in enumerate(cxyz): for
var, data in data_dict.items(): gridded_data[var][...] += data[i]`) doing
scatter-add into a 4D output grid, once per individual per timestep.

By Amdahl's law: even a perfect, zero-overhead parallelization of
`update_lifestage` across infinite cores caps total speedup at roughly
**1 / (1 − 0.45) ≈ 1.8×**, because `resolve_spatial` alone would remain as
~44% serial residual. The old guides' "4-8x on 8 cores" claim was never
achievable with `resolve_spatial` left untouched, independent of whether
`coupler_parallel.py`'s multiprocessing approach itself worked correctly
(it doesn't — see below).

Two secondary findings, both easy wins, neither requiring multiprocessing:
- `resolve_spatial`'s Python loop is a textbook `numpy` scatter-add
  (`np.add.at` or index-based bincount) — should vectorize away almost
  entirely.
- `active_supindividuals()` reconstructs a numpy object array from the
  Python list via `np.asarray(self.supindividuals)[mask]` on every call,
  and is called 6,277 times in this 730-step run (multiple times per
  timestep, from `log_spatial`, `gene_hunt`, `respawn`, `clean_dead`,
  `environment_indices`) — an O(N) reconversion repeated needlessly.
  `numpy.asarray` alone costs 2.1s (10% of total runtime).

## Vectorization results (resolve_spatial + active_supindividuals dedup)

Same scenario, same seed (200 super-individuals, 0.5 sim-years, seed=0),
measured before and after the changes below:

```
                  before      after      change
wall_time_s       20.282      11.202     1.81x faster
ms_per_timestep   27.784      15.346
log_spatial       8.909s      0.994s     8.96x faster (44% -> 9% of runtime)
active_supindividuals() calls   6,277    2,934     -53%
numpy.asarray tottime           2.110s   1.072s    -49%
final_population_size (identical seed)  770555.7696734982  770555.7696734982  bit-identical
```

`resolve_spatial` no longer appears in the top-20-by-cumtime at all.
`update_lifestage` is now unambiguously the dominant cost at **8.858s /
11.202s = 79%** of total runtime (up from 45%, simply because the thing it
was sharing the profile with is mostly gone) — which is exactly what Phase 2
(parallelizing it) needs to be true for multiprocessing to pay off close to
core-count.

Two changes, in `data_logger.py` and `coupler.py`:

1. **`data_logger.py::resolve_spatial()`** — replaced the nested Python
   loop with a single vectorized `np.add.at` scatter-add. This is a
   **behavior fix, not just a speedup**: the original code had an outer
   `for d in range(devstages): if any individual is in stage d: <loop over
   ALL individuals>` wrapper. Because the inner loop wasn't restricted to
   stage-`d` individuals, every individual's contribution was re-added once
   per *distinct* developmental stage present that timestep — with a
   population spread across most of the 13 stages (the normal case for a
   mixed-age cohort), spatial output (`output_ps.nc`'s `nvindividuals`,
   `structuralmass`, etc.) was inflated by up to ~13x. A regression test
   (`tests/test_data_logger.py::test_resolve_spatial_conserves_total_mass_across_multiple_stages`)
   reproduces this precisely: the old code overcounted by 3,303,455 /
   254,112 ≈ **13.0x** on a population spanning all 13 stages. **If anyone
   has already used `coupler.py`'s spatial NetCDF output for analysis, it
   should be treated as unreliable and rerun** — this bug predates any of
   this parallelization work and is independent of it.
2. **`coupler.py`** — `active_supindividuals()` (which rebuilds a numpy
   object array from the Python list + a boolean mask on every call) was
   being called multiple times per method body — most severely in
   `clean_dead()`, where it was called once per removed individual inside
   two separate loops. Consolidated to one call per method, cached in a
   local variable and reused. This is a pure call-site refactor; nothing
   about mutation ordering changed (verified: in every case, the cached
   snapshot is only read from before `self.supindividuals` is mutated
   later in the same method).

Full test suite (`tests/`, 5 tests: smoke x2, data_logger x3) passes
throughout.

### Revised phase order

1. ~~Phase 0: fix test harness~~ — done.
2. ~~Phase 1: profile~~ — done.
3. ~~Vectorize `resolve_spatial`, dedup `active_supindividuals()`~~ — done,
   1.81x on this scenario, plus a real correctness bug fixed as a byproduct.
4. **Next: Phase 2** — fix `coupler_parallel.py`'s per-individual full-array
   pickling problem for `update_lifestage`, now that it's 79% of runtime and
   no longer capped by a serial `log_spatial`.
5. Phase 3: correctness validation (sequential vs. parallel, using the
   `stochastic=False` determinism check already in `tests/test_smoke.py`).
6. Phase 4: revisit whether vectorizing `update_lifestage` itself
   (structure-of-arrays across individuals sharing a developmental stage)
   beats or complements multiprocessing — `stage_3_9`/vertical-migration
   calls are plain scalar arithmetic per the profile, so this is feasible;
   worth a decision once Phase 2/3 numbers exist.
7. Phase 5: container + HPC scaling test.

This file will be extended (not replaced) as each phase lands, with real
numbers each time — no illustrative/example output going forward.
