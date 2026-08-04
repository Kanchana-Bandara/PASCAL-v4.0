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

## Phase 2: fixing coupler_parallel.py - and a negative result

### The bug fixed

Every `SuperIndividual` holds a direct reference to the *entire* shared
per-timestep environment arrays (`self.environment`, `self.environment_profiles`
- see `individual.py`'s `__init__` and `coupler.py`'s `seed()`), not just its
own column. In the advection case (`PascalAdvection`, one OpenDrift tracker
element per super-individual - as opposed to `Pascal1D`, where every
individual shares a single column and this doesn't apply) those arrays have
shape `(n_depth, n_super_individuals)`. The original `coupler_parallel.py`
did `pool.map(_update_individual_worker, self.supindividuals)`, which
pickles each `SuperIndividual` independently - meaning the *entire* shared
environment array got serialized once per individual, per timestep, in both
directions. This was never benchmarked (see top of this document); it's
exactly the kind of thing that looks fine in a code review and falls over
under load.

Fixed by extracting each individual's own small slice of just the
variables `update_lifestage()`'s call tree actually reads - now declared
explicitly as `individual.py::PROFILE_ENVIRONMENT_VARIABLES` /
`SCALAR_ENVIRONMENT_VARIABLES`, colocated with the code that reads them -
before dispatch, and restoring the individual's real
`environment_index`/`environment`/`environment_profiles` (needed by
`coupler.py`'s `log_spatial`/`gene_hunt`/`respawn`, which index into the
tracker's element arrays by `environment_index`) once results come back.
`individual.py` itself is unchanged.

Also fixed: workers previously never got their own RNG seed, so multiple
persistent pool workers would draw statistically correlated random streams
for the life of the pool (sex determination, diapause strategy, gene
crossover/mutation all use numpy's global RNG). `_init_worker_rng` now
seeds each worker independently via `SeedSequence.spawn`, indexed by a
Pool-scoped atomic counter (deliberately *not*
`multiprocessing.current_process()._identity`, which is global to the
parent process's lifetime, not reset per `Pool` - an early version of this
fix used it and broke the moment a test created a second `Pool` in the same
process). Tested directly in `tests/test_coupler_parallel.py` since a run
short enough to test quickly doesn't reach the stochastic branches.

Correctness (not just the RNG fix, but the environment-slicing itself) is
checked by `tests/test_parallel_matches_sequential_for_deterministic_growth_path`:
a short run where individuals stay in early, non-stochastic stages, so
growth/mortality is a pure function of environment + state, and sequential
vs. parallel must produce bit-identical population trajectories. It does.

### The result: multiprocessing.Pool doesn't pay off here

Benchmarked sequential vs. parallel (4 workers) on the advection scenario,
same machine (8 cores) as all other numbers in this document:

```
n_super=200,  duration=0.5y:  sequential 16.1s   parallel 20.1s   (parallel 25% SLOWER)
n_super=500,  duration=0.3y:  sequential 19.4s   parallel 24.7s   (parallel 27% SLOWER)
n_super=1500, duration=0.3y:  sequential 50.8s   parallel 62.0s   (parallel 22% SLOWER)
```

Population trajectories are identical between sequential and parallel at
every size tested (as expected - these runs don't reach stochastic
branches either, same caveat as above; this confirms the slicing is
correct, not that the RNG fix is exercised end-to-end).

Profiling the *parallel* run's main-process time (500 super-individuals)
shows why: of 30.3s wall time, `{method 'dump' of '_pickle.Pickler'}` alone
is **9.9s (33%)**, and IPC waiting (`_wait_for_updates`, `connection.recv`,
`selectors.poll`, ...) accounts for most of the rest. This is *after*
fixing the environment-duplication bug - the remaining cost is the
fundamental per-call overhead of process-based IPC (pickling ~30-individual
chunks, sending them through a pipe, waiting, unpickling results), paid
once per timestep (438-730 times for these runs), against a payload that's
individually cheap: Phase 1's profiling put `update_lifestage()` at ~60
microseconds per individual. Increasing problem size didn't change the
picture (still ~22-27% slower at 1500 individuals) - the overhead scales
with individual count too, since chunk-pickling cost is roughly
proportional to how much state is in each chunk.

**Conclusion: the `coupler_parallel.py` fix was necessary (the old code was
either wrong-or-worse, never validated) but not sufficient.** Pure
process-based `multiprocessing.Pool`, dispatched once per timestep, is a
poor fit for this workload's granularity regardless of payload size,
because the per-individual compute is simply too cheap relative to IPC
overhead at these problem sizes. This is a genuine, evidence-based negative
result, not a reason to abandon parallelization - just a reason to change
approach. Two directions, not mutually exclusive:

- **Vectorization** (was "Phase 4"): batch same-stage individuals into
  numpy arrays and vectorize the `pascal42_mod_*` scalar-arithmetic calls.
  No IPC at all, so no per-call overhead ceiling - given `update_lifestage`
  is 79% of sequential runtime and its hot functions
  (`stage_3_9`/`verticalmigration_dsc1`/`dsc2`) are plain scalar math per
  Phase 1's profile, this is likely to be the more reliably-profitable
  investment at single-node/small-to-medium scale, and it's now the
  recommended next step ahead of further multiprocessing work.
- **Reduce IPC frequency**: dispatch to workers less often (e.g. hand each
  worker several consecutive timesteps' worth of its individuals' work
  before returning to the main process) rather than once per timestep, so
  pickling cost amortizes over more compute per round-trip. Bigger
  redesign (workers would need several steps' worth of environment data at
  once, not just one slice), not attempted here - worth revisiting if
  vectorization turns out to be insufficient, or specifically for genuine
  multi-node HPC scaling where population sizes are much larger and the
  per-call economics differ from what's measured here.

`coupler_parallel.py` is kept (with the bug fixes) since it's still
correct and may pay off at problem sizes/machines not tested here - just
not recommended as the primary path forward given what's actually been
measured.

## Interlude: a frozen-environment correctness bug (unrelated to parallelization)

Found while deciding exactly what in `update_lifestage()` to vectorize -
worth fixing before vectorizing on top of it, so this landed first.

`update_environment()` creates brand new `environment`/`environment_profiles`
objects every timestep (confirmed: 72 distinct object identities across a
72-step run, in both `Pascal1D` and `PascalAdvection`). `SuperIndividual`
only ever captured those references once, in `seed()` at construction time,
and nothing re-pointed an already-active individual at the new objects
afterward. Practical effect: every super-individual read whatever
temperature/food/irradiance/predation data existed at the exact timestep it
was seeded, frozen for its entire lifespan, no matter how the actual
environment evolved. Verified directly with a temperature ramp - an
individual seeded early was still reading the t=0 value 73 timesteps later.
This predates and is entirely independent of the parallelization work
here, and plausibly explains the "why are the numbers dying weird" question
already in `notes`.

Fixed with `coupler.py::sync_environment_references()`, called from `run()`
right after `update_environment()`. `tests/test_environment_sync.py`
reproduces the bug directly (confirmed failing without the fix).

**This changes population dynamics** compared to every number measured
earlier in this document - individuals now actually respond to a changing
environment instead of a frozen snapshot. The qualitative conclusions above
(resolve_spatial's overcounting bug and fix, multiprocessing's IPC-overhead
problem) aren't affected by this - they're about the *shape* of where time
goes and a data-aggregation bug, not about population trajectories - but
none of the absolute wall-clock/population numbers above should be assumed
to still match run-for-run after this fix; they weren't re-measured against
it.

### Revised phase order

1. ~~Phase 0: fix test harness~~ — done.
2. ~~Phase 1: profile~~ — done.
3. ~~Vectorize `resolve_spatial`, dedup `active_supindividuals()`~~ — done,
   1.81x on this scenario, plus a real correctness bug fixed as a byproduct.
4. ~~Phase 2: fix `coupler_parallel.py`'s environment-pickling bug and RNG
   seeding~~ — done, correctness confirmed, but multiprocessing.Pool nets a
   *loss* (22-27% slower) at all sizes tested due to IPC overhead - see
   above.
5. ~~Fix frozen-environment bug~~ — done, see above.
6. **Next: vectorize `update_lifestage`'s hot path** (structure-of-arrays
   across individuals sharing a developmental stage) - no IPC overhead
   ceiling, and the target functions are already known from Phase 1's
   profile. Will re-profile first, since the environment-sync fix likely
   shifts which functions actually dominate now that individuals
   experience realistic (changing) conditions instead of a frozen
   snapshot.
7. Container + HPC scaling test - now specifically useful for checking
   whether the multiprocessing economics differ at genuinely large
   population sizes / multi-node, independent of whatever vectorization
   achieves single-node.

This file will be extended (not replaced) as each phase lands, with real
numbers each time — no illustrative/example output going forward.
