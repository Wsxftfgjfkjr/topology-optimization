# Benchmark suite

Reproducible measurements of the **`v0.1-reference`** solver, so that later optimized
implementations can be compared against a frozen baseline.

The suite does not modify the solver. `topoopt/`, `examples/` and `tests/` are untouched;
every stage measurement is obtained by wrapping existing callables from the outside.

## Quick start

```bash
python benchmarks/benchmark.py                          # 60x20, 120x40, 240x80
python benchmarks/benchmark.py --cases 60x20 480x160
python benchmarks/benchmark.py --repetitions 1 --warmups 0     # fast pass
python benchmarks/benchmark.py --label optimized        # writes optimized.csv
python benchmarks/benchmark.py --list-cases
```

Results land in `benchmarks/results/`. The run takes roughly 90 seconds on the reference
machine (see *Cost* below).

## What is measured

The timed region is **`TopologyOptimizer.run()` only**. Excluded:

- process start-up and imports
- optimizer construction (`build_filter_kernel`, `element_stiffness_matrix`)
- plotting, CSV and JSON writing
- CLI argument parsing

`verbose=False` is used throughout. That is numerically neutral, not just convenient:
`IterationRecord.time_s` is computed at `optimizer.py:298` and the `print` happens
afterwards at `:303-308`, so the per-iteration timing already excludes printing.

## Methodology

### One process per case

Each mesh size runs in its own subprocess. This is required, not stylistic:
`resource.getrusage().ru_maxrss` is a **high-water mark that never decreases**, so three
cases in one process would all report the largest case's peak memory. Isolation also
removes cross-case allocator contamination.

### Phases

| Phase | What | Timed |
|---|---|---|
| A. Warm-up | `--warmups` (default 1) full runs, discarded | no |
| B. Clean repetitions | `--repetitions` (default 3) runs, **fresh optimizer each** | **yes — authoritative** |
| C. Instrumented run | one fresh optimizer with wrapping active | advisory |

Phase C runs after the peak-memory sample, so instrumentation cannot inflate it.

**A fresh `TopologyOptimizer` per repetition is mandatory.** `run()` mutates
`self.density`; calling it twice on the same instance restarts from the *converged*
design, exits after one iteration and reports `converged=True`. That failure looks
entirely plausible, so phase B asserts all repetitions agree on the iteration count, and
phase C asserts its density digest and iteration count match phase B.

### Instrumentation

Stages are measured by rebinding attributes that production code resolves at call time —
never by editing the solver. `spsolve` is timed through a forwarding shim installed only
on `topoopt.fem.sparse_linalg`, rather than patching `scipy.sparse.linalg` process-wide.

After the instrumented run the suite asserts every wrapped stage recorded at least one
call. This matters: `filter_sensitivities` is imported *by value* in `optimizer.py:31`, so
patching it on its own module would not intercept it and would silently report a stage
that takes exactly zero time.

The instrumentation's own cost is reported as `perturbation_ratio` (instrumented wall ÷
clean wall). Measured: **1.0009–1.0015**, i.e. about 0.1%. Wrapper overhead (~0.09 µs) is
negligible against stages spanning 3.6 µs to 189 ms, so no per-stage correction is
applied.

## The stage table is a partition

`reference_stages.csv` mixes two kinds of row. Only rows with `in_partition=True` form the
partition, and **they sum to 100% of the iteration**:

```
element_moduli + assemble + solve_reduce + solve_numeric
  + sensitivity + sensitivity_filter + oc_update + analysis_tail + loop_tail
  == iterations_total
```

| stage | scope | what it is |
|---|---|---|
| `element_moduli` | measured | SIMP interpolation (`optimizer.py:217`) |
| `assemble` | measured | COO triplet build + `tocsc` (`fem.py:162`) |
| `solve_reduce` | derived | `solve_displacements` − `spsolve`; the `tocsr()` copy and both fancy-index slices (`fem.py:208`) |
| `solve_numeric` | measured | SuperLU LU factorisation + solve (`fem.py:209`) |
| `sensitivity` | measured | analytical `dc/dx_e` (`optimizer.py:251`) |
| `sensitivity_filter` | measured | cone-weighted sparse matvec (`filter.py:83`) |
| `oc_update` | measured | bisection on the volume multiplier (`optimizer.py:81`) |
| `analysis_tail` | derived | `analyze` minus its four measured children |
| `loop_tail` | derived | `iterations_total` minus all other partition rows |

Rows with `in_partition=False` are context and **must not be summed with the partition**.
`analyze_total` is the parent of `element_moduli`/`assemble`/`solve_reduce`/`solve_numeric`
and already contains them; presenting it as a peer would double-count and inflate the
column to ~262%.

### The one limitation that cannot be fixed non-invasively

`compliance` and `element_energy` are **inline expressions** inside `analyze()`
(`optimizer.py:240` and `:242-248`), not function calls. There is no attribute to wrap and
no seam to hook; splitting them would require editing the solver.

Patching `numpy.einsum` globally was considered and rejected: it is a hot, widely used
function, the patch would intercept unrelated calls, and it still would not capture the
`displacements[element_dofs]` gather.

So they are reported two ways:

1. **`analysis_tail`** (`scope=derived`) — an in-loop partition row obtained by
   subtraction. Exact by construction, but it lumps the two expressions together with the
   DOF gather and interpreter overhead.
2. **`compliance` / `element_energy`** (`scope=isolated_estimate`) — the identical
   expressions timed standalone on objects **captured from the real run** (the assembled
   stiffness and the solved displacement vector, hooked from the wrappers — no state is
   fabricated).

**The isolated estimates are not in-loop measurements.** Measured on this baseline they
account for 90% (60×20), 93% (120×40) and 97% (240×80) of `analysis_tail`; the remainder
is the DOF gather plus interpreter overhead. They exist for attribution only — do not add
them to the partition.

## Baseline results (reference machine)

| case | elements | DOFs | iters | converged | final compliance | wall/run | peak RSS |
|---|---:|---:|---:|---|---:|---:|---:|
| 60×20 | 1 200 | 2 562 | 44 | yes | 250.46004256994163 | 0.183 s | 60 MiB |
| 120×40 | 4 800 | 9 922 | 61 | yes | 217.50535787426227 | 2.362 s | 97 MiB |
| 240×80 | 19 200 | 39 042 | 76 | yes | 206.3317750187044 | 15.526 s | 242 MiB |

Per-iteration cost is dominated by the sparse direct solve, and its share grows with mesh
size:

| stage | 60×20 | 120×40 | 240×80 |
|---|---:|---:|---:|
| `solve_numeric` | 72.8% | 89.8% | 92.5% |
| `assemble` | 12.2% | 5.3% | 4.1% |
| `oc_update` | 7.0% | 1.9% | 1.2% |
| `solve_reduce` | 4.9% | 1.9% | 1.3% |
| `analysis_tail` | 2.5% | 1.0% | 0.7% |
| filter / sensitivity / moduli | < 0.3% | < 0.1% | < 0.1% |

Fitted per-call scaling between 60×20 and 240×80: `solve_numeric` ≈ O(n_dofs^1.5) (×62
time for ×15.2 DOFs), `assemble` ≈ O(n_dofs^1.0). Note that `assemble` is 12% of the
iteration at 60×20 but only 4% at 240×80 — **the ranking of hotspots depends on mesh
size**, so profile before optimizing.

All three cases converge within the canonical `max_iterations=200`. Do not assume that
generalises: a 200×60 probe at the same configuration hit the iteration cap. Where it
does, `converged` is recorded honestly as `False` and the affected rows' compliance and
runtime are cap-truncated, not converged, values.

## Output schema

### `reference.csv` — one row per case

Identity: `run_id`, `timestamp_utc`, `label`, `nelx`, `nely`, `n_elements`, `n_nodes`,
`n_dofs`.
Configuration: `volfrac`, `penal`, `rmin`, `max_iterations`, `tolerance`.
Method: `warmups`, `repetitions`.
Numerical: `iterations`, `converged`, `final_compliance`, `final_volume_fraction`,
`final_change`, `result_compliance`, `result_volume_fraction`, `density_min`,
`density_max`, `density_sum`, `density_sha256`, `compliance_spread`,
`numerically_repeatable`.
Performance: `run_wall_s_{mean,median,stdev,min,max}`, `iteration_s_{mean,median,p10,p90}`,
`construct_s_median`.
Memory: `peak_rss_bytes`, `peak_rss_delta_bytes`, `peak_rss_saturated`.
Environment: `python_version`, `numpy_version`, `scipy_version`, `platform_system`,
`platform_machine`, `cpu_count`.
Cross-checks: `process_wall_ratio`, `perturbation_ratio`.

`final_compliance` is `history[-1].compliance`, which describes the design **before** the
final density update (`optimizer.py:293-301`). `result_compliance` is an extra untimed
`analyze()` on the design actually returned. They differ slightly — use
`result_compliance` when you care about the returned design, `final_compliance` when
matching the example's printed output.

### `reference_stages.csv` — one row per (case, stage)

`run_id`, `label`, `nelx`, `nely`, `stage`, `scope`, `in_partition`, `calls`,
`calls_per_iteration`, `total_s`, `mean_per_call_us`, `mean_per_iteration_ms`,
`share_of_iteration`, `note`.

`share_of_iteration` is relative to `iterations_total` (the sum of
`IterationRecord.time_s` over the instrumented run), so partition rows sum to 1.0.

### `environment.json`

Environment and run metadata keyed by `run_id`. Deliberately excludes `hostname`,
`username`, `home directory`, `working directory` and the full interpreter path — only
`os.path.basename(sys.executable)` is stored. Do not add them back.

## Comparing a future build against this baseline

Produce a second label and compare numerically, **not** bit-for-bit:

```bash
python benchmarks/benchmark.py --label optimized
```

Compare `optimized.csv` against `reference.csv` per case:

| column | suggested check | rationale |
|---|---|---|
| `iterations` | exact equality | a different path is a different algorithm |
| `converged` | exact equality | |
| `final_compliance` | `rtol=1e-6` | |
| `final_volume_fraction` | `atol=1e-6` | bisection tolerance is 1e-6 (`optimizer.py:45`) |
| `density_sum` | `rtol=1e-6` | catches whole-field drift |
| `final_change` | `atol=1e-9` | |
| `density_sha256` | **diagnostic only — never a gate** | |
| `run_wall_s_*`, `iteration_s_*` | ratio, e.g. report speedup | machine-dependent |
| `peak_rss_*` | ratio | machine-dependent |

**Do not gate on `density_sha256`.** Any legitimate optimization that changes
floating-point summation order (a different assembly order, a cached sparsity pattern)
will change it while remaining mathematically equivalent. It is recorded to detect
*silent* change in the same code and to back the non-interference assertion — treat a
mismatch as "investigate", never as "fail".

`compliance_spread` and `numerically_repeatable` are determinism checks. On this baseline
all repetitions were bit-identical (`compliance_spread = 0.0`), so a non-zero spread in a
future build is a meaningful signal.

## Measurement caveats

- **Runtime is machine-dependent** and not comparable across machines. Compare ratios
  measured on the same host. `environment.json` records the BLAS thread environment,
  which is the largest single confounder for NumPy-heavy benchmarks.
- **No nanosecond-level precision is claimed.** The smallest stages (`sensitivity` ≈
  3.6 µs) are within an order of magnitude of the timer and wrapper overhead.
- **`peak_rss_bytes` is a process-wide high-water mark** including the interpreter, NumPy
  and SciPy — not "the memory the solver needs". `peak_rss_delta_bytes` is measured
  against a baseline sampled before any solver work; it is a truncated lower bound.
  `peak_rss_saturated` is `True` when the peak equals the baseline, meaning the delta is
  unusable. Below roughly 40×20 the delta is allocator noise.
- `tracemalloc` was evaluated for memory and rejected: SuperLU's factor and workspace are
  C allocations it does not track, so it would undercount the dominant consumer.
- The worker deliberately avoids importing `examples.cantilever`, which would pull in
  Matplotlib (~28 MiB) at module scope. The parent asserts that the benchmark's restated
  problem produces byte-identical `fixed_dofs`, `loads` and `element_dofs`, so the two
  definitions cannot silently diverge.

## Cost

Reference machine, `--warmups 1 --repetitions 3`:

| case | run | case total |
|---|---:|---:|
| 60×20 | 0.18 s | 1.1 s |
| 120×40 | 2.36 s | 12.2 s |
| 240×80 | 15.53 s | 78.7 s |

Whole suite: **~92 s**. `--repetitions 1 --warmups 0` cuts it to roughly a quarter.

Adding **480×160** (76 800 elements, ~154 900 DOFs): extrapolating the measured
`O(n_dofs^1.5)` solve, expect roughly 1.5 s per iteration and 8–10× the 240×80 memory —
order **10 minutes and a few GB** for a 5-run case. It is supported today
(`--cases 480x160`); make sure the machine has the memory before running it.

## Files

```
benchmarks/
    benchmark.py                  # parent + worker
    README.md
    results/
        reference.csv             # baseline metrics, one row per case
        reference_stages.csv      # stage partition, one row per (case, stage)
        environment.json          # environment + methodology metadata
```

Generated artefacts are reproducible from `benchmark.py`; nothing here is a profiling
dump.
