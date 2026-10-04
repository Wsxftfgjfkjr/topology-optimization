# Performance engineering record

How the computational core was measured and improved, in the order it happened,
including the experiment that produced a negative result.

The short version: assembly caching and reduced-system caching removed real
per-iteration work but were worth little end to end, because the sparse solve was
not their bottleneck. A detour into iterative solvers was rejected on evidence.
The change that mattered was the direct solver's fill-reducing ordering.

Every number below comes from the tracked artifacts in
[`benchmarks/results/`](../benchmarks/results/) unless a paragraph says otherwise.
Speedups are multiplicative factors, `old / new`; percentages are computed from
the same values and are given only where they add something.

- [1. Benchmark methodology](#1-benchmark-methodology)
- [2. Frozen reference baseline](#2-frozen-reference-baseline)
- [3. Experiment 1 — sparse assembly structure caching](#3-experiment-1--sparse-assembly-structure-caching)
- [4. Experiment 2 — reduced-system extraction caching](#4-experiment-2--reduced-system-extraction-caching)
- [5. Experiment 3A — solver investigation](#5-experiment-3a--solver-investigation)
- [6. Experiment 3B — SuperLU fill-reducing ordering](#6-experiment-3b--superlu-fill-reducing-ordering)
- [7. Final performance summary](#7-final-performance-summary)
- [8. Current bottleneck and scaling limit](#8-current-bottleneck-and-scaling-limit)
- [9. Reproducing the benchmarks](#9-reproducing-the-benchmarks)

## 1. Benchmark methodology

The harness is [`benchmarks/benchmark.py`](../benchmarks/benchmark.py); the full
methodology and its caveats are in [`benchmarks/README.md`](../benchmarks/README.md).
The parts that matter for reading the numbers here:

**Cases.** Three canonical meshes at fixed parameters (`volfrac` 0.4, `penal` 3.0,
`rmin` 1.5, `max_iterations` 200, convergence tolerance 1e-2):

| case | elements | nodes | DOFs |
|---|---:|---:|---:|
| 60×20 | 1 200 | 1 281 | 2 562 |
| 120×40 | 4 800 | 4 961 | 9 922 |
| 240×80 | 19 200 | 19 521 | 39 042 |

**Timed region.** `TopologyOptimizer.run()` only. Process start-up, imports,
optimizer construction, plotting, and CSV/JSON writing are excluded. Presentation
work is excluded because it is I/O and rendering, not computation, and it varies
with output directory and disk state; including it would add noise to a signal
measured in milliseconds. The one consequence worth stating plainly is that
*construction is excluded*, and two of the experiments below moved invariant setup
work from the loop into construction. Section 3 quantifies that.

**Phases.** One discarded warm-up run, then three clean timed runs, then one
instrumented run for the stage breakdown. The timed runs are authoritative; the
instrumented run is advisory and runs after the peak-memory sample so it cannot
inflate it. Each repetition constructs a **fresh optimizer** — `run()` mutates
`self.density`, so reusing an instance would restart from the converged design and
finish in one iteration while still reporting `converged=True`.

**Subprocess isolation.** Each mesh size runs in its own process. This is required
rather than stylistic: `resource.getrusage().ru_maxrss` is a high-water mark that
never decreases, so three cases in one process would all report the largest case's
peak.

**Instrumentation.** Stages are measured by rebinding attributes that production
code resolves at call time, never by editing the solver. The instrumentation's own
cost is reported as `perturbation_ratio` (instrumented wall ÷ clean wall); across
all four tracked runs it stays between 0.996 and 1.006.

The stage rows form a partition — `element_moduli + assemble + solve_reduce +
solve_numeric + sensitivity + sensitivity_filter + oc_update + analysis_tail +
loop_tail` sums to 100% of the iteration. Rows outside the partition
(`analyze_total`, `run_wall`) are context and must not be summed with it.

**Peak memory.** `peak_rss_bytes` is a process-wide high-water mark including the
interpreter, NumPy and SciPy — not "the memory the solver needs".
`peak_rss_delta_bytes` is measured against a baseline sampled before any solver
work and is a truncated lower bound. Below roughly 40×20 the delta is allocator
noise. All runs here report `peak_rss_saturated = False`, so the deltas are usable.

**Numerical regression metrics.** Every run records iteration count, convergence
status, final compliance, final volume fraction, density sum, final density change,
a density digest, and a determinism check. The digest is a **diagnostic, never a
gate**: any legitimate change that alters floating-point summation order will move
it while remaining mathematically equivalent. `compliance_spread` is the spread
across the three clean repetitions; it is `0.0` for all four tracked builds,
meaning each build is internally bit-reproducible.

**Environment.** Each run records interpreter, NumPy and SciPy versions, platform,
CPU model, CPU count and the BLAS thread environment variables, which are the
largest single confounder for NumPy-heavy benchmarks. It deliberately excludes
hostname, username, home directory, working directory and the full interpreter
path.

## 2. Frozen reference baseline

`v0.1-reference` is the reference solver and `v0.2-benchmark` is the harness plus
the baseline it measured. The baseline was frozen **before** any performance work
so that later builds are compared against a fixed point rather than against a
moving one. It is tracked as `reference.csv` and `reference_stages.csv`.

| case | elements | DOFs | iterations | converged | final compliance | wall/run | peak RSS |
|---|---:|---:|---:|---|---:|---:|---:|
| 60×20 | 1 200 | 2 562 | 44 | yes | 250.4600425699 | 0.182848 s | 60.4 MiB |
| 120×40 | 4 800 | 9 922 | 61 | yes | 217.5053578743 | 2.362309 s | 97.1 MiB |
| 240×80 | 19 200 | 39 042 | 76 | yes | 206.3317750187 | 15.526240 s | 241.8 MiB |

Wall times are `run_wall_s_mean` over three clean repetitions.

Per-iteration stage shares at the baseline, which set the agenda for everything
that follows:

| stage | 60×20 | 120×40 | 240×80 |
|---|---:|---:|---:|
| `solve_numeric` | 72.81% | 89.75% | 92.50% |
| `assemble` | 12.17% | 5.28% | 4.12% |
| `solve_reduce` | 4.95% | 1.87% | 1.35% |
| `oc_update` | 7.01% | 1.92% | 1.22% |
| everything else | 3.07% | 1.18% | 0.82% |

The ranking depends on mesh size. `assemble` is the second-largest stage at 60×20
and a rounding error by 240×80; `solve_numeric` moves the other way. The baseline
documentation records the fitted per-call scaling of `solve_numeric` between the
smallest and largest case as roughly `O(n_dofs^1.5)` — ×62 time for ×15.2 DOFs.

## 3. Experiment 1 — sparse assembly structure caching

Artifacts: `optimized_assembly.csv`, `optimized_assembly_stages.csv`.

**The invariant.** `Mesh.element_dofs` is built once at construction and never
mutated. Everything structural about the assembly therefore follows from the mesh
alone: which `(row, column)` pair each of the 64 entries of an element matrix
scatters to, the CSC sparsity pattern of the assembled matrix, and the data slot
each entry accumulates into. None of that depends on the element moduli, so none of
it changes between topology iterations.

**The repeated work.** The reference assembled by building `rows` and `columns`
arrays of length `64 × n_elements`, constructing a COO matrix, and letting
`tocsc()` sum duplicates and sort. Every iteration rebuilt and re-sorted the same
index arrays to produce the same pattern.

**The implementation.** `StiffnessAssemblyPlan` computes the pattern once. It packs
each `(row, column)` pair into a single integer key, sorts by key, and derives the
CSC `indptr`/`indices` plus a `slot` array mapping every one of the
`64 × n_elements` entries to the data slot it accumulates into. Assembly then
becomes one `np.bincount` scatter-add over precomputed slots. The plan is cached on
the `Mesh` and primed at optimizer construction.

**Local effect.** `assemble`, in ms per iteration:

| case | reference | cached | speedup |
|---|---:|---:|---:|
| 60×20 | 0.5063 | 0.1160 | 4.37× |
| 120×40 | 2.0490 | 0.3936 | 5.21× |
| 240×80 | 8.4162 | 1.6919 | 4.97× |

**End-to-end effect.** 1.098× / 1.046× / 1.034×.

**Amdahl's law, and why the end-to-end win is small.** A 5× local improvement that
produces a 3.4% end-to-end gain looks disappointing until you look at the share.
At 240×80 `assemble` was 4.12% of the iteration, so the ceiling on the total from
optimizing it at all is `1 / (1 − 0.0412 + 0.0412/4.97) = 1.034`. Measured:
1.034. The arithmetic predicts the result almost exactly, and the same holds at
the other sizes. Both columns are computed on the iteration partition — share and
measured speedup come from the same stage rows — so they compare like with like:

| case | `assemble` share | speedup ceiling | measured |
|---|---:|---:|---:|
| 60×20 | 12.17% | 1.104× | 1.095× |
| 120×40 | 5.28% | 1.045× | 1.045× |
| 240×80 | 4.12% | 1.034× | 1.034× |

The lesson is not that the optimization was wasted. It is that the profile decides
what an optimization is worth, and the expensive stage was somewhere else.

**Memory trade-off.** The plan is retained for the whole run: a `slot` array of
`64 × n_elements` machine integers, plus the CSC `indices` and `indptr`. Peak
RSS delta rose 7.1 → 7.8 MiB (60×20), 43.7 → 49.0 MiB (120×40) and
188.2 → 200.3 MiB (240×80). That is a real cost paid for the speed.

**Numerical equivalence.** The cached scatter produces the same CSC structure as
the explicit reference path exactly, and the same values to summation roundoff.
Two tests hold it to that, including across unrelated moduli fields.

Bit identity is deliberately *not* required, because it is not portable here.
Both paths sum each slot's contributions in COO input order, but SciPy reaches
that order through `sum_duplicates` → `sort_indices`, which sorts
`(index, value)` pairs with `std::sort`. That sort is not stable, so entries
tying on the row index may be permuted, and which permutation comes out depends
on the C++ standard library and on the run length — `std::sort` is a stable
insertion sort for short runs and an unstable introsort beyond that, and a
column of this mesh holds up to 18 entries. So the last bits of a slot that
receives three or four contributions are a property of the standard library, not
of this code: macOS/libc++ reproduces the input order, Linux/libstdc++ does not.
An early version of these tests asserted bit equality and passed on macOS while
failing on Linux for exactly this reason.

On the reference environment the two paths did come out bit-identical, which is
why the tracked `density_sha256` for `optimized_assembly` matches `reference` on
all three meshes. That is an observation about that environment, not a property
the code guarantees.

## 4. Experiment 2 — reduced-system extraction caching

Artifacts: `optimized_reduced.csv`, `optimized_reduced_stages.csv`.

**The repeated work.** The reference reduced the system inside every solve with

```python
reduced = stiffness.tocsr()[free_dofs, :][:, free_dofs].tocsc()
```

which converts the whole matrix to CSR, performs two fancy-index slices, and
converts back to CSC — four passes over the entry list, per iteration, returning
the same answer every time.

**The invariant.** The reduced matrix consists of exactly those global CSC entries
whose row *and* column are free, kept in global CSC order. Nothing is summed and
nothing is dropped, because the assembled matrix is canonical. The reduced `data`
array is therefore a plain selection of the global one, and *which* entries are
selected depends only on the structure and the free DOF set.

**The implementation.** `ReducedSystemPlan` computes a boolean keep-mask over the
global entry list once, along with the reduced `indptr` and `indices`. Extraction
becomes `stiffness.data[keep]`. Requiring `free_dofs` to be strictly increasing is
what keeps the selection a subsequence, so no entry has to be reordered; the
constructor enforces it and the plan validates the structure it is handed rather
than trusting it, because the alternative to an exception there is a wrong answer.

**Local effect.** `solve_reduce`, in ms per iteration:

| case | reference | cached | speedup |
|---|---:|---:|---:|
| 60×20 | 0.2058 | 0.0302 | 6.84× |
| 120×40 | 0.7254 | 0.1046 | 6.94× |
| 240×80 | 2.7530 | 0.3843 | 7.16× |

**End-to-end effect — incremental versus cumulative.** This distinction matters,
because the cumulative figure includes Experiment 1:

| case | incremental (assembly → reduced) | cumulative (reference → reduced) |
|---|---:|---:|
| 60×20 | 1.052× | 1.156× |
| 120×40 | 1.020× | 1.067× |
| 240×80 | 1.011× | 1.045× |

The stages are independent and multiply: 1.098 × 1.052 = 1.156, and so on for the
other two meshes.

**Memory trade-off, and one that goes the other way.** What the plan retains is a
boolean mask over the global entries plus the reduced CSC structure — roughly five
bytes per stored entry against the twelve the assembled matrix occupies. Peak RSS
delta moved 7.8 → 7.6 MiB (60×20), 49.0 → 49.4 MiB (120×40) and
200.3 → 206.1 MiB (240×80). At the two larger meshes it went **up**, because both
plans are now constructed eagerly. The saving is per-iteration allocation churn,
not peak footprint.

**Construction cost.** Both plans are built at optimizer construction rather than
on first use, which moves invariant work out of the timed loop. Construction grew
from 0.377 to 0.668 ms (60×20), 1.024 to 2.085 ms (120×40) and 3.474 to 7.379 ms
(240×80). Against a full run that is 0.42%, 0.09% and 0.05% respectively, so the
trade is clearly favourable — but it is a trade, and it is only invisible because
the methodology excludes construction from the measured region.

**Numerical equivalence.** Bit-exact — and unlike Experiment 1, that claim is
portable. The reduction performs no arithmetic at all: it is the boolean
selection `stiffness.data[keep]`, so there is no summation order to disagree
about and nothing for a different standard library to permute. The cached
extraction is held to exact equality with the reference slicing on every array,
structure included. `optimized_reduced`'s tracked `density_sha256` matches
`reference` on all three meshes.

## 5. Experiment 3A — solver investigation

Artifacts: none. This experiment was exploratory, and its scripts and raw output
were temporary rather than tracked. The repository therefore does not contain its
condition numbers, iteration counts or error magnitudes, and none are reproduced
here. What follows are the conclusions and the reasoning, which are what informed
the production decision.

**The question.** With `solve_numeric` at 73–93% of the iteration, the obvious
question was whether a better *solver* would beat the direct factorization. The
reduced system is symmetric (its asymmetry is at the rounding level, a
consequence of scatter-add accumulation order) and is expected to be positive
definite once the rigid-body modes are restrained. Conjugate gradient is the
textbook method for exactly that structure, so it was a legitimate candidate
rather than a long shot.

It was not adopted automatically, and the alternative was not adopted
automatically either. Both were measured.

**Why the problem is hostile to iterative solvers.** SIMP with
`void_modulus_ratio` 1e-9, `penal` 3 and `min_density` 1e-3 puts the
element moduli across a range of about nine orders of magnitude. As the
optimization proceeds, a large fraction of elements sit at the lower density
bound, so the assembled system contains large near-void regions coupled to stiff
solid regions. The one property CG needs — a well-conditioned SPD matrix — is
exactly the property the physics gives up.

**The finding that decided it.** Residual convergence is not sufficient evidence
that a solve is usable here. The quantities the optimizer actually consumes are
the element strain energies, which feed the sensitivity field, and through it the
Optimality Criteria update. A solve can reach a reasonable-looking relative
residual while the sensitivity field is inaccurate, because the solution error is
concentrated in near-null modes that the compliance barely sees but the element
energies do. Judging an iterative solver by compliance error alone would have been
misleading.

That is the substantive engineering point of this experiment, and it generalizes
beyond CG: **for a topology optimization loop, a solver's acceptance criterion has
to be validated against the sensitivity field, not just the residual or the
objective.**

**What was tested, and what was concluded.** Unpreconditioned CG, Jacobi
(diagonal) preconditioning, warm starts carried over from the previous topology
iteration, and a block-Jacobi variant that inverts the 2×2 node blocks exactly.
All were run across a ladder of stopping tolerances, on initial, intermediate and
near-converged designs, at the three canonical mesh sizes, and compared against
the direct solve on displacement error, compliance error and sensitivity error.

- Unpreconditioned CG was **not viable** on intermediate and near-converged
  designs at any tested tolerance.
- Jacobi preconditioning was necessary for CG to converge at all, and warm starts
  measurably reduced late-iteration counts. Neither made CG competitive with the
  direct factorization end to end at accuracy levels that preserved the
  optimization result. Where CG did approach the direct solver in speed, it did so
  by adopting a stopping tolerance loose enough to change the converged design.
- Block Jacobi reduced iteration counts only marginally while costing substantially
  more per application, which indicates the 2×2 node coupling is not where the
  conditioning problem lives.

**Rejected, not never-to-be-revisited.** This conclusion is scoped to the tested
problem family, the three canonical meshes, SciPy's `cg`, and diagonal and
block-diagonal preconditioning. It is **not** a claim that iterative solvers are
unsuitable for topology optimization in general. Multilevel preconditioning and
incomplete factorization were never tested; either could change the answer, and
both are outside what the existing dependency stack provides.

**What the experiment did produce.** Profiling the factorization directly showed
that its cost and its memory both scale with fill-in, which pointed at the
fill-reducing ordering as a cheaper, lower-risk lever than replacing the solver.
That became Experiment 3B.

The negative result is arguably the most useful thing here. A CG implementation
would have been more code, a new set of failure modes, and a solver that was
faster only on the initial well-conditioned design — which is the one state where
it did not matter.

## 6. Experiment 3B — SuperLU fill-reducing ordering

Artifacts: `optimized_ordering.csv`, `optimized_ordering_stages.csv`. Production
change in [`topoopt/fem.py`](../topoopt/fem.py); commit
`cb62e4c`, tagged `v0.3-performance`.

**The change.** SuperLU permits choosing the column ordering applied before
factorization. The default is `COLAMD`, an approximate minimum-degree ordering
designed for unsymmetric matrices. The reduced stiffness matrix is symmetric, and
for a symmetric matrix the structure of $A^{\mathsf{T}} + A$ is the structure of
$A$ itself, so `MMD_AT_PLUS_A` — minimum degree on $A^{\mathsf{T}} + A$ — is the
ordering matched to the problem. The production change selects it and nothing
else.

**Why the mechanism should work.** A sparse direct factorization costs what its
fill-in costs, in both time and memory. A fill-reducing ordering chosen for the
actual symmetry of the matrix should produce fewer factor entries than one chosen
for a general unsymmetric matrix, and the factorization should get both faster and
smaller together.

**Fill-in.** The fill counts were measured during the investigation rather than
recorded in the tracked artifact set, so the counts themselves are not reproduced
here. The direction — fewer factor entries with `MMD_AT_PLUS_A` than with
`COLAMD` — is what the timings below are the consequence of, and the commit
message records it.

**Local effect.** `solve_numeric`, in ms per iteration, comparing the build before
and after this change:

| case | `optimized_reduced` | `optimized_ordering` | speedup |
|---|---:|---:|---:|
| 60×20 | 3.0309 | 2.7418 | **1.105×** |
| 120×40 | 34.6503 | 19.0212 | **1.822×** |
| 240×80 | 189.2537 | 125.2724 | **1.511×** |

**End-to-end effect.** 1.092× / 1.757× / 1.497×. Here the local and end-to-end
numbers are close, because this change targets the stage that dominates the loop —
the opposite of Experiments 1 and 2. Amdahl's law again predicts it: at 240×80
`solve_numeric` was 96.85% of the iteration, giving a ceiling of
`1 / (1 − 0.9685 + 0.9685/1.511) = 1.487` on iteration time, against 1.488
measured on the same basis.

**Effect on memory.** Unlike Experiments 1 and 2, this one reduces peak RSS rather
than trading memory for speed:

| case | `optimized_reduced` | `optimized_ordering` | change |
|---|---:|---:|---:|
| 60×20 | 7.6 MiB | 7.6 MiB | no change |
| 120×40 | 49.4 MiB | 40.5 MiB | −8.9 MiB |
| 240×80 | 206.1 MiB | 165.9 MiB | −40.2 MiB |

(peak RSS *delta* over the pre-solve baseline). At 60×20 the fill reduction is
small and the change is not resolvable.

One measurement caveat. SuperLU's `L` and `U` properties materialise fresh CSC
copies on access, so inspecting the factors allocates a second copy of their
contents. Nothing in the official benchmark path does this — it records
`peak_rss_bytes` from the normal run — and any factor inspection must be kept in
a separate process or it will be mistaken for solver memory.

**Numerical differences.** Changing the ordering changes the order the
factorization sums in, so results are numerically equivalent but **not
bit-identical**. This is the first of the three optimizations for which the tracked
`density_sha256` differs from the reference. The differences, computed from the
tracked case CSVs:

| case | final compliance (rel. diff vs reference) | volume fraction (abs. diff) | density sum (rel. diff) |
|---|---:|---:|---:|
| 60×20 | 2.9e-13 | 8.7e-14 | 1.1e-13 |
| 120×40 | 4.8e-12 | 1.3e-12 | 2.7e-12 |
| 240×80 | 4.2e-11 | 1.0e-11 | 2.5e-11 |

The largest is 4.2e-11 relative, against a convergence tolerance of 1e-2 — about
eight orders of magnitude below the tolerance that decides when the optimization
stops. The benchmark documentation's guidance is to treat a digest mismatch as
"investigate", never "fail"; this is the case it was written for.

**Convergence behaviour is unchanged.** All three cases converge, in the same
iteration count (44 / 61 / 76) as every other build, and every repetition is
internally bit-reproducible (`compliance_spread = 0.0`). Regression tests were
added holding the two orderings to numerical equivalence with a documented
tolerance, and holding the default to the selected ordering.

**Cumulative versus incremental.** This experiment's own contribution is
1.092× / 1.757× / 1.497×. The cumulative figures against the frozen reference
(1.262× / 1.875× / 1.565×) also include Experiments 1 and 2 and must not be
attributed to the ordering change.

## 7. Final performance summary

Runtime progression, `run_wall_s_mean` in seconds:

| case | `reference` | `optimized_assembly` | `optimized_reduced` | `optimized_ordering` |
|---|---:|---:|---:|---:|
| 60×20 | 0.182848 | 0.166475 | 0.158216 | **0.144951** |
| 120×40 | 2.362309 | 2.258295 | 2.213366 | **1.259841** |
| 240×80 | 15.526240 | 15.022805 | 14.855837 | **9.922328** |

Incremental and cumulative speedups:

| case | Exp 1 | Exp 2 | Exp 3B | cumulative vs `reference` |
|---|---:|---:|---:|---:|
| 60×20 | 1.098× | 1.052× | 1.092× | **1.262×** |
| 120×40 | 1.046× | 1.020× | 1.757× | **1.875×** |
| 240×80 | 1.034× | 1.011× | 1.497× | **1.565×** |

The incremental factors multiply to the cumulative factor in every row, which is
the check that the three changes are independent.

Peak RSS *delta*, in MiB. Memory does not follow the same shape as runtime:

| case | `reference` | `optimized_assembly` | `optimized_reduced` | `optimized_ordering` |
|---|---:|---:|---:|---:|
| 60×20 | 7.1 | 7.8 | 7.6 | 7.6 |
| 120×40 | 43.7 | 49.0 | 49.4 | **40.5** |
| 240×80 | 188.2 | 200.3 | 206.1 | **165.9** |

Caching the two invariant structures cost memory; replacing the ordering more
than paid it back at the larger meshes. Net against the reference, the current
build uses 11.9% less solver memory at 240×80 and about 8% more at 60×20, where
the retained plans are a larger fraction of a much smaller footprint.

## 8. Current bottleneck and scaling limit

`solve_numeric` is still dominant, and more so as the mesh grows:

| build | 60×20 | 120×40 | 240×80 |
|---|---:|---:|---:|
| `reference` | 72.81% | 89.75% | 92.50% |
| `optimized_ordering` | 83.26% | 92.02% | **95.38%** |

The next largest stage at 240×80 is now `oc_update` at 1.88%, followed by
`assemble` at 1.19%. The ordering change lowered the cost of the dominant stage
without changing what the dominant stage is.

Its share rising is not a regression; it is the arithmetic consequence of
optimizing the other stages. The absolute cost still grew: `solve_numeric` went
from 3.03 to 189.25 ms per iteration between the smallest and largest mesh, while
the problem grew by ×15.2 in DOFs. The baseline documentation records the fitted
growth as roughly `O(n_dofs^1.5)`, faster than the problem itself.

That superlinear growth is the real limitation, and it applies to memory before
it applies to time: fill-in grows faster than the matrix, and the factorization's
storage is a multiple of its fill. Extrapolating, memory becomes the binding
constraint before runtime does — the baseline documentation estimates a 480×160
case at roughly 8–10× the 240×80 memory.

Directions worth investigating, none attempted here and none promised:

- **Alternative sparse direct solvers** with better fill or memory behaviour on
  this problem class.
- **Symbolic factorization reuse.** The sparsity pattern is invariant across
  iterations — only the values change — so the symbolic phase of the
  factorization is recomputed for identical input every iteration. SuperLU
  supports same-pattern refactorization, but SciPy's public API does not expose
  it, so exploiting this would mean a different library or a native dependency.
  This is the most clearly identified remaining inefficiency.
- **Stronger iterative preconditioning** — multilevel or incomplete
  factorization. Section 5 explains why the simple preconditioners tested did not
  suffice and why that result does not settle the question.

## 9. Reproducing the benchmarks

The harness writes `<label>.csv`, `<label>_stages.csv` and `environment.json` into
its output directory. **`--label` defaults to `reference`**, so a bare
`python benchmarks/benchmark.py` overwrites the frozen baseline. Always pass a
label, and an output directory if you do not intend to add a tracked artifact:

```bash
python benchmarks/benchmark.py --label my-experiment --output-dir /tmp/bench-out
```

Other useful forms:

```bash
python benchmarks/benchmark.py --list-cases
python benchmarks/benchmark.py --cases 60x20 --repetitions 1 --warmups 0 \
    --label quick --output-dir /tmp/bench-out
```

Reproducing a tracked build means using the label recorded in its
`*_environment.json`: `reference`, `optimized_assembly`, `optimized_reduced`, or
`optimized_ordering`. The full suite takes roughly 90 seconds at the default
methodology on the reference machine — less on the current build, which is faster.

Compare builds by ratio, never by absolute seconds, and only within one host.
Results vary with CPU, OS, SciPy and NumPy builds, thermal state and background
load, and `environment.json` records the BLAS thread environment because it is the
largest single confounder. Compare numerically rather than bit-for-bit:
`benchmarks/README.md` gives the per-column suggested tolerances and explains why
the density digest is diagnostic rather than a pass/fail gate.
