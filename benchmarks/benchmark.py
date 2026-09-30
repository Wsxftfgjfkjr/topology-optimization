#!/usr/bin/env python3
"""Reproducible benchmark suite for the topoopt reference solver.

This measures the *frozen* ``v0.1-reference`` implementation so that later
optimized versions can be compared against it.  It does not modify the solver:
every stage measurement is obtained by wrapping existing callables from the
outside.  ``topoopt/``, ``examples/`` and ``tests/`` are untouched.

Two entry points share this file:

* the **parent** (default) orchestrates one subprocess per mesh size and writes
  the CSVs;
* the **worker** (``--worker``) runs exactly one case and prints a single
  sentinel-prefixed JSON line.

Each case runs in its own process because ``ru_maxrss`` is a high-water mark
that never decreases -- three cases in one process would all report the largest
case's peak memory.

See ``benchmarks/README.md`` for the methodology and the documented limitations.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import platform
import statistics
import subprocess
import sys
import time
from pathlib import Path

# ---------------------------------------------------------------------------
# sys.path bootstrap.
#
# ``topoopt`` is not an installed distribution -- it is imported from the
# repository root.  Python 3.11+ puts the *script's* directory at sys.path[0],
# not the working directory, so running this file out of benchmarks/ would fail
# to import it.  This block must execute before anything imports topoopt.
# ---------------------------------------------------------------------------
_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

import numpy as np  # noqa: E402

from topoopt import fem  # noqa: E402
from topoopt import optimizer as optimizer_module  # noqa: E402
from topoopt.optimizer import TopologyOptimizer  # noqa: E402

try:
    import resource
except ImportError:  # pragma: no cover - resource is POSIX-only
    resource = None


# ---------------------------------------------------------------------------
# Canonical configuration.
#
# These are module constants rather than CLI flags on purpose: mesh size is the
# only free parameter, so a comparable baseline cannot be produced by accident.
# They mirror the defaults in topoopt/optimizer.py and examples/cantilever.py.
# ---------------------------------------------------------------------------
CONFIG = {
    "volfrac": 0.4,
    "penal": 3.0,
    "rmin": 1.5,
    "max_iterations": 200,
    "tolerance": 1e-2,
}
LOAD_MAGNITUDE = 1.0

DEFAULT_CASES = ("60x20", "120x40", "240x80")
DEFAULT_LABEL = "reference"
DEFAULT_OUTPUT_DIR = Path(__file__).resolve().parent / "results"

_SENTINEL = "__BENCHMARK_JSON__"

# Stages installed by _Instrumentation.  Every one of these must record at
# least one call during the instrumented run, or the run is aborted -- a stage
# that silently records zero would be reported as an impossibly fast stage
# rather than as a bug.  (This is not hypothetical: ``filter_sensitivities`` is
# imported by value in optimizer.py, so patching it on its own module would not
# intercept the call.)
WRAPPED_STAGES = (
    "element_moduli",
    "assemble",
    "solve_displacements",
    "solve_numeric",
    "analyze",
    "sensitivity",
    "sensitivity_filter",
    "oc_update",
)

STAGE_NOTES = {
    "element_moduli": "SIMP interpolation of E_e (optimizer.py:217)",
    "assemble": "COO triplet build + tocsc (fem.py:162)",
    "solve_reduce": "derived: solve_displacements - spsolve; includes the "
                    "tocsr() copy and both fancy-index slices (fem.py:208)",
    "solve_numeric": "SuperLU LU factorisation + solve (fem.py:209)",
    "sensitivity": "analytical dc/dx_e (optimizer.py:251)",
    "sensitivity_filter": "cone-weighted sparse matvec (filter.py:83)",
    "oc_update": "bisection on the volume multiplier (optimizer.py:81)",
    "analysis_tail": "derived: analyze - its four measured children; contains "
                     "the inline compliance matvec, the element_dofs gather and "
                     "the einsum (optimizer.py:240-248) plus interpreter overhead",
    "loop_tail": "derived: iteration_total - all other partition rows; covers "
                 "optimizer.py:292-299 (change, record construction, append)",
    "compliance": "ISOLATED ESTIMATE, not in-loop; see README",
    "element_energy": "ISOLATED ESTIMATE, not in-loop; see README",
    "iterations_total": "reference row, not part of the partition (sum of "
                        "IterationRecord.time_s over the instrumented run)",
    "analyze_total": "reference row, not part of the partition; contains the "
                     "element_moduli/assemble/solve children",
    "run_wall": "reference row, not part of the partition (whole run() wall time)",
    "outside_loop": "derived: run_wall - iterations_total",
}


# ---------------------------------------------------------------------------
# The canonical problem.
#
# This mirrors examples/cantilever.py:build_cantilever exactly.  It is restated
# here rather than imported because importing examples.cantilever pulls in
# Matplotlib (~28 MiB at module scope) which inflates the worker's memory
# baseline and can set the RSS high-water mark before the solver even runs.
# The parent asserts equivalence against the real function -- see
# _assert_problem_equivalence.
# ---------------------------------------------------------------------------
def build_cantilever(nelx, nely, load=LOAD_MAGNITUDE):
    """Left edge fully fixed, downward point load mid-way up the right edge."""
    mesh = fem.Mesh(nelx, nely)

    left_edge = mesh.node_index(0, np.arange(nely + 1))
    fixed_dofs = np.concatenate([2 * left_edge, 2 * left_edge + 1]).astype(np.intp)

    loads = np.zeros(mesh.n_dofs)
    loads[2 * mesh.node_index(nelx, nely // 2) + 1] = -load

    return mesh, fixed_dofs, loads


def _assert_problem_equivalence(cases):
    """Verify the restated problem matches examples.cantilever exactly.

    Runs in the parent, which is allowed to import Matplotlib because it does no
    timing or memory measurement.  Any drift between the two definitions becomes
    a hard failure instead of a silently different benchmark.
    """
    from examples.cantilever import build_cantilever as reference_build

    for nelx, nely in cases:
        ref_mesh, ref_fixed, ref_loads = reference_build(nelx, nely)
        mesh, fixed, loads = build_cantilever(nelx, nely)

        problems = []
        if mesh.n_elements != ref_mesh.n_elements:
            problems.append("n_elements")
        if mesh.n_dofs != ref_mesh.n_dofs:
            problems.append("n_dofs")
        if not np.array_equal(mesh.element_dofs, ref_mesh.element_dofs):
            problems.append("element_dofs")
        if not np.array_equal(fixed, ref_fixed):
            problems.append("fixed_dofs")
        if not np.array_equal(loads, ref_loads):
            problems.append("loads")
        if problems:
            raise RuntimeError(
                f"benchmark problem definition diverged from "
                f"examples.cantilever for {nelx}x{nely}: {', '.join(problems)}"
            )


# ---------------------------------------------------------------------------
# Memory.
# ---------------------------------------------------------------------------
def max_rss_bytes():
    """Peak resident set size of this process, in bytes.

    ``ru_maxrss`` is bytes on Darwin and kibibytes on Linux.  It is a
    high-water mark: it never decreases, which is why each case needs its own
    process.  Returns ``None`` where ``resource`` is unavailable.
    """
    if resource is None:
        return None
    raw = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return int(raw) if sys.platform == "darwin" else int(raw) * 1024


# ---------------------------------------------------------------------------
# Stage timing instrumentation.
# ---------------------------------------------------------------------------
class Recorder:
    """Accumulates wall time and call counts per named stage."""

    def __init__(self):
        self.totals = {}
        self.calls = {}

    def add(self, stage, seconds):
        self.totals[stage] = self.totals.get(stage, 0.0) + seconds
        self.calls[stage] = self.calls.get(stage, 0) + 1

    def total(self, stage):
        return self.totals.get(stage, 0.0)


def _wrap(owner, name, recorder, stage, capture=None):
    """Time every call to ``owner.name``.  Returns a restore callable.

    The callables are resolved by attribute lookup at call time, so rebinding
    the attribute intercepts them.
    """
    original = getattr(owner, name)
    if getattr(original, "_benchmark_wrapped", False):
        raise RuntimeError(f"{name} is already wrapped; refusing to double-install")

    def wrapper(*args, **kwargs):
        start = time.perf_counter()
        try:
            result = original(*args, **kwargs)
        finally:
            recorder.add(stage, time.perf_counter() - start)
        if capture is not None:
            capture["value"] = result
        return result

    wrapper._benchmark_wrapped = True
    wrapper.__wrapped__ = original
    setattr(owner, name, wrapper)

    def restore():
        setattr(owner, name, original)

    return restore


class _SpsolveShim:
    """Forwards to scipy.sparse.linalg but times ``spsolve``.

    ``fem.py`` reaches scipy through the module global ``sparse_linalg``, so
    rebinding that one name inside ``topoopt.fem`` is enough.  This avoids
    patching ``scipy.sparse.linalg`` process-wide -- same measurements, but the
    blast radius is a single module attribute.
    """

    def __init__(self, real, recorder, stage):
        self._real = real
        self._recorder = recorder
        self._stage = stage

    def __getattr__(self, name):
        return getattr(self._real, name)

    def spsolve(self, *args, **kwargs):
        start = time.perf_counter()
        try:
            return self._real.spsolve(*args, **kwargs)
        finally:
            self._recorder.add(self._stage, time.perf_counter() - start)


class Instrumentation:
    """Context manager installing the stage wrappers, restoring on exit."""

    def __init__(self, recorder, captures):
        self.recorder = recorder
        self.captures = captures
        self._restore = []

    def __enter__(self):
        recorder, captures = self.recorder, self.captures
        self._restore.append(
            _wrap(optimizer_module, "optimality_criteria_update", recorder, "oc_update")
        )
        self._restore.append(
            _wrap(TopologyOptimizer, "analyze", recorder, "analyze")
        )
        self._restore.append(
            _wrap(TopologyOptimizer, "element_moduli", recorder, "element_moduli")
        )
        self._restore.append(
            _wrap(TopologyOptimizer, "compliance_sensitivity", recorder, "sensitivity")
        )
        self._restore.append(
            _wrap(TopologyOptimizer, "apply_sensitivity_filter", recorder, "sensitivity_filter")
        )
        self._restore.append(
            _wrap(fem, "assemble_stiffness_matrix", recorder, "assemble",
                  capture=captures.setdefault("stiffness", {}))
        )
        self._restore.append(
            _wrap(fem, "solve_displacements", recorder, "solve_displacements",
                  capture=captures.setdefault("displacements", {}))
        )

        real = fem.sparse_linalg
        if isinstance(real, _SpsolveShim):
            raise RuntimeError("spsolve shim already installed")
        fem.sparse_linalg = _SpsolveShim(real, recorder, "solve_numeric")

        def restore_spsolve():
            fem.sparse_linalg = real
            assert fem.sparse_linalg is real

        self._restore.append(restore_spsolve)
        return self

    def __exit__(self, *exc_info):
        while self._restore:
            self._restore.pop()()
        return False


def _time_callable(fn, rounds=5, inner=100):
    """Best-of-rounds mean wall time of a single call, in seconds."""
    fn()  # warm the caches
    best = float("inf")
    for _ in range(rounds):
        start = time.perf_counter()
        for _ in range(inner):
            fn()
        best = min(best, (time.perf_counter() - start) / inner)
    return best


def _isolated_estimates(captures, element_stiffness, element_dofs):
    """Standalone timings of the two expressions that cannot be wrapped.

    These run on objects *captured from the real run* (the assembled stiffness
    and the solved displacement vector), so no state is fabricated.  They are
    still not in-loop measurements -- see the README.
    """
    stiffness = captures.get("stiffness", {}).get("value")
    displacements = captures.get("displacements", {}).get("value")
    if stiffness is None or displacements is None:
        return {}

    def compliance():
        return float(displacements @ (stiffness @ displacements))

    def element_energy():
        ue = displacements[element_dofs]
        return np.einsum("ei,ij,ej->e", ue, element_stiffness, ue)

    return {
        "compliance": _time_callable(compliance),
        "element_energy": _time_callable(element_energy),
    }


# ---------------------------------------------------------------------------
# Worker: one case, one process.
# ---------------------------------------------------------------------------
def _new_optimizer(mesh, fixed_dofs, loads):
    """Construct a fresh optimizer and time the construction."""
    start = time.perf_counter()
    optimizer = TopologyOptimizer(
        mesh, fixed_dofs, loads, verbose=False, **CONFIG
    )
    return optimizer, time.perf_counter() - start


def _density_digest(density):
    return hashlib.sha256(np.ascontiguousarray(density, dtype=np.float64).tobytes()).hexdigest()


def run_case(nelx, nely, run_id, label, warmups, repetitions):
    """Run every phase for one mesh size and return a JSON-serialisable dict."""
    # Sample the memory baseline before any solver work.  ru_maxrss is a
    # high-water mark, so sampling it after the warm-up would let the warm-up's
    # own allocations set the mark and collapse the reported delta.
    baseline_rss = max_rss_bytes()

    mesh, fixed_dofs, loads = build_cantilever(nelx, nely)

    # --- Phase A: discarded warm-up runs, each on a fresh optimizer ---------
    for _ in range(warmups):
        optimizer, _ = _new_optimizer(mesh, fixed_dofs, loads)
        optimizer.run()

    # --- Phase B: clean repetitions, no instrumentation ---------------------
    run_walls, construction_times, process_times = [], [], []
    iteration_times, per_rep = [], []

    for _ in range(repetitions):
        # A fresh optimizer per repetition is mandatory, not stylistic: run()
        # mutates self.density, so reusing an instance would restart from the
        # converged design and exit after one iteration reporting converged.
        optimizer, construction_s = _new_optimizer(mesh, fixed_dofs, loads)
        construction_times.append(construction_s)

        wall_start = time.perf_counter()
        cpu_start = time.process_time()
        result = optimizer.run()
        wall = time.perf_counter() - wall_start
        process_times.append(time.process_time() - cpu_start)

        run_walls.append(wall)
        iteration_times.extend(record.time_s for record in result.history)
        per_rep.append(
            {
                "iterations": result.iterations,
                "converged": result.converged,
                "compliance": result.history[-1].compliance,
                "volume_fraction": result.history[-1].volume_fraction,
                "change": result.history[-1].change,
                "digest": _density_digest(result.density),
                "iteration_sum": sum(r.time_s for r in result.history),
            }
        )

    final = per_rep[-1]

    # Structural invariant: every repetition must take the same path.  If this
    # fails, the repetitions are not comparable and the run is meaningless.
    iteration_counts = {rep["iterations"] for rep in per_rep}
    if len(iteration_counts) != 1:
        raise RuntimeError(
            f"{nelx}x{nely}: repetitions disagreed on iteration count "
            f"{sorted(iteration_counts)}; results are not comparable"
        )

    # Extra untimed analysis of the returned design.  history[-1].compliance
    # describes the design *before* the final update (optimizer.py:293-301), so
    # this is the compliance of the design actually returned.
    _, result_compliance, _ = optimizer.analyze()
    density = result.density

    # --- Memory: sample the high-water mark before phase C ------------------
    peak_rss = max_rss_bytes()

    # --- Phase C: instrumented run on a fresh optimizer ---------------------
    recorder = Recorder()
    captures = {}
    optimizer_c, _ = _new_optimizer(mesh, fixed_dofs, loads)
    with Instrumentation(recorder, captures):
        wall_start = time.perf_counter()
        result_c = optimizer_c.run()
        instrumented_wall = time.perf_counter() - wall_start

    missing = [s for s in WRAPPED_STAGES if recorder.calls.get(s, 0) == 0]
    if missing:
        raise RuntimeError(
            f"{nelx}x{nely}: instrumentation did not intercept {missing}; "
            f"the wrapper targets are wrong"
        )

    # The whole point of wrapping-only instrumentation: it must not perturb the
    # numerics.  If this fires, the harness is not measuring the frozen solver.
    digest_c = _density_digest(result_c.density)
    if digest_c != final["digest"] or result_c.iterations != final["iterations"]:
        raise RuntimeError(
            f"{nelx}x{nely}: instrumentation changed the result "
            f"({result_c.iterations} vs {final['iterations']} iterations)"
        )

    # --- Stage partition ----------------------------------------------------
    iterations_total = sum(r.time_s for r in result_c.history)
    element_moduli = recorder.total("element_moduli")
    assemble = recorder.total("assemble")
    solve_numeric = recorder.total("solve_numeric")
    solve_reduce = recorder.total("solve_displacements") - solve_numeric
    sensitivity = recorder.total("sensitivity")
    sensitivity_filter = recorder.total("sensitivity_filter")
    oc_update = recorder.total("oc_update")
    analyze_total = recorder.total("analyze")

    analysis_tail = analyze_total - (
        element_moduli + assemble + solve_reduce + solve_numeric
    )
    loop_tail = iterations_total - (
        analyze_total + sensitivity + sensitivity_filter + oc_update
    )

    estimates = _isolated_estimates(captures, optimizer_c.element_stiffness, mesh.element_dofs)

    n_iterations = max(result_c.iterations, 1)
    partition = [
        ("element_moduli", "measured", True, element_moduli),
        ("assemble", "measured", True, assemble),
        ("solve_reduce", "derived", True, solve_reduce),
        ("solve_numeric", "measured", True, solve_numeric),
        ("sensitivity", "measured", True, sensitivity),
        ("sensitivity_filter", "measured", True, sensitivity_filter),
        ("oc_update", "measured", True, oc_update),
        ("analysis_tail", "derived", True, analysis_tail),
        ("loop_tail", "derived", True, loop_tail),
    ]
    context = [
        ("iterations_total", "reference", False, iterations_total),
        ("analyze_total", "reference", False, analyze_total),
        ("run_wall", "reference", False, instrumented_wall),
        ("outside_loop", "derived", False, instrumented_wall - iterations_total),
    ]
    for name, seconds in estimates.items():
        context.append((name, "isolated_estimate", False, seconds))

    peak_delta = None
    saturated = None
    if peak_rss is not None and baseline_rss is not None:
        peak_delta = peak_rss - baseline_rss
        saturated = peak_delta == 0

    case_row = {
        "run_id": run_id,
        "timestamp_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "label": label,
        "nelx": nelx,
        "nely": nely,
        "n_elements": mesh.n_elements,
        "n_nodes": mesh.n_nodes,
        "n_dofs": mesh.n_dofs,
        "volfrac": CONFIG["volfrac"],
        "penal": CONFIG["penal"],
        "rmin": CONFIG["rmin"],
        "max_iterations": CONFIG["max_iterations"],
        "tolerance": CONFIG["tolerance"],
        "warmups": warmups,
        "repetitions": repetitions,
        "iterations": final["iterations"],
        "converged": final["converged"],
        "final_compliance": final["compliance"],
        "final_volume_fraction": final["volume_fraction"],
        "final_change": final["change"],
        "result_compliance": result_compliance,
        "result_volume_fraction": float(density.mean()),
        "density_min": float(density.min()),
        "density_max": float(density.max()),
        "density_sum": float(density.sum()),
        "density_sha256": final["digest"],
        "compliance_spread": max(r["compliance"] for r in per_rep)
        - min(r["compliance"] for r in per_rep),
        "numerically_repeatable": len({r["digest"] for r in per_rep}) == 1,
        "run_wall_s_mean": statistics.fmean(run_walls),
        "run_wall_s_median": statistics.median(run_walls),
        "run_wall_s_stdev": statistics.stdev(run_walls) if len(run_walls) > 1 else 0.0,
        "run_wall_s_min": min(run_walls),
        "run_wall_s_max": max(run_walls),
        "iteration_s_mean": statistics.fmean(iteration_times),
        "iteration_s_median": statistics.median(iteration_times),
        "iteration_s_p10": _percentile(iteration_times, 10),
        "iteration_s_p90": _percentile(iteration_times, 90),
        "construct_s_median": statistics.median(construction_times),
        "peak_rss_bytes": peak_rss if peak_rss is not None else "",
        "peak_rss_delta_bytes": peak_delta if peak_delta is not None else "",
        "peak_rss_saturated": saturated if saturated is not None else "",
        "process_wall_ratio": statistics.fmean(process_times) / statistics.fmean(run_walls),
        "perturbation_ratio": instrumented_wall / statistics.fmean(run_walls),
        "python_version": platform.python_version(),
        "numpy_version": np.__version__,
        "scipy_version": _scipy_version(),
        "platform_system": platform.system(),
        "platform_machine": platform.machine(),
        "cpu_count": os.cpu_count(),
    }

    stage_rows = []
    for stage, scope, in_partition, seconds in partition + context:
        calls = recorder.calls.get(stage, 0)
        stage_rows.append(
            {
                "run_id": run_id,
                "label": label,
                "nelx": nelx,
                "nely": nely,
                "stage": stage,
                "scope": scope,
                "in_partition": in_partition,
                "calls": calls,
                "calls_per_iteration": calls / n_iterations if calls else "",
                "total_s": seconds,
                "mean_per_call_us": (seconds / calls * 1e6) if calls else "",
                "mean_per_iteration_ms": seconds / n_iterations * 1e3,
                "share_of_iteration": seconds / iterations_total if iterations_total else "",
                "note": STAGE_NOTES.get(stage, ""),
            }
        )

    return {"case": case_row, "stages": stage_rows}


def _percentile(values, percent):
    ordered = sorted(values)
    if not ordered:
        return ""
    position = (len(ordered) - 1) * percent / 100.0
    lower = int(position)
    upper = min(lower + 1, len(ordered) - 1)
    return ordered[lower] + (ordered[upper] - ordered[lower]) * (position - lower)


def _scipy_version():
    import scipy
    return scipy.__version__


# ---------------------------------------------------------------------------
# Environment capture.  Deliberately free of identifying machine information.
# ---------------------------------------------------------------------------
def collect_environment(run_id, label, cases, warmups, repetitions):
    return {
        "run_id": run_id,
        "timestamp_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "label": label,
        "python": {
            "version": platform.python_version(),
            "implementation": platform.python_implementation(),
            # basename only: the full path contains the home directory
            "executable": os.path.basename(sys.executable),
        },
        "numpy": {"version": np.__version__},
        "scipy": {"version": _scipy_version()},
        "platform": {
            "system": platform.system(),
            "release": platform.release(),
            "machine": platform.machine(),
            "cpu_model": _cpu_model(),
            "cpu_count": os.cpu_count(),
        },
        "thread_environment": {
            name: os.environ.get(name)
            for name in (
                "OMP_NUM_THREADS",
                "OPENBLAS_NUM_THREADS",
                "MKL_NUM_THREADS",
                "VECLIB_MAXIMUM_THREADS",
                "NUMEXPR_NUM_THREADS",
            )
        },
        "methodology": {
            "subprocess_per_case": True,
            "warmups": warmups,
            "repetitions": repetitions,
            "config": dict(CONFIG),
            "timed_region": "TopologyOptimizer.run() only",
            "excluded_from_timing": [
                "process start-up and imports",
                "optimizer construction",
                "plotting",
                "CSV and JSON writing",
            ],
        },
        "cases": [f"{nelx}x{nely}" for nelx, nely in cases],
        "deliberately_excluded": [
            "hostname",
            "username",
            "home directory",
            "working directory",
            "full interpreter path",
            "environment variables other than the BLAS thread settings above",
        ],
    }


def _cpu_model():
    """Best-effort CPU model string.  Never returns a hostname."""
    if sys.platform == "darwin":
        try:
            completed = subprocess.run(
                ["sysctl", "-n", "hw.model"],
                capture_output=True, text=True, timeout=5, check=False,
            )
            if completed.returncode == 0 and completed.stdout.strip():
                return completed.stdout.strip()
        except (OSError, subprocess.SubprocessError):
            pass
    return platform.processor() or ""


# ---------------------------------------------------------------------------
# Parent: orchestration and output.
# ---------------------------------------------------------------------------
def parse_case(text):
    nelx, _, nely = text.lower().partition("x")
    try:
        return int(nelx), int(nely)
    except ValueError:
        raise argparse.ArgumentTypeError(
            f"invalid case {text!r}; expected NELXxNELY such as 60x20"
        ) from None


def run_in_subprocess(case, run_id, label, warmups, repetitions):
    """Run one case in its own process and return its payload."""
    command = [
        sys.executable, str(Path(__file__).resolve()),
        "--worker",
        "--nelx", str(case[0]),
        "--nely", str(case[1]),
        "--run-id", run_id,
        "--label", label,
        "--warmups", str(warmups),
        "--repetitions", str(repetitions),
    ]
    completed = subprocess.run(command, capture_output=True, text=True, check=False)

    for line in completed.stdout.splitlines():
        if line.startswith(_SENTINEL):
            return json.loads(line[len(_SENTINEL):])

    raise RuntimeError(
        f"worker for {case[0]}x{case[1]} produced no result "
        f"(exit {completed.returncode})\n"
        f"--- stdout ---\n{completed.stdout}\n--- stderr ---\n{completed.stderr}"
    )


def write_csv(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def result_paths(output_dir, label):
    """The artifacts one benchmark run writes, split by scope.

    Returns ``(label_paths, shared_paths)``.  The label-scoped CSVs are named
    after the label, so an existing one means this run would replace that
    label's results.  ``environment.json`` carries no label and is rewritten by
    every run in a directory by design, which is why it is tracked separately:
    it is part of what a run replaces, but it is not on its own a collision.
    """
    output_dir = Path(output_dir)
    label_paths = (
        output_dir / f"{label}.csv",
        output_dir / f"{label}_stages.csv",
    )
    shared_paths = (output_dir / "environment.json",)
    return label_paths, shared_paths


def refuse_overwrite(label_paths, shared_paths, overwrite):
    """Stop before measuring anything if this run would replace a result.

    A bare ``python benchmarks/benchmark.py`` uses the default ``reference``
    label, whose artifacts are the frozen baseline.  Refusing by default turns
    replacing an existing result into a deliberate act (``--overwrite``) rather
    than something a mistyped command does quietly.

    Only the label-scoped artifacts can trigger a refusal.  Blocking on the
    shared ``environment.json`` would mean that adding a *second* label to an
    existing results directory needed ``--overwrite`` even though no result
    would be lost, so it is reported as something the run replaces without
    being a reason to stop.  The check runs before any case is measured, so a
    collision costs nothing but the time to read the message, and nothing is
    ever renamed to dodge one -- a caller who wants a different name passes a
    different ``--label``.
    """
    if overwrite:
        return

    collisions = [path for path in label_paths if path.exists()]
    if not collisions:
        return

    replaced = collisions + [path for path in shared_paths if path.exists()]
    noun = "file" if len(replaced) == 1 else "files"
    listed = "\n".join(f"  {path}" for path in replaced)
    raise SystemExit(
        f"error: refusing to overwrite {len(replaced)} existing result {noun}:\n"
        f"{listed}\n"
        "\n"
        "Nothing was measured and no file was written.\n"
        "\n"
        "To resolve this, either:\n"
        "  - choose another label:    --label <new-label>\n"
        "  - choose another output:   --output-dir <directory>\n"
        f"  - or replace the {noun} above on purpose: --overwrite\n"
    )


def main(argv=None):
    parser = argparse.ArgumentParser(
        description="Benchmark the topoopt reference solver.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--cases", nargs="+", default=list(DEFAULT_CASES),
                        metavar="NELXxNELY", help="mesh sizes to benchmark")
    parser.add_argument("--repetitions", type=int, default=3,
                        help="timed clean runs per case")
    parser.add_argument("--warmups", type=int, default=1,
                        help="discarded warm-up runs per case")
    parser.add_argument("--label", default=DEFAULT_LABEL,
                        help="output stem, e.g. 'optimized' for a later build")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--overwrite", action="store_true",
                        help="replace existing result files instead of refusing to run")
    parser.add_argument("--in-process", action="store_true",
                        help="debug only: skip subprocesses (memory readings become unreliable)")
    parser.add_argument("--list-cases", action="store_true",
                        help="print the case list with problem sizes and exit")
    # Worker mode (internal).
    parser.add_argument("--worker", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--nelx", type=int, default=60, help=argparse.SUPPRESS)
    parser.add_argument("--nely", type=int, default=20, help=argparse.SUPPRESS)
    parser.add_argument("--run-id", default="", help=argparse.SUPPRESS)

    args = parser.parse_args(argv)

    if args.worker:
        payload = run_case(
            args.nelx, args.nely, args.run_id, args.label,
            args.warmups, args.repetitions,
        )
        print(_SENTINEL + json.dumps(payload))
        return 0

    cases = [parse_case(text) for text in args.cases]

    if args.list_cases:
        print(f"{'case':>10} {'elements':>10} {'nodes':>10} {'dofs':>10}")
        for nelx, nely in cases:
            mesh = fem.Mesh(nelx, nely)
            print(f"{f'{nelx}x{nely}':>10} {mesh.n_elements:>10} "
                  f"{mesh.n_nodes:>10} {mesh.n_dofs:>10}")
        return 0

    # Refuse before measuring anything: the check is only useful ahead of the
    # run, since afterwards the baseline it protects has already been replaced.
    # Only the label-scoped artifacts can refuse; the shared environment.json is
    # reported among the files the run replaces, but does not on its own block a
    # new label written into a directory that already has one.
    output_dir = Path(args.output_dir)
    label_paths, shared_paths = result_paths(output_dir, args.label)
    refuse_overwrite(label_paths, shared_paths, args.overwrite)
    case_path, stage_path = label_paths
    (environment_path,) = shared_paths

    _assert_problem_equivalence(cases)

    run_id = time.strftime("%Y%m%dT%H%M%SZ", time.gmtime())

    print(f"topoopt reference benchmark  (run {run_id}, label {args.label})")
    print(f"warm-ups: {args.warmups}   repetitions: {args.repetitions}   "
          f"cases: {', '.join(f'{a}x{b}' for a, b in cases)}")
    print()

    case_rows, stage_rows = [], []
    for nelx, nely in cases:
        print(f"  running {nelx}x{nely} ...", end="", flush=True)
        start = time.perf_counter()
        if args.in_process:
            payload = run_case(nelx, nely, run_id, args.label,
                               args.warmups, args.repetitions)
        else:
            payload = run_in_subprocess((nelx, nely), run_id, args.label,
                                        args.warmups, args.repetitions)
        elapsed = time.perf_counter() - start

        case_rows.append(payload["case"])
        stage_rows.extend(payload["stages"])
        row = payload["case"]
        status = "converged" if row["converged"] else "hit iteration cap"
        print(f" {row['iterations']} iters, {status}, "
              f"compliance {row['final_compliance']:.6e}, {elapsed:.1f}s")

    write_csv(case_path, case_rows)
    write_csv(stage_path, stage_rows)
    output_dir.mkdir(parents=True, exist_ok=True)
    with open(environment_path, "w") as handle:
        json.dump(
            collect_environment(run_id, args.label, cases,
                                args.warmups, args.repetitions),
            handle, indent=2, sort_keys=False,
        )
        handle.write("\n")

    print()
    print(f"wrote {case_path}")
    print(f"wrote {stage_path}")
    print(f"wrote {environment_path}")

    non_converged = [f"{r['nelx']}x{r['nely']}" for r in case_rows if not r["converged"]]
    if non_converged:
        print()
        print(f"NOTE: did not converge within {CONFIG['max_iterations']} iterations: "
              f"{', '.join(non_converged)}. Their final compliance and total runtime "
              f"are cap-truncated, not converged values.")

    return 0


if __name__ == "__main__":
    sys.exit(main())
