"""Classic 2D cantilever minimum-compliance topology optimization.

A rectangular domain is fully fixed along its left edge and loaded by a single
downward point load at the middle of its right edge::

    python -m examples.cantilever --nelx 60 --nely 20 --volfrac 0.4 --penal 3.0 --rmin 1.5

The run prints one line per iteration and writes three files into the output
directory (``results/`` by default):

* ``topology.png``     -- the optimized density field
* ``convergence.png``  -- compliance, volume fraction and density change
* ``history.csv``      -- the same per-iteration numbers, for later analysis

Plotting uses the non-interactive Agg backend, so the example runs from a plain
terminal without a display.
"""

from __future__ import annotations

import argparse
import csv
import dataclasses
from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # must be set before pyplot is imported

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.colors import LinearSegmentedColormap  # noqa: E402

from topoopt.fem import Mesh  # noqa: E402
from topoopt.optimizer import IterationRecord, TopologyOptimizer  # noqa: E402

DEFAULT_OUTPUT_DIR = Path("results")
LOAD_MAGNITUDE = 1.0

# Chart chrome and series colours (light surface).
SURFACE = "#fcfcfb"
INK_PRIMARY = "#0b0b0b"
INK_SECONDARY = "#52514e"
INK_MUTED = "#898781"
GRIDLINE = "#e1e0d9"
AXIS = "#c3c2b7"
SERIES = "#2a78d6"

# Single-hue sequential ramp for density: light means void, dark means solid.
DENSITY_RAMP = (
    "#cde2fb", "#b7d3f6", "#9ec5f4", "#86b6ef", "#6da7ec", "#5598e7", "#3987e5",
    "#2a78d6", "#256abf", "#1c5cab", "#184f95", "#104281", "#0d366b",
)


def build_cantilever(nelx, nely, load=LOAD_MAGNITUDE):
    """Build the canonical cantilever: fixed left edge, point load on the right.

    Returns the mesh, the restrained degrees of freedom (both components of
    every node on the left edge) and the nodal load vector (a single downward
    force at the middle of the right edge).
    """
    mesh = Mesh(nelx, nely)

    left_edge = mesh.node_index(0, np.arange(nely + 1))
    fixed_dofs = np.concatenate([2 * left_edge, 2 * left_edge + 1]).astype(np.intp)

    loads = np.zeros(mesh.n_dofs)
    loads[2 * mesh.node_index(nelx, nely // 2) + 1] = -load

    return mesh, fixed_dofs, loads


def write_history(history, path):
    """Write one CSV row per iteration, using the record fields as columns."""
    fieldnames = [field.name for field in dataclasses.fields(IterationRecord)]
    with open(path, "w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for record in history:
            writer.writerow(dataclasses.asdict(record))


def _configure_matplotlib():
    plt.rcParams.update(
        {
            "figure.facecolor": SURFACE,
            "savefig.facecolor": SURFACE,
            "axes.facecolor": SURFACE,
            "font.family": "sans-serif",
            "font.size": 9,
            "savefig.dpi": 150,
            "figure.dpi": 150,
        }
    )


def _style_axes(axes, grid_axis="y"):
    """Hairline chrome: recessive spines, grid and tick labels.

    ``grid_axis`` is ``None`` for image panels, where gridlines would only
    scribble over the data.
    """
    axes.set_axisbelow(True)
    if grid_axis is None:
        axes.grid(False)
    else:
        axes.grid(True, axis=grid_axis, color=GRIDLINE, linewidth=0.8, linestyle="-")
    for side in ("top", "right"):
        axes.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        axes.spines[side].set_color(AXIS)
        axes.spines[side].set_linewidth(0.8)
    axes.tick_params(colors=INK_MUTED, labelsize=8, length=3, width=0.8)


def _label_endpoint(axes, x, y, text):
    """Direct-label the final point of a series rather than every point."""
    axes.plot(
        [x[-1]],
        [y[-1]],
        marker="o",
        markersize=5,
        color=SERIES,
        markeredgecolor=SURFACE,
        markeredgewidth=1.5,
        zorder=3,
    )
    axes.annotate(
        text,
        xy=(x[-1], y[-1]),
        xytext=(6, 0),
        textcoords="offset points",
        color=INK_SECONDARY,
        fontsize=8,
        va="center",
        ha="left",
    )


def plot_topology(density, path):
    """Render the optimized density field."""
    nely, nelx = density.shape
    cmap = LinearSegmentedColormap.from_list("density", DENSITY_RAMP)

    figure, axes = plt.subplots(figsize=(9.0, 3.6), layout="constrained")
    image = axes.imshow(
        density,
        origin="lower",
        extent=(0, nelx, 0, nely),
        cmap=cmap,
        vmin=0.0,
        vmax=1.0,
        interpolation="nearest",
    )

    axes.set_aspect("equal")
    axes.set_xlabel("x [element units]", color=INK_SECONDARY)
    axes.set_ylabel("y [element units]", color=INK_SECONDARY)
    _style_axes(axes, grid_axis=None)
    axes.set_title(
        "Cantilever: optimized material layout",
        color=INK_PRIMARY,
        fontsize=12,
        loc="left",
    )

    colorbar = figure.colorbar(image, ax=axes, pad=0.02)
    colorbar.set_label("element density", color=INK_SECONDARY)
    colorbar.ax.tick_params(colors=INK_MUTED, labelsize=8)
    colorbar.outline.set_edgecolor(AXIS)

    figure.savefig(path, bbox_inches="tight")
    plt.close(figure)


def plot_convergence(history, path, volfrac, tolerance):
    """Small multiples of the three per-iteration measures.

    Each measure gets its own panel and its own axis; they span different
    ranges and must not share a scale.
    """
    iterations = np.array([record.iteration for record in history])
    compliance = np.array([record.compliance for record in history])
    volume_fraction = np.array([record.volume_fraction for record in history])
    change = np.array([record.change for record in history])

    figure, panels = plt.subplots(1, 3, figsize=(11.0, 3.1), layout="constrained")

    _single_series_panel(
        panels[0], iterations, compliance, "compliance", "Compliance"
    )
    _label_endpoint(panels[0], iterations, compliance, f"{compliance[-1]:.3e}")

    _single_series_panel(
        panels[1], iterations, volume_fraction, "volume fraction", "Volume fraction"
    )
    panels[1].axhline(volfrac, color=INK_MUTED, linewidth=0.9, linestyle="--", zorder=1)
    panels[1].annotate(
        f"target {volfrac:.2f}",
        xy=(iterations[0], volfrac),
        xytext=(4, 4),
        textcoords="offset points",
        color=INK_MUTED,
        fontsize=8,
    )
    span = max(float(np.ptp(volume_fraction)), 1e-4)
    panels[1].set_ylim(volfrac - 4 * span, volfrac + 4 * span)

    positive = change > 0.0
    _single_series_panel(
        panels[2],
        iterations[positive],
        change[positive],
        "max density change",
        "Density change",
        log_scale=True,
    )
    panels[2].axhline(tolerance, color=INK_MUTED, linewidth=0.9, linestyle="--", zorder=1)
    panels[2].annotate(
        f"tolerance {tolerance:g}",
        xy=(iterations[0], tolerance),
        xytext=(4, 4),
        textcoords="offset points",
        color=INK_MUTED,
        fontsize=8,
    )

    figure.suptitle(
        f"Convergence history ({len(history)} iterations)",
        color=INK_PRIMARY,
        fontsize=12,
        x=0.005,
        ha="left",
    )
    figure.savefig(path, bbox_inches="tight")
    plt.close(figure)


def _single_series_panel(axes, x, y, ylabel, title, log_scale=False):
    axes.plot(x, y, color=SERIES, linewidth=1.8, solid_capstyle="round")
    if log_scale:
        axes.set_yscale("log")
    axes.set_xlabel("iteration", color=INK_SECONDARY)
    axes.set_ylabel(ylabel, color=INK_SECONDARY)
    axes.set_title(title, color=INK_PRIMARY, fontsize=10, loc="left")
    axes.set_xlim(x[0], x[-1] + max(1, 0.06 * (x[-1] - x[0])))
    _style_axes(axes)


def parse_args(argv=None):
    parser = argparse.ArgumentParser(
        description="Minimum-compliance topology optimization of a 2D cantilever."
    )
    parser.add_argument("--nelx", type=int, default=60, help="elements along x")
    parser.add_argument("--nely", type=int, default=20, help="elements along y")
    parser.add_argument("--volfrac", type=float, default=0.4, help="volume fraction")
    parser.add_argument("--penal", type=float, default=3.0, help="SIMP penalty exponent")
    parser.add_argument("--rmin", type=float, default=1.5, help="filter radius, in elements")
    parser.add_argument("--max-iter", type=int, default=200, help="iteration cap")
    parser.add_argument("--tol", type=float, default=1e-2, help="convergence tolerance")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help="directory for the plots and the history file",
    )
    return parser.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    _configure_matplotlib()

    mesh, fixed_dofs, loads = build_cantilever(args.nelx, args.nely)

    print(
        f"Cantilever {args.nelx} x {args.nely} elements "
        f"({mesh.n_elements} elements, {mesh.n_dofs} DOFs)"
    )
    print(
        f"volfrac = {args.volfrac}, penal = {args.penal}, rmin = {args.rmin}, "
        f"tol = {args.tol}, max_iter = {args.max_iter}"
    )
    print()

    optimizer = TopologyOptimizer(
        mesh,
        fixed_dofs,
        loads,
        volfrac=args.volfrac,
        penal=args.penal,
        rmin=args.rmin,
        max_iterations=args.max_iter,
        tolerance=args.tol,
    )
    result = optimizer.run()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    write_history(result.history, output_dir / "history.csv")
    plot_topology(result.density, output_dir / "topology.png")
    plot_convergence(result.history, output_dir / "convergence.png", args.volfrac, args.tol)

    print()
    print(f"wrote {output_dir / 'topology.png'}")
    print(f"wrote {output_dir / 'convergence.png'}")
    print(f"wrote {output_dir / 'history.csv'}")

    return result


if __name__ == "__main__":
    main()
