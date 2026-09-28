# Topology Optimization

A lightweight 2D structural topology optimization platform focused on numerical
algorithms, software engineering, and performance optimization.

## Purpose

The goal is a small, readable, and correct implementation of density-based
topology optimization that can serve as a **reference baseline**. A later phase
will add optimized implementations that are measured against this one, so this
version deliberately favors clarity over speed: every step is written out
explicitly rather than compressed.

## V0.1 scope

This milestone implements a single classic problem: **minimum-compliance
topology optimization of a 2D cantilever** on a structured quadrilateral mesh.

In scope:

- structured rectangular mesh, 4-node (Q4) plane-stress elements, 2 DOF per node
- linear elastic material, SIMP density interpolation
- compliance minimization under a volume-fraction constraint
- analytical compliance sensitivities
- sensitivity filtering to suppress checkerboarding
- Optimality Criteria (OC) density update
- convergence on the maximum density change
- sparse assembly and a sparse direct linear solver

Deliberately out of scope for now: 3D, unstructured or arbitrary meshes, multiple
load cases, additional physics (thermal, buckling, stress constraints), and the
FastAPI/Docker/database/async scaffolding that a hosted service would need.

## Numerical method

The optimizer solves

```
minimize    c(x) = Uᵀ K(x) U
subject to  K(x) U = F
            Σ xₑ / nₑ ≤ volfrac
            x_min ≤ xₑ ≤ 1
```

1. **Discretization.** The domain is a regular grid of `nelx × nely` bilinear
   quadrilaterals of unit edge length, so the design domain is
   `[0, nelx] × [0, nely]`. Nodes are numbered row-major with the x index
   fastest; each element's four nodes are ordered counter-clockwise from its
   lower-left corner.

2. **Element stiffness.** `Kₑ = ∫ Bᵀ D B dΩ` for the plane-stress constitutive
   matrix `D`, evaluated with 2×2 Gauss quadrature. The rule is exact here: for
   a rectangle the integrand is at most quadratic in each natural coordinate.

3. **Material interpolation (SIMP).** `Eₑ(xₑ) = E_min + xₑ^penal (E₀ − E_min)`,
   with `E_min = 10⁻⁹ E₀` so the stiffness matrix stays non-singular.

4. **Assembly and solve.** The global stiffness matrix is assembled in sparse
   (COO → CSC) format; the restrained degrees of freedom are eliminated and the
   reduced system is solved with `scipy.sparse.linalg.spsolve`.

5. **Sensitivities.** For a fixed load the problem is self-adjoint, so

   ```
   ∂c/∂xₑ = −penal · xₑ^(penal−1) · (E₀ − E_min) · uₑᵀ k₀ uₑ
   ```

   where `k₀` is the unit-modulus element stiffness matrix. Note that `c` equals
   the sum of the element strain energies `Eₑ · uₑᵀ k₀ uₑ`, which the test suite
   checks directly.

6. **Sensitivity filter.** Raw element sensitivities are replaced by a cone
   weighted average of their neighbourhood,

   ```
   ∂̃c/∂xₑ = Σ_f H[e,f] · x_f · ∂c/∂x_f / ( xₑ · Σ_f H[e,f] )
   H[e,f] = max(0, rmin − ‖cₑ − c_f‖)
   ```

   where `rmin` is measured in **element units**, so it does not correspond to a
   fixed physical length if the mesh is refined without changing it. This is the
   Sigmund–Petersson filter in its sensitivity form; it leaves the finite element
   analysis untouched. `rmin ≤ 1` has no effect, since no neighbour lies inside
   the cone.

7. **Design update.** Optimality Criteria: stationarity of the Lagrangian gives
   `xₑ ← xₑ √(−∂̃c/∂xₑ / λ)`, clipped to the move limit and to
   `[x_min, 1]`, with the volume-constraint multiplier `λ` found by bisection.

8. **Convergence.** The loop stops when `max|x_new − x|` falls below the
   tolerance, or at the iteration cap.

## Installation

Python 3.10 or newer. The only runtime dependencies are NumPy, SciPy and
Matplotlib; pytest is needed for the test suite.

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e .
```

## Running the cantilever example

```bash
python -m examples.cantilever --nelx 60 --nely 20 --volfrac 0.4 --penal 3.0 --rmin 1.5
```

The run prints one line per iteration (iteration, compliance, volume fraction,
maximum density change, and the time for that iteration) and writes three files
into `results/`:

| File | Contents |
|---|---|
| `topology.png` | the optimized density field |
| `convergence.png` | compliance, volume fraction and density change per iteration |
| `history.csv` | the same per-iteration numbers, for later analysis |

`history.csv` carries the per-iteration compliance, volume fraction, maximum
density change and wall-clock time, which is the raw material for the
benchmarking phase. Plotting uses the non-interactive Agg backend, so the
example runs from a terminal without a display. `--max-iter`, `--tol` and
`--output-dir` are also available; run with `--help` for the full list.

## Running the tests

```bash
pytest
```

The suite covers the numerical building blocks rather than only the end-to-end
run: the element stiffness matrix (symmetry, positive semi-definiteness, its
three rigid-body null modes, and its strain energy against the constitutive law
for uniaxial and shear strain), mesh and DOF mapping, a constant-strain patch
test with analytical consistent nodal loads, the filter kernel and filtering
behaviour, and the OC update's bounds and volume constraint. A small cantilever
smoke test checks that the full loop finishes and returns finite values.

## Status

V0.1 — reference implementation. Not yet optimized for performance, and not
validated against commercial finite element software.
