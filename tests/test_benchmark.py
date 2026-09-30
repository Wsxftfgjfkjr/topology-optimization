"""Tests for the benchmark harness's output-overwrite guard.

The harness defaults to the ``reference`` label, whose artifacts are the frozen
baseline, so a bare invocation used to be able to replace it.  These tests pin
the guard that stops that: a run refuses to write over an existing result
unless ``--overwrite`` is passed.

Every case here runs against ``tmp_path``.  Nothing in this file may touch the
tracked ``benchmarks/results/`` directory, and the tiny ``4x2`` case keeps the
runs short enough to belong in the ordinary test suite.
"""

import pytest

from benchmarks import benchmark

# Small enough to run in well under a second, and in-process so the tests do
# not pay interpreter start-up per case.  Only the overwrite guard is under
# test here; the numbers a run produces are irrelevant to it.
TINY_CASE = ("--cases", "4x2", "--repetitions", "1", "--warmups", "0",
             "--in-process")


def run_cli(*args):
    """Run the benchmark CLI in process.

    Returns ``0`` when the run completed.  When the guard refuses, it raises
    ``SystemExit`` carrying the message, which is returned instead.
    """
    try:
        benchmark.main(list(args))
    except SystemExit as exit_signal:
        return exit_signal.code
    return 0


def test_fresh_output_location_succeeds(tmp_path):
    """A run into an empty directory writes all three artifacts."""
    output_dir = tmp_path / "out"

    assert run_cli(*TINY_CASE, "--label", "alpha",
                   "--output-dir", str(output_dir)) == 0
    assert (output_dir / "alpha.csv").is_file()
    assert (output_dir / "alpha_stages.csv").is_file()
    assert (output_dir / "environment.json").is_file()


def test_existing_result_is_refused_before_any_case_runs(tmp_path, monkeypatch):
    """The guard fires ahead of the measurements, not after them.

    ``run_case`` and ``run_in_subprocess`` are replaced by tripwires: if the
    guard let the run start, one of them would raise instead of the guard
    returning its message.
    """
    output_dir = tmp_path / "out"
    output_dir.mkdir()
    existing = output_dir / "alpha.csv"
    existing.write_text("iteration,compliance\n")

    def tripwire(*args, **kwargs):
        raise AssertionError("a case was measured despite the output collision")

    monkeypatch.setattr(benchmark, "run_case", tripwire)
    monkeypatch.setattr(benchmark, "run_in_subprocess", tripwire)

    message = run_cli(*TINY_CASE, "--label", "alpha",
                      "--output-dir", str(output_dir))

    assert isinstance(message, str), "the run proceeded instead of refusing"
    assert existing.read_text() == "iteration,compliance\n"


def test_refusal_explains_the_conflict_and_the_ways_out(tmp_path):
    """The message names the file and every option for resolving it."""
    output_dir = tmp_path / "out"
    output_dir.mkdir()
    (output_dir / "alpha.csv").write_text("x")

    message = run_cli(*TINY_CASE, "--label", "alpha",
                      "--output-dir", str(output_dir))

    assert "alpha.csv" in message
    assert "--label" in message
    assert "--output-dir" in message
    assert "--overwrite" in message
    assert str(output_dir) in message


def test_refusal_reports_every_artifact_that_would_be_replaced(tmp_path):
    """Both label-scoped files and the shared metadata are listed."""
    output_dir = tmp_path / "out"
    output_dir.mkdir()
    (output_dir / "alpha.csv").write_text("x")
    (output_dir / "alpha_stages.csv").write_text("x")
    (output_dir / "environment.json").write_text("{}")

    message = run_cli(*TINY_CASE, "--label", "alpha",
                      "--output-dir", str(output_dir))

    assert "alpha.csv" in message
    assert "alpha_stages.csv" in message
    assert "environment.json" in message
    assert "3 existing result files" in message


def test_overwrite_permits_intentional_replacement(tmp_path):
    """``--overwrite`` is the explicit opt-in, and it rewrites the files."""
    output_dir = tmp_path / "out"
    output_dir.mkdir()
    target = output_dir / "alpha.csv"
    target.write_text("stale")

    assert run_cli(*TINY_CASE, "--label", "alpha", "--overwrite",
                   "--output-dir", str(output_dir)) == 0
    assert target.read_text() != "stale"
    assert target.read_text().startswith("run_id,")


def test_default_label_is_protected(tmp_path, monkeypatch):
    """The bare invocation targets ``reference``; a baseline must survive it."""
    output_dir = tmp_path / "results"
    output_dir.mkdir()
    files = {
        "reference.csv": b"run_id,nelx\n",
        "reference_stages.csv": b"run_id,stage\n",
        "environment.json": b"{}\n",
    }
    for name, payload in files.items():
        (output_dir / name).write_bytes(payload)

    def tripwire(*args, **kwargs):
        raise AssertionError("a case was measured despite the output collision")

    monkeypatch.setattr(benchmark, "run_case", tripwire)
    monkeypatch.setattr(benchmark, "run_in_subprocess", tripwire)

    # No --label, so the default `reference` label is used.
    message = run_cli(*TINY_CASE, "--output-dir", str(output_dir))

    assert isinstance(message, str)
    assert "reference.csv" in message
    for name, payload in files.items():
        assert (output_dir / name).read_bytes() == payload


def test_different_label_does_not_require_overwrite(tmp_path):
    """A new label collides with nothing, so the guard must stay out of the way.

    The shared ``environment.json`` is rewritten by every run in a directory by
    design; treating it as a collision would make adding a second label need
    ``--overwrite`` even though no result is lost.  This is the documented way
    to add a comparison label, so it is pinned here.
    """
    output_dir = tmp_path / "out"
    output_dir.mkdir()
    (output_dir / "environment.json").write_text("{}")

    assert run_cli(*TINY_CASE, "--label", "alpha",
                   "--output-dir", str(output_dir)) == 0
    assert run_cli(*TINY_CASE, "--label", "beta",
                   "--output-dir", str(output_dir)) == 0

    assert (output_dir / "alpha.csv").is_file()
    assert (output_dir / "beta.csv").is_file()


def test_list_cases_is_not_guarded(tmp_path):
    """``--list-cases`` writes nothing, so it needs no opt-in."""
    output_dir = tmp_path / "out"
    output_dir.mkdir()
    (output_dir / "reference.csv").write_text("x")

    assert run_cli("--list-cases", "--output-dir", str(output_dir)) == 0


@pytest.mark.parametrize("label,expected", [
    ("alpha", ("alpha.csv", "alpha_stages.csv")),
    ("optimized_ordering", ("optimized_ordering.csv",
                            "optimized_ordering_stages.csv")),
])
def test_result_paths_are_named_after_the_label(tmp_path, label, expected):
    label_paths, shared_paths = benchmark.result_paths(tmp_path, label)

    assert tuple(path.name for path in label_paths) == expected
    assert tuple(path.parent for path in label_paths) == (tmp_path, tmp_path)
    assert tuple(path.name for path in shared_paths) == ("environment.json",)
