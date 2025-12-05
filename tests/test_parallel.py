import numpy as np
import pytest

from pdex import _parallel as parallel


def build_chunk() -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    X_chunk = np.array(
        [
            [1, 2, 3],
            [2, 3, 4],
            [5, 6, 7],
            [3, 4, 5],
        ],
        dtype=np.float32,
    )
    reference_mask = np.array([False, False, True, True])
    X_ref = X_chunk[reference_mask, :]
    means_ref = X_ref.mean(axis=0)
    return X_chunk, X_ref, means_ref


def test_get_default_parallelization(monkeypatch):
    monkeypatch.setattr(parallel.os, "cpu_count", lambda: 32)
    workers, threads = parallel.get_default_parallelization()
    assert workers == 4
    assert threads is None

    monkeypatch.setattr(parallel.os, "cpu_count", lambda: 2)
    workers_small, threads_small = parallel.get_default_parallelization()
    assert workers_small == 1
    assert threads_small is None


def test_set_numba_threads_none(monkeypatch):
    monkeypatch.setattr(parallel, "get_num_threads", lambda: 8)
    assert parallel.set_numba_threads(None) == 8


def test_set_numba_threads_explicit(monkeypatch):
    captured: dict[str, int] = {}

    def fake_set_num_threads(value: int) -> None:
        captured["value"] = value

    monkeypatch.setattr(parallel, "set_num_threads", fake_set_num_threads)
    assert parallel.set_numba_threads(5) == 5
    assert captured["value"] == 5


def test_process_target_in_chunk_uses_numba(monkeypatch):
    X_chunk, X_ref, means_ref = build_chunk()
    target_mask = np.array([True, True, False, False])
    gene_names = np.array(["g0", "g1", "g2"])

    def fake_vectorized(X_target: np.ndarray, X_reference: np.ndarray):
        assert X_reference is X_ref
        return (
            np.full(X_target.shape[1], 0.5, dtype=np.float64),
            np.arange(X_target.shape[1], dtype=np.float64),
        )

    monkeypatch.setattr(parallel, "vectorized_ranksum_test", fake_vectorized)
    results = parallel.process_target_in_chunk(
        target="target_a",
        reference="target_b",
        X_chunk=X_chunk,
        X_ref=X_ref,
        target_mask=target_mask,
        means_ref=means_ref,
        gene_names=gene_names,
        chunk_start=0,
        metric="wilcoxon",
        tie_correct=True,
        is_log1p=False,
        exp_post_agg=True,
        clip_value=20.0,
        use_numba=True,
    )

    assert len(results) == X_chunk.shape[1]
    assert all(result["p_value"] == 0.5 for result in results)
    assert [result["statistic"] for result in results] == [0.0, 1.0, 2.0]
    assert {result["feature"] for result in results} == {"g0", "g1", "g2"}


def test_process_target_in_chunk_metric_fallback(monkeypatch):
    X_chunk, X_ref, means_ref = build_chunk()
    target_mask = np.array([True, False, False, False])
    gene_names = np.array(["g0", "g1", "g2"])

    call_counter = {"count": 0}

    def fail_vectorized(*_args, **_kwargs):
        call_counter["count"] += 1
        raise AssertionError("Vectorized test should not run for anderson")

    monkeypatch.setattr(parallel, "vectorized_ranksum_test", fail_vectorized)

    results = parallel.process_target_in_chunk(
        target="target_a",
        reference="target_b",
        X_chunk=X_chunk,
        X_ref=X_ref,
        target_mask=target_mask,
        means_ref=means_ref,
        gene_names=gene_names,
        chunk_start=0,
        metric="anderson",
        tie_correct=True,
        is_log1p=False,
        exp_post_agg=True,
        clip_value=20.0,
        use_numba=True,
    )

    assert call_counter["count"] == 0
    assert len(results) == X_chunk.shape[1]
    assert all("p_value" in result for result in results)


def test_process_targets_parallel_sequential():
    targets = ["t1", "t2", "t3"]

    def process_fn(*, target: str, scale: int) -> list[dict]:
        return [{"target": target, "value": scale}]

    results = parallel.process_targets_parallel(
        targets=targets,
        process_fn=process_fn,
        num_workers=1,
        show_progress=False,
        scale=2,
    )

    assert [r["target"] for r in results] == targets
    assert all(r["value"] == 2 for r in results)


def test_process_targets_parallel_threaded():
    targets = [f"t{i}" for i in range(5)]

    def process_fn(*, target: str) -> list[dict]:
        return [{"target": target}]

    results = parallel.process_targets_parallel(
        targets=targets,
        process_fn=process_fn,
        num_workers=2,
        show_progress=False,
    )

    assert sorted(r["target"] for r in results) == sorted(targets)


def test_vectorized_ranksum_test_calls_kernel(monkeypatch):
    # Use float values (not integers) to test the sorting kernel path
    X_target = np.array([[1.5, 2.5], [3.5, 4.5]], dtype=np.float32)
    X_ref = np.array([[0.5, 1.5], [1.5, 1.5]], dtype=np.float32)
    captured: dict[str, np.ndarray | tuple] = {}

    def fake_sorting_kernel(Xt, Xr):
        captured["kernel_inputs"] = (Xt, Xr)
        return (
            np.array([0.1, 0.2], dtype=np.float64),
            np.array([1.0, 2.0], dtype=np.float64),
        )

    monkeypatch.setattr(parallel, "_ranksum_kernel_sorting", fake_sorting_kernel)

    p_vals, stats = parallel.vectorized_ranksum_test(X_target, X_ref)

    Xt, Xr = captured["kernel_inputs"]
    assert Xt.flags["C_CONTIGUOUS"]
    assert Xr.flags["C_CONTIGUOUS"]
    assert np.allclose(p_vals, [0.1, 0.2])
    assert np.allclose(stats, [1.0, 2.0])


def test_is_integer_data_detects_integer_values():
    assert parallel.is_integer_data(np.array([0.0, 1.0, 2.0]))


def test_is_integer_data_detects_float_values():
    assert not parallel.is_integer_data(np.array([0.0, 0.5, 1.0]))


def test_should_use_numba_respects_requirements():
    ints = np.array([[0, 1], [2, 3]], dtype=np.float32)
    floats = np.array([[0.1, 1.0], [2.0, 3.0]], dtype=np.float32)

    # With dual-kernel support, numba is used for both int and float data
    assert parallel.should_use_numba(ints, "wilcoxon", 2)
    assert parallel.should_use_numba(floats, "wilcoxon", 2)
    # Still disabled for non-wilcoxon metrics and single-threaded mode
    assert not parallel.should_use_numba(ints, "anderson", 2)
    assert not parallel.should_use_numba(ints, "wilcoxon", 1)
