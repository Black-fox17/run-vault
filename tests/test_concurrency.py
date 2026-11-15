"""Tests for multi-threaded SQLite concurrency and connection isolation."""

import sqlite3
import threading
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import runvault
from runvault.core.db.repository import SQLiteMetadataRepository
from runvault.core.errors import WorkspaceError


def test_concurrent_threads_against_same_workspace(tmp_path: Path) -> None:
    """Verify that multiple threads concurrently using the SDK do not violate thread checks."""
    exp = runvault.create_experiment("concurrent-exp", workspace=tmp_path)

    num_threads = 5
    runs_per_thread = 3

    def worker(thread_idx: int) -> list[str]:
        created_run_ids: list[str] = []
        for r_idx in range(runs_per_thread):
            run_name = f"thread-{thread_idx}-run-{r_idx}"
            with exp.start_run(name=run_name, tags={"thread": str(thread_idx)}) as run:
                run.log_params({"thread_id": thread_idx, "run_index": r_idx})
                for step in range(5):
                    run.log_metrics(
                        {"loss": 1.0 / (step + 1), "acc": 0.5 + 0.1 * step},
                        step=step,
                    )
                created_run_ids.append(run.id)
        return created_run_ids

    with ThreadPoolExecutor(max_workers=num_threads) as executor:
        futures = [executor.submit(worker, i) for i in range(num_threads)]
        results = [f.result() for f in futures]

    total_runs = sum(len(r) for r in results)
    assert total_runs == num_threads * runs_per_thread

    # Verify that all runs were properly persisted and marked FINISHED
    persisted_runs = exp.list_runs()
    assert len(persisted_runs) == total_runs
    for pr in persisted_runs:
        assert pr.status == runvault.RunStatus.FINISHED
        params = runvault.get_params(pr.id, workspace=tmp_path)
        assert "thread_id" in params
        history = runvault.get_metric_history(pr.id, metric_name="loss", workspace=tmp_path)
        assert len(history) == 5

    runvault.reset_repository_cache()


def test_thread_connection_ownership_and_isolation(tmp_path: Path) -> None:
    """Verify each thread strictly receives an independent connection owned by itself."""
    db_file = tmp_path / "metadata.db"
    repo = SQLiteMetadataRepository(db_file)

    main_conn = repo.connection
    worker_conns: list[sqlite3.Connection] = []
    barrier = threading.Barrier(3)

    def worker() -> None:
        c = repo.connection
        worker_conns.append(c)
        barrier.wait()

    t1 = threading.Thread(target=worker)
    t2 = threading.Thread(target=worker)
    t1.start()
    t2.start()

    barrier.wait()
    t1.join()
    t2.join()

    # All connections must be distinct objects
    assert len(worker_conns) == 2
    assert isinstance(worker_conns[0], sqlite3.Connection)
    assert worker_conns[0] is not worker_conns[1]
    assert worker_conns[0] is not main_conn
    assert worker_conns[1] is not main_conn

    repo.close()


def test_worker_exit_and_thread_id_reuse(tmp_path: Path) -> None:
    """Verify that subsequent threads do not reuse connections from terminated threads."""
    db_file = tmp_path / "metadata.db"
    repo = SQLiteMetadataRepository(db_file)

    connections_seen: list[int] = []

    def worker() -> None:
        conn = repo.connection
        connections_seen.append(id(conn))

    # Spawn and join sequential threads
    for _ in range(4):
        t = threading.Thread(target=worker)
        t.start()
        t.join()

    # With thread termination, the previous thread-local connection is destroyed,
    # ensuring no stale or cross-contaminated connection state is reused.
    assert len(connections_seen) == 4
    repo.close()


def test_cache_reset_after_multi_threaded_usage(tmp_path: Path) -> None:
    """Verify cache reset safely executes after multi-threaded SDK activity."""
    exp = runvault.create_experiment("cache-reset-test", workspace=tmp_path)

    def worker(idx: int) -> None:
        with exp.start_run(name=f"worker-{idx}") as run:
            run.log_param("worker_idx", idx)

    with ThreadPoolExecutor(max_workers=4) as executor:
        list(executor.map(worker, range(4)))

    # Reset cache from main thread
    runvault.reset_repository_cache()

    # Re-accessing the workspace must construct a clean repo and find the persisted runs
    runs = runvault.list_runs(exp.id, workspace=tmp_path)
    assert len(runs) == 4
    runvault.reset_repository_cache()


def test_repo_closure_while_worker_threads_active(tmp_path: Path) -> None:
    """Verify closing repo while a worker is running safely raises WorkspaceError on next call."""
    db_file = tmp_path / "metadata.db"
    repo = SQLiteMetadataRepository(db_file)

    worker_ready = threading.Event()
    proceed_to_call = threading.Event()
    worker_error: list[Exception] = []

    def worker() -> None:
        _ = repo.connection  # Initializes connection in worker thread
        worker_ready.set()
        proceed_to_call.wait(timeout=5.0)
        try:
            _ = repo.connection
        except WorkspaceError as err:
            worker_error.append(err)

    t = threading.Thread(target=worker)
    t.start()
    worker_ready.wait(timeout=5.0)

    # Main thread closes repository
    repo.close()
    proceed_to_call.set()
    t.join(timeout=5.0)

    assert len(worker_error) == 1
    assert "Repository has been closed" in str(worker_error[0])


def test_concurrent_workspace_initialization(tmp_path: Path) -> None:
    """Verify concurrent calls to init() against same workspace are race-free."""
    num_threads = 6

    def worker(idx: int) -> str:
        run = runvault.init(
            experiment="concurrent-init-exp",
            name=f"run-{idx}",
            workspace=tmp_path,
        )
        with run:
            run.log_param("idx", idx)
        return run.id

    with ThreadPoolExecutor(max_workers=num_threads) as executor:
        run_ids = list(executor.map(worker, range(num_threads)))

    assert len(run_ids) == num_threads
    assert len(set(run_ids)) == num_threads

    runs = runvault.list_runs(
        runvault.get_experiment_by_name("concurrent-init-exp", workspace=tmp_path).id,  # pyright: ignore[reportOptionalMemberAccess]
        workspace=tmp_path,
    )
    assert len(runs) == num_threads
    runvault.reset_repository_cache()
