"""End-to-end SDK tests for RunVault public API and run lifecycle."""

from pathlib import Path

import pytest

import runvault
from runvault import api
from runvault.core.domain import RunStatus
from runvault.core.errors import (
    ConflictingParameterError,
    InvalidLifecycleTransitionError,
    TerminalRunError,
)


def test_public_sdk_surface() -> None:
    """Ensure expected public functions and classes are directly exported."""
    expected_exports = [
        "init",
        "create_experiment",
        "get_experiment",
        "get_experiment_by_name",
        "list_experiments",
        "get_run",
        "list_runs",
        "get_metric_history",
        "get_params",
        "get_tags",
        "reset_repository_cache",
        "ActiveRun",
        "ExperimentHandle",
        "Experiment",
        "Run",
        "RunStatus",
        "Metric",
        "Parameter",
        "Tag",
        "Workspace",
        "Checkpoint",
        "CURRENT_CHECKPOINT_FORMAT_VERSION",
        "save_checkpoint",
        "get_checkpoint",
        "list_checkpoints",
        "get_latest_checkpoint",
        "get_latest_valid_checkpoint",
        "restore_checkpoint",
        "RunVaultError",
        "CheckpointError",
        "CheckpointCreationError",
        "CheckpointNotFoundError",
        "CheckpointCorruptError",
        "CorruptCheckpointError",
        "UnsupportedCheckpointFormatError",
        "CheckpointRecoveryError",
        "CheckpointRecoveryResult",
        "InvalidParameterError",
        "InvalidLifecycleTransitionError",
        "TerminalRunError",
    ]
    for name in expected_exports:
        assert hasattr(runvault, name), f"Missing public export: {name}"


def test_experiment_creation_and_run_lifecycle_success(tmp_path: Path) -> None:
    """Test standard successful experiment and run context execution."""
    exp = runvault.create_experiment(
        "mnist-classifier",
        description="Convolutional net baseline",
        workspace=tmp_path,
    )
    assert exp.id is not None
    assert exp.name == "mnist-classifier"

    # Execute a run
    with exp.start_run(name="trial-1", tags={"author": "alice"}) as run:
        assert run.status == RunStatus.RUNNING

        # Log parameters
        run.log_params({"lr": 0.001, "batch_size": 64})
        run.log_param("optimizer", "adam")

        # Log metrics across steps
        for step in range(3):
            run.log_metrics({"loss": 1.0 / (step + 1), "acc": 0.8 + 0.05 * step}, step=step)

        # Set tags
        run.set_tags({"phase": "train"})
        run.set_tag("precision", "fp32")

    # Outside the context block, run must be FINISHED
    assert run.status == RunStatus.FINISHED

    # Verify persisted data across clean retrieval
    persisted_run = runvault.get_run(run.id, workspace=tmp_path)
    assert persisted_run is not None
    assert persisted_run.status == RunStatus.FINISHED
    assert persisted_run.duration_seconds is not None
    assert persisted_run.duration_seconds >= 0.0

    # Verify parameters
    params = runvault.get_params(run.id, workspace=tmp_path)
    assert params == {"batch_size": 64, "lr": 0.001, "optimizer": "adam"}

    # Verify metrics
    loss_history = runvault.get_metric_history(run.id, metric_name="loss", workspace=tmp_path)
    assert len(loss_history) == 3
    assert [m.step for m in loss_history] == [0, 1, 2]

    # Verify tags
    tags = runvault.get_tags("run", run.id, workspace=tmp_path)
    assert tags == {"author": "alice", "phase": "train", "precision": "fp32"}


def test_run_context_exception_records_failure_and_reraises(tmp_path: Path) -> None:
    """An uncaught exception inside the context marks the run FAILED and records traceback."""
    exp = runvault.create_experiment("failure-test", workspace=tmp_path)

    with (
        pytest.raises(ZeroDivisionError, match="division by zero"),
        exp.start_run(name="faulty-run") as run,
    ):
        run.log_param("initial_step", 0)
        _ = 1 / 0

    assert run.status == RunStatus.FAILED

    persisted = runvault.get_run(run.id, workspace=tmp_path)
    assert persisted is not None
    assert persisted.status == RunStatus.FAILED
    assert persisted.error_type == "ZeroDivisionError"
    assert persisted.error_message == "division by zero"
    assert persisted.error_traceback is not None
    assert "ZeroDivisionError" in persisted.error_traceback


def test_explicit_run_termination(tmp_path: Path) -> None:
    """Verify explicit run termination."""
    exp = runvault.create_experiment("term-test", workspace=tmp_path)

    run = exp.start_run(name="early-abort")
    run.log_param("epochs", 100)
    run.terminate(reason="Validation loss diverged")

    assert run.status == RunStatus.TERMINATED

    persisted = runvault.get_run(run.id, workspace=tmp_path)
    assert persisted is not None
    assert persisted.status == RunStatus.TERMINATED
    assert persisted.error_message == "Validation loss diverged"


def test_idempotent_close_and_terminal_write_rejection(tmp_path: Path) -> None:
    """Closing a run multiple times is safe; writing to a terminal run is rejected."""
    exp = runvault.create_experiment("idempotence-test", workspace=tmp_path)

    with exp.start_run(name="terminal-checks") as run:
        run.log_param("val", 10)

    # Already finished
    assert run.status == RunStatus.FINISHED

    # Calling finish again is a safe no-op
    run.finish()
    assert run.status == RunStatus.FINISHED

    # Attempting to log params after completion raises TerminalRunError
    with pytest.raises(TerminalRunError, match="terminal state"):
        run.log_param("val2", 20)

    # Attempting to log metrics after completion raises TerminalRunError
    with pytest.raises(TerminalRunError, match="terminal state"):
        run.log_metric("loss", 0.1)

    # Attempting to set tags after completion raises TerminalRunError
    with pytest.raises(TerminalRunError, match="terminal state"):
        run.set_tag("status", "done")


def test_parameter_conflict_enforcement(tmp_path: Path) -> None:
    """Overwriting an existing parameter with a conflicting value raises error."""
    exp = runvault.create_experiment("param-conflict-test", workspace=tmp_path)

    with exp.start_run() as run:
        run.log_param("learning_rate", 0.01)

        # Idempotent logging of the exact same value
        run.log_param("learning_rate", 0.01)

        # Conflicting logging raises ConflictingParameterError
        with pytest.raises(ConflictingParameterError):
            run.log_param("learning_rate", 0.05)


def test_experiment_and_run_listing(tmp_path: Path) -> None:
    """Verify experiment and run listing and filtering."""
    exp1 = runvault.create_experiment("exp-alpha", workspace=tmp_path)
    exp2 = runvault.create_experiment("exp-beta", workspace=tmp_path)

    run1 = exp1.start_run(name="alpha-1")
    run1.finish()
    run2 = exp1.start_run(name="alpha-2")
    run2.finish()

    run3 = exp2.start_run(name="beta-1")
    run3.finish()

    # List experiments
    experiments = runvault.list_experiments(workspace=tmp_path)
    assert len(experiments) == 2
    exp_names = {e.name for e in experiments}
    assert exp_names == {"exp-alpha", "exp-beta"}

    # List runs by experiment
    alpha_runs = exp1.list_runs()
    assert len(alpha_runs) == 2
    assert {r.name for r in alpha_runs} == {"alpha-1", "alpha-2"}

    beta_runs = runvault.list_runs(exp2.id, workspace=tmp_path)
    assert len(beta_runs) == 1
    assert beta_runs[0].name == "beta-1"


def test_run_initializing_state_before_context_entry(tmp_path: Path) -> None:
    """A run is persisted in INITIALIZING with no start_time until entered."""
    exp = runvault.create_experiment("init-state-exp", workspace=tmp_path)
    run = exp.start_run(name="delayed-start")

    # Persisted status before context entry must be INITIALIZING
    persisted_initial = runvault.get_run(run.id, workspace=tmp_path)
    assert persisted_initial is not None
    assert persisted_initial.status == RunStatus.INITIALIZING
    assert persisted_initial.start_time is None

    # Entering context transitions to RUNNING and sets start_time
    with run:
        assert run.status == RunStatus.RUNNING
        assert run.start_time is not None
        persisted_active = runvault.get_run(run.id, workspace=tmp_path)
        assert persisted_active is not None
        assert persisted_active.status == RunStatus.RUNNING
        assert persisted_active.start_time is not None

    # After exit, it is FINISHED
    assert run.status == RunStatus.FINISHED

    # Re-entering an already finished run raises TerminalRunError
    with pytest.raises(TerminalRunError, match="terminal state"), run:
        pass


def test_failure_metadata_is_bounded(tmp_path: Path) -> None:
    """Failure message and traceback lengths are bounded to prevent database bloat."""
    exp = runvault.create_experiment("bounded-error-exp", workspace=tmp_path)

    huge_message = "X" * 5000

    with (
        pytest.raises(RuntimeError),
        exp.start_run(name="deep-error-run") as run,
    ):
        raise RuntimeError(huge_message)

    persisted = runvault.get_run(run.id, workspace=tmp_path)
    assert persisted is not None
    assert persisted.status == RunStatus.FAILED
    assert persisted.error_message is not None
    assert len(persisted.error_message) <= 1024
    assert persisted.error_traceback is not None
    assert len(persisted.error_traceback) <= 4096


def test_workspace_isolation_and_cache_cleanup(tmp_path: Path) -> None:
    """Different workspace paths remain strictly isolated and can be cleanly closed."""
    ws1 = tmp_path / "workspace_alpha"
    ws2 = tmp_path / "workspace_beta"

    runvault.create_experiment("exp-in-ws1", workspace=ws1)
    runvault.create_experiment("exp-in-ws2", workspace=ws2)

    assert len(runvault.list_experiments(workspace=ws1)) == 1
    assert runvault.list_experiments(workspace=ws1)[0].name == "exp-in-ws1"

    assert len(runvault.list_experiments(workspace=ws2)) == 1
    assert runvault.list_experiments(workspace=ws2)[0].name == "exp-in-ws2"

    # Clean up repository cache
    runvault.reset_repository_cache()


def test_init_top_level_entrypoint(tmp_path: Path) -> None:
    """Verify runvault.init() creates experiments on the fly and starts runs."""
    run = runvault.init(
        experiment="vision-resnet",
        name="trial-init",
        tags={"dataset": "cifar10"},
        workspace=tmp_path,
    )
    with run:
        run.log_params({"epochs": 10, "lr": 0.001})
        run.log_metric("accuracy", 0.92, step=1)

    assert run.status == RunStatus.FINISHED

    exp = runvault.get_experiment_by_name("vision-resnet", workspace=tmp_path)
    assert exp is not None
    assert exp.name == "vision-resnet"

    persisted_run = runvault.get_run(run.id, workspace=tmp_path)
    assert persisted_run is not None
    assert persisted_run.status == RunStatus.FINISHED
    assert persisted_run.name == "trial-init"

    params = runvault.get_params(run.id, workspace=tmp_path)
    assert params == {"epochs": 10, "lr": 0.001}

    metrics = runvault.get_metric_history(run.id, workspace=tmp_path)
    assert len(metrics) == 1
    assert metrics[0].value == 0.92

    runvault.reset_repository_cache()


def test_double_context_entry_rejected(tmp_path: Path) -> None:
    """A run cannot be entered twice with a context manager."""
    run = runvault.init(experiment="double-entry-exp", workspace=tmp_path)
    with run, pytest.raises(InvalidLifecycleTransitionError), run:
        pass


def test_terminate_before_context_entry_does_not_activate(tmp_path: Path) -> None:
    """Terminating an unstarted run transitions directly to TERMINATED with no start_time."""
    run = runvault.init(experiment="abort-exp", name="unstarted-run", workspace=tmp_path)
    assert run.status == RunStatus.INITIALIZING

    run.terminate(reason="Cancelled by operator before start")
    assert run.status == RunStatus.TERMINATED
    assert run.start_time is None

    persisted = runvault.get_run(run.id, workspace=tmp_path)
    assert persisted is not None
    assert persisted.status == RunStatus.TERMINATED
    assert persisted.start_time is None
    assert persisted.duration_seconds == 0.0
    assert persisted.error_message == "Cancelled by operator before start"


def test_logging_to_terminal_run_rejected(tmp_path: Path) -> None:
    """Attempting to log parameters, metrics, or tags to a terminal run raises TerminalRunError."""
    run = runvault.init(experiment="term-log-exp", workspace=tmp_path)
    with run:
        run.log_param("step_0", True)

    assert run.status == RunStatus.FINISHED

    with pytest.raises(TerminalRunError, match="terminal state"):
        run.log_param("new_param", 123)

    with pytest.raises(TerminalRunError, match="terminal state"):
        run.log_params({"p2": 456})

    with pytest.raises(TerminalRunError, match="terminal state"):
        run.log_metric("loss", 0.1)

    with pytest.raises(TerminalRunError, match="terminal state"):
        run.log_metrics({"m2": 0.2})

    with pytest.raises(TerminalRunError, match="terminal state"):
        run.set_tag("tag1", "val1")

    with pytest.raises(TerminalRunError, match="terminal state"):
        run.set_tags({"t2": "v2"})


def test_workspace_discovery_hierarchy(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Test workspace resolution: explicit path > RUNVAULT_DIR > upward discovery > cwd default."""
    # 1. Database file path resolution to parent
    db_file = tmp_path / "custom" / "metadata.db"
    ws = runvault.Workspace.from_path(db_file)
    assert ws.path == (tmp_path / "custom").resolve()
    assert ws.db_path == db_file.resolve()

    # 2. RUNVAULT_DIR environment variable
    env_dir = tmp_path / "env_vault"
    monkeypatch.setenv("RUNVAULT_DIR", str(env_dir))
    resolved_env = runvault.Workspace.from_path(None)
    assert resolved_env.path == env_dir.resolve()
    monkeypatch.delenv("RUNVAULT_DIR", raising=False)

    # 3. Upward discovery
    project_root = tmp_path / "my_project"
    vault_dir = project_root / ".runvault"
    vault_dir.mkdir(parents=True)
    deep_subdir = project_root / "src" / "models" / "deep"
    deep_subdir.mkdir(parents=True)

    monkeypatch.chdir(deep_subdir)
    resolved_upward = runvault.Workspace.from_path(None)
    assert resolved_upward.path == vault_dir.resolve()


def test_checkpoint_error_hierarchy_and_compatibility_aliases() -> None:
    """Verify consistent exception exports, inheritance hierarchies, and compatibility aliases."""
    # Verify CorruptCheckpointError is an exact alias of CheckpointCorruptError
    assert runvault.CorruptCheckpointError is runvault.CheckpointCorruptError
    assert issubclass(runvault.CheckpointCorruptError, runvault.CheckpointError)
    assert issubclass(runvault.CheckpointCreationError, runvault.CheckpointError)
    assert issubclass(runvault.CheckpointNotFoundError, runvault.CheckpointError)
    assert issubclass(runvault.CheckpointRecoveryError, runvault.CheckpointError)
    assert issubclass(runvault.UnsupportedCheckpointFormatError, runvault.CheckpointError)
    assert issubclass(runvault.CheckpointError, runvault.RunVaultError)

    # Verify exports in runvault.api match
    assert api.CheckpointCorruptError is runvault.CheckpointCorruptError
    assert api.CorruptCheckpointError is runvault.CheckpointCorruptError
    assert api.CheckpointCreationError is runvault.CheckpointCreationError
    assert api.CheckpointNotFoundError is runvault.CheckpointNotFoundError
    assert api.CheckpointRecoveryError is runvault.CheckpointRecoveryError
    assert api.UnsupportedCheckpointFormatError is runvault.UnsupportedCheckpointFormatError
    assert api.CheckpointError is runvault.CheckpointError
