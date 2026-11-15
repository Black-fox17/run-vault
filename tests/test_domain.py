"""Unit tests for RunVault domain models and lifecycle rules."""

import pytest

from runvault.core.domain import (
    Experiment,
    Metric,
    Parameter,
    Run,
    RunStatus,
    Tag,
    check_parameter_conflict,
    validate_status_transition,
)
from runvault.core.errors import (
    ConflictingParameterError,
    InvalidLifecycleTransitionError,
    InvalidMetricError,
    InvalidParameterError,
)


def test_experiment_validation() -> None:
    """Experiment requires non-empty id and name."""
    exp = Experiment(id="exp-1", name="baseline")
    assert exp.id == "exp-1"
    assert exp.name == "baseline"
    assert exp.description is None
    assert exp.created_at is not None

    with pytest.raises(ValueError, match="Experiment id must be a non-empty string"):
        Experiment(id="", name="baseline")

    with pytest.raises(ValueError, match="Experiment name must be a non-empty string"):
        Experiment(id="exp-1", name="   ")


def test_run_validation() -> None:
    """Run requires non-empty id and experiment_id."""
    run = Run(id="run-1", experiment_id="exp-1")
    assert run.id == "run-1"
    assert run.experiment_id == "exp-1"
    assert run.status == RunStatus.INITIALIZING

    with pytest.raises(ValueError, match="Run id must be a non-empty string"):
        Run(id="", experiment_id="exp-1")

    with pytest.raises(ValueError, match="Run experiment_id must be a non-empty string"):
        Run(id="run-1", experiment_id="")


def test_lifecycle_transitions() -> None:
    """Verify valid transitions and enforce rejection of invalid/terminal transitions."""
    # Valid: INITIALIZING -> RUNNING
    validate_status_transition(RunStatus.INITIALIZING, RunStatus.RUNNING, "run-1")

    # Valid: RUNNING -> FINISHED / FAILED / TERMINATED
    validate_status_transition(RunStatus.RUNNING, RunStatus.FINISHED, "run-1")
    validate_status_transition(RunStatus.RUNNING, RunStatus.FAILED, "run-1")
    validate_status_transition(RunStatus.RUNNING, RunStatus.TERMINATED, "run-1")

    # Invalid: Terminal states cannot transition to anything
    for terminal in (RunStatus.FINISHED, RunStatus.FAILED, RunStatus.TERMINATED):
        assert terminal.is_terminal
        for target in RunStatus:
            with pytest.raises(InvalidLifecycleTransitionError) as exc_info:
                validate_status_transition(terminal, target, "run-1")
            assert exc_info.value.current_status == terminal.value
            assert exc_info.value.target_status == target.value

    # Invalid: INITIALIZING cannot jump directly to FINISHED
    with pytest.raises(InvalidLifecycleTransitionError):
        validate_status_transition(RunStatus.INITIALIZING, RunStatus.FINISHED, "run-1")


def test_parameter_serialization_and_conflict() -> None:
    """Parameters must be JSON serializable and handle conflicts."""
    param = Parameter(run_id="run-1", name="learning_rate", value=0.001)
    serialized = param.serialize_value()
    assert Parameter.deserialize_value(serialized) == 0.001

    # Structured values (dicts, lists) are supported
    nested_param = Parameter(
        run_id="run-1",
        name="model_cfg",
        value={"layers": [64, 32], "activation": "relu"},
    )
    assert Parameter.deserialize_value(nested_param.serialize_value()) == {
        "layers": [64, 32],
        "activation": "relu",
    }

    # Non-JSON serializable values are rejected
    with pytest.raises(ValueError, match="is not JSON-serializable"):
        Parameter(run_id="run-1", name="bad_param", value=object())

    # Same value is idempotent (no conflict)
    check_parameter_conflict("lr", 0.01, 0.01, "run-1")

    # Different value raises ConflictingParameterError
    with pytest.raises(ConflictingParameterError) as exc_info:
        check_parameter_conflict("lr", 0.01, 0.02, "run-1")
    assert exc_info.value.name == "lr"
    assert exc_info.value.existing_value == 0.01
    assert exc_info.value.new_value == 0.02


def test_metric_validation() -> None:
    """Metrics must reject non-finite values and negative steps."""
    m = Metric(run_id="run-1", name="loss", value=0.45, step=10)
    assert m.value == 0.45
    assert m.step == 10

    # Non-finite values rejected
    for bad_val in (float("nan"), float("inf"), float("-inf")):
        with pytest.raises(InvalidMetricError, match="finite number"):
            Metric(run_id="run-1", name="loss", value=bad_val)

    # Negative step rejected
    with pytest.raises(InvalidMetricError, match="non-negative integer"):
        Metric(run_id="run-1", name="loss", value=0.5, step=-1)

    # Empty name rejected
    with pytest.raises(InvalidMetricError, match="non-empty string"):
        Metric(run_id="run-1", name="   ", value=0.5)

    # Boolean values rejected
    with pytest.raises(InvalidMetricError, match="expected float or int, got bool"):
        Metric(run_id="run-1", name="loss", value=True)  # pyright: ignore[reportArgumentType]

    with pytest.raises(InvalidMetricError, match="step must be a non-negative integer"):
        Metric(run_id="run-1", name="loss", value=0.5, step=True)  # pyright: ignore[reportArgumentType]


def test_tag_validation() -> None:
    """Tags require non-empty keys and string values."""
    tag = Tag(entity_type="run", entity_id="run-1", key="env", value="prod")
    assert tag.key == "env"
    assert tag.value == "prod"

    with pytest.raises(ValueError, match="Tag key must be a non-empty string"):
        Tag(entity_type="run", entity_id="run-1", key="", value="prod")

    # Non-string value rejected
    with pytest.raises(ValueError, match="Tag value must be a string"):
        # We test runtime validation by passing a non-string object
        Tag(entity_type="run", entity_id="run-1", key="k", value=123)  # pyright: ignore[reportArgumentType]


def test_exhaustive_lifecycle_transition_table() -> None:
    """Exhaustively verify the 5x5 state transition matrix."""
    valid_pairs = {
        (RunStatus.INITIALIZING, RunStatus.RUNNING),
        (RunStatus.INITIALIZING, RunStatus.TERMINATED),
        (RunStatus.RUNNING, RunStatus.FINISHED),
        (RunStatus.RUNNING, RunStatus.FAILED),
        (RunStatus.RUNNING, RunStatus.TERMINATED),
    }

    all_statuses = list(RunStatus)
    for src in all_statuses:
        for dst in all_statuses:
            if (src, dst) in valid_pairs:
                validate_status_transition(src, dst, "run-matrix")
            else:
                with pytest.raises(InvalidLifecycleTransitionError) as exc_info:
                    validate_status_transition(src, dst, "run-matrix")
                assert exc_info.value.current_status == src.value
                assert exc_info.value.target_status == dst.value


def test_parameter_invalid_types_raise_invalid_parameter_error() -> None:
    """Non-serializable types and non-finite floats raise InvalidParameterError."""
    # Sets are not JSON serializable
    with pytest.raises(InvalidParameterError, match="is not JSON-serializable"):
        Parameter(run_id="run-1", name="p_set", value={1, 2, 3})

    def dummy_func() -> None:
        pass

    # Callables
    with pytest.raises(InvalidParameterError, match="is not JSON-serializable"):
        Parameter(run_id="run-1", name="p_fn", value=dummy_func)

    # NaN in parameter
    with pytest.raises(InvalidParameterError, match="is not JSON-serializable"):
        Parameter(run_id="run-1", name="p_nan", value=float("nan"))
