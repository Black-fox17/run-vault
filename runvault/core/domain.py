"""Domain models and lifecycle rules for RunVault."""

import json
import math
from dataclasses import dataclass, field
from datetime import UTC, datetime
from enum import StrEnum
from typing import Any, Final

from runvault.core.errors import (
    ConflictingParameterError,
    InvalidLifecycleTransitionError,
    InvalidMetricError,
    InvalidParameterError,
)


class RunStatus(StrEnum):
    """Explicit lifecycle status for an experiment run."""

    INITIALIZING = "INITIALIZING"
    RUNNING = "RUNNING"
    FINISHED = "FINISHED"
    FAILED = "FAILED"
    TERMINATED = "TERMINATED"

    @property
    def is_terminal(self) -> bool:
        """Return True if the status represents a final terminal state."""
        return self in _TERMINAL_STATES


_TERMINAL_STATES: Final[frozenset[RunStatus]] = frozenset(
    {RunStatus.FINISHED, RunStatus.FAILED, RunStatus.TERMINATED}
)

_VALID_TRANSITIONS: Final[dict[RunStatus, frozenset[RunStatus]]] = {
    RunStatus.INITIALIZING: frozenset({RunStatus.RUNNING, RunStatus.TERMINATED}),
    RunStatus.RUNNING: frozenset({RunStatus.FINISHED, RunStatus.FAILED, RunStatus.TERMINATED}),
    RunStatus.FINISHED: frozenset(),
    RunStatus.FAILED: frozenset(),
    RunStatus.TERMINATED: frozenset(),
}


def validate_status_transition(current: RunStatus, target: RunStatus, run_id: str) -> None:
    """Enforce valid run lifecycle transitions.

    Raises:
        InvalidLifecycleTransitionError: If the transition is not allowed.
    """
    allowed = _VALID_TRANSITIONS.get(current, frozenset())
    if target not in allowed:
        raise InvalidLifecycleTransitionError(
            current_status=current.value,
            target_status=target.value,
            run_id=run_id,
        )


def utc_now() -> datetime:
    """Return the current timezone-aware UTC datetime."""
    return datetime.now(UTC)


@dataclass(frozen=True)
class Experiment:
    """An experiment grouping related training and evaluation runs."""

    id: str
    name: str
    description: str | None = None
    created_at: datetime = field(default_factory=utc_now)
    updated_at: datetime = field(default_factory=utc_now)

    def __post_init__(self) -> None:
        if not self.id or not self.id.strip():
            raise ValueError("Experiment id must be a non-empty string")
        if not self.name or not self.name.strip():
            raise ValueError("Experiment name must be a non-empty string")


@dataclass(frozen=True)
class Run:
    """A single execution run of an experiment."""

    id: str
    experiment_id: str
    name: str | None = None
    status: RunStatus = RunStatus.INITIALIZING
    start_time: datetime | None = None
    end_time: datetime | None = None
    duration_seconds: float | None = None
    error_message: str | None = None
    error_type: str | None = None
    error_traceback: str | None = None
    created_at: datetime = field(default_factory=utc_now)

    def __post_init__(self) -> None:
        if not self.id or not self.id.strip():
            raise ValueError("Run id must be a non-empty string")
        if not self.experiment_id or not self.experiment_id.strip():
            raise ValueError("Run experiment_id must be a non-empty string")


def validate_json_serializable(value: Any, param_name: str) -> None:
    """Validate that a parameter value is JSON-serializable."""
    try:
        json.dumps(value, allow_nan=False)
    except (TypeError, ValueError) as err:
        msg = (
            f"Parameter '{param_name}' value of type '{type(value).__name__}' "
            f"is not JSON-serializable: {err}"
        )
        raise InvalidParameterError(param_name, value, msg) from err


@dataclass(frozen=True)
class Parameter:
    """A configuration parameter logged for a run."""

    run_id: str
    name: str
    value: Any

    def __post_init__(self) -> None:
        if not self.name or not self.name.strip():
            raise ValueError("Parameter name must be a non-empty string")
        validate_json_serializable(self.value, self.name)

    def serialize_value(self) -> str:
        """Serialize parameter value to standard JSON string."""
        return json.dumps(self.value, allow_nan=False, sort_keys=True)

    @classmethod
    def deserialize_value(cls, raw_json: str) -> Any:
        """Deserialize parameter value from standard JSON string."""
        return json.loads(raw_json)


def check_parameter_conflict(
    param_name: str,
    existing_value: Any,
    new_value: Any,
    run_id: str,
) -> None:
    """Check for parameter conflict. Logging the exact same value is idempotent.

    Raises:
        ConflictingParameterError: If new_value does not equal existing_value.
    """
    if existing_value != new_value:
        raise ConflictingParameterError(
            name=param_name,
            existing_value=existing_value,
            new_value=new_value,
            run_id=run_id,
        )


@dataclass(frozen=True)
class Metric:
    """A scalar numeric measurement recorded during run execution."""

    run_id: str
    name: str
    value: float
    step: int | None = None
    timestamp: datetime = field(default_factory=utc_now)

    def __post_init__(self) -> None:
        if not self.name or not self.name.strip():
            msg = "metric name must be a non-empty string"
            raise InvalidMetricError(self.name, self.value, msg)
        if type(self.value) is bool or type(self.value) not in (int, float):
            raise InvalidMetricError(
                self.name, self.value, f"expected float or int, got {type(self.value).__name__}"
            )
        float_val = float(self.value)
        if not math.isfinite(float_val):
            msg = "metric value must be a finite number (NaN, +inf, -inf rejected)"
            raise InvalidMetricError(self.name, self.value, msg)
        if self.step is not None and (
            type(self.step) is bool or type(self.step) is not int or self.step < 0
        ):
            msg = f"step must be a non-negative integer, got {self.step}"
            raise InvalidMetricError(self.name, self.value, msg)


@dataclass(frozen=True)
class Tag:
    """A string key-value tag associated with an experiment or run."""

    entity_type: str
    entity_id: str
    key: str
    value: str

    def __post_init__(self) -> None:
        if not self.key or not self.key.strip():
            raise ValueError("Tag key must be a non-empty string")
        if type(self.value) is not str:
            raise ValueError(f"Tag value must be a string, got {type(self.value).__name__}")
