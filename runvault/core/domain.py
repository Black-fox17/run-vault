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


CURRENT_CHECKPOINT_FORMAT_VERSION: Final[int] = 1
SHA256_HEX_LENGTH: Final[int] = 64


def _validate_identifier(value: str, field_name: str) -> None:
    if not value or not value.strip():
        raise ValueError(f"Checkpoint {field_name} must be a non-empty string")
    if "/" in value or "\\" in value or ".." in value:
        raise ValueError(
            f"Checkpoint {field_name} '{value}' cannot contain path separators or traversal"
        )


@dataclass(frozen=True)
class Checkpoint:
    """An immutable record of a persisted training checkpoint artifact."""

    id: str
    run_id: str
    name: str
    size_bytes: int
    checksum_sha256: str
    artifact_path: str = ""
    step: int | None = None
    format_version: int = CURRENT_CHECKPOINT_FORMAT_VERSION
    metadata: dict[str, Any] = field(default_factory=dict)
    created_at: datetime = field(default_factory=utc_now)
    commit_seq: int | None = None
    storage_provider: str = "local"
    storage_key: str = ""

    def __post_init__(self) -> None:
        key = self.storage_key or self.artifact_path
        if not key or not key.strip():
            raise ValueError("Checkpoint storage_key or artifact_path must be a non-empty string")
        object.__setattr__(self, "storage_key", key)
        object.__setattr__(self, "artifact_path", key)

        _validate_identifier(self.id, "id")
        _validate_identifier(self.run_id, "run_id")
        if not self.name or not self.name.strip():
            raise ValueError("Checkpoint name must be a non-empty string")
        if "/" in self.name or "\\" in self.name or ".." in self.name:
            raise ValueError(
                f"Checkpoint name '{self.name}' must be a label, not a filesystem path"
            )
        if ".." in key:
            raise ValueError(f"Checkpoint artifact_path '{key}' cannot contain traversal")
        if not self.storage_provider or not self.storage_provider.strip():
            raise ValueError("Checkpoint storage_provider must be a non-empty string")
        if "/" in self.storage_provider or "\\" in self.storage_provider:
            raise ValueError(
                f"Checkpoint storage_provider '{self.storage_provider}' "
                "cannot contain path separators"
            )
        if self.size_bytes < 0:
            raise ValueError(f"Checkpoint size_bytes must be >= 0, got {self.size_bytes}")
        if not self.checksum_sha256 or len(self.checksum_sha256) != SHA256_HEX_LENGTH:
            raise ValueError("Checkpoint checksum_sha256 must be a 64-character hex string")
        if self.step is not None and (
            type(self.step) is bool or type(self.step) is not int or self.step < 0
        ):
            raise ValueError(f"Checkpoint step must be a non-negative integer, got {self.step}")
        if self.commit_seq is not None and (
            type(self.commit_seq) is bool or type(self.commit_seq) is not int or self.commit_seq < 1
        ):
            raise ValueError(
                f"Checkpoint commit_seq must be a positive integer, got {self.commit_seq}"
            )
        if self.format_version < 1:
            raise ValueError(f"Checkpoint format_version must be >= 1, got {self.format_version}")
        validate_json_serializable(self.metadata, f"checkpoint-{self.id}-metadata")

    def serialize_metadata(self) -> str:
        """Serialize checkpoint metadata to JSON string."""
        return json.dumps(self.metadata, allow_nan=False, sort_keys=True)

    @classmethod
    def deserialize_metadata(cls, raw_json: str | None) -> dict[str, Any]:
        """Deserialize checkpoint metadata from JSON string."""
        if not raw_json:
            return {}
        res = json.loads(raw_json)
        if isinstance(res, dict):
            return res
        return {}
