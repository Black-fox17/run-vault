"""Domain exceptions for RunVault."""


class RunVaultError(Exception):
    """Base exception for all RunVault errors."""


class ExperimentNotFoundError(RunVaultError):
    """Raised when an experiment with the specified ID or name is not found."""

    def __init__(self, identifier: str) -> None:
        super().__init__(f"Experiment not found: '{identifier}'")
        self.identifier = identifier


class RunNotFoundError(RunVaultError):
    """Raised when a run with the specified ID is not found."""

    def __init__(self, run_id: str) -> None:
        super().__init__(f"Run not found: '{run_id}'")
        self.run_id = run_id


class InvalidLifecycleTransitionError(RunVaultError):
    """Raised when an illegal run lifecycle state transition is attempted."""

    def __init__(self, current_status: str, target_status: str, run_id: str) -> None:
        super().__init__(
            f"Invalid lifecycle transition for run '{run_id}': "
            f"cannot transition from {current_status} to {target_status}"
        )
        self.current_status = current_status
        self.target_status = target_status
        self.run_id = run_id


class TerminalRunError(RunVaultError):
    """Raised when attempting to log metrics, parameters, or tags to an ended run."""

    def __init__(self, run_id: str, status: str, operation: str) -> None:
        msg = (
            f"Cannot perform '{operation}' on run '{run_id}' "
            f"because it is in terminal state '{status}'"
        )
        super().__init__(msg)
        self.run_id = run_id
        self.status = status
        self.operation = operation


class ConflictingParameterError(RunVaultError):
    """Raised when logging a parameter with a different value than previously recorded."""

    def __init__(
        self,
        name: str,
        existing_value: object,
        new_value: object,
        run_id: str,
    ) -> None:
        super().__init__(
            f"Parameter '{name}' already logged for run '{run_id}' with value {existing_value!r}; "
            f"conflicting new value: {new_value!r}"
        )
        self.name = name
        self.existing_value = existing_value
        self.new_value = new_value
        self.run_id = run_id


class InvalidParameterError(RunVaultError, ValueError):
    """Raised when an invalid parameter name or unsupported value is supplied."""

    def __init__(self, name: str, value: object, reason: str) -> None:
        super().__init__(f"Invalid parameter '{name}' with value {value!r}: {reason}")
        self.name = name
        self.value = value
        self.reason = reason


class InvalidMetricError(RunVaultError):
    """Raised when an invalid metric value or step is supplied."""

    def __init__(self, name: str, value: object, reason: str) -> None:
        super().__init__(f"Invalid metric '{name}' with value {value!r}: {reason}")
        self.name = name
        self.value = value
        self.reason = reason


class WorkspaceError(RunVaultError):
    """Raised when a local workspace cannot be initialized, accessed, or validated."""

    def __init__(self, path: str, reason: str, original_error: Exception | None = None) -> None:
        super().__init__(f"Workspace error at '{path}': {reason}")
        self.path = path
        self.reason = reason
        self.original_error = original_error
