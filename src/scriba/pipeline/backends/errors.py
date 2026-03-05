"""Backend and model-client error types."""


class BackendError(RuntimeError):
    """Raised when backend startup or lifecycle checks fail."""


class ModelClientError(RuntimeError):
    """Raised when a model completion request fails."""


class RateLimitError(ModelClientError):
    """Raised when provider rejects requests due to rate limits."""

    def __init__(self, message: str, *, retry_after_s: float | None = None) -> None:
        super().__init__(message)
        self.retry_after_s = retry_after_s


class ContextWindowError(ModelClientError):
    """Raised when backend rejects request due to context length."""


class ModelRequestTimeoutError(ModelClientError):
    """Raised when backend request exceeds timeout."""
