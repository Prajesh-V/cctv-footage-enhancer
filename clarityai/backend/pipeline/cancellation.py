import threading


class CancellationToken:
    """A simple, thread-safe cancellation token used to stop long-running pipelines.

    Consumers should periodically check `token.cancelled` and abort processing when True.
    """

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._cancelled = False

    @property
    def cancelled(self) -> bool:
        with self._lock:
            return self._cancelled

    def cancel(self) -> None:
        with self._lock:
            self._cancelled = True
