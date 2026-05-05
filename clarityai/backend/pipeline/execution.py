from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor, Future


_EXECUTOR = ThreadPoolExecutor(max_workers=1)


def submit(fn, *args, **kwargs) -> Future:
    """Submit a callable to the single-worker thread pool.

    Returns a concurrent.futures.Future representing the execution.
    """
    return _EXECUTOR.submit(fn, *args, **kwargs)


def shutdown(wait: bool = True) -> None:
    """Shutdown the executor gracefully."""
    _EXECUTOR.shutdown(wait=wait)
