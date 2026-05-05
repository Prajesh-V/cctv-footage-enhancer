from typing import Any, Dict

PipelineOutput = object  # Either a numpy array or a filesystem path (string)


class PipelineResult(dict):
    """Typed-ish dict for pipeline results.

    Expected keys:
      - output: PipelineOutput (np.ndarray or path string)
      - metadata: dict with processing_time, roi_applied, errors, etc.
    """

    def __init__(self, output: PipelineOutput, metadata: Dict[str, Any] | None = None):
        super().__init__(output=output, metadata=metadata or {})
