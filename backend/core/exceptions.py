class ChurnAnalysisError(Exception):
    """Base exception — every custom error inherits from this."""
    pass


class MLPredictionError(ChurnAnalysisError):
    """Raised when the ML model cannot produce a prediction."""
    pass


class LLMServiceError(ChurnAnalysisError):
    """Raised when the LLM service is unavailable or returns garbage."""
    pass


class ValidationError(ChurnAnalysisError):
    """Raised when incoming data fails business-rule validation."""
    pass


class DataStorageError(ChurnAnalysisError):
    """Raised when history can't be read from or written to disk."""
    pass
