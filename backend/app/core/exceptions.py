
class OptiRAGError(Exception):
    """Base exception for expected application errors."""

    status_code = 500
    public_message = "An unexpected OptiRAG error occurred."


class RetrievalError(OptiRAGError):
    """Raised when document retrieval fails."""

    status_code = 503
    public_message = "Document retrieval is temporarily unavailable."


class GenerationError(OptiRAGError):
    """Raised when answer generation fails."""

    status_code = 503
    public_message = "Answer generation is temporarily unavailable."


class ResourceUnavailableError(OptiRAGError):
    """Raised when a required resource is unavailable."""

    status_code = 503
    public_message = "A required OptiRAG resource is unavailable."
