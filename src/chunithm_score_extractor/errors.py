class ExtractionError(Exception):
    """Base class for expected input and extraction failures."""

    code = "extraction_error"


class InvalidImageError(ExtractionError):
    code = "invalid_image"


class UnsupportedLayoutError(ExtractionError):
    code = "unsupported_layout"


class UncertainResultError(ExtractionError):
    code = "uncertain_result"


class ConflictingReadingsError(ExtractionError):
    code = "conflicting_readings"


class ModelInitializationError(ExtractionError):
    code = "model_initialization_failed"


class ProcessingError(ExtractionError):
    code = "processing_failed"
