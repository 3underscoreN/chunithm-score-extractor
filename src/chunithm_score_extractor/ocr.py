from dataclasses import dataclass
import json
import logging
import re
from typing import Protocol, Sequence
import warnings

from typing import Literal

import numpy as np

from .errors import ModelInitializationError, ProcessingError


@dataclass(frozen=True, slots=True)
class TextReading:
    text: str
    confidence: float
    left: float
    top: float
    right: float
    bottom: float


class OcrEngine(Protocol):
    def read(self, image: np.ndarray) -> Sequence[TextReading]: ...


class PaddleOcrEngine:
    """
    A PaddleOCR-based implementation of the OcrEngine protocol.
    """
    def __init__(self, *, size: Literal["tiny", "small", "medium"] = "small", device: str | None = None) -> None:
        # simply validating the device string format
        if device is not None and not re.fullmatch(r"cpu|gpu(?::\d+)?", device):
            raise ValueError("device must be 'cpu', 'gpu', or 'gpu:<index>'.")

        # this is a filter to suppress warnings and info messages from PaddleOCR startup
        class StartupNoiseFilter(logging.Filter):
            def filter(self, record: logging.LogRecord) -> bool:
                message = record.getMessage()
                if record.levelno == logging.WARNING and (
                    message.startswith("Checking connectivity to the model hosters")
                    or message.startswith("Connectivity check to the model hoster has been skipped")
                ):
                    return False
                if record.levelno != logging.INFO:
                    return True
                return not (
                    message.startswith("Creating model:")
                    or message.startswith("Model files already exist.")
                    or message.startswith("Using official model (")
                )

        # configure the PaddleOCR logger to use the noise filter
        paddle_logger = logging.getLogger("paddlex")
        noise_filter = StartupNoiseFilter()
        paddle_logger.addFilter(noise_filter)
        try:
            with warnings.catch_warnings():
                # suppress specific warnings from PaddleOCR compilation
                warnings.filterwarnings(
                    "ignore",
                    message=r"No ccache found\. Please be aware that recompiling all source files may be required\.",
                    category=UserWarning,
                    module=r"paddle\.utils\.cpp_extension\.extension_utils",
                )
                from paddleocr import PaddleOCR

                if device is not None and device.startswith("gpu"):
                    # check if PaddlePaddle is compiled with CUDA for GPU usage,
                    # and if the numbered GPU device index is valid
                    import paddle

                    if not paddle.is_compiled_with_cuda(): # pyright: ignore[reportPrivateImportUsage]
                        raise ModelInitializationError(
                            "GPU requested, but PaddlePaddle was not built with CUDA. "
                            "Use a Python environment with paddlepaddle-gpu installed; "
                            "see the NVIDIA GPU setup in README.md."
                        )
                    index = int(device.split(":", 1)[1]) if ":" in device else 0
                    if index >= paddle.device.cuda.device_count():
                        raise ModelInitializationError(
                            f"GPU device {index} is not available to PaddlePaddle."
                        )

                # underlying OCR object initialization
                self._ocr = PaddleOCR(
                    text_detection_model_name=f"PP-OCRv6_{size}_det",
                    text_recognition_model_name=f"PP-OCRv6_{size}_rec",
                    use_doc_orientation_classify=False,
                    use_doc_unwarping=False,
                    use_textline_orientation=False,
                    enable_mkldnn=False,
                    device=device,
                )
        except ModelInitializationError:
            raise
        except ImportError as exc:
            raise ModelInitializationError(
                "Install the OCR dependencies with the 'ocr' extra."
            ) from exc
        except Exception as exc:
            raise ModelInitializationError("PaddleOCR could not initialize its models.") from exc
        finally:
            paddle_logger.removeFilter(noise_filter)

    def read(self, image: np.ndarray) -> Sequence[TextReading]:
        """
        Reads a sequence of text readings from the given image.

        Parameters
        ----------
        image : np.ndarray
            The image to perform OCR on.

        Returns
        -------
        Sequence[TextReading]
            A sequence of recognized text readings with their bounding boxes and confidence scores.

        Raises
        ------
        ProcessingError
            If there is an error during OCR inference.
        """
        try:
            outputs = self._ocr.predict(image)
            readings = []
            for output in outputs:
                readings.extend(_parse_result(_result_mapping(output)))
            return tuple(readings)
        except ProcessingError:
            raise
        except Exception as exc:
            raise ProcessingError("OCR inference failed.") from exc


def _result_mapping(output: object) -> dict:
    """
    Maps the output from PaddleOCR to a dictionary format.

    Parameters
    ----------
    output : object
        The raw output from PaddleOCR.

    Returns
    -------
    dict
        The mapped result as a dictionary.

    Raises
    ------
    ProcessingError
        If the output format is unsupported.
    """
    if isinstance(output, dict):
        return output
    
    if hasattr(output, "json"):
        serialized = output.json # pyright: ignore[reportAttributeAccessIssue]
        if callable(serialized):
            serialized = serialized()
        if isinstance(serialized, str):
            serialized = json.loads(serialized)
        if isinstance(serialized, dict):
            payload = serialized.get("res", serialized)
            if isinstance(payload, dict):
                return payload

    if hasattr(output, "to_dict"):
        value = output.to_dict() # pyright: ignore[reportAttributeAccessIssue]
        if isinstance(value, dict):
            payload = value.get("res", value)
            if isinstance(payload, dict):
                return payload
            
    raise ProcessingError("PaddleOCR returned an unsupported result.")


def _parse_result(output: dict) -> tuple[TextReading, ...]:
    """
    Parses the mapped result from PaddleOCR into a sequence of text readings.

    Parameters
    ----------
    output : dict
        The mapped result from PaddleOCR.

    Returns
    -------
    tuple[TextReading, ...]
        A tuple of recognized text readings with their bounding boxes and confidence scores.

    Raises
    ------
    ProcessingError
        If the recognition data is inconsistent or invalid.
    """
    texts = output.get("rec_texts", ())
    scores = output.get("rec_scores", ())
    boxes = output.get("rec_polys", output.get("rec_boxes", ()))

    if not (len(texts) == len(scores) == len(boxes)):
        raise ProcessingError("PaddleOCR returned inconsistent recognition data.")

    readings = []
    for text, confidence, points in zip(texts, scores, boxes, strict=True):
        # coords
        coordinates = np.asarray(points, dtype=float)
        if coordinates.shape == (4,):
            left, top, right, bottom = coordinates
        elif coordinates.ndim == 2 and coordinates.shape[1] == 2 and coordinates.size:
            left, top = coordinates.min(axis=0)
            right, bottom = coordinates.max(axis=0)
        else:
            raise ProcessingError("PaddleOCR returned an invalid text bounding box.")

        # scores
        score = float(confidence)
        if not np.isfinite(score) or not 0 <= score <= 1:
            raise ProcessingError("PaddleOCR returned an invalid recognition confidence.")
        
        readings.append(
            TextReading(
                text=str(text),
                confidence=score,
                left=float(left),
                top=float(top),
                right=float(right),
                bottom=float(bottom),
            )
        )
    return tuple(readings)
