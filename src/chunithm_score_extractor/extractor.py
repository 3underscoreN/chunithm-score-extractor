import re
from collections.abc import Sequence

from typing import Literal

import cv2
import numpy as np

from .errors import (
    ConflictingReadingsError,
    ExtractionError,
    ProcessingError,
    UncertainResultError,
    UnsupportedLayoutError,
)
from .image import ImageLimits, decode_image
from .layout import SCREEN_HEIGHT, SCREEN_WIDTH, rectify_screen
from .models import Difficulty, ExtractionResult
from .ocr import OcrEngine, PaddleOcrEngine, TextReading


CONFIDENCE_THRESHOLD = 0.75
REGION_OVERLAP_THRESHOLD = 0.80
_DIFFICULTY_NAMES = {
    "BASIC": Difficulty.BAS,
    "ADVANCED": Difficulty.ADV,
    "EXPERT": Difficulty.EXP,
    "MASTER": Difficulty.MAS,
    "ULTIMA": Difficulty.ULT,
}
_SCORE_PATTERN = re.compile(r"(?<!\d)(?:\d[\d,.\s]{4,}\d)(?!\d)")
_WORLD_END_PATTERN = re.compile(r"WORLD[\s'’]*S?\s*END|WORLDEND", re.IGNORECASE)


class ChunithmExtractor:
    """
    The ChunithmExtractor class.
    """
    def __init__(
        self,
        ocr: OcrEngine | None = None,
        image_limits: ImageLimits = ImageLimits(),
        confidence_threshold: float = CONFIDENCE_THRESHOLD,
        *,
        size: Literal["tiny", "small", "medium"] = "small",
        device: str | None = None,
    ) -> None:
        """
        Initialize a ChunithmExtractor.

        Parameters
        ----------
        ocr : OcrEngine | None, optional
            The OCR engine to use. If None, a default PaddleOcrEngine is used.
        image_limits : ImageLimits, optional
            The limits for decoding the image.
        confidence_threshold : float, optional
            The confidence threshold for accepting OCR readings.
        size : Literal["tiny", "small", "medium"], optional
            The size of the OCR model to use if a custom engine is not provided.
        device : str | None, optional
            The device to use for the OCR engine if a custom engine is not provided.
        """
        if not 0 <= confidence_threshold <= 1:
            raise ValueError("confidence_threshold must be between 0 and 1.")
        if ocr is not None and device is not None:
            raise ValueError("device cannot be specified with a custom OCR engine.")
        self._ocr = ocr if ocr is not None else PaddleOcrEngine(size=size, device=device)
        self._image_limits = image_limits
        self._confidence_threshold = confidence_threshold

    def extract(self, image_bytes: bytes) -> ExtractionResult:
        """
        Extract the score and title from the given image bytes.

        Parameters
        ----------
        image_bytes : bytes
            The image data to extract information from.

        Returns
        -------
        ExtractionResult
            The result of the extraction, including the score and title.

        Raises
        ------
        ExtractionError
            If there is an error during the extraction process.
        ProcessingError
            If there is an error during image processing.
        UncertainResultError
            If a unique difficulty label cannot be located.
        """
        # decode the image from bytes
        image = decode_image(image_bytes, self._image_limits)

        # rectify the screen to a standard layout
        try:
            screen = rectify_screen(image)
        except ExtractionError:
            raise
        except Exception as exc:
            raise ProcessingError("Display localization failed.") from exc

        # First pass: perform OCR on the entire screen to identify potential anchors and readings
        readings = _read_ocr(self._ocr, screen)
        # Get possible anchors
        score_anchor, rank_anchor = _layout_anchors(readings, self._confidence_threshold)

        # Get the difficulty lines within the specified region
        difficulty_lines = _in_region(readings, (0.65, 0.025, 1.0, 0.15))
        # Ensure only one difficulty exists in the region
        difficulty = _extract_difficulty(difficulty_lines, self._confidence_threshold)
        # Ensure only one difficulty anchor is identified
        difficulty_anchors = [
            line for line in difficulty_lines
            if line.confidence >= self._confidence_threshold
            and _difficulty_in_text(line.text) == {difficulty}
        ]
        if len(difficulty_anchors) != 1:
            raise UncertainResultError("Could not locate a unique difficulty label.")
        # gets the difficulty to use
        difficulty_anchor = difficulty_anchors[0]

        # get the field regions for score and title based on the identified anchors
        score_region, title_region = _field_regions(
            score_anchor, rank_anchor, difficulty_anchor
        )

        # extracting score
        header_lines = _in_region(readings, (0.35, 0.0, 0.72, 0.10))
        first_score_lines = _overlapping_region(readings, score_region)
        try:
            # hacky code to check if the first score lines contain a valid score reading
            _extract_score(first_score_lines, self._confidence_threshold)

            # if no exception is raised, the first score lines are valid
            score = _current_score(first_score_lines, header_lines, self._confidence_threshold)

            if any(
                line.confidence < self._confidence_threshold
                and _SCORE_PATTERN.search(line.text)
                for line in first_score_lines
            ): # if any score line is below the confidence threshold and contains a score pattern
                raise UncertainResultError("A score reading is below the confidence threshold.")
        except (UncertainResultError, ConflictingReadingsError) as exc:
            score_lines = _in_region(_read_region(self._ocr, screen, score_region), score_region)
            if isinstance(exc, ConflictingReadingsError):
                # Do not resolve a known disagreement using only the original header
                _extract_score(score_lines, self._confidence_threshold)

            score = _current_score(score_lines, header_lines, self._confidence_threshold)

        # title extraction
        first_title_lines = _title_candidates(
            _overlapping_region(readings, title_region),
            readings, difficulty_anchor, self._confidence_threshold,
        )
        try:
            song_name = _extract_title(first_title_lines, self._confidence_threshold)
        except (UncertainResultError, ConflictingReadingsError):
            # if title extraction results in uncertainty or conflict, crop the title region and retry
            title_lines = _read_region(self._ocr, screen, title_region)
            song_name = _extract_title(
                _title_candidates(
                    _in_region(title_lines, title_region),
                    readings, difficulty_anchor, self._confidence_threshold,
                ),
                self._confidence_threshold,
            )
        return ExtractionResult(score=score, song_name=song_name, difficulty=difficulty)


def _field_regions(
    score_anchor: TextReading,
    rank_anchor: TextReading,
    difficulty_anchor: TextReading,
) -> tuple[tuple[float, float, float, float], tuple[float, float, float, float]]:
    """
    Define the regions for score and title fields based on the identified anchors.
    Returns a tuple containing the score region and the title region.
    """
    score_region = (
        max(0.0, score_anchor.left / SCREEN_WIDTH - 0.02),
        min(1.0, score_anchor.bottom / SCREEN_HEIGHT + 0.025),
        min(1.0, rank_anchor.left / SCREEN_WIDTH - 0.005),
        min(1.0, score_anchor.bottom / SCREEN_HEIGHT + 0.19),
    )
    title_region = (
        max(0.0, difficulty_anchor.left / SCREEN_WIDTH - 0.02),
        min(1.0, difficulty_anchor.bottom / SCREEN_HEIGHT + 0.002),
        0.993,
        min(1.0, difficulty_anchor.bottom / SCREEN_HEIGHT + 0.05),
    )
    return score_region, title_region


def _overlapping_region(
    readings: Sequence[TextReading],
    region: tuple[float, float, float, float],
) -> tuple[TextReading, ...]:
    """
    Returns the readings that overlap with the specified region.

    Parameters
    ----------
    readings : Sequence[TextReading]
        The OCR readings from the screen.
    region : tuple[float, float, float, float]
        The region to check for overlap, specified as (left, top, right, bottom) in normalized coordinates.

    Returns
    -------
    tuple[TextReading, ...]
        The readings that overlap with the specified region.
    """
    left, top, right, bottom = region
    left, right = left * SCREEN_WIDTH, right * SCREEN_WIDTH
    top, bottom = top * SCREEN_HEIGHT, bottom * SCREEN_HEIGHT
    selected = []
    for line in readings:
        width, height = line.right - line.left, line.bottom - line.top
        if width <= 0 or height <= 0:
            continue
        intersection = max(0.0, min(line.right, right) - max(line.left, left)) * max(
            0.0, min(line.bottom, bottom) - max(line.top, top)
        )
        if intersection >= REGION_OVERLAP_THRESHOLD * width * height:
            selected.append(line)
    return tuple(selected)


def _title_candidates(
    lines: Sequence[TextReading],
    screen_readings: Sequence[TextReading],
    difficulty_anchor: TextReading,
    threshold: float,
) -> tuple[TextReading, ...]:
    """
    Returns the candidate title lines that are below the difficulty anchor and not part of the cabinet banner.

    Parameters
    ----------
    lines : Sequence[TextReading]
        The OCR readings from the screen.
    screen_readings : Sequence[TextReading]
        The OCR readings from the entire screen.
    difficulty_anchor : TextReading
        The difficulty anchor reading.
    threshold : float
        The confidence threshold for considering a reading.

    Returns
    -------
    tuple[TextReading, ...]
        The candidate title lines that meet the criteria.
    """
    return tuple(
        line for line in lines
        if line.top / SCREEN_HEIGHT <= difficulty_anchor.bottom / SCREEN_HEIGHT + 0.027
        and not _is_cabinet_banner_fragment(line, screen_readings, difficulty_anchor, threshold)
    )


def _is_cabinet_banner_fragment(
    line: TextReading,
    screen_readings: Sequence[TextReading],
    difficulty_anchor: TextReading,
    threshold: float,
) -> bool:
    """
    Determine if the given line is part of the cabinet banner in the screen readings.

    Parameters
    ----------
    line : TextReading
        The line to check.
    screen_readings : Sequence[TextReading]
        The OCR readings from the screen.
    difficulty_anchor : TextReading
        The difficulty anchor reading.
    threshold : float
        The confidence threshold for considering a reading.
    
    Returns
    -------
    bool
        True if the line is part of the cabinet banner, False otherwise.
    """
    if line.bottom - line.top <= 2 * (line.right - line.left):
        return False
    center_x = (line.left + line.right) / 2
    center_y = (line.top + line.bottom) / 2
    for banner in screen_readings:
        normalized = re.sub(r"[\s\-–—]+", "", banner.text.upper())
        if (
            banner.confidence >= threshold
            and normalized == "CABINETTOCABINET"
            and banner.left > difficulty_anchor.right
            and banner.bottom - banner.top > 2 * (banner.right - banner.left)
            and banner.left <= center_x <= banner.right
            and banner.top <= center_y <= banner.bottom
        ):
            return True
    return False


def _layout_anchors(
    readings: Sequence[TextReading],
    confidence_threshold: float,
) -> tuple[TextReading, TextReading]:
    """
    Gets the layout anchors ("SCORE" and "RANK") for the score and rank regions in the screen readings.

    Parameters
    ----------
    readings : Sequence[TextReading]
        The OCR readings from the screen.
    confidence_threshold : float
        The minimum confidence required for a reading to be considered.

    Returns
    -------
    tuple[TextReading, TextReading]
        A tuple containing the score anchor and rank anchor.

    Raises
    ------
    UnsupportedLayoutError
        If the layout does not match the expected configuration.
    """
    scores = []
    ranks = []

    # get all "SCORE" and "RANK" readings with high confidence
    for reading in readings:
        # skip low confidence readings
        if reading.confidence < confidence_threshold:
            continue

        text = reading.text.strip().upper()
        x, y = _center(reading)
        if text == "SCORE" and 0.10 <= x <= 0.42 and 0.12 <= y <= 0.38:
            scores.append(reading)
        if text == "RANK" and 0.40 <= x <= 0.75 and 0.12 <= y <= 0.38:
            ranks.append(reading)
    pairs = [
        (score, rank)
        for score in scores
        for rank in ranks
        if rank.left > score.right
        and abs(_center(score)[1] - _center(rank)[1]) <= 0.06
    ]
    if len(pairs) != 1:
        raise UnsupportedLayoutError("The photographed screen does not match the known layout.")
    return pairs[0]


def _in_region(
    readings: Sequence[TextReading],
    region: tuple[float, float, float, float],
) -> tuple[TextReading, ...]:
    """
    Helper function that filters OCR readings to those within a specified region.

    Parameters
    ----------
    readings : Sequence[TextReading]
        All of the OCR readings from the screen.
    region : tuple[float, float, float, float]
        The normalized (left, top, right, bottom) coordinates of the region.

    Returns
    -------
    tuple[TextReading, ...]
        The OCR readings that fall within the specified region.
    """
    left, top, right, bottom = region
    return tuple(
        reading
        for reading in readings
        if left <= _center(reading)[0] <= right and top <= _center(reading)[1] <= bottom
    )


def _read_region(
    engine: OcrEngine,
    screen: np.ndarray,
    region: tuple[float, float, float, float],
) -> tuple[TextReading, ...]:
    """
    Reads OCR readings from a specified region of the screen.

    Unlike `_in_region`, this function performs OCR on the specified region of the screen.

    Unlike `_read_ocr`, this function focuses on a specific region of the screen rather than the entire image.

    Parameters
    ----------
    engine : OcrEngine
        The OCR engine to use for reading the image.
    screen : np.ndarray
        The full screen image.
    region : tuple[float, float, float, float]
        The normalized (left, top, right, bottom) coordinates of the region to read.

    Returns
    -------
    tuple[TextReading, ...]
        The OCR readings within the specified region.
    """
    left, top, right, bottom = region
    x0 = int(left * SCREEN_WIDTH)
    y0 = int(top * SCREEN_HEIGHT)
    x1 = int(right * SCREEN_WIDTH)
    y1 = int(bottom * SCREEN_HEIGHT)
    scale = 2
    crop = screen[y0:y1, x0:x1]
    enlarged = cv2.resize(
        crop,
        (crop.shape[1] * scale, crop.shape[0] * scale),
        interpolation=cv2.INTER_CUBIC,
    )
    readings = _read_ocr(engine, enlarged)
    return tuple(
        TextReading(
            text=reading.text,
            confidence=reading.confidence,
            left=reading.left / scale + x0,
            top=reading.top / scale + y0,
            right=reading.right / scale + x0,
            bottom=reading.bottom / scale + y0,
        )
        for reading in readings
    )


def _read_ocr(engine: OcrEngine, image: np.ndarray) -> Sequence[TextReading]:
    """
    Reads text from an image using the specified OCR engine.

    Unlike `_read_region`, this function operates on the entire image provided 
    and does not perform any cropping or scaling.

    Parameters
    ----------
    engine : OcrEngine
        The OCR engine to use for reading the image.
    image : np.ndarray
        The image to perform OCR on.

    Returns
    -------
    Sequence[TextReading]
        The OCR readings obtained from the image.

    Raises
    ------
    ExtractionError
        If the OCR engine fails to extract text from the image.
    ProcessingError
        If an unexpected error occurs during OCR processing.
    """
    try:
        return engine.read(image)
    except ExtractionError:
        raise
    except Exception as exc:
        raise ProcessingError("OCR processing failed.") from exc


def _extract_score(lines: Sequence[TextReading], threshold: float) -> int:
    """
    Extracts the current-play score from the given OCR readings.

    This function converts TextReading sequences to integer scores.

    Parameters
    ----------
    lines : Sequence[TextReading]
        The OCR readings to extract the score from.
    threshold : float
        The confidence threshold for considering a reading valid.

    Returns
    -------
    int
        The extracted score.

    Raises
    ------
    ConflictingReadingsError
        If multiple conflicting scores are found.
    UncertainResultError
        If no confident score can be determined.
    """
    values: set[int] = set()
    for line in lines:
        # skip readings with low confidence
        if line.confidence < threshold:
            continue
        for match in _SCORE_PATTERN.finditer(line.text):
            digits = re.sub(r"\D", "", match.group())
            value = int(digits)
            if 0 <= value <= 1_010_000:
                values.add(value)
    if len(values) > 1:
        raise ConflictingReadingsError("OCR returned conflicting current-play scores.")
    if not values:
        raise UncertainResultError("The current-play score could not be read confidently.")
    return values.pop()


def _current_score(
    main_lines: Sequence[TextReading],
    header_lines: Sequence[TextReading],
    threshold: float,
) -> int:
    """
    Extracts the current-play score from the given OCR readings.

    The function first attempts to extract the score from the main lines. If it fails,
    it falls back to the header lines. If both sources provide conflicting scores,
    an error is raised.

    Parameters
    ----------
    main_lines : Sequence[TextReading]
        The OCR readings from the main region of the screen.
    header_lines : Sequence[TextReading]
        The OCR readings from the header region of the screen.
    threshold : float
        The confidence threshold for considering a reading valid.

    Returns
    -------
    int
        The extracted current-play score.

    Raises
    ------
    ConflictingReadingsError
        If the main and header scores conflict.
    UncertainResultError
        If the score cannot be determined confidently.
    """

    # Filter header lines to include only those that likely contain the score, 
    # excluding high scores
    header_scores = [
        line for line in header_lines
        if re.search(r"\bSCORE\s*[:：]", line.text, re.IGNORECASE)
        and "HIGH" not in line.text.upper()
    ]
    try:
        main_score = _extract_score(main_lines, threshold)
    except UncertainResultError:
        return _extract_score(header_scores, threshold)
    try:
        header_score = _extract_score(header_scores, threshold)
    except UncertainResultError:
        return main_score
    if main_score != header_score:
        raise ConflictingReadingsError("Current-play score readings disagree.")
    return main_score


def _extract_title(lines: Sequence[TextReading], threshold: float) -> str:
    """
    Extracts the song title from the given OCR readings.

    Parameters
    ----------
    lines : Sequence[TextReading]
        The OCR readings from which to extract the song title.
    threshold : float
        The confidence threshold for considering a reading valid.

    Returns
    -------
    str
        The extracted song title.

    Raises
    ------
    ConflictingReadingsError
        If the OCR returned conflicting song-title readings.
    UncertainResultError
        If the song title cannot be determined confidently.
    """
    candidates = [line for line in lines if line.text.strip()]
    if not candidates:
        raise UncertainResultError("The visible song title could not be read confidently.")

    first = min(candidates, key=lambda line: line.top)
    height = max(1.0, first.bottom - first.top)
    title_lines = [
        line for line in candidates
        if abs(line.top - first.top) <= height * 0.5
    ]
    if any(line.confidence < threshold for line in title_lines):
        raise UncertainResultError("The visible song title could not be read confidently.")
    normalized = {_normalize_title(line.text) for line in title_lines}
    normalized.discard("")
    if len(normalized) > 1:
        raise ConflictingReadingsError("OCR returned conflicting song-title readings.")
    if not normalized:
        raise UncertainResultError("The visible song title could not be read confidently.")
    return normalized.pop()


def _extract_difficulty(lines: Sequence[TextReading], threshold: float) -> Difficulty:
    """
    Extracts the difficulty level from the given OCR readings.

    Parameters
    ----------
    lines : Sequence[TextReading]
        The OCR readings from which to extract the difficulty.
    threshold : float
        The confidence threshold for considering a reading valid.

    Returns
    -------
    Difficulty
        The extracted difficulty level.
    """
    found: set[Difficulty] = set()
    for line in lines:
        if line.confidence < threshold:
            continue
        found.update(_difficulty_in_text(line.text))
    if len(found) > 1:
        raise ConflictingReadingsError("OCR returned conflicting difficulty readings.")
    if not found:
        raise UncertainResultError("The difficulty could not be read confidently.")
    return found.pop()


def _difficulty_in_text(text: str) -> set[Difficulty]:
    """
    Identifies (and maps) the difficulty levels mentioned in the given text.

    Parameters
    ----------
    text : str
        The text to search for difficulty indicators.

    Returns
    -------
    set[Difficulty]
        The set of difficulty levels found in the text.
    """
    text = text.upper().replace(" ", "")
    found = {difficulty for name, difficulty in _DIFFICULTY_NAMES.items() if name in text}
    if _WORLD_END_PATTERN.search(text):
        found.add(Difficulty.WORLDEND)
    return found


def _normalize_title(value: str) -> str:
    return " ".join(value.strip().split())


def _center(reading: TextReading) -> tuple[float, float]:
    """
    Helper function that calculates the normalized center coordinates of a text reading relative to the screen dimensions.

    Parameters
    ----------
    reading : TextReading
        The OCR reading for which to calculate the center coordinates.

    Returns
    -------
    tuple[float, float]
        The normalized (x, y) center coordinates of the reading relative to the screen dimensions.
    """
    return (
        (reading.left + reading.right) / (2 * SCREEN_WIDTH),
        (reading.top + reading.bottom) / (2 * SCREEN_HEIGHT),
    )
