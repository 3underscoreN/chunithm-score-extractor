from collections.abc import Sequence

import cv2
import numpy as np

from .errors import UnsupportedLayoutError


SCREEN_WIDTH = 1600
SCREEN_HEIGHT = 1000


def rectify_screen(image: np.ndarray) -> np.ndarray:
    """
    Rectifies the given image of a CHUNITHM display to a standard screen layout.

    Parameters
    ----------
    image : np.ndarray
        The input image to rectify.

    Returns
    -------
    np.ndarray
        The rectified image with the standard screen dimensions.

    Raises
    ------
    UnsupportedLayoutError
        If the image does not contain a recognizable CHUNITHM display.
    """
    image_area = image.shape[0] * image.shape[1]
    candidates: list[tuple[float, np.ndarray]] = []
    # detect edges
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    edges = cv2.Canny(blurred, 40, 130)

    # collect candidates of potential display regions from the contours
    contours, _ = cv2.findContours(edges, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    _collect_candidates(contours, image_area, candidates)

    # collect candidates from thresholding at multiple levels
    close_size = min(61, max(15, int(min(image.shape[:2]) * 0.03) | 1))
    close_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (close_size, close_size))
    for level in (40, 60, 80, 100, 120, 140, 150, 160, 180, 200, 220):
        mask = cv2.threshold(gray, level, 255, cv2.THRESH_BINARY)[1]
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, close_kernel)
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        _collect_candidates(contours, image_area, candidates)

    if not candidates:
        raise UnsupportedLayoutError("Could not locate a supported CHUNITHM display.")

    # select the candidate with the largest area as the display region
    _, corners = max(candidates, key=lambda candidate: candidate[0])
    target = np.array(
        [
            [0, 0],
            [SCREEN_WIDTH - 1, 0],
            [SCREEN_WIDTH - 1, SCREEN_HEIGHT - 1],
            [0, SCREEN_HEIGHT - 1],
        ],
        dtype=np.float32,
    )
    # transform the corners to the target screen layout
    transform = cv2.getPerspectiveTransform(corners, target)
    return cv2.warpPerspective(image, transform, (SCREEN_WIDTH, SCREEN_HEIGHT))


def _collect_candidates(
    contours: Sequence[np.ndarray],
    image_area: int,
    candidates: list[tuple[float, np.ndarray]],
) -> None:
    """
    Given a list of contours, the image area, and a list to store candidates, this 
    function collects potential CHUNITHM display candidates based on contour properties.

    Parameters
    ----------
    contours : Sequence[np.ndarray]
        The contours detected in the image.
    image_area : int
        The area of the image.
    candidates : list[tuple[float, np.ndarray]]
        The list to store candidate contours along with their areas.

    Returns
    -------
    None
    """
    # Iterate through each contour to find potential CHUNITHM display candidates
    for contour in contours:
        # Skip small contours that are unlikely to be the display
        perimeter = cv2.arcLength(contour, True)
        if perimeter <= 0:
            continue

        # Approximate the contour to a polygon and check if it could be a display rectangle
        # calculate the polygon approximation and its area
        polygon = cv2.approxPolyDP(contour, 0.02 * perimeter, True)
        area = cv2.contourArea(polygon)

        # Skip contours that do not form a quadrilateral or are too small or not convex
        if len(polygon) != 4 or area < image_area * 0.20 or not cv2.isContourConvex(polygon):
            continue

        # Gets the approximate aspect ratio of the quadrilateral
        ordered = _order_corners(polygon.reshape(4, 2).astype(np.float32))
        width = max(
            np.linalg.norm(ordered[1] - ordered[0]),
            np.linalg.norm(ordered[2] - ordered[3]),
        )
        height = max(
            np.linalg.norm(ordered[3] - ordered[0]),
            np.linalg.norm(ordered[2] - ordered[1]),
        )
        ratio = width / max(height, 1)
        # if aspect ratio falls in a reasonable range for the display, add to candidate
        if 1.4 <= ratio <= 2.5:
            candidates.append((float(area), ordered))


def _order_corners(points: np.ndarray) -> np.ndarray:
    """
    Helper function to order the corners of a quadrilateral in a consistent manner.

    Parameters
    ----------
    points : np.ndarray
        An array of shape (4, 2) representing the corners of a quadrilateral.

    Returns
    -------
    np.ndarray
        The ordered corners of the quadrilateral starting from the top-left corner and proceeding clockwise.
    """
    center = points.mean(axis=0)
    angles = np.arctan2(points[:, 1] - center[1], points[:, 0] - center[0])
    ordered = points[np.argsort(angles)]
    top_left_index = int(np.argmin(ordered.sum(axis=1)))
    return np.roll(ordered, -top_left_index, axis=0).astype(np.float32)
