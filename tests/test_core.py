import io
import unittest

import cv2
import numpy as np
from PIL import Image

from chunithm_score_extractor import ChunithmExtractor, ExtractionResult
from chunithm_score_extractor.errors import (
    ConflictingReadingsError,
    InvalidImageError,
    UncertainResultError,
    UnsupportedLayoutError,
)
from chunithm_score_extractor.extractor import (
    _current_score,
    _extract_difficulty,
    _extract_score,
    _extract_title,
    _is_cabinet_banner_fragment,
    _layout_anchors,
)
from chunithm_score_extractor.image import ImageLimits, decode_image
from chunithm_score_extractor.layout import SCREEN_HEIGHT, SCREEN_WIDTH, rectify_screen
from chunithm_score_extractor.models import Difficulty
from chunithm_score_extractor.ocr import TextReading, _parse_result


def reading(
    text: str,
    confidence: float = 0.95,
    box: tuple[int, int, int, int] = (0, 0, 100, 30),
) -> TextReading:
    return TextReading(text, confidence, *box)


class ImageValidationTests(unittest.TestCase):
    def test_decodes_supported_image(self) -> None:
        output = io.BytesIO()
        Image.new("RGB", (20, 10), "white").save(output, format="PNG")

        image = decode_image(output.getvalue())

        self.assertEqual(image.shape, (10, 20, 3))

    def test_rejects_bad_and_oversized_input(self) -> None:
        with self.assertRaises(InvalidImageError):
            decode_image(b"not an image")
        with self.assertRaises(InvalidImageError):
            decode_image(b"")

        output = io.BytesIO()
        Image.new("RGB", (21, 10), "white").save(output, format="PNG")
        with self.assertRaises(InvalidImageError):
            decode_image(output.getvalue(), ImageLimits(max_pixels=200))


class FieldParsingTests(unittest.TestCase):
    def test_reads_formatted_score(self) -> None:
        self.assertEqual(_extract_score([reading("1,007,584")], 0.75), 1_007_584)

    def test_header_score_fallback_and_conflicts(self) -> None:
        header = [reading("SCORE: 1,007,584")]
        with self.assertRaises(ConflictingReadingsError):
            _current_score([reading("1,007,585")], header, 0.75)
        with self.assertRaises(UncertainResultError):
            _current_score([], [reading("HIGH SCORE: 1,007,584")], 0.75)

    def test_rejects_uncertain_and_conflicting_scores(self) -> None:
        with self.assertRaises(UncertainResultError):
            _extract_score([reading("1,007,584", 0.4)], 0.75)
        with self.assertRaises(ConflictingReadingsError):
            _extract_score([reading("1,007,584"), reading("1,007,585")], 0.75)

    def test_reads_difficulty_names(self) -> None:
        self.assertEqual(
            _extract_difficulty([reading("MASTER")], 0.75),
            Difficulty.MAS,
        )
        self.assertEqual(
            _extract_difficulty([reading("WORLD'S END")], 0.75),
            Difficulty.WORLDEND,
        )

    def test_reads_and_preserves_unicode_title(self) -> None:
        title = "ヒプノシスマイク -Division Battle Anthem-"
        self.assertEqual(_extract_title([reading(title)], 0.75), title)
        with self.assertRaises(ConflictingReadingsError):
            _extract_title([reading(title), reading("別の曲")], 0.75)

    def test_low_confidence_title_does_not_fall_through_to_artist(self) -> None:
        with self.assertRaises(UncertainResultError):
            _extract_title(
                [
                    reading("title", 0.7, (0, 0, 200, 20)),
                    reading("artist", 0.99, (0, 30, 200, 45)),
                ],
                0.75,
            )

    def test_layout_anchors_must_form_one_matching_pair(self) -> None:
        score = reading("SCORE", box=(220, 180, 320, 210))
        rank = reading("RANK", box=(680, 190, 780, 220))
        self.assertEqual(_layout_anchors([score, rank], 0.75), (score, rank))
        with self.assertRaises(UnsupportedLayoutError):
            _layout_anchors([score, reading("RANK", box=(680, 300, 780, 330))], 0.75)
        with self.assertRaises(UnsupportedLayoutError):
            _layout_anchors([score, rank, rank], 0.75)

    def test_only_confirmed_vertical_banner_fragments_are_excluded(self) -> None:
        difficulty = reading("EXPERT", box=(1256, 64, 1422, 102))
        banner = reading("CABINET-TO-CABINET", box=(1572, 24, 1588, 149))
        fragment = reading("CABINET", box=(1573, 104, 1586, 150))
        title = reading("CABINET-TO-CABINET", box=(1261, 113, 1409, 138))

        self.assertTrue(_is_cabinet_banner_fragment(fragment, [banner], difficulty, 0.75))
        self.assertFalse(_is_cabinet_banner_fragment(title, [banner], difficulty, 0.75))
        self.assertFalse(
            _is_cabinet_banner_fragment(
                reading("Title", box=(1530, 115, 1630, 135)), [banner], difficulty, 0.75
            )
        )
        self.assertFalse(
            _is_cabinet_banner_fragment(
                fragment, [reading(banner.text, 0.7, (1572, 24, 1588, 149))], difficulty, 0.75
            )
        )
        self.assertFalse(
            _is_cabinet_banner_fragment(
                fragment, [reading(banner.text, box=(1500, 100, 1600, 130))], difficulty, 0.75
            )
        )

    def test_banner_filter_does_not_hide_missing_or_conflicting_titles(self) -> None:
        difficulty = reading("EXPERT", box=(1256, 64, 1422, 102))
        banner = reading("CABINET-TO-CABINET", box=(1572, 24, 1588, 149))
        fragment = reading("CABINET", box=(1573, 104, 1586, 150))
        first_title = reading("First", box=(1261, 113, 1409, 138))
        second_title = reading("Second", box=(1261, 114, 1409, 139))

        def filter_banner(lines: list[TextReading], screen: list[TextReading]) -> list[TextReading]:
            return [
                line for line in lines
                if not _is_cabinet_banner_fragment(line, screen, difficulty, 0.75)
            ]

        with self.assertRaises(UncertainResultError):
            _extract_title(filter_banner([fragment], [banner]), 0.75)
        with self.assertRaises(ConflictingReadingsError):
            _extract_title(filter_banner([first_title, second_title, fragment], [banner]), 0.75)
        with self.assertRaises(ConflictingReadingsError):
            _extract_title(filter_banner([first_title, fragment], []), 0.75)


class ExtractionPipelineTests(unittest.TestCase):
    def test_extracts_all_fields_after_screen_rectification(self) -> None:
        image = np.zeros((480, 640, 3), dtype=np.uint8)
        corners = np.array([[70, 90], [570, 90], [570, 390], [70, 390]])
        cv2.fillConvexPoly(image, corners, (220, 220, 220))
        success, encoded = cv2.imencode(".png", image)
        self.assertTrue(success)
        engine = SequenceOcrEngine(
            [
                [
                    reading("SCORE", box=(250, 280, 350, 310)),
                    reading("RANK", box=(900, 280, 1000, 310)),
                    reading("MASTER", box=(1250, 70, 1450, 110)),
                ],
                [reading("1,007,584", box=(100, 100, 500, 140))],
                [reading("テスト曲 / Imaginary Title", box=(20, 10, 260, 30))],
            ]
        )

        result = ChunithmExtractor(ocr=engine).extract(encoded.tobytes())

        self.assertEqual(
            result,
            ExtractionResult(1_007_584, "テスト曲 / Imaginary Title", Difficulty.MAS),
        )
        self.assertEqual(
            engine.image_shapes[0], (SCREEN_HEIGHT, SCREEN_WIDTH, 3)
        )
        self.assertEqual(len(engine.image_shapes), 3)

    def test_rectifies_a_steep_trapezoid(self) -> None:
        image = np.zeros((600, 900, 3), dtype=np.uint8)
        corners = np.array([[140, 80], [730, 90], [890, 420], [10, 410]])
        cv2.fillConvexPoly(image, corners, (210, 210, 210))
        screen = rectify_screen(image)
        self.assertEqual(screen.shape, (SCREEN_HEIGHT, SCREEN_WIDTH, 3))

    def test_rejects_unrecognized_layout_before_reading_fields(self) -> None:
        image = np.zeros((480, 640, 3), dtype=np.uint8)
        corners = np.array([[70, 90], [570, 90], [570, 390], [70, 390]])
        cv2.fillConvexPoly(image, corners, (220, 220, 220))
        success, encoded = cv2.imencode(".png", image)
        self.assertTrue(success)
        engine = SequenceOcrEngine([[]])

        with self.assertRaises(UnsupportedLayoutError):
            ChunithmExtractor(ocr=engine).extract(encoded.tobytes())
        self.assertEqual(len(engine.image_shapes), 1)

    def test_artist_is_not_accepted_when_title_is_missing(self) -> None:
        image = np.zeros((480, 640, 3), dtype=np.uint8)
        corners = np.array([[70, 90], [570, 90], [570, 390], [70, 390]])
        cv2.fillConvexPoly(image, corners, (220, 220, 220))
        success, encoded = cv2.imencode(".png", image)
        self.assertTrue(success)
        engine = SequenceOcrEngine(
            [
                [
                    reading("SCORE", box=(250, 280, 350, 310)),
                    reading("RANK", box=(900, 280, 1000, 310)),
                    reading("MASTER", box=(1250, 70, 1450, 110)),
                ],
                [reading("1,007,584", box=(100, 100, 500, 140))],
                [reading("artist", box=(20, 62, 200, 80))],
            ]
        )

        with self.assertRaises(UncertainResultError):
            ChunithmExtractor(ocr=engine).extract(encoded.tobytes())

    def test_title_ignores_confirmed_cabinet_banner_fragment(self) -> None:
        image = np.zeros((480, 640, 3), dtype=np.uint8)
        cv2.fillConvexPoly(image, np.array([[70, 90], [570, 90], [570, 390], [70, 390]]), (220, 220, 220))
        success, encoded = cv2.imencode(".png", image)
        self.assertTrue(success)
        engine = SequenceOcrEngine(
            [
                [
                    reading("SCORE", box=(250, 280, 350, 310)),
                    reading("RANK", box=(900, 280, 1000, 310)),
                    reading("EXPERT", box=(1256, 64, 1422, 102)),
                    reading("CABINET-TO-CABINET", box=(1572, 24, 1588, 149)),
                ],
                [reading("1,008,638", box=(100, 100, 500, 140))],
                [
                    reading("CABINET", box=(688, 0, 714, 93)),
                    reading("HeinousAeaper", box=(66, 19, 357, 66)),
                ],
            ]
        )
        result = ChunithmExtractor(ocr=engine).extract(encoded.tobytes())
        self.assertEqual(result, ExtractionResult(1_008_638, "HeinousAeaper", Difficulty.EXP))

    def test_parses_paddleocr_text_boxes(self) -> None:
        readings = _parse_result(
            {
                "rec_texts": ["SCORE"],
                "rec_scores": [0.98],
                "rec_polys": [[[10, 20], [60, 20], [60, 40], [10, 40]]],
            }
        )

        self.assertEqual(readings, (TextReading("SCORE", 0.98, 10, 20, 60, 40),))


class SequenceOcrEngine:
    def __init__(self, results: list[list[TextReading]]) -> None:
        self.results = results
        self.image_shapes: list[tuple[int, ...]] = []

    def read(self, image: np.ndarray) -> list[TextReading]:
        self.image_shapes.append(image.shape)
        return self.results.pop(0)


if __name__ == "__main__":
    unittest.main()
