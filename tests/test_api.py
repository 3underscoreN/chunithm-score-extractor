from __future__ import annotations

import io
import os
import unittest
from unittest.mock import patch

from PIL import Image

try:
    from fastapi.testclient import TestClient

    from chunithm_score_extractor import ChunithmExtractor
    from chunithm_score_extractor.api import (
        MAX_REQUEST_BYTES,
        MAX_UPLOAD_BYTES,
        create_app,
    )
    from chunithm_score_extractor.errors import (
        ModelInitializationError,
        ProcessingError,
        UnsupportedLayoutError,
    )
    from chunithm_score_extractor.models import Difficulty, ExtractionResult
    from chunithm_score_extractor.ocr import TextReading

    API_AVAILABLE = True
except ModuleNotFoundError:
    API_AVAILABLE = False


@unittest.skipUnless(API_AVAILABLE, "Install the 'dev' extra to run API tests.")
class ApiTests(unittest.TestCase):
    def test_extract_returns_stable_result_shape(self) -> None:
        app = create_app(StubExtractor())
        with TestClient(app) as client:
            response = client.post(
                "/extract",
                files={"image": ("result.png", _png(), "image/png")},
            )

        self.assertEqual(response.status_code, 200)
        self.assertEqual(
            response.json(),
            {
                "score": 1_007_584,
                "song_name": "Visible title",
                "difficulty": "MAS",
            },
        )

    def test_maps_input_extraction_and_internal_errors(self) -> None:
        app = create_app(StubExtractor(UnsupportedLayoutError("not a result screen")))
        invalid_app = create_app(ChunithmExtractor(ocr=EmptyOcrEngine()))
        with TestClient(app) as client:
            unsupported = client.post(
                "/extract",
                files={"image": ("result.png", _png(), "image/png")},
            )
        with TestClient(invalid_app) as client:
            invalid = client.post(
                "/extract",
                files={"image": ("bad.png", b"not an image", "image/png")},
            )

        self.assertEqual(unsupported.status_code, 422)
        self.assertEqual(unsupported.json()["detail"]["code"], "unsupported_layout")
        self.assertEqual(invalid.status_code, 400)
        self.assertEqual(invalid.json()["detail"]["code"], "invalid_image")

        app = create_app(StubExtractor(ProcessingError("inference failed")))
        with TestClient(app) as client:
            internal = client.post(
                "/extract",
                files={"image": ("result.png", _png(), "image/png")},
            )
        self.assertEqual(internal.status_code, 500)
        self.assertEqual(internal.json()["detail"]["code"], "processing_failed")

    def test_rejects_unsupported_and_oversized_uploads(self) -> None:
        app = create_app(StubExtractor())
        with TestClient(app) as client:
            unsupported = client.post(
                "/extract",
                files={"image": ("result.txt", b"data", "text/plain")},
            )
            oversized_file = client.post(
                "/extract",
                files={
                    "image": (
                        "large.png",
                        b"x" * (MAX_UPLOAD_BYTES + 1),
                        "image/png",
                    )
                },
            )
            oversized_body = client.post(
                "/extract",
                content=b"x" * (MAX_REQUEST_BYTES + 1),
                headers={"content-type": "application/octet-stream"},
            )

        self.assertEqual(unsupported.status_code, 415)
        self.assertEqual(oversized_file.status_code, 413)
        self.assertEqual(oversized_body.status_code, 413)

    def test_reports_model_initialization_failure_as_not_ready(self) -> None:
        with patch(
            "chunithm_score_extractor.api.ChunithmExtractor",
            side_effect=ModelInitializationError("model unavailable"),
        ):
            app = create_app()
            with TestClient(app) as client:
                live = client.get("/health/live")
                ready = client.get("/health/ready")

        self.assertEqual(live.status_code, 200)
        self.assertEqual(ready.status_code, 503)
        self.assertEqual(ready.json()["detail"]["code"], "model_initialization_failed")

    def test_forwards_device_from_environment(self) -> None:
        extractor = StubExtractor()
        with patch.dict(os.environ, {"CHUNITHM_OCR_DEVICE": "gpu:0"}), patch(
            "chunithm_score_extractor.api.ChunithmExtractor",
            return_value=extractor,
        ) as constructor:
            with TestClient(create_app()) as client:
                self.assertEqual(client.get("/health/ready").status_code, 200)
        constructor.assert_called_once_with(device="gpu:0")


class StubExtractor:
    def __init__(self, error: Exception | None = None) -> None:
        self.error = error

    def extract(self, image: bytes) -> ExtractionResult:
        if self.error is not None:
            raise self.error
        return ExtractionResult(1_007_584, "Visible title", Difficulty.MAS)


class EmptyOcrEngine:
    def read(self, image: object) -> list[TextReading]:
        return []


def _png() -> bytes:
    image = io.BytesIO()
    Image.new("RGB", (20, 10), "white").save(image, format="PNG")
    return image.getvalue()


if __name__ == "__main__":
    unittest.main()
