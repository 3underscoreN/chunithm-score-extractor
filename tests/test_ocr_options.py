import logging
import sys
import types
import unittest
import warnings
from unittest.mock import Mock, patch

from chunithm_score_extractor import ChunithmExtractor, ModelInitializationError
from chunithm_score_extractor.ocr import PaddleOcrEngine


class PaddleOptionsTests(unittest.TestCase):
    def test_device_is_forwarded_to_paddleocr(self) -> None:
        constructor = Mock()
        paddleocr = types.ModuleType("paddleocr")
        paddleocr.PaddleOCR = constructor
        paddle = types.ModuleType("paddle")
        paddle.is_compiled_with_cuda = lambda: True
        paddle.device = types.SimpleNamespace(cuda=types.SimpleNamespace(device_count=lambda: 1))

        with patch.dict(sys.modules, {"paddleocr": paddleocr, "paddle": paddle}):
            PaddleOcrEngine(device="gpu:0")
            self.assertEqual(constructor.call_args.kwargs["device"], "gpu:0")
            PaddleOcrEngine(device="cpu")
            self.assertEqual(constructor.call_args.kwargs["device"], "cpu")
            PaddleOcrEngine()
            self.assertIsNone(constructor.call_args.kwargs["device"])

    def test_gpu_must_be_available_and_cuda_enabled(self) -> None:
        paddleocr = types.ModuleType("paddleocr")
        paddleocr.PaddleOCR = Mock()
        paddle = types.ModuleType("paddle")
        paddle.is_compiled_with_cuda = lambda: False
        with patch.dict(sys.modules, {"paddleocr": paddleocr, "paddle": paddle}):
            with self.assertRaisesRegex(ModelInitializationError, "not built with CUDA"):
                PaddleOcrEngine(device="gpu")

        paddle.is_compiled_with_cuda = lambda: True
        paddle.device = types.SimpleNamespace(cuda=types.SimpleNamespace(device_count=lambda: 1))
        with patch.dict(sys.modules, {"paddleocr": paddleocr, "paddle": paddle}):
            with self.assertRaisesRegex(ModelInitializationError, "GPU device 1"):
                PaddleOcrEngine(device="gpu:1")

    def test_invalid_device_and_custom_engine_combination_are_rejected(self) -> None:
        with self.assertRaises(ValueError):
            PaddleOcrEngine(device="gpu:0,1")
        with self.assertRaises(ValueError):
            ChunithmExtractor(ocr=Mock(), device="cpu")

    def test_only_specific_startup_noise_is_filtered_and_filter_is_removed(self) -> None:
        logger = logging.getLogger("paddlex")
        captured = []

        class Capture(logging.Handler):
            def emit(self, record: logging.LogRecord) -> None:
                captured.append(record.getMessage())

        handler = Capture()
        paddleocr = types.ModuleType("paddleocr")

        def construct(**kwargs: object) -> object:
            logger.info("Creating model: example")
            logger.warning("Checking connectivity to the model hosters, please wait")
            logger.warning("A genuine model warning")
            warnings.warn_explicit(
                "No ccache found. Please be aware that recompiling all source files may be required.",
                UserWarning,
                "extension_utils.py",
                1,
                module="paddle.utils.cpp_extension.extension_utils",
            )
            warnings.warn("A genuine Python warning", UserWarning)
            return object()

        paddleocr.PaddleOCR = construct
        previous_level = logger.level
        logger.setLevel(logging.INFO)
        logger.addHandler(handler)
        try:
            with patch.dict(sys.modules, {"paddleocr": paddleocr}), warnings.catch_warnings(record=True) as seen:
                warnings.simplefilter("always")
                PaddleOcrEngine(device="cpu")
            logger.info("Creating model: after initialization")
        finally:
            logger.removeHandler(handler)
            logger.setLevel(previous_level)

        self.assertIn("A genuine model warning", captured)
        self.assertIn("Creating model: after initialization", captured)
        self.assertNotIn("Creating model: example", captured)
        self.assertEqual([str(item.message) for item in seen], ["A genuine Python warning"])


if __name__ == "__main__":
    unittest.main()
