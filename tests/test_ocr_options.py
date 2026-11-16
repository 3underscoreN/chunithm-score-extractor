from __future__ import annotations

import builtins
import logging
import os
import sys
import types
import unittest
import warnings
from unittest.mock import Mock, patch

from chunithm_score_extractor import ChunithmExtractor, ModelInitializationError
from chunithm_score_extractor.ocr import PaddleOcrEngine


class PaddleOptionsTests(unittest.TestCase):
    def test_import_error_preserves_native_library_failure(self) -> None:
        original_import = builtins.__import__
        failure = ImportError("libgomp.so.1: cannot open shared object file")
        onnxruntime = _fake_onnxruntime({"CPUExecutionProvider"})

        def failing_import(name: str, *args: object, **kwargs: object) -> object:
            if name == "paddleocr":
                raise failure
            return original_import(name, *args, **kwargs)

        with patch.dict(sys.modules, {"onnxruntime": onnxruntime}), patch(
            "builtins.__import__", side_effect=failing_import
        ):
            with self.assertRaises(ModelInitializationError) as caught:
                PaddleOcrEngine()

        self.assertIn("libgomp.so.1", str(caught.exception))
        self.assertIs(caught.exception.__cause__, failure)

    def test_device_is_forwarded_as_onnxruntime_providers(self) -> None:
        paddleocr, constructor = _fake_paddleocr()
        onnxruntime = _fake_onnxruntime(
            {"CPUExecutionProvider", "CUDAExecutionProvider", "OpenVINOExecutionProvider"}
        )

        with patch.dict(
            sys.modules, {"paddleocr": paddleocr, "onnxruntime": onnxruntime}
        ):
            PaddleOcrEngine(device="gpu:0")
            self.assertEqual(
                constructor.call_args.kwargs["engine_config"],
                {
                    "providers": [
                        "CUDAExecutionProvider",
                        "CPUExecutionProvider",
                    ],
                    "provider_options": [{"device_id": 0}, {}],
                },
            )
            PaddleOcrEngine(device="cpu")
            self.assertEqual(
                constructor.call_args.kwargs["engine_config"],
                {"providers": ["CPUExecutionProvider"]},
            )
            PaddleOcrEngine()
            self.assertEqual(
                constructor.call_args.kwargs["engine_config"],
                {"providers": ["CPUExecutionProvider"]},
            )
            PaddleOcrEngine(device="openvino")
            self.assertEqual(
                constructor.call_args.kwargs["engine_config"],
                {
                    "providers": [
                        "OpenVINOExecutionProvider",
                        "CPUExecutionProvider",
                    ],
                    "provider_options": [{"device_type": "AUTO"}, {}],
                },
            )
            PaddleOcrEngine(device="openvino:gpu")
            self.assertEqual(
                constructor.call_args.kwargs["engine_config"],
                {
                    "providers": [
                        "OpenVINOExecutionProvider",
                        "CPUExecutionProvider",
                    ],
                    "provider_options": [{"device_type": "GPU"}, {}],
                },
            )

    def test_gpu_requires_the_cuda_provider(self) -> None:
        paddleocr, _ = _fake_paddleocr()
        onnxruntime = _fake_onnxruntime({"CPUExecutionProvider"})

        with patch.dict(
            sys.modules, {"paddleocr": paddleocr, "onnxruntime": onnxruntime}
        ):
            with self.assertRaisesRegex(ModelInitializationError, "CUDA"):
                PaddleOcrEngine(device="gpu")

    def test_openvino_requires_the_openvino_provider(self) -> None:
        paddleocr, _ = _fake_paddleocr()
        onnxruntime = _fake_onnxruntime({"CPUExecutionProvider"})

        with patch.dict(
            sys.modules, {"paddleocr": paddleocr, "onnxruntime": onnxruntime}
        ):
            with self.assertRaisesRegex(ModelInitializationError, "OpenVINO"):
                PaddleOcrEngine(device="openvino")

    def test_invalid_device_and_custom_engine_combination_are_rejected(self) -> None:
        with self.assertRaises(ValueError):
            PaddleOcrEngine(device="gpu:0,1")
        with self.assertRaises(ValueError):
            PaddleOcrEngine(device="openvino:tpu")
        with self.assertRaises(ValueError):
            ChunithmExtractor(ocr=Mock(), device="cpu")

    def test_device_environment_is_used_and_explicit_device_wins(self) -> None:
        with patch.dict(os.environ, {"DEVICE": "gpu:0"}), patch(
            "chunithm_score_extractor.extractor.PaddleOcrEngine"
        ) as engine:
            ChunithmExtractor()
            engine.assert_called_once_with(size="small", device="gpu:0")

        with patch.dict(os.environ, {"DEVICE": "gpu:0"}), patch(
            "chunithm_score_extractor.extractor.PaddleOcrEngine"
        ) as engine:
            ChunithmExtractor(device="cpu")
            engine.assert_called_once_with(size="small", device="cpu")

    def test_only_specific_startup_noise_is_filtered_and_filter_is_removed(self) -> None:
        logger = logging.getLogger("paddlex")
        captured = []
        onnxruntime = _fake_onnxruntime({"CPUExecutionProvider"})

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
            with patch.dict(
                sys.modules, {"paddleocr": paddleocr, "onnxruntime": onnxruntime}
            ), warnings.catch_warnings(record=True) as seen:
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


def _fake_paddleocr() -> tuple[types.ModuleType, Mock]:
    module = types.ModuleType("paddleocr")
    constructor = Mock(return_value=object())
    module.PaddleOCR = constructor
    return module, constructor


def _fake_onnxruntime(providers: set[str]) -> types.ModuleType:
    module = types.ModuleType("onnxruntime")
    module.get_available_providers = lambda: list(providers)
    return module


if __name__ == "__main__":
    unittest.main()
