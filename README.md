# CHUNITHM Score Extractor

A Python library and optional self-hosted HTTP API for extracting the current-play score, visible song title, and difficulty from CHUNITHM result photos.

## Requirements

- Python 3.13
- numpy, OpenCV, Pillow
- PaddleOCR 3.7.0 running on ONNX Runtime. CPU, NVIDIA GPU, and Intel OpenVINO runtimes are supported, and you can also bring your own custom OCR engine.
- Docker (optional)

Install the library and CPU OCR runtime with uv:

```shell
uv sync
```

For pip, specify a runtime extra as described below.

## Python API

```python
from pathlib import Path

from chunithm_score_extractor import ChunithmExtractor

extractor = ChunithmExtractor()
result = extractor.extract(Path("result.jpg").read_bytes())

print(result.score)
print(result.song_name)
print(result.difficulty.value)
```

The first initialization downloads PaddleOCR model files.

Success returns an `ExtractionResult`. See below for the available fields.

Invalid images, unsupported layouts, uncertain readings, conflicting readings, model initialization errors, and processing failures raise errors. See the `chunithm_score_extractor.errors` module for the full list of exceptions.

The library accepts JPEG, PNG, and WebP bytes, with a default 15 MiB input limit, 25-megapixel decoded-pixel limit, and 10,000-pixel dimension limit. The limit can be adjusted during creation of the extractor object.

> [!IMPORTANT]
> There's no guarantee that OCR would be able to parse 100% matching text from the images. For example, for really long song names, or for weird character sequences, the OCR may produce incorrect or incomplete readings. Fuzzy matching to a known song database, and appropriate error handling should be applied when using the extracted results.

### Some API I guess?

#### Extractor
```python
class ChunithmExtractor:
    def __init__(
      self, 
      ocr: OcrEngine | None = None, # Custom OCR engine to use instead of the built-in one, see below for details
      image_limits: ImageLimits = ImageLimits(), # Limits for input image size and dimensions
      confidence_threshold: float = CONFIDENCE_THRESHOLD, # Minimum confidence for OCR readings to be considered valid, defaults to 0.75
      *,
      size: Literal["tiny", "small", "medium"] = "small", # Built-in OCR model size, defaults to "small"
      device: str | None = None # Defaults to the DEVICE environment variable, then "cpu". Only used by the built-in OCR engine.
    ):
        ...
    
    def extract(self, image_bytes: bytes) -> ExtractionResult:
        ...
```

#### OcrEngine
```python
class OcrEngine(Protocol):
    def read(self, image: np.ndarray) -> Sequence[TextReading]:
        ...
```

#### Data classes and enums
```python
@dataclass
class ImageLimits:
    max_input_size: int = 15 * 1024 * 1024 # 15 MiB
    max_decoded_pixels: int = 25_000_000 # 25 megapixels
    max_dimension: int = 10_000 # 10,000 pixels

class ExtractionResult:
    score: int
    song_name: str
    difficulty: Difficulty

class Difficulty(StrEnum):
    BAS = "BAS"
    ADV = "ADV"
    EXP = "EXP"
    MAS = "MAS"
    ULT = "ULT"
    WORLDEND = "WORLDEND"
```

### Choosing an ONNX Runtime

`uv sync` installs the CPU runtime by default. To use the GPU runtime in the same uv-managed environment, disable the default CPU group and select the GPU extra:

```shell
uv sync --no-group onnx-cpu --extra gpu
```

Use `uv sync --no-group onnx-cpu --extra openvino` for the OpenVINO runtime. Running plain `uv sync` again restores the CPU runtime. Keep one runtime per environment because all three packages provide the `onnxruntime` import.

For pip, choose one extra in a fresh environment:

```shell
python -m pip install ".[cpu]"       # CPU
python -m pip install ".[gpu]"       # NVIDIA GPU
python -m pip install ".[openvino]"  # Intel OpenVINO
```

`pip install .` alone does not install an ONNX Runtime. Extras only add packages, so pip will not replace a runtime that is already installed.

The built-in OCR engine reads the `DEVICE` environment variable when the `device` argument is not set. It defaults to `cpu`. Accepted values are `cpu`, `gpu`, `gpu:0`, `openvino`, `openvino:cpu`, `openvino:gpu`, `openvino:npu`, and `openvino:auto`. OpenVINO's `auto` mode lets the runtime choose an Intel CPU, GPU, or NPU.

The GPU build needs a supported platform with CUDA and cuDNN. OpenVINO wheels are available for Linux x86_64 and Windows; Apple Silicon uses the CPU runtime. See the [ONNX Runtime installation guide](https://onnxruntime.ai/docs/install/) for platform requirements and execution provider setup.

### Bring Your Own OCR

If you prefer to use your own OCR setup (e.g. external APIs), you can implement a custom OCR engine and pass that to the extractor instead of the built-in OCR:
```python
class CustomOCREngine:
    def read(self, image: np.ndarray) -> Sequence[TextReading]:
        # Implement your OCR logic here
        ...

extractor = ChunithmExtractor(ocr=CustomOCREngine())
```
The `TextReading` class is defined as below:

```python
@dataclass(frozen=True, slots=True)
class TextReading:
    text: str
    confidence: float
    left: float
    top: float
    right: float
    bottom: float
```
In this case, you may choose to skip installing an ONNX Runtime entirely.

## Optional API

You can set up this project as a standalone API server, and a Dockerfile is conveniently
provided for containerized deployment.

> [!NOTE]
> Each published image contains one ONNX Runtime variant. Tags end in `-cpu`, `-gpu`, or `-openvino`, and `latest` points to the CPU image. The GPU image includes the CUDA 12.9 and cuDNN runtime, but the host still needs a compatible NVIDIA driver. The OpenVINO image uses `DEVICE=openvino` to let OpenVINO choose an Intel CPU, GPU, or NPU. `DEVICE` only affects the built-in OCR engine.

```shell
# CPU
docker run --rm -p 8000:8000 ghcr.io/OWNER/REPO:latest

# NVIDIA GPU
docker run --rm --gpus all -e DEVICE=gpu -p 8000:8000 ghcr.io/OWNER/REPO:latest-gpu

# OpenVINO
docker run --rm -e DEVICE=openvino -p 8000:8000 ghcr.io/OWNER/REPO:latest-openvino
```

To build a variant locally, pass it as a build argument:

```shell
docker build --build-arg ORT_VARIANT=cpu -t chunithm-score-extractor:cpu .
docker build --build-arg ORT_VARIANT=gpu -t chunithm-score-extractor:gpu .
docker build --build-arg ORT_VARIANT=openvino -t chunithm-score-extractor:openvino .
```

The published images target `linux/amd64`. On an Apple Silicon host, add `--platform linux/amd64` to the local build command.

The API also reads `CHUNITHM_OCR_DEVICE` when `DEVICE` is not set.

`POST /extract` accepts `multipart/form-data` with an `image` file field. It returns:

```json
{
  "score": 1007584,
  "song_name": "visible title",
  "difficulty": "MAS"
}
```

Paths for this API server are denoted as follows:
| Path | Method | Description | Request | Response |
|------|--------|-------------|---------|----------|
| /extract | POST | Extract Chunithm score from an image | form-data with `image` file field | JSON with `score`, `song_name`, and `difficulty` |
| /health/live | GET | Check if the process is alive | None | 200 if alive |
| /health/ready | GET | Check if the service is ready | None | 200 if ready, 503 if not |

`/extract` can return HTTP 400, 413, 415, or 422 with a machine-readable `detail.code`.

The API server is limited to one concurrent request per process by default. You can adjust this limit with the `CHUNITHM_MAX_CONCURRENCY` env var.

Run the API server behind a reverse proxy and add authentication before exposing it beyond a private network.

## Visual pipeline walkthrough

To see how the extractor works, open [pipeline_walkthrough.ipynb](./pipeline_walkthrough.ipynb). It walks through screen localization, OCR readings, field crops, and the final result. Install the notebook dependencies alongside one ONNX Runtime extra.

For CPU, run 
```shell
uv sync --extra notebook
``` 
or 
```shell
pip install -e "[cpu,notebook]"
```
Select that environment as the notebook kernel, choose an image and device
in the first code cell, then run the cells in order. The first model
initialization may download model files.
Notebook outputs can contain your photos and OCR text; clear them before
sharing or committing the notebook if those images are private.

## Testing
Run the unit tests with the development dependencies:

```shell
python -m pip install -e ".[dev,cpu]"
python -m unittest discover -s tests -v
```