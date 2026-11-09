# CHUNITHM Score Extractor

A Python library and optional self-hosted HTTP API for extracting the current-play score, visible song title, and difficulty from CHUNITHM result photos.

## Requirements

- Python 3.13
- numpy, OpenCV, Pillow
- PaddleOCR 3.4.1 with PaddlePaddle 3.3.1 (CPU/GPU build). It is possible to bring your own custom OCR engine, which we will mention below.
- Docker (optional)

Install the library and OCR runtime:

```shell
python -m pip install ".[ocr]"
```

PaddleOCR and PaddlePaddle are licensed under [Apache-2.0 license](https://github.com/PaddlePaddle/Paddle/blob/develop/LICENSE).

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

The first initialization may acquire PaddleOCR model files.

Success returns an `ExtractionResult`. See below for the available fields.

Invalid images, unsupported layouts, uncertain readings, conflicting readings, model initialization errors, and processing failures raise errors. See the `chunithm_score_extractor.errors` module for the full list of exceptions.

The library accepts JPEG, PNG, and WebP bytes, with a default 15 MiB input limit, 25-megapixel decoded-pixel limit, and 10,000-pixel dimension limit. The limit can be adjusted if necessary.

> [!IMPORTANT]
> There's no guarantee that OCR would be able to parse 100% matching text from the images. For example, for really long song names, or for weird character sequences, the OCR may produce incorrect or incomplete readings. Fuzzy matching to a known song database, and appropriate error handling should be applied when using the extracted results.

### Some API I guess?
```python
class ChunithmExtractor:
    def __init__(
      self, 
      ocr_engine: OcrEngine | None = None, # Custom OCR engine to use instead of the built-in one, see below for details
      image_limits: ImageLimits = ImageLimits(), # Limits for input image size and dimensions
      confidence_threshold: float = CONFIDENCE_THRESHOLD, # Minimum confidence for OCR readings to be considered valid, defaults to 0.75
      *,
      device: str | None = None # Device to run the OCR on, e.g., "cpu" or "gpu:0", defaults to None. NOTE: Only used when the built-in OCR engine is used
    ):
        ...
    
    def extract(self, image_bytes: bytes) -> ExtractionResult:
        ...
```

```python
class OcrEngine(Protocol):
    def read(self, image: np.ndarray) -> Sequence[TextReading]:
        ...
```

Some data classes and enums used by the API are defined below.
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

### Using the OCR extras

The `ocr` extra installs the CPU build. In order to speed up the OCR process, a separate environment with the `ocr-gpu` is available.

Use a separate environment for GPU. Do not install both extras or both Paddle builds in the same environment.

When installing from this checkout, `uv` uses the configured Paddle CUDA 12.6
index automatically. With `pip`, the index must be provided explicitly:

```powershell
python -m pip install ".[ocr-gpu]" --extra-index-url "https://www.paddlepaddle.org.cn/packages/stable/cu126/"
```

Package indexes are not embedded in published Python metadata. `uv sync` and
`uv run` enforce the project's conflicting extras, but `uv pip install` and
`pip` do not. Choose only one extra with either pip interface.

Choose the CUDA build and prerequisites for your system using the
[PaddlePaddle installation guide](https://www.paddlepaddle.org.cn/documentation/docs/en/install/pip/windows-pip_en.html).
On the tested Windows RTX 2060 SUPER, CUDA 12.6 inference ran, but Paddle
reported that its binary was built with cuDNN 9.9 while its wheel declared and
installed cuDNN 9.5. Treat this configuration as experimental until upstream
compatibility is resolved; do not ignore the warning for production use.
The supplied Dockerfile installs the CPU runtime; GPU deployment needs a
separately configured GPU-enabled image and NVIDIA container runtime.

### Bring Your Own OCR

If you prefer to use your own OCR setup (or even external API), you can implement a custom OCR engine and pass that to the extractor instead of the built-in OCR:
```python
class CustomOCREngine:
    def read(self, image: np.ndarray) -> Sequence[TextReading]:
        # Implement your OCR logic here
        ...

extractor = ChunithmExtractor(ocr_engine=CustomOCREngine())
```
In this case, you may choose to skip installing the built-in OCR extras entirely.

## Optional API

You can set up this project as a standalone API server, and a Dockerfile is conveniently
provided for containerized deployment.

> [!NOTE]
> Review the Dockerfile before deployment. If you want CPU inference, make sure to adjust the build accordingly.

```shell
docker build -t chunithm-score-extractor .
docker run --rm -p 8000:8000 chunithm-score-extractor
```

`POST /extract` accepts `multipart/form-data` with an `image` file field. It returns:

```json
{
  "score": 1007584,
  "song_name": "visible title",
  "difficulty": "MAS"
}
```

Expected input/extraction failures return HTTP 400, 413, 415, or 422 with a machine-readable `detail.code`. Internal service failures return HTTP 500. `/health/live` reports process liveness; `/health/ready` returns 503 until the OCR model has initialized. Inference is run outside the async event loop and limited to one concurrent request per process by default; set `CHUNITHM_MAX_CONCURRENCY` to adjust this per-worker limit. Put the API behind an appropriately trusted reverse proxy and authentication layer before exposing it beyond a private network.

Set `CHUNITHM_OCR_DEVICE=cpu`, `gpu`, or `gpu:0` to choose the API inference
device. Without it, PaddleOCR selects the device. Each worker loads its own
model and has its own concurrency limit.

## Visual pipeline walkthrough

For people that are curious on how the extractor works under the hood, a walkthrough Jupyter notebook is provided. 
Open [pipeline_walkthrough.ipynb](./pipeline_walkthrough.ipynb) to inspect screen localization, OCR readings, field crops, and the
final result. Install the notebook dependencies alongside **one** OCR extra only.

For CPU, install `.[ocr,notebook]` instead. Select that environment as the
notebook kernel, choose an image and device in the first code cell, then run
the cells in order. The first model initialization may download model files.
Notebook outputs can contain your photos and OCR text; clear them before
sharing or committing the notebook if those images are private.

## Testing
Run the unit tests with the development dependencies:

```shell
python -m pip install -e ".[dev]"
python -m unittest discover -s tests -v
```