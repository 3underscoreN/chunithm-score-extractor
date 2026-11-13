# CHUNITHM Score Extractor

A Python library and optional self-hosted HTTP API for extracting the current-play score, visible song title, and difficulty from CHUNITHM result photos.

## Requirements

- Python 3.13
- numpy, OpenCV, Pillow
- PaddleOCR 3.7.0 with ONNX Runtime (CPU or GPU build). It is possible to bring your own custom OCR engine, which we will mention below.
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

The first initialization may acquire PaddleOCR model files.

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

`uv sync` installs the CPU runtime by default. To switch the same uv-managed
environment to the GPU runtime, disable the default CPU group and opt in to
the GPU extra:

```shell
uv sync --no-group onnx-cpu --extra gpu
```

To switch back, run `uv sync` again. uv removes the previously selected runtime
when syncing. Do not install both runtimes in the same environment: they share
the `onnxruntime` import namespace.

For pip or other installers, choose exactly one extra, preferably in a fresh environment:

```shell
python -m pip install ".[cpu]"
# Or, instead:
python -m pip install ".[gpu]"
```

Plain `pip install .` does not install an ONNX Runtime. Pip extras only add
dependencies and do not uninstall a previously installed runtime. Therefore, it is suggested to use a fresh
environment when switching. 

The GPU build requires a supported platform and CUDA/cuDNN setup. See [ONNX Runtime installation guide](https://onnxruntime.ai/docs/install/).

### Bring Your Own OCR

If you prefer to use your own OCR setup (e.g. external APIs), you can implement a custom OCR engine and pass that to the extractor instead of the built-in OCR:
```python
class CustomOCREngine:
    def read(self, image: np.ndarray) -> Sequence[TextReading]:
        # Implement your OCR logic here
        ...

extractor = ChunithmExtractor(ocr_engine=CustomOCREngine())
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
> The Dockerfile defaults to CPU inference. For GPU inference, select the GPU
> install line instead of the CPU line and provide the required GPU runtime.
> You also need to set `CHUNITHM_OCR_DEVICE=gpu`, as well as passing GPU with `--gpus` flag to the `docker run` command for GPU inference.

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

Paths for this API server are denoted as follows:
| Path | Method | Description | Request | Response |
|------|--------|-------------|---------|----------|
| /extract | POST | Extract Chunithm score from an image | form-data with `image` file field | JSON with `score`, `song_name`, and `difficulty` |
| /health/live | GET | Check if the process is alive | None | 200 if alive |
| /health/ready | GET | Check if the service is ready | None | 200 if ready, 503 if not |

Possible error code for /extract include HTTP 400, 413, 415, or 422 with a machine-readable `detail.code`. 

The API server is limited to one concurrent request per process by default. You can adjust this limit with the `CHUNITHM_MAX_CONCURRENCY` env var.

It is suggested to run the API server behind a reverse proxy and implement proper authentication before exposing it beyond a private network.

## Visual pipeline walkthrough

For people that are curious on how the extractor works under the hood, a walkthrough Jupyter notebook is provided. 
Open [pipeline_walkthrough.ipynb](./pipeline_walkthrough.ipynb) to inspect screen localization, OCR readings, field crops, and the
final result. Install the notebook dependencies alongside **one** ONNX Runtime.

For CPU, run ```shell
uv sync --extra notebook
``` 
or 
```shell
pip install -e "[cpu,notebook]"
```
Select that environment as thenotebook kernel, choose an image and device 
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