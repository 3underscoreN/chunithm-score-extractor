# syntax=docker/dockerfile:1.7

# Build one ONNX Runtime variant at a time:
#   docker build --build-arg ORT_VARIANT=cpu      -t chunithm:cpu .
#   docker build --build-arg ORT_VARIANT=gpu      -t chunithm:gpu .
#   docker build --build-arg ORT_VARIANT=openvino -t chunithm:openvino .
ARG ORT_VARIANT=cpu

FROM astral/uv:python3.13-trixie-slim AS base-cpu

FROM base-cpu AS base-openvino

FROM nvidia/cuda:12.9.1-cudnn-runtime-ubuntu24.04 AS base-gpu

FROM base-${ORT_VARIANT} AS final

ARG ORT_VARIANT=cpu

# uv is the only tool copied into the GPU base image; the CPU base already has it.
COPY --from=base-cpu /usr/local/bin/uv /usr/local/bin/uvx /usr/local/bin/

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    UV_PYTHON_INSTALL_DIR=/opt/python \
    PADDLE_PDX_CACHE_HOME=/opt/models/paddlex \
    PATH="/app/.venv/bin:$PATH"

RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        ca-certificates \
        libgl1 \
        libglib2.0-0 \
        libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# Let uv provide the same Python version on every base image.
RUN uv python install 3.13

WORKDIR /app

COPY pyproject.toml uv.lock README.md LICENSE ./
COPY src ./src

# The onnx-cpu dependency group is only for local uv sync; select exactly one
# runtime through the variant extra here.
RUN uv sync --frozen --no-dev --no-group onnx-cpu \
    --extra api --extra "${ORT_VARIANT}"

# uv sync installs into /app/.venv, so put its entry points (uvicorn, ...) on PATH.
ENV PATH="/app/.venv/bin:$PATH"

# Add user
RUN groupadd --system app \
    && useradd --system --gid app --create-home app \
    && mkdir -p /opt/models \
    && chown -R app:app /app /opt/python /opt/models

EXPOSE 8000

USER app:app

CMD ["uvicorn", "chunithm_score_extractor.api:app", "--host", "0.0.0.0", "--port", "8000"]
