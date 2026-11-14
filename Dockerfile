FROM astral/uv:python3.13-trixie-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PADDLE_PDX_CACHE_HOME=/opt/paddle/.paddlex

RUN apt-get update \
    && apt-get install -y --no-install-recommends libgl1 libglib2.0-0 libgomp1 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY pyproject.toml README.md LICENSE ./
COPY src ./src

# Choose exactly one ONNX Runtime variant.
RUN uv sync --extra api
# RUN uv sync --no-group onnx-cpu --extra api --extra gpu

# uv sync installs into /app/.venv, so put its entry points (uvicorn, ...) on PATH.
ENV PATH="/app/.venv/bin:$PATH"

# Add user
RUN groupadd --system app \
    && useradd --system --gid app --create-home app \
    && mkdir -p /opt/paddle \
    && chown -R app:app /app /opt/paddle

EXPOSE 8000

USER app:app

CMD ["uvicorn", "chunithm_score_extractor.api:app", "--host", "0.0.0.0", "--port", "8000"]
