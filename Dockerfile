FROM python:3.13-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PADDLE_PDX_CACHE_HOME=/opt/paddle/.paddlex

RUN apt-get update \
    && apt-get install -y --no-install-recommends libgl1 libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY pyproject.toml README.md LICENSE ./
COPY src ./src

RUN python -m pip install --no-cache-dir ".[api,ocr-gpu]"
RUN groupadd --system app \
    && useradd --system --gid app --create-home app \
    && mkdir -p /opt/paddle \
    && chown -R app:app /app /opt/paddle

EXPOSE 8000

USER app:app

CMD ["uvicorn", "chunithm_score_extractor.api:app", "--host", "0.0.0.0", "--port", "8000"]
