FROM python:3.13-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PADDLE_PDX_CACHE_HOME=/opt/paddle/.paddlex

RUN apt-get update \
    && apt-get install -y --no-install-recommends libgl1 libglib2.0-0 libgomp1 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY pyproject.toml README.md LICENSE ./
COPY src ./src

RUN python -m pip install --no-cache-dir ".[api]"
# Comment out the lines you wish to skip
# Since we are not using uv here, extra index URL for GPU is mandatory
# RUN python -m pip install --no-cache-dir ".[ocr]"
RUN python -m pip install --no-cache-dir ".[ocr-gpu]" \
    --extra-index-url "https://www.paddlepaddle.org.cn/packages/stable/cu126/"

# Add user
RUN groupadd --system app \
    && useradd --system --gid app --create-home app \
    && mkdir -p /opt/paddle \
    && chown -R app:app /app /opt/paddle

EXPOSE 8000

USER app:app

CMD ["uvicorn", "chunithm_score_extractor.api:app", "--host", "0.0.0.0", "--port", "8000"]
