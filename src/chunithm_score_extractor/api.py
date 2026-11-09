import asyncio
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
import logging
import os
from typing import Any, Awaitable, Callable, Protocol

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.concurrency import run_in_threadpool
from pydantic import BaseModel
from starlette.responses import JSONResponse

from .errors import (
    ConflictingReadingsError,
    ExtractionError,
    InvalidImageError,
    ProcessingError,
    UncertainResultError,
    UnsupportedLayoutError,
)
from .extractor import ChunithmExtractor
from .image import ImageLimits
from .models import ExtractionResult


MAX_UPLOAD_BYTES = ImageLimits().max_bytes
MAX_REQUEST_BYTES = MAX_UPLOAD_BYTES + 64 * 1024
logger = logging.getLogger(__name__)


class ImageExtractor(Protocol):
    def extract(self, image_bytes: bytes) -> ExtractionResult: ...


class ResultResponse(BaseModel):
    score: int
    song_name: str
    difficulty: str


class _RequestBodyTooLarge(Exception):
    pass


class RequestBodyLimitMiddleware:
    def __init__(self, app: Any, max_bytes: int) -> None:
        self.app = app
        self.max_bytes = max_bytes

    async def __call__(
        self,
        scope: dict[str, Any],
        receive: Callable[[], Awaitable[dict[str, Any]]],
        send: Callable[[dict[str, Any]], Awaitable[None]],
    ) -> None:
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return

        headers = dict(scope.get("headers", ()))
        content_length = headers.get(b"content-length")
        if content_length is not None:
            try:
                if int(content_length) > self.max_bytes:
                    await _send_too_large(scope, receive, send)
                    return
            except ValueError:
                response = JSONResponse(
                    status_code=400,
                    content={"detail": {"code": "invalid_content_length"}},
                )
                await response(scope, receive, send)
                return

        received = 0

        async def limited_receive() -> dict[str, Any]:
            nonlocal received
            message = await receive()
            if message["type"] == "http.request":
                received += len(message.get("body", b""))
                if received > self.max_bytes:
                    raise _RequestBodyTooLarge
            return message

        try:
            await self.app(scope, limited_receive, send)
        except _RequestBodyTooLarge:
            await _send_too_large(scope, receive, send)


async def _send_too_large(
    scope: dict[str, Any],
    receive: Callable[[], Awaitable[dict[str, Any]]],
    send: Callable[[dict[str, Any]], Awaitable[None]],
) -> None:
    response = JSONResponse(
        status_code=413,
        content={"detail": {"code": "image_too_large"}},
    )
    await response(scope, receive, send)


def create_app(extractor: ImageExtractor | None = None, max_concurrency: int = 1) -> FastAPI:
    if max_concurrency < 1:
        raise ValueError("max_concurrency must be at least 1.")

    instance: ImageExtractor | None = extractor
    initialization_error: ExtractionError | None = None
    semaphore = asyncio.Semaphore(max_concurrency)

    @asynccontextmanager
    async def lifespan(_: FastAPI) -> AsyncIterator[None]:
        nonlocal instance, initialization_error
        if instance is None:
            try:
                device = os.environ.get("CHUNITHM_OCR_DEVICE")
                instance = await run_in_threadpool(ChunithmExtractor, device=device)
            except ExtractionError as exc:
                initialization_error = exc
                logger.error("OCR model initialization failed: %s", exc)
        yield

    application = FastAPI(title="CHUNITHM Result Extractor", lifespan=lifespan)
    application.add_middleware(
        RequestBodyLimitMiddleware,
        max_bytes=MAX_REQUEST_BYTES,
    )

    @application.get("/health/live")
    async def live() -> dict[str, str]:
        return {"status": "live"}

    @application.get("/health/ready")
    async def ready() -> dict[str, str]:
        if instance is None:
            code = (
                initialization_error.code
                if initialization_error is not None
                else "model_not_ready"
            )
            raise HTTPException(status_code=503, detail={"code": code})
        return {"status": "ready"}

    @application.post("/extract", response_model=ResultResponse)
    async def extract(file: UploadFile = File(..., alias="image")) -> ResultResponse:
        try:
            if file.content_type not in {"image/jpeg", "image/png", "image/webp"}:
                raise HTTPException(
                    status_code=415,
                    detail={"code": "unsupported_media_type"},
                )
            data = await file.read(MAX_UPLOAD_BYTES + 1)
            if len(data) > MAX_UPLOAD_BYTES:
                raise HTTPException(status_code=413, detail={"code": "image_too_large"})
            if not data:
                raise HTTPException(status_code=400, detail={"code": "invalid_image"})
            async with semaphore:
                if instance is None:
                    code = (
                        initialization_error.code
                        if initialization_error is not None
                        else "model_not_ready"
                    )
                    raise HTTPException(status_code=503, detail={"code": code})
                result = await run_in_threadpool(instance.extract, data)
            return ResultResponse(
                score=result.score,
                song_name=result.song_name,
                difficulty=result.difficulty.value,
            )
        except InvalidImageError as exc:
            raise HTTPException(
                status_code=400,
                detail={"code": exc.code},
            ) from exc
        except (
            UnsupportedLayoutError,
            UncertainResultError,
            ConflictingReadingsError,
        ) as exc:
            raise HTTPException(
                status_code=422,
                detail={"code": exc.code},
            ) from exc
        except ProcessingError as exc:
            raise HTTPException(
                status_code=500,
                detail={"code": exc.code},
            ) from exc
        except ExtractionError as exc:
            raise HTTPException(
                status_code=422,
                detail={"code": exc.code},
            ) from exc
        finally:
            await file.close()

    return application


app = create_app(max_concurrency=int(os.environ.get("CHUNITHM_MAX_CONCURRENCY", "1")))
