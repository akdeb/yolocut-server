"""FastAPI server for indexing and searching footage."""

from __future__ import annotations

import asyncio
import json
import mimetypes
import os
import shutil
import tempfile
import threading
import uuid
from datetime import datetime, timezone
from pathlib import Path
from urllib.parse import quote, urlencode, urlparse
from urllib.request import Request as UrlRequest
from urllib.request import urlopen
from urllib.error import HTTPError

import uvicorn
from dotenv import load_dotenv
from fastapi import BackgroundTasks, FastAPI, HTTPException, Query, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, StreamingResponse
from pydantic import BaseModel, Field, field_validator, model_validator

from .cli import _apply_overlay_to_clip, _embed_with_retry, _fmt_time

_ENV_PATH = os.path.join(os.path.expanduser("~"), ".sentrysearch", ".env")
load_dotenv(_ENV_PATH)
load_dotenv()
load_dotenv(".env.local")


app = FastAPI(
    title="SentrySearch API",
    version="0.1.0",
    description="HTTP API for indexing and searching dashcam footage.",
)

_jobs: dict[str, dict] = {}
_jobs_lock = threading.Lock()
_work_lock = threading.Lock()
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:3000",
        os.getenv("FRONTEND_ORIGIN", "https://yolocut.vercel.app"),
    ],
    allow_credentials=False,
    allow_methods=["GET", "POST", "OPTIONS"],
    allow_headers=["Content-Type"],
)


class IndexRequest(BaseModel):
    customer_id: str = Field(..., min_length=1)
    chunk_duration: int = Field(3, gt=0)
    overlap: int = Field(1, ge=0)
    preprocess: bool = True
    target_resolution: int = Field(480, gt=0)
    target_fps: int = Field(5, gt=0)
    backend: str | None = Field(None, pattern="^(gemini|local)$")
    model: str | None = None
    quantize: bool | None = None
    retry_failed: bool = False
    skip_still: bool = False
    verbose: bool = False

    @model_validator(mode="after")
    def _validate_chunking(self) -> "IndexRequest":
        if self.overlap >= self.chunk_duration:
            raise ValueError(
                "overlap must be less than chunk_duration"
            )
        return self


class SearchRequest(BaseModel):
    customer_id: str = Field(..., min_length=1)
    query: str = Field(..., min_length=1)
    results: int = Field(5, ge=1, le=100)
    output_dir: str = "~/sentrysearch_clips"
    trim: bool = True
    save_top: int | None = Field(None, ge=1)
    threshold: float = 0.41
    force_trim_low_confidence: bool = False
    overlay: bool = False
    backend: str | None = Field(None, pattern="^(gemini|local)$")
    model: str | None = None
    quantize: bool | None = None
    verbose: bool = False

    @field_validator("output_dir")
    @classmethod
    def _expand_output_dir(cls, value: str) -> str:
        return os.path.abspath(os.path.expanduser(value))


class BatchSearchItem(BaseModel):
    visual_broll: str = Field(..., min_length=1)


class BatchSearchRequest(BaseModel):
    customer_id: str = Field(..., min_length=1)
    items: list[BatchSearchItem] = Field(..., min_length=1)
    results: int = Field(5, ge=1, le=100)
    output_dir: str = "~/sentrysearch_clips"
    trim: bool = True
    save_top: int | None = Field(None, ge=1)
    threshold: float = 0.41
    force_trim_low_confidence: bool = False
    overlay: bool = False
    backend: str | None = Field(None, pattern="^(gemini|local)$")
    model: str | None = None
    quantize: bool | None = None
    verbose: bool = False

    @field_validator("output_dir")
    @classmethod
    def _expand_output_dir(cls, value: str) -> str:
        return os.path.abspath(os.path.expanduser(value))


class ImageSearchRequest(BaseModel):
    customer_id: str = Field(..., min_length=1)
    image_path: str = Field(..., description="Image file to use as the query.")
    results: int = Field(5, ge=1, le=100)
    output_dir: str = "~/sentrysearch_clips"
    trim: bool = True
    save_top: int | None = Field(None, ge=1)
    threshold: float = 0.41
    force_trim_low_confidence: bool = False
    overlay: bool = False
    backend: str | None = Field(None, pattern="^(gemini|local)$")
    model: str | None = None
    quantize: bool | None = None
    verbose: bool = False

    @field_validator("image_path", "output_dir")
    @classmethod
    def _expand_paths(cls, value: str) -> str:
        return os.path.abspath(os.path.expanduser(value))


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _set_job(job_id: str, **updates) -> None:
    with _jobs_lock:
        job = _jobs[job_id]
        job.update(updates)
        job["updated_at"] = _now()


def _add_event(job_id: str, message: str) -> None:
    with _jobs_lock:
        job = _jobs[job_id]
        job.setdefault("events", []).append({"at": _now(), "message": message})
        job["updated_at"] = _now()


def _public_job(row: dict) -> dict:
    public = dict(row)
    status = public.get("status")
    public["job_id"] = str(public.get("job_id"))
    public["done"] = status in {"succeeded", "failed"}
    public["succeeded"] = status == "succeeded"
    public["failed"] = status == "failed"
    public.setdefault("progress", 1.0 if public["done"] else 0.0)
    public.setdefault("files_done", 0)
    public.setdefault("total_files", 0)
    public.setdefault("current_file", None)
    public.setdefault("current_broll_id", None)
    public.setdefault("error", None)
    return public


def _sse_event(payload: dict, event: str = "message") -> str:
    return f"event: {event}\ndata: {json.dumps(payload)}\n\n"


def _normalize_backend(
    backend: str | None,
    model: str | None,
    *,
    for_index: bool,
) -> tuple[str, str | None]:
    from .local_embedder import detect_default_model, normalize_model_key
    from .store import detect_index

    if model is not None and backend is None:
        backend = "local"
    if for_index:
        backend = backend or "gemini"
        if backend == "local" and model is None:
            model = detect_default_model()
    else:
        if model is not None:
            model = normalize_model_key(model)
        if backend is None:
            detected_backend, detected_model = detect_index()
            backend = detected_backend or "gemini"
            if model is None:
                model = detected_model
        elif backend == "local" and model is None:
            _, model = detect_index()

    if backend == "local":
        model = normalize_model_key(model)
    return backend, model


def _clip_url(path: str | Path) -> str:
    return f"/clips?path={quote(str(path), safe='')}"


def _absolute_url(base_url: str | None, path: str | None) -> str | None:
    if base_url is None or path is None:
        return None
    return f"{base_url.rstrip('/')}{path}"


def _is_remote_url(path: str | None) -> bool:
    if not path:
        return False
    parsed = urlparse(path)
    return parsed.scheme in {"http", "https"}


def _format_result(
    result: dict,
    rank: int,
    clip_path: str | None = None,
    *,
    base_url: str | None = None,
) -> dict:
    clip_url = None
    clip_stream_url = None
    if clip_path is not None:
        clip_url = _clip_url(clip_path)
        clip_stream_url = _absolute_url(base_url, clip_url)
    elif _is_remote_url(result.get("blob_url")):
        clip_stream_url = result.get("blob_url")
    elif _is_remote_url(result.get("source_file")):
        clip_stream_url = result.get("source_file")
    return {
        "rank": rank,
        "source_file": result["source_file"],
        "source_basename": os.path.basename(result["source_file"]),
        "start_time": result["start_time"],
        "end_time": result["end_time"],
        "start_time_formatted": _fmt_time(result["start_time"]),
        "end_time_formatted": _fmt_time(result["end_time"]),
        "similarity_score": result["similarity_score"],
        "broll_id": result.get("broll_id"),
        "customer_id": result.get("customer_id"),
        "title": result.get("title"),
        "creator": result.get("creator"),
        "blob_url": result.get("blob_url"),
        "clip_path": clip_path,
        "clip_url": clip_url,
        "clip_stream_url": clip_stream_url,
    }


def _index_progress(file_idx: int, total_files: int, chunk_idx: int = 0, total_chunks: int = 0) -> float:
    if total_files <= 0:
        return 1.0
    file_progress = 0.0
    if total_chunks > 0:
        file_progress = min(chunk_idx / total_chunks, 1.0)
    return min(((file_idx - 1) + file_progress) / total_files, 1.0)


def _supabase_config() -> tuple[str, str]:
    url = os.getenv("SUPABASE_URL", "https://orjrkzierhpmkamhwejb.supabase.co")
    key = os.getenv("SUPABASE_PUBLISHABLE_KEY")
    if not key:
        raise RuntimeError("SUPABASE_PUBLISHABLE_KEY is not configured.")
    return url.rstrip("/"), key


def _supabase_headers() -> dict[str, str]:
    _url, key = _supabase_config()
    return {
        "apikey": key,
        "Authorization": f"Bearer {key}",
        "Content-Type": "application/json",
    }


def _supabase_request(
    method: str,
    path: str,
    body: dict | None = None,
    *,
    prefer: str | None = None,
) -> object:
    base_url, _key = _supabase_config()
    data = None if body is None else json.dumps(body).encode()
    headers = _supabase_headers()
    if prefer is not None:
        headers["Prefer"] = prefer
    elif method == "PATCH":
        headers["Prefer"] = "return=minimal"
    request = UrlRequest(
        f"{base_url}{path}",
        data=data,
        headers=headers,
        method=method,
    )
    with urlopen(request, timeout=30) as response:
        payload = response.read()
        if not payload:
            return None
        return json.loads(payload.decode())


def _tenant_where(
    customer_id: str,
    backend: str,
    model: str | None,
) -> dict:
    clauses = [
        {"customer_id": customer_id},
        {"embedding_backend": backend},
    ]
    if model is not None:
        clauses.append({"embedding_model": model})
    return clauses[0] if len(clauses) == 1 else {"$and": clauses}


def _fetch_unindexed_brolls(customer_id: str) -> list[dict]:
    filters = [
        ("select", "*"),
        ("customer_id", f"eq.{customer_id}"),
        ("indexed", "eq.false"),
    ]
    query = urlencode(filters)
    result = _supabase_request("GET", f"/rest/v1/brolls?{query}")
    if not isinstance(result, list):
        raise RuntimeError("Unexpected Supabase response while fetching brolls.")
    return result


def _update_broll_indexed(
    broll_id: str,
    customer_id: str,
) -> None:
    filters = [
        ("broll_id", f"eq.{broll_id}"),
        ("customer_id", f"eq.{customer_id}"),
    ]
    query = urlencode(filters)
    _supabase_request("PATCH", f"/rest/v1/brolls?{query}", {"indexed": True})


def _broll_id(row: dict) -> str:
    value = row.get("broll_id") or row.get("id")
    if value is None:
        raise RuntimeError("Supabase broll row is missing broll_id.")
    return str(value)


def _blob_authorization_header(token: str) -> str:
    token = token.strip()
    if token.lower().startswith("bearer "):
        return token
    return f"Bearer {token}"


def _download_blob(blob_url: str, directory: str, fallback_name: str) -> str:
    parsed = urlparse(blob_url)
    suffix = Path(parsed.path).suffix or ".mp4"
    target = Path(directory) / f"{fallback_name}{suffix}"
    headers = {"User-Agent": "sentrysearch-api/0.1"}
    if parsed.hostname and parsed.hostname.endswith(".blob.vercel-storage.com"):
        token = os.getenv("BLOB_READ_WRITE_TOKEN")
        if not token:
            raise RuntimeError(
                "BLOB_READ_WRITE_TOKEN is required to download private Vercel Blob videos."
            )
        headers["Authorization"] = _blob_authorization_header(token)
    request = UrlRequest(blob_url, headers=headers)
    try:
        with urlopen(request, timeout=120) as response:
            content_type = response.headers.get("content-type", "")
            with open(target, "wb") as out:
                shutil.copyfileobj(response, out)
    except HTTPError as exc:
        body = exc.read().decode("utf-8", errors="replace")
        raise RuntimeError(
            f"Failed to download blob {blob_url}: HTTP {exc.code} {body}"
        ) from exc

    size = target.stat().st_size
    if size < 1024 and "json" in content_type.lower():
        error_text = target.read_text(errors="replace")
        raise RuntimeError(
            f"Downloaded blob is not a video: content-type={content_type}, "
            f"size={size} bytes, body={error_text}"
        )
    return str(target)


def _trim_results(
    results: list[dict],
    *,
    output_dir: str,
    trim: bool,
    save_top: int | None,
    low_confidence: bool,
    force_trim_low_confidence: bool,
    overlay: bool,
) -> tuple[list[str], str | None]:
    if not results or not (trim or save_top is not None):
        return [], None
    if low_confidence and not force_trim_low_confidence:
        return [], "low_confidence"
    count = save_top if save_top is not None else 1
    if any(_is_remote_url(r.get("source_file")) for r in results[:count]):
        return [], "remote_source"

    from .trimmer import trim_top_results

    clip_paths = trim_top_results(results, output_dir, count=count)
    if overlay:
        for idx, clip_path in enumerate(clip_paths):
            result = results[idx]
            _apply_overlay_to_clip(
                clip_path,
                result["source_file"],
                result["start_time"],
                result["end_time"],
            )
    return clip_paths, None


def _run_index(job_id: str, request: IndexRequest) -> None:
    reset_embedder = None
    try:
        from .chunker import (
            _get_video_duration,
            chunk_video,
            expected_chunk_spans,
            is_still_frame_chunk,
            preprocess_chunk,
        )
        from .dlq import DeadLetterQueue
        from .embedder import get_embedder, reset_embedder as _reset_embedder
        from .store import SentryStore

        reset_embedder = _reset_embedder
        _set_job(job_id, status="running", started_at=_now())

        with _work_lock:
            backend, model = _normalize_backend(
                request.backend, request.model, for_index=True,
            )
            _set_job(job_id, backend=backend, model=model)

            store = SentryStore(backend=backend, model=model)
            brolls = _fetch_unindexed_brolls(request.customer_id)
            total_files = len(brolls)
            _set_job(job_id, total_files=total_files)

            if not brolls:
                _set_job(
                    job_id,
                    status="succeeded",
                    finished_at=_now(),
                    progress=1.0,
                    files_done=0,
                    total_files=0,
                    result={
                        "message": f"No unindexed brolls found for customer {request.customer_id}.",
                        "new_chunks": 0,
                        "new_files": 0,
                    },
                )
                return

            embedder = get_embedder(backend, model=model, quantize=request.quantize)
            dlq = DeadLetterQueue()
            new_files = 0
            new_chunks = 0
            skipped_files = 0
            skipped_chunks = 0
            failed_chunks = 0

            for file_idx, broll in enumerate(brolls, 1):
                broll_id = _broll_id(broll)
                blob_url = broll.get("blob_url")
                if not blob_url:
                    raise RuntimeError(f"Broll {broll_id} is missing blob_url.")

                title = broll.get("title") or ""
                creator = broll.get("creator") or ""
                display_name = title or os.path.basename(blob_url) or broll_id
                _set_job(
                    job_id,
                    current_broll_id=broll_id,
                    current_file=display_name,
                    current_chunk=0,
                    total_chunks_in_file=0,
                    files_done=file_idx - 1,
                    total_files=total_files,
                    progress=_index_progress(file_idx, total_files),
                )

                with tempfile.TemporaryDirectory(prefix="sentrysearch_broll_") as tmp_dir:
                    video_path = _download_blob(blob_url, tmp_dir, broll_id)

                    try:
                        duration = _get_video_duration(video_path)
                        spans = expected_chunk_spans(
                            duration,
                            chunk_duration=request.chunk_duration,
                            overlap=request.overlap,
                        )
                        chunk_key = f"{request.customer_id}:{broll_id}:{blob_url}"
                        if spans and all(
                            store.has_chunk(store.make_chunk_id(chunk_key, start))
                            for start, _ in spans
                        ):
                            skipped_files += 1
                            _update_broll_indexed(broll_id, request.customer_id)
                            _set_job(
                                job_id,
                                files_done=file_idx,
                                progress=_index_progress(file_idx + 1, total_files),
                            )
                            continue
                    except Exception:
                        pass

                    chunks = chunk_video(
                        video_path,
                        chunk_duration=request.chunk_duration,
                        overlap=request.overlap,
                    )
                    files_to_cleanup: list[str] = []
                    file_new_chunks = 0

                    for chunk_idx, chunk in enumerate(chunks, 1):
                        _set_job(
                            job_id,
                            current_broll_id=broll_id,
                            current_file=display_name,
                            current_chunk=chunk_idx,
                            total_chunks_in_file=len(chunks),
                            files_done=file_idx - 1,
                            progress=_index_progress(
                                file_idx,
                                total_files,
                                chunk_idx - 1,
                                len(chunks),
                            ),
                        )
                        chunk_key = f"{request.customer_id}:{broll_id}:{blob_url}"
                        chunk_id = store.make_chunk_id(chunk_key, chunk["start_time"])

                        if store.has_chunk(chunk_id):
                            files_to_cleanup.append(chunk["chunk_path"])
                            continue
                        if dlq.contains(chunk_id):
                            if request.retry_failed:
                                dlq.remove(chunk_id)
                            else:
                                files_to_cleanup.append(chunk["chunk_path"])
                                continue
                        if request.skip_still and is_still_frame_chunk(
                            chunk["chunk_path"], verbose=request.verbose,
                        ):
                            skipped_chunks += 1
                            files_to_cleanup.append(chunk["chunk_path"])
                            continue

                        embed_path = chunk["chunk_path"]
                        if request.preprocess:
                            embed_path = preprocess_chunk(
                                embed_path,
                                target_resolution=request.target_resolution,
                                target_fps=request.target_fps,
                            )
                            if embed_path != chunk["chunk_path"]:
                                files_to_cleanup.append(embed_path)

                        metadata = {
                            "chunk_id": chunk_id,
                            "source_file": blob_url,
                            "broll_id": broll_id,
                            "customer_id": request.customer_id,
                            "blob_url": blob_url,
                            "start_time": chunk["start_time"],
                            "end_time": chunk["end_time"],
                        }
                        if title:
                            metadata["title"] = title
                        if creator:
                            metadata["creator"] = creator

                        embedding = _embed_with_retry(
                            embedder,
                            embed_path,
                            metadata,
                            dlq,
                            verbose=request.verbose,
                        )
                        files_to_cleanup.append(chunk["chunk_path"])
                        if embedding is None:
                            failed_chunks += 1
                            continue

                        store.add_chunk(chunk_id, embedding, metadata)
                        file_new_chunks += 1

                    for path in files_to_cleanup:
                        try:
                            os.unlink(path)
                        except OSError:
                            pass
                    if chunks:
                        shutil.rmtree(os.path.dirname(chunks[0]["chunk_path"]), ignore_errors=True)

                if file_new_chunks:
                    new_files += 1
                    new_chunks += file_new_chunks
                    _update_broll_indexed(broll_id, request.customer_id)

                _set_job(
                    job_id,
                    files_done=file_idx,
                    progress=_index_progress(file_idx + 1, total_files),
                )

            _set_job(
                job_id,
                status="succeeded",
                finished_at=_now(),
                progress=1.0,
                files_done=total_files,
                total_files=total_files,
                result={
                    "new_chunks": new_chunks,
                    "new_files": new_files,
                    "skipped_files": skipped_files,
                    "skipped_still_chunks": skipped_chunks,
                    "failed_chunks": failed_chunks,
                },
            )
    except Exception as exc:
        _set_job(job_id, status="failed", finished_at=_now(), error=str(exc))
    finally:
        if reset_embedder is not None:
            reset_embedder()


def _search(
    *,
    customer_id: str,
    query: str | None,
    image_path: str | None,
    results_count: int,
    output_dir: str,
    trim: bool,
    save_top: int | None,
    threshold: float,
    force_trim_low_confidence: bool,
    overlay: bool,
    backend: str | None,
    model: str | None,
    quantize: bool | None,
    verbose: bool,
    base_url: str | None = None,
) -> dict:
    from .embedder import get_embedder, reset_embedder
    from .search import search_footage, search_footage_by_image
    from .store import SentryStore, detect_index

    try:
        with _work_lock:
            backend, model = _normalize_backend(backend, model, for_index=False)
            store = SentryStore(backend=backend, model=model)
            stats = store.get_stats()
            if stats["total_chunks"] == 0:
                detected_backend, detected_model = detect_index()
                detail = "No indexed footage found. Run POST /index first."
                if detected_backend and detected_backend != backend:
                    detail = (
                        f"No footage indexed with backend {backend}. "
                        f"Your index uses {detected_backend}."
                    )
                elif detected_model and detected_model != model:
                    detail = (
                        f"No footage indexed with model {model}. "
                        f"Your index uses {detected_model}."
                    )
                raise HTTPException(status_code=404, detail=detail)

            get_embedder(backend, model=model, quantize=quantize)
            if save_top is not None and save_top > results_count:
                results_count = save_top
            where = _tenant_where(customer_id, backend, model)

            raw_results = (
                search_footage(
                    query or "",
                    store,
                    n_results=results_count,
                    verbose=verbose,
                    where=where,
                )
                if image_path is None
                else search_footage_by_image(
                    image_path,
                    store,
                    n_results=results_count,
                    verbose=verbose,
                    where=where,
                )
            )
            best_score = raw_results[0]["similarity_score"] if raw_results else None
            low_confidence = best_score is not None and best_score < threshold
            clip_paths, trim_skipped_reason = _trim_results(
                raw_results,
                output_dir=output_dir,
                trim=trim,
                save_top=save_top,
                low_confidence=low_confidence,
                force_trim_low_confidence=force_trim_low_confidence,
                overlay=overlay,
            )

            formatted = []
            for idx, result in enumerate(raw_results, 1):
                clip_path = clip_paths[idx - 1] if idx <= len(clip_paths) else None
                formatted.append(_format_result(result, idx, clip_path, base_url=base_url))

            return {
                "query": query,
                "image_path": image_path,
                "customer_id": customer_id,
                "backend": backend,
                "model": model,
                "threshold": threshold,
                "best_score": best_score,
                "low_confidence": low_confidence,
                "trim_skipped_reason": trim_skipped_reason,
                "results": formatted,
            }
    finally:
        reset_embedder()


@app.get("/health")
def health() -> dict:
    return {"ok": True}


@app.get("/stats")
def stats() -> dict:
    from .store import SentryStore, detect_index

    backend, model = detect_index()
    store = SentryStore(backend=backend or "gemini", model=model)
    return {"backend": backend, "model": model, **store.get_stats()}


@app.post("/index", status_code=202)
def index(request: IndexRequest, background_tasks: BackgroundTasks) -> dict:
    job_id = uuid.uuid4().hex
    with _jobs_lock:
        _jobs[job_id] = {
            "job_id": job_id,
            "status": "queued",
            "created_at": _now(),
            "updated_at": _now(),
            "customer_id": request.customer_id,
            "progress": 0.0,
            "files_done": 0,
            "total_files": 0,
            "current_broll_id": None,
            "current_file": None,
            "current_chunk": 0,
            "total_chunks_in_file": 0,
            "error": None,
            "events": [],
        }
    background_tasks.add_task(_run_index, job_id, request)
    return {
        "job_id": job_id,
        "status": "queued",
    }


@app.get("/jobs/{job_id}")
def job(job_id: str) -> dict:
    with _jobs_lock:
        row = _jobs.get(job_id)
        if row is None:
            raise HTTPException(status_code=404, detail="Job not found")
        return _public_job(row)


@app.get("/jobs/{job_id}/events")
async def job_events(job_id: str):
    with _jobs_lock:
        if job_id not in _jobs:
            raise HTTPException(status_code=404, detail="Job not found")

    async def stream():
        while True:
            with _jobs_lock:
                payload = _public_job(_jobs[job_id])
            event = "complete" if payload["done"] else "progress"
            yield _sse_event(payload, event=event)
            if payload["done"]:
                break
            await asyncio.sleep(1)

    return StreamingResponse(
        stream(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
        },
    )


@app.post("/search")
def search(request: SearchRequest, http_request: Request) -> dict:
    return _search(
        customer_id=request.customer_id,
        query=request.query,
        image_path=None,
        results_count=request.results,
        output_dir=request.output_dir,
        trim=request.trim,
        save_top=request.save_top,
        threshold=request.threshold,
        force_trim_low_confidence=request.force_trim_low_confidence,
        overlay=request.overlay,
        backend=request.backend,
        model=request.model,
        quantize=request.quantize,
        verbose=request.verbose,
        base_url=str(http_request.base_url),
    )


@app.post("/search/batch")
def search_batch(request: BatchSearchRequest, http_request: Request) -> dict:
    rows = []
    for index, item in enumerate(request.items):
        result = _search(
            customer_id=request.customer_id,
            query=item.visual_broll,
            image_path=None,
            results_count=request.results,
            output_dir=request.output_dir,
            trim=request.trim,
            save_top=request.save_top,
            threshold=request.threshold,
            force_trim_low_confidence=request.force_trim_low_confidence,
            overlay=request.overlay,
            backend=request.backend,
            model=request.model,
            quantize=request.quantize,
            verbose=request.verbose,
            base_url=str(http_request.base_url),
        )
        rows.append(
            {
                "index": index,
                "visual_broll": item.visual_broll,
                "query": result["query"],
                "results": result["results"],
            }
        )
    return {"rows": rows}


@app.post("/search/image")
def search_by_image(request: ImageSearchRequest, http_request: Request) -> dict:
    if not os.path.isfile(request.image_path):
        raise HTTPException(status_code=404, detail=f"Image not found: {request.image_path}")
    return _search(
        customer_id=request.customer_id,
        query=None,
        image_path=request.image_path,
        results_count=request.results,
        output_dir=request.output_dir,
        trim=request.trim,
        save_top=request.save_top,
        threshold=request.threshold,
        force_trim_low_confidence=request.force_trim_low_confidence,
        overlay=request.overlay,
        backend=request.backend,
        model=request.model,
        quantize=request.quantize,
        verbose=request.verbose,
        base_url=str(http_request.base_url),
    )


@app.get("/clips")
def clips(path: str = Query(..., description="Absolute path returned as clip_path.")):
    clip_path = Path(os.path.abspath(os.path.expanduser(path)))
    if not clip_path.is_file():
        raise HTTPException(status_code=404, detail="Clip not found")
    media_type = mimetypes.guess_type(clip_path.name)[0] or "application/octet-stream"
    return FileResponse(str(clip_path), media_type=media_type, filename=clip_path.name)


def main() -> None:
    uvicorn.run("sentrysearch.api:app", host="0.0.0.0", port=8080, reload=False)


if __name__ == "__main__":
    main()
