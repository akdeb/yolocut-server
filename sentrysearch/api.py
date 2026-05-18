"""FastAPI server for indexing and searching footage."""

from __future__ import annotations

import asyncio
import json
import mimetypes
import os
import shutil
import threading
import uuid
from datetime import datetime, timezone
from pathlib import Path
from urllib.parse import quote

import uvicorn
from fastapi import BackgroundTasks, FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, StreamingResponse
from pydantic import BaseModel, Field, field_validator, model_validator

from .cli import _apply_overlay_to_clip, _embed_with_retry, _fmt_time


app = FastAPI(
    title="SentrySearch API",
    version="0.1.0",
    description="HTTP API for indexing and searching dashcam footage.",
)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)

_jobs: dict[str, dict] = {}
_jobs_lock = threading.Lock()
_work_lock = threading.Lock()
PROJECT_ROOT = Path(__file__).resolve().parent.parent
VIDEOS_DIR = PROJECT_ROOT / "videos"


class IndexRequest(BaseModel):
    path: str | None = Field(
        None,
        description="Video file or directory to index. Defaults to the repo videos folder.",
    )
    chunk_duration: int = Field(30, gt=0)
    overlap: int = Field(5, ge=0)
    preprocess: bool = True
    target_resolution: int = Field(480, gt=0)
    target_fps: int = Field(5, gt=0)
    backend: str | None = Field(None, pattern="^(gemini|local)$")
    model: str | None = None
    quantize: bool | None = None
    retry_failed: bool = False
    skip_still: bool = False
    verbose: bool = False

    @field_validator("path")
    @classmethod
    def _expand_path(cls, value: str | None) -> str | None:
        if value is None:
            return None
        return os.path.abspath(os.path.expanduser(value))

    @model_validator(mode="after")
    def _validate_chunking(self) -> "IndexRequest":
        if self.overlap >= self.chunk_duration:
            raise ValueError(
                "overlap must be less than chunk_duration"
            )
        return self


class SearchRequest(BaseModel):
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


def _public_job(job_id: str) -> dict:
    with _jobs_lock:
        job = _jobs.get(job_id)
        if job is None:
            raise HTTPException(status_code=404, detail="Job not found")
        public = dict(job)
    status = public.get("status")
    public["done"] = status in {"succeeded", "failed"}
    public["succeeded"] = status == "succeeded"
    public["failed"] = status == "failed"
    public.setdefault("progress", 1.0 if public["done"] else 0.0)
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


def _format_result(result: dict, rank: int, clip_path: str | None = None) -> dict:
    clip_url = None
    if clip_path is not None:
        clip_url = _clip_url(clip_path)
    return {
        "rank": rank,
        "source_file": result["source_file"],
        "source_basename": os.path.basename(result["source_file"]),
        "start_time": result["start_time"],
        "end_time": result["end_time"],
        "start_time_formatted": _fmt_time(result["start_time"]),
        "end_time_formatted": _fmt_time(result["end_time"]),
        "similarity_score": result["similarity_score"],
        "clip_path": clip_path,
        "clip_url": clip_url,
    }


def _indexed_source_files(backend: str | None = None, model: str | None = None) -> set[str]:
    from .store import SentryStore, detect_index

    if backend is None:
        backend, detected_model = detect_index()
        if model is None:
            model = detected_model
    store = SentryStore(backend=backend or "gemini", model=model)
    return {str(Path(path).resolve()) for path in store.get_stats()["source_files"]}


def _list_broll_videos(indexed_sources: set[str] | None = None) -> list[dict]:
    from .chunker import SUPPORTED_VIDEO_EXTENSIONS

    if not VIDEOS_DIR.is_dir():
        return []
    indexed_sources = indexed_sources or set()

    videos = [
        path
        for path in VIDEOS_DIR.rglob("*")
        if path.is_file() and path.suffix.lower() in SUPPORTED_VIDEO_EXTENSIONS
    ]
    videos.sort(key=lambda path: path.relative_to(VIDEOS_DIR).as_posix().lower())

    manifest = []
    for path in videos:
        stat = path.stat()
        absolute_path = path.resolve()
        indexed = str(absolute_path) in indexed_sources
        relative_path = path.relative_to(VIDEOS_DIR).as_posix()
        manifest.append(
            {
                "name": path.stem,
                "filename": path.name,
                "relative_path": relative_path,
                "path": str(absolute_path),
                "url": _clip_url(absolute_path),
                "indexed": indexed,
                "size_bytes": stat.st_size,
                "modified_at": datetime.fromtimestamp(
                    stat.st_mtime, timezone.utc
                ).isoformat(),
            }
        )
    return manifest


def _filter_unindexed_videos(
    videos: list[str],
    indexed_sources: set[str],
    *,
    job_id: str | None = None,
) -> tuple[list[str], int]:
    videos_to_index = []
    skipped_files = 0
    for video_path in videos:
        abs_path = os.path.abspath(video_path)
        if abs_path in indexed_sources:
            skipped_files += 1
            if job_id is not None:
                _add_event(
                    job_id,
                    f"Skipping {os.path.basename(video_path)} (already indexed)",
                )
        else:
            videos_to_index.append(video_path)
    return videos_to_index, skipped_files


def _index_progress(file_idx: int, total_files: int, chunk_idx: int = 0, total_chunks: int = 0) -> float:
    if total_files <= 0:
        return 1.0
    file_progress = 0.0
    if total_chunks > 0:
        file_progress = min(chunk_idx / total_chunks, 1.0)
    return min(((file_idx - 1) + file_progress) / total_files, 1.0)


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

    from .trimmer import trim_top_results

    count = save_top if save_top is not None else 1
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
    from .chunker import (
        SUPPORTED_VIDEO_EXTENSIONS,
        _get_video_duration,
        chunk_video,
        expected_chunk_spans,
        is_still_frame_chunk,
        preprocess_chunk,
        scan_directory,
    )
    from .dlq import DeadLetterQueue
    from .embedder import get_embedder, reset_embedder
    from .store import SentryStore

    _set_job(job_id, status="running", started_at=_now())

    try:
        with _work_lock:
            if request.overlap >= request.chunk_duration:
                raise ValueError(
                    f"overlap ({request.overlap}s) must be less than "
                    f"chunk_duration ({request.chunk_duration}s)."
                )
            if not os.path.exists(request.path):
                raise FileNotFoundError(f"Path not found: {request.path}")

            backend, model = _normalize_backend(
                request.backend, request.model, for_index=True,
            )
            _set_job(job_id, backend=backend, model=model)

            videos = [request.path] if os.path.isfile(request.path) else scan_directory(request.path)
            if not videos:
                supported = ", ".join(SUPPORTED_VIDEO_EXTENSIONS)
                message = f"No supported video files found ({supported})."
                _add_event(job_id, message)
                _set_job(
                    job_id,
                    status="succeeded",
                    finished_at=_now(),
                    progress=1.0,
                    result={"message": message, "new_chunks": 0, "new_files": 0},
                )
                return

            store = SentryStore(backend=backend, model=model)
            indexed_sources = {
                os.path.abspath(source_file)
                for source_file in store.get_stats()["source_files"]
            }
            videos_to_index, skipped_files = _filter_unindexed_videos(
                videos,
                indexed_sources,
                job_id=job_id,
            )

            if not videos_to_index:
                stats = store.get_stats()
                message = "All supported video files are already indexed."
                _add_event(job_id, message)
                _set_job(
                    job_id,
                    status="succeeded",
                    finished_at=_now(),
                    progress=1.0,
                    files_done=len(videos),
                    total_files=len(videos),
                    result={
                        "message": message,
                        "new_chunks": 0,
                        "new_files": 0,
                        "skipped_files": skipped_files,
                        "skipped_still_chunks": 0,
                        "failed_chunks": 0,
                        "total_chunks": stats["total_chunks"],
                        "unique_source_files": stats["unique_source_files"],
                        "source_files": stats["source_files"],
                    },
                )
                return

            embedder = get_embedder(backend, model=model, quantize=request.quantize)
            dlq = DeadLetterQueue()
            total_files = len(videos_to_index)
            new_files = 0
            new_chunks = 0
            skipped_chunks = 0
            dlq_chunks = 0

            for file_idx, video_path in enumerate(videos_to_index, 1):
                abs_path = os.path.abspath(video_path)
                basename = os.path.basename(video_path)
                _set_job(
                    job_id,
                    current_file=abs_path,
                    files_done=file_idx - 1,
                    total_files=total_files,
                    progress=_index_progress(file_idx, total_files),
                )

                try:
                    duration = _get_video_duration(abs_path)
                    expected_spans = expected_chunk_spans(
                        duration,
                        chunk_duration=request.chunk_duration,
                        overlap=request.overlap,
                    )
                    if expected_spans and all(
                        store.has_chunk(store.make_chunk_id(abs_path, start))
                        for start, _ in expected_spans
                    ):
                        _add_event(job_id, f"Skipping {basename} (already indexed)")
                        continue
                except Exception:
                    pass

                chunks = chunk_video(
                    abs_path,
                    chunk_duration=request.chunk_duration,
                    overlap=request.overlap,
                )
                files_to_cleanup: list[str] = []
                file_new_chunks = 0

                for chunk_idx, chunk in enumerate(chunks, 1):
                    _set_job(
                        job_id,
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
                    chunk_id = store.make_chunk_id(abs_path, chunk["start_time"])

                    if store.has_chunk(chunk_id):
                        files_to_cleanup.append(chunk["chunk_path"])
                        continue
                    if dlq.contains(chunk_id):
                        if request.retry_failed:
                            dlq.remove(chunk_id)
                        else:
                            _add_event(
                                job_id,
                                f"Skipping {basename} chunk {chunk_idx}/{len(chunks)} (in DLQ)",
                            )
                            files_to_cleanup.append(chunk["chunk_path"])
                            continue
                    if request.skip_still and is_still_frame_chunk(
                        chunk["chunk_path"], verbose=request.verbose,
                    ):
                        skipped_chunks += 1
                        files_to_cleanup.append(chunk["chunk_path"])
                        continue

                    _add_event(
                        job_id,
                        f"Indexing file {file_idx}/{total_files}: {basename} "
                        f"[chunk {chunk_idx}/{len(chunks)}]",
                    )
                    embed_path = chunk["chunk_path"]
                    if request.preprocess:
                        embed_path = preprocess_chunk(
                            embed_path,
                            target_resolution=request.target_resolution,
                            target_fps=request.target_fps,
                        )
                        if embed_path != chunk["chunk_path"]:
                            files_to_cleanup.append(embed_path)

                    embedding = _embed_with_retry(
                        embedder,
                        embed_path,
                        {
                            "chunk_id": chunk_id,
                            "source_file": abs_path,
                            "start_time": chunk["start_time"],
                            "end_time": chunk["end_time"],
                        },
                        dlq,
                        verbose=request.verbose,
                    )
                    files_to_cleanup.append(chunk["chunk_path"])
                    if embedding is None:
                        dlq_chunks += 1
                        continue
                    store.add_chunk(
                        chunk_id,
                        embedding,
                        {
                            "source_file": abs_path,
                            "start_time": chunk["start_time"],
                            "end_time": chunk["end_time"],
                        },
                    )
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
                _set_job(
                    job_id,
                    files_done=file_idx,
                    progress=_index_progress(file_idx + 1, total_files),
                )

            stats = store.get_stats()
            result = {
                "new_chunks": new_chunks,
                "new_files": new_files,
                "skipped_files": skipped_files,
                "skipped_still_chunks": skipped_chunks,
                "failed_chunks": dlq_chunks,
                "total_chunks": stats["total_chunks"],
                "unique_source_files": stats["unique_source_files"],
                "source_files": stats["source_files"],
            }
            _set_job(
                job_id,
                status="succeeded",
                finished_at=_now(),
                progress=1.0,
                result=result,
            )
    except Exception as exc:
        _set_job(job_id, status="failed", finished_at=_now(), error=str(exc))
    finally:
        reset_embedder()


def _search(
    *,
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

            raw_results = (
                search_footage(query or "", store, n_results=results_count, verbose=verbose)
                if image_path is None
                else search_footage_by_image(image_path, store, n_results=results_count, verbose=verbose)
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
                formatted.append(_format_result(result, idx, clip_path))

            return {
                "query": query,
                "image_path": image_path,
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


@app.get("/videos")
def videos() -> dict:
    indexed_sources = _indexed_source_files()
    entries = _list_broll_videos(indexed_sources)
    indexed_count = sum(1 for entry in entries if entry["indexed"])
    return {
        "directory": str(VIDEOS_DIR.resolve()),
        "count": len(entries),
        "indexed_count": indexed_count,
        "unindexed_count": len(entries) - indexed_count,
        "fully_indexed": bool(entries) and indexed_count == len(entries),
        "videos": entries,
    }


@app.post("/index", status_code=202)
def index(request: IndexRequest, background_tasks: BackgroundTasks) -> dict:
    index_path = request.path or str(VIDEOS_DIR.resolve())
    if not os.path.exists(index_path):
        raise HTTPException(status_code=404, detail=f"Path not found: {index_path}")
    request = request.model_copy(update={"path": index_path})
    job_id = uuid.uuid4().hex
    with _jobs_lock:
        _jobs[job_id] = {
            "job_id": job_id,
            "type": "index",
            "status": "queued",
            "created_at": _now(),
            "updated_at": _now(),
            "events": [],
        }
    background_tasks.add_task(_run_index, job_id, request)
    return {
        "job_id": job_id,
        "status": "queued",
        "path": index_path,
        "job_url": f"/jobs/{job_id}",
    }


@app.get("/jobs/{job_id}")
def job(job_id: str) -> dict:
    return _public_job(job_id)


@app.get("/jobs/{job_id}/events")
async def job_events(job_id: str):
    _public_job(job_id)

    async def stream():
        while True:
            payload = _public_job(job_id)
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
def search(request: SearchRequest) -> dict:
    return _search(
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
    )


@app.post("/search/batch")
def search_batch(request: BatchSearchRequest) -> dict:
    rows = []
    for index, item in enumerate(request.items):
        result = _search(
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
def search_by_image(request: ImageSearchRequest) -> dict:
    if not os.path.isfile(request.image_path):
        raise HTTPException(status_code=404, detail=f"Image not found: {request.image_path}")
    return _search(
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
