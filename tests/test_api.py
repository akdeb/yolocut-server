"""Tests for the FastAPI server."""

from fastapi.testclient import TestClient

import sentrysearch.api as api


def test_videos_lists_broll_manifest(monkeypatch, tmp_path):
    videos_dir = tmp_path / "videos"
    videos_dir.mkdir()
    (videos_dir / "beta.mov").write_bytes(b"mov")
    (videos_dir / "Alpha Clip.MP4").write_bytes(b"mp4")
    (videos_dir / "notes.txt").write_text("not a video")

    monkeypatch.setattr(api, "VIDEOS_DIR", videos_dir)
    monkeypatch.setattr(
        api,
        "_indexed_source_files",
        lambda: {str((videos_dir / "Alpha Clip.MP4").resolve())},
    )

    client = TestClient(api.app)
    response = client.get("/videos")

    assert response.status_code == 200
    payload = response.json()
    assert payload["directory"] == str(videos_dir.resolve())
    assert payload["count"] == 2
    assert payload["indexed_count"] == 1
    assert payload["unindexed_count"] == 1
    assert payload["fully_indexed"] is False
    assert [video["filename"] for video in payload["videos"]] == [
        "Alpha Clip.MP4",
        "beta.mov",
    ]
    assert payload["videos"][0]["name"] == "Alpha Clip"
    assert payload["videos"][0]["relative_path"] == "Alpha Clip.MP4"
    assert payload["videos"][0]["path"] == str((videos_dir / "Alpha Clip.MP4").resolve())
    assert payload["videos"][0]["url"].startswith("/clips?path=")
    assert payload["videos"][0]["indexed"] is True
    assert payload["videos"][0]["size_bytes"] == 3
    assert payload["videos"][0]["modified_at"]
    assert payload["videos"][1]["indexed"] is False


def test_videos_handles_missing_directory(monkeypatch, tmp_path):
    monkeypatch.setattr(api, "VIDEOS_DIR", tmp_path / "missing")
    monkeypatch.setattr(api, "_indexed_source_files", lambda: set())

    client = TestClient(api.app)
    response = client.get("/videos")

    assert response.status_code == 200
    assert response.json() == {
        "directory": str((tmp_path / "missing").resolve()),
        "count": 0,
        "indexed_count": 0,
        "unindexed_count": 0,
        "fully_indexed": False,
        "videos": [],
    }


def test_index_defaults_to_videos_directory(monkeypatch, tmp_path):
    videos_dir = tmp_path / "videos"
    videos_dir.mkdir()
    captured = {}

    def fake_run_index(_job_id, request):
        captured["path"] = request.path

    monkeypatch.setattr(api, "VIDEOS_DIR", videos_dir)
    monkeypatch.setattr(api, "_run_index", fake_run_index)

    client = TestClient(api.app)
    response = client.post("/index", json={"chunk_duration": 2, "overlap": 1})

    assert response.status_code == 202
    payload = response.json()
    assert payload["job_id"]
    assert payload["path"] == str(videos_dir.resolve())
    assert captured["path"] == str(videos_dir.resolve())
    assert api.IndexRequest().skip_still is False


def test_job_response_includes_completion_flags():
    try:
        api._jobs["test-job"] = {
            "job_id": "test-job",
            "status": "succeeded",
        }

        client = TestClient(api.app)
        response = client.get("/jobs/test-job")

        assert response.status_code == 200
        payload = response.json()
        assert payload["done"] is True
        assert payload["succeeded"] is True
        assert payload["failed"] is False
        assert payload["progress"] == 1.0
    finally:
        api._jobs.pop("test-job", None)


def test_job_events_streams_sse_until_complete():
    try:
        api._jobs["stream-job"] = {
            "job_id": "stream-job",
            "status": "succeeded",
            "progress": 1.0,
        }

        client = TestClient(api.app)
        response = client.get("/jobs/stream-job/events")

        assert response.status_code == 200
        assert response.headers["content-type"].startswith("text/event-stream")
        body = response.text
        assert "event: complete" in body
        assert '"job_id": "stream-job"' in body
        assert '"progress": 1.0' in body
    finally:
        api._jobs.pop("stream-job", None)


def test_search_batch_preserves_order_and_maps_results(monkeypatch):
    calls = []

    def fake_search(**kwargs):
        calls.append(kwargs)
        query = kwargs["query"]
        if "no match" in query:
            return {"query": query, "results": []}
        return {
            "query": query,
            "results": [
                {
                    "rank": 1,
                    "source_file": "/source/video.mov",
                    "source_basename": "video.mov",
                    "start_time": 12.0,
                    "end_time": 14.0,
                    "start_time_formatted": "00:12",
                    "end_time_formatted": "00:14",
                    "similarity_score": 0.83,
                    "clip_path": "/clips/generated.mp4",
                    "clip_url": "/clips?path=%2Fclips%2Fgenerated.mp4",
                }
            ],
        }

    monkeypatch.setattr(api, "_search", fake_search)

    client = TestClient(api.app)
    response = client.post(
        "/search/batch",
        json={
            "items": [
                {"visual_broll": "close-up of green supplement packet"},
                {"visual_broll": "no match for this prompt"},
            ],
            "results": 5,
            "save_top": 5,
            "trim": True,
            "force_trim_low_confidence": True,
            "backend": "gemini",
            "model": None,
            "quantize": None,
            "verbose": False,
        },
    )

    assert response.status_code == 200
    payload = response.json()
    assert [row["index"] for row in payload["rows"]] == [0, 1]
    assert payload["rows"][0]["visual_broll"] == "close-up of green supplement packet"
    assert payload["rows"][0]["query"] == "close-up of green supplement packet"
    assert payload["rows"][0]["results"][0]["rank"] == 1
    assert payload["rows"][1] == {
        "index": 1,
        "visual_broll": "no match for this prompt",
        "query": "no match for this prompt",
        "results": [],
    }
    assert calls[0]["results_count"] == 5
    assert calls[0]["save_top"] == 5
    assert calls[0]["trim"] is True
    assert calls[0]["force_trim_low_confidence"] is True
    assert calls[0]["backend"] == "gemini"


def test_filter_unindexed_videos_skips_already_indexed_source_files(tmp_path):
    videos_dir = tmp_path / "videos"
    videos_dir.mkdir()
    indexed_video = videos_dir / "already.mp4"
    new_video = videos_dir / "new.mp4"
    indexed_video.write_bytes(b"indexed")
    new_video.write_bytes(b"new")

    videos_to_index, skipped_files = api._filter_unindexed_videos(
        [str(indexed_video), str(new_video)],
        {str(indexed_video.resolve())},
    )

    assert videos_to_index == [str(new_video)]
    assert skipped_files == 1


def test_index_progress_includes_current_chunk_fraction():
    assert api._index_progress(file_idx=1, total_files=5, chunk_idx=54, total_chunks=59) == (
        54 / 59
    ) / 5
    assert api._index_progress(file_idx=6, total_files=5) == 1.0
