"""Tests for the dead-letter queue."""

import json

import pytest

from sentrysearch.dlq import DeadLetterQueue


@pytest.fixture
def dlq_path(tmp_path):
    return tmp_path / "dlq.json"


class TestDeadLetterQueue:
    def test_empty_when_file_missing(self, dlq_path):
        dlq = DeadLetterQueue(dlq_path)
        assert len(dlq) == 0
        assert not dlq.contains("anything")

    def test_record_persists_to_disk(self, dlq_path):
        dlq = DeadLetterQueue(dlq_path)
        dlq.record(
            "abc", source_file="/v.mp4", start_time=0.0, end_time=30.0,
            error="OOM", attempts=3,
        )
        assert dlq_path.exists()
        data = json.loads(dlq_path.read_text())
        assert "abc" in data
        assert data["abc"]["error"] == "OOM"
        assert data["abc"]["attempts"] == 3

    def test_record_survives_reload(self, dlq_path):
        dlq = DeadLetterQueue(dlq_path)
        dlq.record(
            "abc", source_file="/v.mp4", start_time=0.0, end_time=30.0,
            error="OOM", attempts=3,
        )
        dlq2 = DeadLetterQueue(dlq_path)
        assert dlq2.contains("abc")
        assert len(dlq2) == 1

    def test_contains(self, dlq_path):
        dlq = DeadLetterQueue(dlq_path)
        assert not dlq.contains("abc")
        dlq.record(
            "abc", source_file="/v.mp4", start_time=0.0, end_time=30.0,
            error="e", attempts=1,
        )
        assert dlq.contains("abc")

    def test_remove(self, dlq_path):
        dlq = DeadLetterQueue(dlq_path)
        dlq.record(
            "abc", source_file="/v.mp4", start_time=0.0, end_time=30.0,
            error="e", attempts=1,
        )
        assert dlq.remove("abc") is True
        assert not dlq.contains("abc")
        assert dlq.remove("abc") is False

    def test_clear(self, dlq_path):
        dlq = DeadLetterQueue(dlq_path)
        dlq.record(
            "a", source_file="/v.mp4", start_time=0.0, end_time=30.0,
            error="e", attempts=1,
        )
        dlq.record(
            "b", source_file="/v.mp4", start_time=30.0, end_time=60.0,
            error="e", attempts=1,
        )
        removed = dlq.clear()
        assert removed == 2
        assert len(dlq) == 0

    def test_corrupt_file_is_ignored(self, dlq_path):
        dlq_path.parent.mkdir(parents=True, exist_ok=True)
        dlq_path.write_text("not json")
        dlq = DeadLetterQueue(dlq_path)
        assert len(dlq) == 0

    def test_error_message_truncated(self, dlq_path):
        dlq = DeadLetterQueue(dlq_path)
        dlq.record(
            "abc", source_file="/v.mp4", start_time=0.0, end_time=30.0,
            error="x" * 2000, attempts=1,
        )
        assert len(dlq.entries()["abc"]["error"]) == 500
