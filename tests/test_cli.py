"""Tests for sentrysearch.cli (Click CLI)."""

from unittest.mock import MagicMock, patch

import pytest
from click.testing import CliRunner

from sentrysearch.cli import _fmt_time, cli


@pytest.fixture
def runner():
    return CliRunner()


class TestFmtTime:
    def test_zero(self):
        assert _fmt_time(0) == "00:00"

    def test_minutes(self):
        assert _fmt_time(125) == "02:05"


class TestCliGroup:
    def test_help(self, runner):
        result = runner.invoke(cli, ["--help"])
        assert result.exit_code == 0
        assert "Search dashcam footage" in result.output or "search" in result.output.lower()


class TestStatsCommand:
    def test_stats_empty(self, runner):
        with patch("sentrysearch.store.SentryStore") as MockStore:
            inst = MagicMock()
            inst.get_stats.return_value = {
                "total_chunks": 0, "unique_source_files": 0, "source_files": [],
            }
            MockStore.return_value = inst
            result = runner.invoke(cli, ["stats"])
            assert result.exit_code == 0
            assert "empty" in result.output.lower() or "0" in result.output

    def test_stats_with_data(self, runner):
        with patch("sentrysearch.store.SentryStore") as MockStore:
            inst = MagicMock()
            inst.get_stats.return_value = {
                "total_chunks": 10,
                "unique_source_files": 2,
                "source_files": ["/a/video1.mp4", "/b/video2.mp4"],
            }
            MockStore.return_value = inst
            result = runner.invoke(cli, ["stats"])
            assert result.exit_code == 0
            assert "10" in result.output


class TestSearchCommand:
    def test_search_empty_index(self, runner):
        with patch("sentrysearch.store.SentryStore") as MockStore:
            inst = MagicMock()
            inst.get_stats.return_value = {"total_chunks": 0}
            MockStore.return_value = inst
            result = runner.invoke(cli, ["search", "red car"])
            assert result.exit_code == 0
            assert "No indexed footage" in result.output


class TestIndexCommand:
    def test_index_no_mp4s(self, runner, tmp_path):
        empty_dir = tmp_path / "empty"
        empty_dir.mkdir()
        with patch("sentrysearch.store.SentryStore") as MockStore, \
             patch("sentrysearch.embedder.get_embedder", return_value=MagicMock()):
            MockStore.return_value = MagicMock()
            result = runner.invoke(cli, ["index", str(empty_dir)])
            assert result.exit_code == 0
            assert "No mp4 files" in result.output or "No mp4" in result.output

    def test_index_accepts_backend_option(self, runner, tmp_path):
        empty_dir = tmp_path / "empty"
        empty_dir.mkdir()
        with patch("sentrysearch.store.SentryStore") as MockStore, \
             patch("sentrysearch.embedder.get_embedder", return_value=MagicMock()):
            MockStore.return_value = MagicMock()
            result = runner.invoke(cli, ["index", str(empty_dir), "--backend", "local"])
            assert result.exit_code == 0


class TestIndexLocalFlags:
    def test_index_passes_model_to_embedder(self, runner, tmp_path):
        empty_dir = tmp_path / "empty"
        empty_dir.mkdir()
        with patch("sentrysearch.store.SentryStore") as MockStore, \
             patch("sentrysearch.embedder.get_embedder", return_value=MagicMock()) as mock_get:
            MockStore.return_value = MagicMock()
            result = runner.invoke(cli, [
                "index", str(empty_dir), "--backend", "local", "--model", "qwen2b",
            ])
            assert result.exit_code == 0
            mock_get.assert_called_once()
            assert mock_get.call_args[1]["model"] == "qwen2b"

    def test_index_passes_quantize_to_embedder(self, runner, tmp_path):
        empty_dir = tmp_path / "empty"
        empty_dir.mkdir()
        with patch("sentrysearch.store.SentryStore") as MockStore, \
             patch("sentrysearch.embedder.get_embedder", return_value=MagicMock()) as mock_get:
            MockStore.return_value = MagicMock()
            result = runner.invoke(cli, [
                "index", str(empty_dir), "--backend", "local", "--quantize",
            ])
            assert result.exit_code == 0
            mock_get.assert_called_once()
            assert mock_get.call_args[1]["quantize"] is True

    def test_index_passes_no_quantize_to_embedder(self, runner, tmp_path):
        empty_dir = tmp_path / "empty"
        empty_dir.mkdir()
        with patch("sentrysearch.store.SentryStore") as MockStore, \
             patch("sentrysearch.embedder.get_embedder", return_value=MagicMock()) as mock_get:
            MockStore.return_value = MagicMock()
            result = runner.invoke(cli, [
                "index", str(empty_dir), "--backend", "local", "--no-quantize",
            ])
            assert result.exit_code == 0
            mock_get.assert_called_once()
            assert mock_get.call_args[1]["quantize"] is False

    def test_index_default_model_is_qwen8b(self, runner, tmp_path):
        empty_dir = tmp_path / "empty"
        empty_dir.mkdir()
        with patch("sentrysearch.store.SentryStore") as MockStore, \
             patch("sentrysearch.embedder.get_embedder", return_value=MagicMock()) as mock_get:
            MockStore.return_value = MagicMock()
            result = runner.invoke(cli, [
                "index", str(empty_dir), "--backend", "local",
            ])
            assert result.exit_code == 0
            assert mock_get.call_args[1]["model"] == "qwen8b"

    def test_index_passes_backend_to_store(self, runner, tmp_path):
        d = tmp_path / "vids"
        d.mkdir()
        (d / "test.mp4").write_bytes(b"fake")
        with patch("sentrysearch.store.SentryStore") as MockStore, \
             patch("sentrysearch.embedder.get_embedder", return_value=MagicMock()):
            mock_inst = MagicMock()
            mock_inst.is_indexed.return_value = True
            MockStore.return_value = mock_inst
            runner.invoke(cli, ["index", str(d), "--backend", "local"])
            MockStore.assert_called_once_with(backend="local")


class TestSearchLocalFlags:
    def test_search_passes_model_to_embedder(self, runner):
        with patch("sentrysearch.store.SentryStore") as MockStore, \
             patch("sentrysearch.embedder.get_embedder", return_value=MagicMock()) as mock_get, \
             patch("sentrysearch.store.detect_backend", return_value="local"), \
             patch("sentrysearch.search.search_footage", return_value=[]):
            inst = MagicMock()
            inst.get_stats.return_value = {"total_chunks": 5}
            MockStore.return_value = inst
            result = runner.invoke(cli, [
                "search", "test query", "--backend", "local", "--model", "qwen2b",
            ])
            assert result.exit_code == 0
            mock_get.assert_called_with("local", model="qwen2b", quantize=None)

    def test_search_passes_quantize_to_embedder(self, runner):
        with patch("sentrysearch.store.SentryStore") as MockStore, \
             patch("sentrysearch.embedder.get_embedder", return_value=MagicMock()) as mock_get, \
             patch("sentrysearch.store.detect_backend", return_value="local"), \
             patch("sentrysearch.search.search_footage", return_value=[]):
            inst = MagicMock()
            inst.get_stats.return_value = {"total_chunks": 5}
            MockStore.return_value = inst
            result = runner.invoke(cli, [
                "search", "test query", "--backend", "local", "--quantize",
            ])
            assert result.exit_code == 0
            mock_get.assert_called_with("local", model="qwen8b", quantize=True)


class TestHandleError:
    def test_local_model_error(self, runner):
        from sentrysearch.local_embedder import LocalModelError

        with patch("sentrysearch.store.SentryStore") as MockStore:
            inst = MagicMock()
            inst.get_stats.return_value = {"total_chunks": 5}
            MockStore.return_value = inst

            with patch(
                "sentrysearch.embedder.get_embedder",
                side_effect=LocalModelError("no torch"),
            ):
                result = runner.invoke(cli, ["search", "test query", "--backend", "local"])
                assert result.exit_code == 1
                assert "no torch" in result.output

    def test_backend_mismatch_error(self, runner):
        from sentrysearch.store import BackendMismatchError

        with patch("sentrysearch.store.SentryStore") as MockStore:
            inst = MagicMock()
            inst.get_stats.return_value = {"total_chunks": 5}
            MockStore.return_value = inst

            with patch(
                "sentrysearch.embedder.get_embedder",
                side_effect=BackendMismatchError("built with gemini"),
            ):
                result = runner.invoke(cli, ["search", "test", "--backend", "local"])
                assert result.exit_code == 1
                assert "gemini" in result.output


class TestResetCommand:
    def test_reset_empty_index(self, runner):
        with patch("sentrysearch.store.SentryStore") as MockStore, \
             patch("sentrysearch.store.detect_backend", return_value="gemini"):
            inst = MagicMock()
            inst.get_stats.return_value = {
                "total_chunks": 0, "unique_source_files": 0, "source_files": [],
            }
            MockStore.return_value = inst
            result = runner.invoke(cli, ["reset", "--yes"])
            assert result.exit_code == 0
            assert "already empty" in result.output.lower()

    def test_reset_removes_all(self, runner):
        with patch("sentrysearch.store.SentryStore") as MockStore, \
             patch("sentrysearch.store.detect_backend", return_value="gemini"):
            inst = MagicMock()
            inst.get_stats.return_value = {
                "total_chunks": 10, "unique_source_files": 2,
                "source_files": ["/a/v1.mp4", "/b/v2.mp4"],
            }
            inst.remove_file.return_value = 5
            MockStore.return_value = inst
            result = runner.invoke(cli, ["reset", "--yes"])
            assert result.exit_code == 0
            assert "10" in result.output
            assert inst.remove_file.call_count == 2


class TestRemoveCommand:
    def test_remove_matching_file(self, runner):
        with patch("sentrysearch.store.SentryStore") as MockStore, \
             patch("sentrysearch.store.detect_backend", return_value="gemini"):
            inst = MagicMock()
            inst.get_stats.return_value = {
                "total_chunks": 10, "unique_source_files": 2,
                "source_files": ["/a/video1.mp4", "/b/video2.mp4"],
            }
            inst.remove_file.return_value = 5
            MockStore.return_value = inst
            result = runner.invoke(cli, ["remove", "video1"])
            assert result.exit_code == 0
            assert "Removed 5 chunks" in result.output
            inst.remove_file.assert_called_once_with("/a/video1.mp4")

    def test_remove_no_match(self, runner):
        with patch("sentrysearch.store.SentryStore") as MockStore, \
             patch("sentrysearch.store.detect_backend", return_value="gemini"):
            inst = MagicMock()
            inst.get_stats.return_value = {
                "total_chunks": 10, "unique_source_files": 1,
                "source_files": ["/a/video1.mp4"],
            }
            MockStore.return_value = inst
            result = runner.invoke(cli, ["remove", "nonexistent"])
            assert result.exit_code == 0
            assert "No indexed files matching" in result.output

    def test_remove_empty_index(self, runner):
        with patch("sentrysearch.store.SentryStore") as MockStore, \
             patch("sentrysearch.store.detect_backend", return_value="gemini"):
            inst = MagicMock()
            inst.get_stats.return_value = {
                "total_chunks": 0, "unique_source_files": 0, "source_files": [],
            }
            MockStore.return_value = inst
            result = runner.invoke(cli, ["remove", "anything"])
            assert result.exit_code == 0
            assert "empty" in result.output.lower()
