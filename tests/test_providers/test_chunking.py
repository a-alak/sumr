from unittest.mock import MagicMock, patch

import pytest

from sumr.providers.base import TranscriberLimits, TranscriptionResult
from sumr.providers.chunking import ChunkingTranscriber


def _mock_inner(text="ok"):
    inner = MagicMock()
    inner.transcribe.return_value = TranscriptionResult(text=text, model="test-model")
    return inner


def _limits(max_bytes=100, max_duration=None):
    return TranscriberLimits(max_upload_bytes=max_bytes, max_chunk_duration_secs=max_duration)


class TestChunkingTranscriber:
    def test_small_file_passes_through(self, tmp_path):
        inner = _mock_inner("transcript")
        audio = tmp_path / "small.mp3"
        audio.touch()  # 0 bytes
        result = ChunkingTranscriber(inner, _limits(max_bytes=100)).transcribe(audio)
        assert result.text == "transcript"
        inner.transcribe.assert_called_once_with(
            audio, language=None, prompt=None, response_format="text"
        )

    def test_file_not_found_raises(self, tmp_path):
        inner = _mock_inner()
        with pytest.raises(FileNotFoundError):
            ChunkingTranscriber(inner, _limits()).transcribe(tmp_path / "missing.mp3")

    @patch("sumr.providers.chunking.compress_audio")
    def test_compression_path(self, mock_compress, tmp_path):
        """File > size limit, compressed < limit -> inner called once with compressed."""
        inner = _mock_inner("compressed transcript")

        def fake_compress(src, dst):
            dst.write_bytes(b"\x00" * 5)

        mock_compress.side_effect = fake_compress

        audio = tmp_path / "big.mp3"
        audio.write_bytes(b"\x00" * 200)

        result = ChunkingTranscriber(inner, _limits(max_bytes=10)).transcribe(audio)
        assert result.text == "compressed transcript"
        mock_compress.assert_called_once()
        assert inner.transcribe.call_count == 1

    @patch("sumr.providers.chunking.chunk_audio_file")
    @patch("sumr.providers.chunking.compress_audio")
    def test_chunking_path_joins_transcripts(self, mock_compress, mock_chunk, tmp_path):
        """File > size limit, compressed > limit -> chunks joined."""
        inner = MagicMock()
        inner.transcribe.side_effect = [
            TranscriptionResult(text="part one", model="m"),
            TranscriptionResult(text="part two", model="m"),
        ]

        def fake_compress(src, dst):
            dst.write_bytes(b"\x00" * 200)

        mock_compress.side_effect = fake_compress

        chunk_files = [tmp_path / "chunk_000.mp3", tmp_path / "chunk_001.mp3"]
        for c in chunk_files:
            c.write_bytes(b"\x00" * 5)
        mock_chunk.return_value = chunk_files

        audio = tmp_path / "huge.mp3"
        audio.write_bytes(b"\x00" * 200)

        result = ChunkingTranscriber(inner, _limits(max_bytes=10)).transcribe(audio)
        assert result.text == "part one\npart two"
        assert result.model == "m"
        assert inner.transcribe.call_count == 2

    @patch(
        "sumr.providers.chunking.compress_audio",
        side_effect=FileNotFoundError("ffmpeg not found"),
    )
    def test_ffmpeg_not_found_propagates(self, _mock, tmp_path):
        audio = tmp_path / "big.mp3"
        audio.write_bytes(b"\x00" * 200)
        with pytest.raises(FileNotFoundError, match="ffmpeg not found"):
            ChunkingTranscriber(MagicMock(), _limits(max_bytes=10)).transcribe(audio)

    def test_satisfies_transcriber_protocol(self, tmp_path):
        from sumr.providers.base import Transcriber

        inner = _mock_inner()
        t = ChunkingTranscriber(inner, TranscriberLimits())
        assert isinstance(t, Transcriber)

    # --- Duration-based chunking tests ---

    @patch("sumr.providers.chunking.get_audio_duration")
    def test_duration_within_limit_passes_through(self, mock_duration, tmp_path):
        """Small file, duration under limit -> no compression."""
        mock_duration.return_value = 300.0  # 5 min
        inner = _mock_inner("ok")
        audio = tmp_path / "short.mp3"
        audio.touch()

        result = ChunkingTranscriber(
            inner, _limits(max_bytes=10**9, max_duration=480)
        ).transcribe(audio)

        assert result.text == "ok"
        inner.transcribe.assert_called_once()

    @patch("sumr.providers.chunking.chunk_audio_file")
    @patch("sumr.providers.chunking.compress_audio")
    @patch("sumr.providers.chunking.get_audio_duration")
    def test_duration_exceeded_triggers_chunking(
        self, mock_duration, mock_compress, mock_chunk, tmp_path
    ):
        """Small file but long audio -> compress then chunk."""
        mock_duration.return_value = 600.0  # 10 min, exceeds 480 s limit

        inner = MagicMock()
        inner.transcribe.side_effect = [
            TranscriptionResult(text="first half", model="m"),
            TranscriptionResult(text="second half", model="m"),
        ]

        def fake_compress(src, dst):
            dst.write_bytes(b"\x00" * 50)

        mock_compress.side_effect = fake_compress

        chunk_files = [tmp_path / "chunk_000.mp3", tmp_path / "chunk_001.mp3"]
        for c in chunk_files:
            c.write_bytes(b"\x00" * 5)
        mock_chunk.return_value = chunk_files

        audio = tmp_path / "long.mp3"
        audio.write_bytes(b"\x00" * 50)  # small file, but long duration

        result = ChunkingTranscriber(
            inner, _limits(max_bytes=10**9, max_duration=480)
        ).transcribe(audio)

        assert result.text == "first half\nsecond half"
        mock_compress.assert_called_once()
        mock_chunk.assert_called_once()

    @patch("sumr.providers.chunking.chunk_audio_file")
    @patch("sumr.providers.chunking.compress_audio")
    @patch("sumr.providers.chunking.get_audio_duration")
    def test_chunk_bytes_bounded_by_duration_limit(
        self, mock_duration, mock_compress, mock_chunk, tmp_path
    ):
        """max_chunk_bytes passed to chunk_audio_file must respect the duration limit."""
        mock_duration.return_value = 600.0

        inner = MagicMock()
        inner.transcribe.return_value = TranscriptionResult(text="x", model="m")
        mock_chunk.return_value = [tmp_path / "c.mp3"]
        (tmp_path / "c.mp3").touch()

        compressed_size = 240_000  # 240 kB -> 400 bytes/sec at 600 s

        def fake_compress(src, dst):
            dst.write_bytes(b"\x00" * compressed_size)

        mock_compress.side_effect = fake_compress

        audio = tmp_path / "long.mp3"
        audio.write_bytes(b"\x00" * 50)

        max_upload = 10**9  # huge size limit — duration should be the binding constraint
        max_duration = 480  # 480 s
        ChunkingTranscriber(
            inner, _limits(max_bytes=max_upload, max_duration=max_duration)
        ).transcribe(audio)

        # bytes_per_sec = 240_000 / 600 = 400; duration_max = int(400 * 480 * 0.95) = 182_400
        expected_max_chunk = 182_400
        actual_max_chunk = mock_chunk.call_args[0][1]
        assert actual_max_chunk == expected_max_chunk
