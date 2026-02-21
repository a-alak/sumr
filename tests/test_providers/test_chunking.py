from unittest.mock import MagicMock, patch

import pytest

from sumr.providers.base import TranscriptionResult
from sumr.providers.chunking import ChunkingTranscriber


def _mock_inner(text="ok"):
    inner = MagicMock()
    inner.transcribe.return_value = TranscriptionResult(text=text, model="test-model")
    return inner


class TestChunkingTranscriber:
    def test_small_file_passes_through(self, tmp_path):
        inner = _mock_inner("transcript")
        audio = tmp_path / "small.mp3"
        audio.touch()  # 0 bytes
        result = ChunkingTranscriber(inner, max_upload_bytes=100).transcribe(audio)
        assert result.text == "transcript"
        inner.transcribe.assert_called_once_with(
            audio, language=None, prompt=None, response_format="text"
        )

    def test_file_not_found_raises(self, tmp_path):
        inner = _mock_inner()
        with pytest.raises(FileNotFoundError):
            ChunkingTranscriber(inner, max_upload_bytes=100).transcribe(
                tmp_path / "missing.mp3"
            )

    @patch("sumr.providers.chunking.compress_audio")
    def test_compression_path(self, mock_compress, tmp_path):
        """File > limit, compressed < limit -> inner called once with compressed."""
        inner = _mock_inner("compressed transcript")

        def fake_compress(src, dst):
            dst.write_bytes(b"\x00" * 5)

        mock_compress.side_effect = fake_compress

        audio = tmp_path / "big.mp3"
        audio.write_bytes(b"\x00" * 200)

        result = ChunkingTranscriber(inner, max_upload_bytes=10).transcribe(audio)
        assert result.text == "compressed transcript"
        mock_compress.assert_called_once()
        assert inner.transcribe.call_count == 1

    @patch("sumr.providers.chunking.chunk_audio_file")
    @patch("sumr.providers.chunking.compress_audio")
    def test_chunking_path_joins_transcripts(self, mock_compress, mock_chunk, tmp_path):
        """File > limit, compressed > limit -> chunks joined."""
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

        result = ChunkingTranscriber(inner, max_upload_bytes=10).transcribe(audio)
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
            ChunkingTranscriber(MagicMock(), max_upload_bytes=10).transcribe(audio)

    def test_satisfies_transcriber_protocol(self, tmp_path):
        from sumr.providers.base import Transcriber

        inner = _mock_inner()
        t = ChunkingTranscriber(inner, max_upload_bytes=25 * 1024 * 1024)
        assert isinstance(t, Transcriber)
