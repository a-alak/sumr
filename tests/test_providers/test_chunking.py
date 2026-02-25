from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from sumr.providers.base import TranscriptionResult
from sumr.providers.chunking import ChunkingTranscriber


def _mock_inner(texts=("ok",)):
    inner = MagicMock()
    inner.atranscribe = AsyncMock(
        side_effect=[TranscriptionResult(text=t, model="test-model") for t in texts]
    )
    return inner


class TestChunkingTranscriber:
    def test_file_not_found_raises(self, tmp_path):
        inner = _mock_inner()
        with pytest.raises(FileNotFoundError):
            ChunkingTranscriber(inner).transcribe(tmp_path / "missing.mp3")

    @patch("sumr.providers.chunking.chunk_audio_file")
    def test_single_chunk_transcribed(self, mock_chunk, tmp_path):
        audio = tmp_path / "audio.mp3"
        audio.touch()
        chunk = tmp_path / "chunk_000.mp3"
        chunk.touch()
        mock_chunk.return_value = [chunk]

        inner = _mock_inner(texts=("hello world",))
        result = ChunkingTranscriber(inner).transcribe(audio)

        assert result.text == "hello world"
        assert result.model == "test-model"
        inner.atranscribe.assert_called_once()

    @patch("sumr.providers.chunking.chunk_audio_file")
    def test_multiple_chunks_joined_in_order(self, mock_chunk, tmp_path):
        audio = tmp_path / "audio.mp3"
        audio.touch()
        chunks = [tmp_path / f"chunk_{i:03d}.mp3" for i in range(3)]
        for c in chunks:
            c.touch()
        mock_chunk.return_value = chunks

        inner = _mock_inner(texts=("part one", "part two", "part three"))
        result = ChunkingTranscriber(inner).transcribe(audio)

        assert result.text == "part one\npart two\npart three"
        assert result.model == "test-model"
        assert inner.atranscribe.call_count == 3

    @patch("sumr.providers.chunking.chunk_audio_file")
    def test_passes_kwargs_to_atranscribe(self, mock_chunk, tmp_path):
        audio = tmp_path / "audio.mp3"
        audio.touch()
        chunk = tmp_path / "chunk_000.mp3"
        chunk.touch()
        mock_chunk.return_value = [chunk]

        inner = _mock_inner(texts=("text",))
        ChunkingTranscriber(inner).transcribe(
            audio, language="en", prompt="notes", response_format="text"
        )

        inner.atranscribe.assert_called_once_with(
            chunk, language="en", prompt="notes", response_format="text"
        )

    @patch(
        "sumr.providers.chunking.chunk_audio_file",
        side_effect=FileNotFoundError("ffmpeg not found"),
    )
    def test_ffmpeg_not_found_propagates(self, _mock, tmp_path):
        audio = tmp_path / "audio.mp3"
        audio.touch()
        with pytest.raises(FileNotFoundError, match="ffmpeg not found"):
            ChunkingTranscriber(MagicMock()).transcribe(audio)

    def test_satisfies_transcriber_protocol(self):
        from sumr.providers.base import Transcriber

        inner = _mock_inner()
        t = ChunkingTranscriber(inner)
        assert isinstance(t, Transcriber)
