import tempfile
from pathlib import Path

from sumr.providers.base import Transcriber, TranscriptionResult
from sumr.utils import chunk_audio_file, compress_audio


class ChunkingTranscriber:
    """Wraps any Transcriber to transparently handle files exceeding the upload limit.

    Strategy:
    1. File <= limit  ->  pass through to inner transcriber.
    2. File > limit  ->  compress to 32 kbps mono MP3.
    3. Compressed <= limit  ->  transcribe compressed file.
    4. Compressed > limit  ->  chunk at silence boundaries, transcribe each chunk,
       join transcripts with newline.
    """

    def __init__(self, inner: Transcriber, max_upload_bytes: int) -> None:
        self._inner = inner
        self._max_bytes = max_upload_bytes

    def transcribe(
        self,
        audio_path: Path,
        *,
        language: str | None = None,
        prompt: str | None = None,
        response_format: str = "text",
    ) -> TranscriptionResult:
        if not audio_path.exists():
            raise FileNotFoundError(f"Audio file not found: {audio_path}")

        kwargs = dict(language=language, prompt=prompt, response_format=response_format)

        if audio_path.stat().st_size <= self._max_bytes:
            return self._inner.transcribe(audio_path, **kwargs)

        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp = Path(tmp_dir)
            compressed = tmp / "compressed.mp3"
            compress_audio(audio_path, compressed)

            if compressed.stat().st_size <= self._max_bytes:
                return self._inner.transcribe(compressed, **kwargs)

            chunks = chunk_audio_file(compressed, self._max_bytes, tmp)
            results = [self._inner.transcribe(c, **kwargs) for c in chunks]
            return TranscriptionResult(
                text="\n".join(r.text for r in results),
                model=results[0].model,
            )
