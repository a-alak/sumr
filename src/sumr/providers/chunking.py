import tempfile
from pathlib import Path

from sumr.providers.base import Transcriber, TranscriberLimits, TranscriptionResult
from sumr.utils import chunk_audio_file, compress_audio, get_audio_duration


class ChunkingTranscriber:
    """Wraps any Transcriber to enforce provider-specific upload and duration limits.

    Strategy:
    1. File within size limit AND duration within limit  ->  pass through as-is.
    2. File too large OR audio too long  ->  compress to 32 kbps mono MP3.
    3. Compressed file now fits both limits  ->  transcribe compressed file.
    4. Still exceeds a limit  ->  split at silence boundaries so each chunk
       satisfies both limits, transcribe each chunk, join with newline.
    """

    def __init__(self, inner: Transcriber, limits: TranscriberLimits) -> None:
        self._inner = inner
        self._limits = limits

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

        # Fast path: skip ffmpeg entirely when the file already satisfies all limits.
        size_ok = audio_path.stat().st_size <= self._limits.max_upload_bytes
        if size_ok:
            duration_ok = self._limits.max_chunk_duration_secs is None or (
                get_audio_duration(audio_path) <= self._limits.max_chunk_duration_secs
            )
            if duration_ok:
                return self._inner.transcribe(audio_path, **kwargs)

        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp = Path(tmp_dir)
            compressed = tmp / "compressed.mp3"
            compress_audio(audio_path, compressed)

            compressed_size = compressed.stat().st_size

            # Compute duration once if we need it (avoid a second ffprobe call later).
            compressed_duration: float | None = None
            if self._limits.max_chunk_duration_secs is not None:
                compressed_duration = get_audio_duration(compressed)

            size_ok = compressed_size <= self._limits.max_upload_bytes
            duration_ok = compressed_duration is None or (
                compressed_duration <= self._limits.max_chunk_duration_secs  # type: ignore[operator]
            )

            if size_ok and duration_ok:
                return self._inner.transcribe(compressed, **kwargs)

            # Determine the max bytes per chunk, honouring both limits.
            max_chunk_bytes = self._limits.max_upload_bytes
            if self._limits.max_chunk_duration_secs is not None:
                if compressed_duration is None:
                    compressed_duration = get_audio_duration(compressed)
                bytes_per_sec = compressed_size / compressed_duration
                duration_max_bytes = int(
                    bytes_per_sec * self._limits.max_chunk_duration_secs * 0.95
                )
                max_chunk_bytes = min(max_chunk_bytes, duration_max_bytes)

            chunks = chunk_audio_file(compressed, max_chunk_bytes, tmp)
            results = [self._inner.transcribe(c, **kwargs) for c in chunks]
            return TranscriptionResult(
                text="\n".join(r.text for r in results),
                model=results[0].model,
            )
