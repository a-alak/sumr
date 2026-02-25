import asyncio
import tempfile
from pathlib import Path
from typing import Any

from sumr.providers.base import TranscriptionResult
from sumr.utils import chunk_audio_file

TARGET_CHUNK_DURATION: float = 60.0


class ChunkingTranscriber:
    """Splits audio into ~60s chunks and transcribes them in parallel."""

    def __init__(self, inner: Any) -> None:
        self._inner = inner

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

        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp = Path(tmp_dir)
            chunks = chunk_audio_file(audio_path, TARGET_CHUNK_DURATION, tmp)

            async def _gather_all() -> list[TranscriptionResult]:
                return list(
                    await asyncio.gather(
                        *[self._inner.atranscribe(c, **kwargs) for c in chunks]
                    )
                )

            results = asyncio.run(_gather_all())

        return TranscriptionResult(
            text="\n".join(r.text for r in results),
            model=results[0].model,
        )
