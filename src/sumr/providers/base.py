from dataclasses import dataclass
from pathlib import Path
from typing import Protocol, runtime_checkable


@dataclass(frozen=True)
class TranscriberLimits:
    max_upload_bytes: int = 25 * 1024 * 1024  # 25 MB
    max_chunk_duration_secs: int | None = None  # None = no duration limit


@dataclass(frozen=True)
class TranscriptionResult:
    text: str
    model: str


@dataclass(frozen=True)
class SummarizationResult:
    text: str
    model: str


@runtime_checkable
class Transcriber(Protocol):
    def transcribe(
        self,
        audio_path: Path,
        *,
        language: str | None = None,
        prompt: str | None = None,
        response_format: str = "text",
    ) -> TranscriptionResult: ...


@runtime_checkable
class Summarizer(Protocol):
    def summarize(
        self,
        text: str,
        *,
        system_prompt: str | None = None,
    ) -> SummarizationResult: ...
