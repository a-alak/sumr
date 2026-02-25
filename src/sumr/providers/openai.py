from pathlib import Path

from openai import AsyncOpenAI, OpenAI

from sumr.providers.base import (
    SummarizationResult,
    TranscriptionResult,
)

DEFAULT_TRANSCRIPTION_MODEL = "gpt-4o-mini-transcribe"

_DIARIZATION_MODELS = {"gpt-4o-transcribe-diarize"}


class OpenAITranscriber:
    def __init__(self, api_key: str, model: str = DEFAULT_TRANSCRIPTION_MODEL) -> None:
        self._client = OpenAI(api_key=api_key)
        self._async_client = AsyncOpenAI(api_key=api_key)
        self._model = model

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

        effective_format = (
            "diarized_json"
            if self._model in _DIARIZATION_MODELS and response_format == "text"
            else response_format
        )

        kwargs: dict = {
            "model": self._model,
            "file": audio_path,
            "response_format": effective_format,
        }
        if language is not None:
            kwargs["language"] = language
        if prompt is not None:
            kwargs["prompt"] = prompt
        if self._model in _DIARIZATION_MODELS:
            kwargs["chunking_strategy"] = "auto"

        response = self._client.audio.transcriptions.create(**kwargs)
        return TranscriptionResult(text=self._parse_text(response), model=self._model)

    async def atranscribe(
        self,
        audio_path: Path,
        *,
        language: str | None = None,
        prompt: str | None = None,
        response_format: str = "text",
    ) -> TranscriptionResult:
        if not audio_path.exists():
            raise FileNotFoundError(f"Audio file not found: {audio_path}")

        effective_format = (
            "diarized_json"
            if self._model in _DIARIZATION_MODELS and response_format == "text"
            else response_format
        )

        kwargs: dict = {
            "model": self._model,
            "file": audio_path,
            "response_format": effective_format,
        }
        if language is not None:
            kwargs["language"] = language
        if prompt is not None:
            kwargs["prompt"] = prompt
        if self._model in _DIARIZATION_MODELS:
            kwargs["chunking_strategy"] = "auto"

        response = await self._async_client.audio.transcriptions.create(**kwargs)
        return TranscriptionResult(text=self._parse_text(response), model=self._model)

    def _parse_text(self, response: object) -> str:
        if isinstance(response, str):
            return response
        if hasattr(response, "segments") and response.segments:
            return "\n".join(
                f"[Speaker {s.speaker}]: {s.text}"
                for s in response.segments  # type: ignore[union-attr]
            )
        return response.text  # type: ignore[union-attr]


class OpenAISummarizer:
    def __init__(self, api_key: str, model: str = "gpt-4o-mini") -> None:
        self._client = OpenAI(api_key=api_key)
        self._model = model

    def summarize(
        self,
        text: str,
        *,
        system_prompt: str | None = None,
    ) -> SummarizationResult:
        messages: list[dict] = []
        if system_prompt is not None:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": text})

        response = self._client.chat.completions.create(
            model=self._model,
            messages=messages,
        )
        return SummarizationResult(
            text=response.choices[0].message.content or "",
            model=self._model,
        )
