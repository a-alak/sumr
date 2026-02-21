from pathlib import Path

from openai import OpenAI

from sumr.providers.base import SummarizationResult, TranscriptionResult


class OpenAITranscriber:
    def __init__(self, api_key: str, model: str = "gpt-4o-mini-transcribe") -> None:
        self._client = OpenAI(api_key=api_key)
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

        kwargs: dict = {
            "model": self._model,
            "file": audio_path,
            "response_format": response_format,
        }
        if language is not None:
            kwargs["language"] = language
        if prompt is not None:
            kwargs["prompt"] = prompt

        response = self._client.audio.transcriptions.create(**kwargs)
        text = response if isinstance(response, str) else response.text
        return TranscriptionResult(text=text, model=self._model)


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
