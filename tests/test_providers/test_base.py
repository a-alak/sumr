from pathlib import Path

import pytest

from sumr.providers.base import (
    SummarizationResult,
    Summarizer,
    Transcriber,
    TranscriptionResult,
)


class TestDataclasses:
    def test_transcription_result(self):
        result = TranscriptionResult(text="hello world", model="whisper-1")
        assert result.text == "hello world"
        assert result.model == "whisper-1"

    def test_summarization_result(self):
        result = SummarizationResult(text="summary", model="gpt-4o")
        assert result.text == "summary"
        assert result.model == "gpt-4o"

    def test_results_are_frozen(self):
        result = TranscriptionResult(text="hello", model="m")
        with pytest.raises(AttributeError):
            result.text = "changed"  # type: ignore[misc]


class TestProtocolConformance:
    def test_valid_transcriber_satisfies_protocol(self):
        class MyTranscriber:
            def transcribe(
                self,
                audio_path: Path,
                *,
                language: str | None = None,
                prompt: str | None = None,
                response_format: str = "text",
            ) -> TranscriptionResult:
                return TranscriptionResult(text="", model="")

        assert isinstance(MyTranscriber(), Transcriber)

    def test_missing_method_does_not_satisfy_transcriber(self):
        class NotATranscriber:
            pass

        assert not isinstance(NotATranscriber(), Transcriber)

    def test_valid_summarizer_satisfies_protocol(self):
        class MySummarizer:
            def summarize(
                self,
                text: str,
                *,
                system_prompt: str | None = None,
            ) -> SummarizationResult:
                return SummarizationResult(text="", model="")

        assert isinstance(MySummarizer(), Summarizer)

    def test_missing_method_does_not_satisfy_summarizer(self):
        class NotASummarizer:
            pass

        assert not isinstance(NotASummarizer(), Summarizer)
