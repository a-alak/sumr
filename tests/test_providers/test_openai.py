from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
from openai import APIConnectionError

from sumr.providers.base import Summarizer, Transcriber
from sumr.providers.openai import OpenAISummarizer, OpenAITranscriber


class TestOpenAITranscriber:
    @patch("sumr.providers.openai.OpenAI")
    def test_satisfies_protocol(self, mock_openai_cls):
        t = OpenAITranscriber(api_key="sk-test")
        assert isinstance(t, Transcriber)

    @patch("sumr.providers.openai.OpenAI")
    def test_transcribe_success(self, mock_openai_cls, tmp_path):
        mock_client = mock_openai_cls.return_value
        mock_client.audio.transcriptions.create.return_value = "hello world"

        t = OpenAITranscriber(api_key="sk-test")
        audio = tmp_path / "test.mp3"
        audio.touch()

        result = t.transcribe(audio)
        assert result.text == "hello world"
        assert result.model == "gpt-4o-mini-transcribe"

        call_kwargs = mock_client.audio.transcriptions.create.call_args
        assert call_kwargs.kwargs["model"] == "gpt-4o-mini-transcribe"
        assert call_kwargs.kwargs["file"] == audio

    @patch("sumr.providers.openai.OpenAI")
    def test_transcribe_passes_language_and_prompt(self, mock_openai_cls, tmp_path):
        mock_client = mock_openai_cls.return_value
        mock_client.audio.transcriptions.create.return_value = "transcribed"

        t = OpenAITranscriber(api_key="sk-test")
        audio = tmp_path / "test.mp3"
        audio.touch()

        t.transcribe(audio, language="en", prompt="Meeting notes")
        call_kwargs = mock_client.audio.transcriptions.create.call_args.kwargs
        assert call_kwargs["language"] == "en"
        assert call_kwargs["prompt"] == "Meeting notes"

    @patch("sumr.providers.openai.OpenAI")
    def test_transcribe_file_not_found(self, mock_openai_cls):
        t = OpenAITranscriber(api_key="sk-test")
        with pytest.raises(FileNotFoundError):
            t.transcribe(Path("/nonexistent/audio.mp3"))

    @patch("sumr.providers.openai.OpenAI")
    def test_transcribe_custom_model(self, mock_openai_cls, tmp_path):
        mock_client = mock_openai_cls.return_value
        mock_client.audio.transcriptions.create.return_value = "text"

        t = OpenAITranscriber(api_key="sk-test", model="whisper-1")
        audio = tmp_path / "test.mp3"
        audio.touch()

        result = t.transcribe(audio)
        assert result.model == "whisper-1"

    @patch("sumr.providers.openai.OpenAI")
    def test_transcribe_api_error(self, mock_openai_cls, tmp_path):
        mock_client = mock_openai_cls.return_value
        mock_client.audio.transcriptions.create.side_effect = APIConnectionError(
            request=MagicMock()
        )

        t = OpenAITranscriber(api_key="sk-test")
        audio = tmp_path / "test.mp3"
        audio.touch()

        with pytest.raises(APIConnectionError):
            t.transcribe(audio)


class TestOpenAISummarizer:
    @patch("sumr.providers.openai.OpenAI")
    def test_satisfies_protocol(self, mock_openai_cls):
        s = OpenAISummarizer(api_key="sk-test")
        assert isinstance(s, Summarizer)

    @patch("sumr.providers.openai.OpenAI")
    def test_summarize_success(self, mock_openai_cls):
        mock_client = mock_openai_cls.return_value
        mock_message = MagicMock()
        mock_message.content = "This is a summary"
        mock_choice = MagicMock()
        mock_choice.message = mock_message
        mock_response = MagicMock()
        mock_response.choices = [mock_choice]
        mock_client.chat.completions.create.return_value = mock_response

        s = OpenAISummarizer(api_key="sk-test")
        result = s.summarize("long text here")
        assert result.text == "This is a summary"
        assert result.model == "gpt-4o-mini"

    @patch("sumr.providers.openai.OpenAI")
    def test_summarize_with_system_prompt(self, mock_openai_cls):
        mock_client = mock_openai_cls.return_value
        mock_message = MagicMock()
        mock_message.content = "summary"
        mock_choice = MagicMock()
        mock_choice.message = mock_message
        mock_response = MagicMock()
        mock_response.choices = [mock_choice]
        mock_client.chat.completions.create.return_value = mock_response

        s = OpenAISummarizer(api_key="sk-test")
        s.summarize("text", system_prompt="Be concise")

        call_kwargs = mock_client.chat.completions.create.call_args.kwargs
        messages = call_kwargs["messages"]
        assert messages[0] == {"role": "system", "content": "Be concise"}
        assert messages[1] == {"role": "user", "content": "text"}

    @patch("sumr.providers.openai.OpenAI")
    def test_summarize_without_system_prompt(self, mock_openai_cls):
        mock_client = mock_openai_cls.return_value
        mock_message = MagicMock()
        mock_message.content = "summary"
        mock_choice = MagicMock()
        mock_choice.message = mock_message
        mock_response = MagicMock()
        mock_response.choices = [mock_choice]
        mock_client.chat.completions.create.return_value = mock_response

        s = OpenAISummarizer(api_key="sk-test")
        s.summarize("text")

        call_kwargs = mock_client.chat.completions.create.call_args.kwargs
        messages = call_kwargs["messages"]
        assert len(messages) == 1
        assert messages[0] == {"role": "user", "content": "text"}

    @patch("sumr.providers.openai.OpenAI")
    def test_summarize_api_error(self, mock_openai_cls):
        mock_client = mock_openai_cls.return_value
        mock_client.chat.completions.create.side_effect = APIConnectionError(
            request=MagicMock()
        )

        s = OpenAISummarizer(api_key="sk-test")
        with pytest.raises(APIConnectionError):
            s.summarize("text")
