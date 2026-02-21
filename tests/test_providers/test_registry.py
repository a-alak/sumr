from unittest.mock import patch

import pytest

from sumr.providers import get_summarizer, get_transcriber
from sumr.providers.base import Summarizer, Transcriber


class TestGetTranscriber:
    @patch("sumr.providers.openai.OpenAI")
    def test_valid_provider(self, mock_openai_cls):
        t = get_transcriber("openai", api_key="sk-test")
        assert isinstance(t, Transcriber)

    def test_unknown_provider(self):
        with pytest.raises(ValueError, match="Unknown provider"):
            get_transcriber("unknown", api_key="sk-test")

    @patch("sumr.providers.openai.OpenAI")
    def test_model_passthrough(self, mock_openai_cls):
        t = get_transcriber("openai", api_key="sk-test", model="whisper-1")
        assert t._inner._model == "whisper-1"


class TestGetSummarizer:
    @patch("sumr.providers.openai.OpenAI")
    def test_valid_provider(self, mock_openai_cls):
        s = get_summarizer("openai", api_key="sk-test")
        assert isinstance(s, Summarizer)

    def test_unknown_provider(self):
        with pytest.raises(ValueError, match="Unknown provider"):
            get_summarizer("unknown", api_key="sk-test")

    @patch("sumr.providers.openai.OpenAI")
    def test_model_passthrough(self, mock_openai_cls):
        s = get_summarizer("openai", api_key="sk-test", model="gpt-4o")
        assert s._model == "gpt-4o"
