from unittest.mock import MagicMock, patch

from openai import APIConnectionError
from typer.testing import CliRunner

from sumr.cli import app
from sumr.providers.base import SummarizationResult, TranscriptionResult

runner = CliRunner()


def _mock_transcriber(text="transcribed text"):
    t = MagicMock()
    t.transcribe.return_value = TranscriptionResult(text=text, model="test-model")
    return t


def _mock_summarizer(text="summary text"):
    s = MagicMock()
    s.summarize.return_value = SummarizationResult(text=text, model="test-model")
    return s


class TestTranscribeCommand:
    def test_missing_api_key(self, tmp_path):
        audio = tmp_path / "test.mp3"
        audio.touch()
        result = runner.invoke(app, ["transcribe", str(audio)])
        assert result.exit_code == 1
        assert "OPENAI_API_KEY" in result.output

    @patch("sumr.cli.get_transcriber")
    def test_file_not_found(self, mock_get, monkeypatch):
        monkeypatch.setenv("OPENAI_API_KEY", "sk-test")
        result = runner.invoke(app, ["transcribe", "/nonexistent/audio.mp3"])
        assert result.exit_code == 1
        assert "not found" in result.output

    @patch("sumr.cli.get_transcriber")
    def test_unsupported_format(self, mock_get, monkeypatch, tmp_path):
        monkeypatch.setenv("OPENAI_API_KEY", "sk-test")
        bad = tmp_path / "audio.xyz"
        bad.touch()
        result = runner.invoke(app, ["transcribe", str(bad)])
        assert result.exit_code == 1
        assert "Unsupported" in result.output

    @patch("sumr.cli.get_transcriber")
    def test_output_file_created(self, mock_get, monkeypatch, tmp_path):
        monkeypatch.setenv("OPENAI_API_KEY", "sk-test")
        mock_get.return_value = _mock_transcriber()

        audio = tmp_path / "meeting.mp3"
        audio.touch()
        out = tmp_path / "out.txt"

        result = runner.invoke(app, ["transcribe", str(audio), "-o", str(out), "-q"])
        assert result.exit_code == 0
        assert out.read_text() == "transcribed text"

    @patch("sumr.cli.sys")
    @patch("sumr.cli.get_transcriber")
    def test_auto_derive_output(self, mock_get, mock_sys, monkeypatch, tmp_path):
        monkeypatch.setenv("OPENAI_API_KEY", "sk-test")
        mock_get.return_value = _mock_transcriber()
        mock_sys.stdout.isatty.return_value = True

        audio = tmp_path / "call.mp3"
        audio.touch()

        result = runner.invoke(app, ["transcribe", str(audio), "-q"])
        assert result.exit_code == 0
        expected = tmp_path / "call_transcript.txt"
        assert expected.read_text() == "transcribed text"

    @patch("sumr.cli.get_transcriber")
    def test_api_error_exits_2(self, mock_get, monkeypatch, tmp_path):
        monkeypatch.setenv("OPENAI_API_KEY", "sk-test")
        t = MagicMock()
        t.transcribe.side_effect = APIConnectionError(request=MagicMock())
        mock_get.return_value = t

        audio = tmp_path / "test.mp3"
        audio.touch()

        result = runner.invoke(app, ["transcribe", str(audio), "-q"])
        assert result.exit_code == 2


class TestSummarizeCommand:
    def test_missing_api_key(self, tmp_path):
        f = tmp_path / "transcript.txt"
        f.write_text("some text")
        result = runner.invoke(app, ["summarize", str(f)])
        assert result.exit_code == 1
        assert "OPENAI_API_KEY" in result.output

    @patch("sumr.cli.get_summarizer")
    def test_file_not_found(self, mock_get, monkeypatch):
        monkeypatch.setenv("OPENAI_API_KEY", "sk-test")
        result = runner.invoke(app, ["summarize", "/nonexistent/file.txt"])
        assert result.exit_code == 1

    @patch("sumr.cli.get_summarizer")
    def test_empty_input(self, mock_get, monkeypatch, tmp_path):
        monkeypatch.setenv("OPENAI_API_KEY", "sk-test")
        f = tmp_path / "empty.txt"
        f.write_text("   ")
        result = runner.invoke(app, ["summarize", str(f)])
        assert result.exit_code == 1
        assert "empty" in result.output.lower()

    @patch("sumr.cli.get_summarizer")
    def test_output_file_created(self, mock_get, monkeypatch, tmp_path):
        monkeypatch.setenv("OPENAI_API_KEY", "sk-test")
        mock_get.return_value = _mock_summarizer()

        f = tmp_path / "transcript.txt"
        f.write_text("long text to summarize")
        out = tmp_path / "summary.txt"

        result = runner.invoke(app, ["summarize", str(f), "-o", str(out), "-q"])
        assert result.exit_code == 0
        assert out.read_text() == "summary text"

    @patch("sumr.cli.get_summarizer")
    def test_custom_prompt_passthrough(self, mock_get, monkeypatch, tmp_path):
        monkeypatch.setenv("OPENAI_API_KEY", "sk-test")
        s = _mock_summarizer()
        mock_get.return_value = s

        f = tmp_path / "transcript.txt"
        f.write_text("text")
        out = tmp_path / "out.txt"

        runner.invoke(
            app,
            ["summarize", str(f), "--prompt", "Be very brief", "-o", str(out), "-q"],
        )
        s.summarize.assert_called_once_with("text", system_prompt="Be very brief")

    @patch("sumr.cli.get_summarizer")
    def test_stdin_input(self, mock_get, monkeypatch, tmp_path):
        monkeypatch.setenv("OPENAI_API_KEY", "sk-test")
        mock_get.return_value = _mock_summarizer()

        out = tmp_path / "out.txt"
        result = runner.invoke(
            app, ["summarize", "-", "-o", str(out), "-q"], input="stdin text"
        )
        assert result.exit_code == 0
        assert out.read_text() == "summary text"

    @patch("sumr.cli.get_summarizer")
    def test_api_error_exits_2(self, mock_get, monkeypatch, tmp_path):
        monkeypatch.setenv("OPENAI_API_KEY", "sk-test")
        s = MagicMock()
        s.summarize.side_effect = APIConnectionError(request=MagicMock())
        mock_get.return_value = s

        f = tmp_path / "t.txt"
        f.write_text("text")

        result = runner.invoke(app, ["summarize", str(f), "-q"])
        assert result.exit_code == 2


class TestDefaultCommand:
    @patch("sumr.cli.get_summarizer")
    @patch("sumr.cli.get_transcriber")
    def test_full_pipeline(self, mock_get_t, mock_get_s, monkeypatch, tmp_path):
        monkeypatch.setenv("OPENAI_API_KEY", "sk-test")
        mock_get_t.return_value = _mock_transcriber("the transcript")
        mock_get_s.return_value = _mock_summarizer("the summary")

        audio = tmp_path / "meeting.mp3"
        audio.touch()
        summary_out = tmp_path / "s.txt"
        transcript_out = tmp_path / "t.txt"

        result = runner.invoke(
            app,
            [
                str(audio),
                "-o",
                str(summary_out),
                "--transcript-output",
                str(transcript_out),
                "-q",
            ],
        )
        assert result.exit_code == 0
        assert summary_out.read_text() == "the summary"
        assert transcript_out.read_text() == "the transcript"

    @patch("sumr.cli.sys")
    @patch("sumr.cli.get_summarizer")
    @patch("sumr.cli.get_transcriber")
    def test_auto_derives_both_files(
        self, mock_get_t, mock_get_s, mock_sys, monkeypatch, tmp_path
    ):
        monkeypatch.setenv("OPENAI_API_KEY", "sk-test")
        mock_get_t.return_value = _mock_transcriber("transcript")
        mock_get_s.return_value = _mock_summarizer("summary")
        mock_sys.stdout.isatty.return_value = True

        audio = tmp_path / "call.mp3"
        audio.touch()

        result = runner.invoke(app, [str(audio), "-q"])
        assert result.exit_code == 0
        assert (tmp_path / "call_transcript.txt").read_text() == "transcript"
        assert (tmp_path / "call_summary.txt").read_text() == "summary"

    def test_missing_api_key(self, tmp_path):
        audio = tmp_path / "test.mp3"
        audio.touch()
        result = runner.invoke(app, [str(audio)])
        assert result.exit_code == 1

    @patch("sumr.cli.get_summarizer")
    @patch("sumr.cli.get_transcriber")
    def test_unknown_provider(self, mock_get_t, mock_get_s, monkeypatch, tmp_path):
        monkeypatch.setenv("OPENAI_API_KEY", "sk-test")
        mock_get_t.side_effect = ValueError("Unknown provider: bad")

        audio = tmp_path / "test.mp3"
        audio.touch()

        result = runner.invoke(app, [str(audio), "-p", "bad", "-q"])
        assert result.exit_code == 1
        assert "Unknown provider" in result.output

    @patch("sumr.cli.get_summarizer")
    @patch("sumr.cli.get_transcriber")
    def test_quiet_suppresses_status(
        self, mock_get_t, mock_get_s, monkeypatch, tmp_path
    ):
        monkeypatch.setenv("OPENAI_API_KEY", "sk-test")
        mock_get_t.return_value = _mock_transcriber()
        mock_get_s.return_value = _mock_summarizer()

        audio = tmp_path / "test.mp3"
        audio.touch()

        result = runner.invoke(app, [str(audio), "-q"])
        assert result.exit_code == 0
        assert "Transcribing" not in result.output
        assert "Summarizing" not in result.output
