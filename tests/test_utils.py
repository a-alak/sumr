from io import StringIO
from pathlib import Path
from unittest.mock import patch

import pytest

from sumr.utils import (
    derive_output_path,
    read_input_text,
    validate_audio_file,
    write_output,
)


class TestDeriveOutputPath:
    def test_transcript_suffix(self, tmp_path):
        result = derive_output_path(tmp_path / "audio.mp3", "_transcript")
        assert result == tmp_path / "audio_transcript.txt"

    def test_summary_suffix(self, tmp_path):
        result = derive_output_path(tmp_path / "meeting.wav", "_summary")
        assert result == tmp_path / "meeting_summary.txt"

    def test_custom_output_dir(self, tmp_path):
        out_dir = tmp_path / "output"
        result = derive_output_path(Path("audio.mp3"), "_transcript", out_dir)
        assert result == out_dir / "audio_transcript.txt"


class TestValidateAudioFile:
    def test_valid_file(self, tmp_path):
        f = tmp_path / "audio.mp3"
        f.touch()
        validate_audio_file(f)  # should not raise

    def test_file_not_found(self, tmp_path):
        with pytest.raises(FileNotFoundError, match="not found"):
            validate_audio_file(tmp_path / "missing.mp3")

    def test_unsupported_extension(self, tmp_path):
        f = tmp_path / "audio.txt"
        f.touch()
        with pytest.raises(ValueError, match="Unsupported audio format"):
            validate_audio_file(f)

    @pytest.mark.parametrize("ext", [".mp3", ".wav", ".flac", ".ogg", ".m4a", ".webm"])
    def test_supported_extensions(self, tmp_path, ext):
        f = tmp_path / f"audio{ext}"
        f.touch()
        validate_audio_file(f)  # should not raise


class TestReadInputText:
    def test_read_from_file(self, tmp_path):
        f = tmp_path / "input.txt"
        f.write_text("hello world")
        assert read_input_text(str(f)) == "hello world"

    def test_file_not_found(self):
        with pytest.raises(FileNotFoundError, match="not found"):
            read_input_text("/nonexistent/file.txt")

    def test_read_from_stdin(self):
        with patch("sumr.utils.sys.stdin", new=StringIO("stdin content")):
            assert read_input_text("-") == "stdin content"


class TestWriteOutput:
    def test_write_to_file(self, tmp_path):
        out = tmp_path / "output.txt"
        write_output("content", out, use_stdout=False)
        assert out.read_text() == "content"

    def test_write_to_stdout(self, capsys):
        write_output("stdout content", None, use_stdout=True)
        assert capsys.readouterr().out == "stdout content"

    def test_creates_parent_dirs(self, tmp_path):
        out = tmp_path / "sub" / "dir" / "output.txt"
        write_output("content", out, use_stdout=False)
        assert out.read_text() == "content"
