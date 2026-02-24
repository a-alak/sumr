import subprocess
from io import StringIO
from pathlib import Path
from unittest.mock import patch

import pytest

from sumr.utils import (
    chunk_audio_file,
    compress_audio,
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


class TestCompressAudio:
    @patch("sumr.utils.subprocess.run")
    def test_ffmpeg_args(self, mock_run, tmp_path):
        compress_audio(tmp_path / "in.m4a", tmp_path / "out.mp3")
        cmd = mock_run.call_args.args[0]
        assert cmd[0] == "ffmpeg" and "-y" in cmd
        assert "-ac" in cmd and cmd[cmd.index("-ac") + 1] == "1"
        assert "-b:a" in cmd and cmd[cmd.index("-b:a") + 1] == "32k"

    @patch("sumr.utils.subprocess.run", side_effect=FileNotFoundError)
    def test_missing_ffmpeg(self, _mock, tmp_path):
        with pytest.raises(FileNotFoundError, match="ffmpeg not found"):
            compress_audio(tmp_path / "in.mp3", tmp_path / "out.mp3")

    @patch(
        "sumr.utils.subprocess.run",
        side_effect=subprocess.CalledProcessError(1, "ffmpeg"),
    )
    def test_nonzero_exit(self, _mock, tmp_path):
        with pytest.raises(RuntimeError, match="ffmpeg compression failed"):
            compress_audio(tmp_path / "in.mp3", tmp_path / "out.mp3")


class TestFindSplitPoints:
    def test_no_silence_produces_hard_cut(self):
        from sumr.utils import _find_split_points

        assert _find_split_points([], max_duration=10.0, total_duration=25.0) == [10.0]

    def test_splits_at_silence_before_limit(self):
        from sumr.utils import _find_split_points

        result = _find_split_points(
            [8.0, 12.0, 22.0, 26.0], max_duration=15.0, total_duration=35.0
        )
        assert result == [12.0, 26.0]

    def test_no_split_needed_for_short_audio(self):
        from sumr.utils import _find_split_points

        result = _find_split_points([5.0, 10.0], max_duration=60.0, total_duration=15.0)
        assert result == []


class TestChunkAudioFile:
    @patch("sumr.utils._extract_chunks")
    @patch("sumr.utils._find_split_points", return_value=[30.0])
    @patch("sumr.utils._detect_silence_midpoints", return_value=[10.0, 30.0, 55.0])
    @patch("sumr.utils.get_audio_duration", return_value=60.0)
    def test_derives_max_duration_from_bitrate(
        self, mock_dur, mock_silence, mock_split, mock_extract, tmp_path
    ):
        src = tmp_path / "compressed.mp3"
        src.write_bytes(b"\x00" * 4000)  # 4000 bytes / 60s ≈ 66 bytes/sec
        chunk_audio_file(src, 1000, tmp_path)
        args = mock_split.call_args.args
        expected = (1000 / (4000 / 60)) * 0.95
        assert abs(args[1] - expected) < 0.01
