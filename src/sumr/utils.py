import subprocess
import sys
from pathlib import Path

SUPPORTED_AUDIO_EXTENSIONS = {
    ".mp3",
    ".mp4",
    ".mpeg",
    ".mpga",
    ".m4a",
    ".wav",
    ".webm",
    ".ogg",
    ".flac",
}


def derive_output_path(
    input_path: Path, suffix: str, output_dir: Path | None = None
) -> Path:
    """Derive output path from input path with a suffix.

    Example: audio.mp3 with suffix '_transcript' → audio_transcript.txt
    """
    stem = input_path.stem
    directory = output_dir if output_dir is not None else input_path.parent
    return directory / f"{stem}{suffix}.txt"


def validate_audio_file(path: Path) -> None:
    """Validate that the audio file exists and has a supported extension."""
    if not path.exists():
        raise FileNotFoundError(f"Audio file not found: {path}")
    if path.suffix.lower() not in SUPPORTED_AUDIO_EXTENSIONS:
        raise ValueError(
            f"Unsupported audio format: {path.suffix}. "
            f"Supported: {', '.join(sorted(SUPPORTED_AUDIO_EXTENSIONS))}"
        )


def read_input_text(path_or_stdin: str) -> str:
    """Read text from a file path or stdin (when path is '-')."""
    if path_or_stdin == "-":
        return sys.stdin.read()
    p = Path(path_or_stdin)
    if not p.exists():
        raise FileNotFoundError(f"File not found: {p}")
    return p.read_text()


def write_output(text: str, output_path: Path | None, use_stdout: bool) -> None:
    """Write text to a file or stdout."""
    if use_stdout:
        sys.stdout.write(text)
    elif output_path is not None:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(text)


def compress_audio(src: Path, dst: Path) -> None:
    """Compress audio to mono, 16 kHz, 32 kbps MP3 via ffmpeg."""
    cmd = [
        "ffmpeg",
        "-y",
        "-i",
        str(src),
        "-ac",
        "1",
        "-ar",
        "16000",
        "-b:a",
        "32k",
        str(dst),
    ]
    try:
        subprocess.run(
            cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL
        )
    except FileNotFoundError:
        raise FileNotFoundError(
            "ffmpeg not found. Install ffmpeg to handle audio files larger than 25 MB."
        ) from None
    except subprocess.CalledProcessError as exc:
        raise RuntimeError(
            f"ffmpeg compression failed (exit {exc.returncode}) for: {src}"
        ) from exc


def chunk_audio_file(src: Path, max_chunk_bytes: int, output_dir: Path) -> list[Path]:
    """Split audio at silence boundaries, keeping each chunk under max_chunk_bytes.

    The max chunk duration is derived from the file's actual bitrate so chunks
    are as large as possible. Falls back to a hard time cut if a segment has
    no detectable silence. Input should already be a compressed MP3.
    """
    duration = _get_audio_duration(src)
    bytes_per_sec = src.stat().st_size / duration
    max_chunk_duration = (max_chunk_bytes / bytes_per_sec) * 0.95  # 5% headroom

    silence_midpoints = _detect_silence_midpoints(src)
    split_points = _find_split_points(silence_midpoints, max_chunk_duration, duration)
    return _extract_chunks(src, split_points, output_dir)


def _get_audio_duration(path: Path) -> float:
    """Return audio duration in seconds via ffprobe."""
    cmd = [
        "ffprobe",
        "-v",
        "error",
        "-show_entries",
        "format=duration",
        "-of",
        "csv=p=0",
        str(path),
    ]
    try:
        result = subprocess.run(cmd, check=True, capture_output=True, text=True)
    except FileNotFoundError:
        raise FileNotFoundError(
            "ffprobe not found. Install ffmpeg (includes ffprobe) to handle"
            " audio files larger than 25 MB."
        ) from None
    except subprocess.CalledProcessError as exc:
        raise RuntimeError(
            f"ffprobe failed (exit {exc.returncode}) for: {path}"
        ) from exc
    return float(result.stdout.strip())


def _detect_silence_midpoints(
    path: Path,
    noise: str = "-40dB",
    min_silence_duration: float = 0.5,
) -> list[float]:
    """Return midpoints of silence intervals (in seconds) via ffmpeg silencedetect."""
    filter_str = f"silencedetect=noise={noise}:d={min_silence_duration}"
    cmd = ["ffmpeg", "-i", str(path), "-af", filter_str, "-f", "null", "-"]
    try:
        # -f null always exits non-zero; don't use check=True
        result = subprocess.run(cmd, capture_output=True, text=True)
    except FileNotFoundError:
        raise FileNotFoundError(
            "ffmpeg not found. Install ffmpeg to handle audio files larger than 25 MB."
        ) from None

    midpoints: list[float] = []
    silence_start: float | None = None
    for line in result.stderr.splitlines():
        if "silence_start:" in line:
            silence_start = float(line.split("silence_start:")[1].strip())
        elif "silence_end:" in line and silence_start is not None:
            silence_end = float(line.split("silence_end:")[1].split("|")[0].strip())
            midpoints.append((silence_start + silence_end) / 2)
            silence_start = None
    return midpoints


def _find_split_points(
    midpoints: list[float],
    max_duration: float,
    total_duration: float,
) -> list[float]:
    """Select split timestamps at silence midpoints so each segment <= max_duration.

    Falls back to a hard cut at max_duration if no silence is found within a segment.
    """
    split_points: list[float] = []
    chunk_start = 0.0
    last_midpoint: float | None = None

    for mp in sorted(midpoints):
        if mp - chunk_start >= max_duration:
            cut = (
                last_midpoint
                if last_midpoint is not None
                else chunk_start + max_duration
            )
            split_points.append(cut)
            chunk_start = cut
            last_midpoint = mp if mp - chunk_start < max_duration else None
        else:
            last_midpoint = mp

    if total_duration - chunk_start > max_duration:
        cut = (
            last_midpoint
            if last_midpoint is not None and last_midpoint - chunk_start < max_duration
            else chunk_start + max_duration
        )
        split_points.append(cut)

    return split_points


def _extract_chunks(
    src: Path, split_points: list[float], output_dir: Path
) -> list[Path]:
    """Extract audio chunks at the given timestamps. Uses -c copy (no re-encode)."""
    starts = [0.0] + split_points
    ends: list[float | None] = split_points + [None]
    chunks: list[Path] = []

    for i, (start, end) in enumerate(zip(starts, ends, strict=False)):
        out = output_dir / f"chunk_{i:03d}.mp3"
        cmd = ["ffmpeg", "-y", "-ss", str(start), "-i", str(src)]
        if end is not None:
            cmd += ["-t", str(end - start)]
        cmd += ["-c", "copy", str(out)]
        try:
            subprocess.run(
                cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL
            )
        except FileNotFoundError:
            raise FileNotFoundError(
                "ffmpeg not found. Install ffmpeg to handle audio"
                " files larger than 25 MB."
            ) from None
        except subprocess.CalledProcessError as exc:
            raise RuntimeError(
                f"ffmpeg chunk extraction failed (exit {exc.returncode})"
            ) from exc
        chunks.append(out)

    return chunks
