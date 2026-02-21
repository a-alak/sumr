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
