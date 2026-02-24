import sys
from pathlib import Path
from typing import Annotated

import typer
from openai import APIError
from rich.console import Console
from typer.core import TyperGroup

from sumr.config import Settings
from sumr.prompts import PromptNotFoundError, load_prompt
from sumr.providers import get_summarizer, get_transcriber
from sumr.utils import (
    derive_output_path,
    read_input_text,
    validate_audio_file,
    write_output,
)


class DefaultGroup(TyperGroup):
    """A Typer group that falls back to the 'run' command for unknown args."""

    def parse_args(self, ctx, args):
        if args and args[0] not in self.commands and not args[0].startswith("-"):
            args = ["run"] + args
        return super().parse_args(ctx, args)


app = typer.Typer(
    name="sumr",
    cls=DefaultGroup,
    help="Transcribe and summarize audio files using LLM providers.",
    no_args_is_help=True,
)
err_console = Console(stderr=True)


def _require_api_key(settings: Settings) -> str:
    if not settings.openai_api_key:
        err_console.print("[red]Error:[/red] OPENAI_API_KEY is not set.")
        raise typer.Exit(code=1)
    return settings.openai_api_key


def _error(message: str, code: int = 1) -> None:
    err_console.print(f"[red]Error:[/red] {message}")
    raise typer.Exit(code=code)


def _api_error(e: APIError) -> None:
    err_console.print(f"[red]API Error:[/red] {e}")
    raise typer.Exit(code=2)


class _nullcontext:
    def __enter__(self):
        return self

    def __exit__(self, *args):
        pass


def _status(message: str, quiet: bool):
    if quiet:
        return _nullcontext()
    return err_console.status(message, spinner="dots")


def _resolve_prompt(
    prompt: str | None, system_prompt: str | None, settings: Settings
) -> str:
    """Resolve which system prompt to use for summarization.

    Priority: --system-prompt (inline) > --prompt (name or file) > default named prompt.
    """
    if system_prompt is not None:
        return system_prompt
    extra_dirs = [settings.sumr_prompts_dir] if settings.sumr_prompts_dir else None
    if prompt is not None:
        p = Path(prompt)
        if p.is_file():
            return p.read_text(encoding="utf-8")
        try:
            return load_prompt(prompt, extra_dirs=extra_dirs)
        except PromptNotFoundError as e:
            _error(str(e))
    try:
        return load_prompt(settings.default_prompt_name, extra_dirs=extra_dirs)
    except PromptNotFoundError as e:
        _error(str(e))
    return ""  # unreachable


@app.command()
def transcribe(
    audio: Annotated[Path, typer.Argument(help="Path to audio file")],
    output: Annotated[
        Path | None, typer.Option("--output", "-o", help="Output file path")
    ] = None,
    model: Annotated[
        str | None, typer.Option("--model", "-m", help="Transcription model")
    ] = None,
    provider: Annotated[
        str | None, typer.Option("--provider", "-P", help="Provider name")
    ] = None,
    language: Annotated[
        str | None, typer.Option("--language", "-l", help="Audio language")
    ] = None,
    prompt: Annotated[
        str | None, typer.Option("--prompt", help="Transcription prompt/context")
    ] = None,
    response_format: Annotated[
        str, typer.Option("--format", "-f", help="Response format")
    ] = "text",
    verbose: Annotated[
        bool, typer.Option("--verbose", "-v", help="Show detailed output")
    ] = False,
    quiet: Annotated[
        bool, typer.Option("--quiet", "-q", help="Suppress status output")
    ] = False,
) -> None:
    """Transcribe an audio file."""
    settings = Settings()
    api_key = _require_api_key(settings)
    prov = provider or settings.default_provider
    mdl = model or settings.default_transcription_model

    try:
        validate_audio_file(audio)
    except (FileNotFoundError, ValueError) as e:
        _error(str(e))

    try:
        transcriber = get_transcriber(prov, api_key, mdl)
    except ValueError as e:
        _error(str(e))

    if verbose and not quiet:
        err_console.print(f"Model: {mdl}, File: {audio} ({audio.stat().st_size} bytes)")

    try:
        with _status("Transcribing...", quiet):
            result = transcriber.transcribe(
                audio,
                language=language,
                prompt=prompt,
                response_format=response_format,
            )
    except APIError as e:
        _api_error(e)

    use_stdout = output is None and not sys.stdout.isatty()
    if output is None and not use_stdout:
        output = derive_output_path(audio, "_transcript")

    write_output(result.text, output, use_stdout)

    if not quiet and not use_stdout:
        err_console.print(f"[green]Transcript saved to {output}[/green]")


@app.command()
def summarize(
    input_file: Annotated[
        str, typer.Argument(help="Path to text file or '-' for stdin")
    ],
    output: Annotated[
        Path | None, typer.Option("--output", "-o", help="Output file path")
    ] = None,
    model: Annotated[
        str | None, typer.Option("--model", "-m", help="Summarization model")
    ] = None,
    provider: Annotated[
        str | None, typer.Option("--provider", "-P", help="Provider name")
    ] = None,
    prompt: Annotated[
        str | None, typer.Option("--prompt", "-p", help="Named prompt or file path")
    ] = None,
    system_prompt_text: Annotated[
        str | None,
        typer.Option("--system-prompt", help="Inline system prompt override"),
    ] = None,
    verbose: Annotated[
        bool, typer.Option("--verbose", "-v", help="Show detailed output")
    ] = False,
    quiet: Annotated[
        bool, typer.Option("--quiet", "-q", help="Suppress status output")
    ] = False,
) -> None:
    """Summarize a text file or stdin."""
    settings = Settings()
    api_key = _require_api_key(settings)
    prov = provider or settings.default_provider
    mdl = model or settings.default_summarization_model
    system_prompt = _resolve_prompt(prompt, system_prompt_text, settings)

    try:
        text = read_input_text(input_file)
    except FileNotFoundError as e:
        _error(str(e))

    if not text.strip():
        _error("Input is empty.")

    try:
        summarizer = get_summarizer(prov, api_key, mdl)
    except ValueError as e:
        _error(str(e))

    if verbose and not quiet:
        err_console.print(f"Model: {mdl}")

    try:
        with _status("Summarizing...", quiet):
            result = summarizer.summarize(text, system_prompt=system_prompt)
    except APIError as e:
        _api_error(e)

    use_stdout = output is None and not sys.stdout.isatty()
    if output is None and not use_stdout:
        if input_file == "-":
            output = Path("summary.txt")
        else:
            output = derive_output_path(Path(input_file), "_summary")

    write_output(result.text, output, use_stdout)

    if not quiet and not use_stdout:
        err_console.print(f"[green]Summary saved to {output}[/green]")


@app.command(hidden=True)
def run(
    audio: Annotated[Path, typer.Argument(help="Path to audio file")],
    output: Annotated[
        Path | None,
        typer.Option("--output", "-o", help="Summary output file path"),
    ] = None,
    transcript_output: Annotated[
        Path | None,
        typer.Option("--transcript-output", help="Transcript output file path"),
    ] = None,
    model: Annotated[
        str | None, typer.Option("--model", "-m", help="Model for both steps")
    ] = None,
    provider: Annotated[
        str | None, typer.Option("--provider", "-P", help="Provider name")
    ] = None,
    language: Annotated[
        str | None, typer.Option("--language", "-l", help="Audio language")
    ] = None,
    prompt: Annotated[
        str | None, typer.Option("--prompt", "-p", help="Named prompt or file path")
    ] = None,
    system_prompt_text: Annotated[
        str | None,
        typer.Option("--system-prompt", help="Inline system prompt override"),
    ] = None,
    response_format: Annotated[
        str, typer.Option("--format", "-f", help="Transcription format")
    ] = "text",
    verbose: Annotated[
        bool, typer.Option("--verbose", "-v", help="Show detailed output")
    ] = False,
    quiet: Annotated[
        bool, typer.Option("--quiet", "-q", help="Suppress status output")
    ] = False,
) -> None:
    """Transcribe and summarize an audio file (default command)."""
    settings = Settings()
    api_key = _require_api_key(settings)
    prov = provider or settings.default_provider
    t_model = model or settings.default_transcription_model
    s_model = model or settings.default_summarization_model
    system_prompt = _resolve_prompt(prompt, system_prompt_text, settings)

    try:
        validate_audio_file(audio)
    except (FileNotFoundError, ValueError) as e:
        _error(str(e))

    try:
        transcriber = get_transcriber(prov, api_key, t_model)
        summarizer = get_summarizer(prov, api_key, s_model)
    except ValueError as e:
        _error(str(e))

    if verbose and not quiet:
        err_console.print(
            f"Transcription model: {t_model}, Summarization model: {s_model}"
        )
        err_console.print(f"File: {audio} ({audio.stat().st_size} bytes)")

    # Transcribe
    try:
        with _status("Transcribing...", quiet):
            t_result = transcriber.transcribe(
                audio,
                language=language,
                response_format=response_format,
            )
    except APIError as e:
        _api_error(e)

    # Save transcript
    is_piped = not sys.stdout.isatty()
    if transcript_output is not None:
        write_output(t_result.text, transcript_output, use_stdout=False)
        if not quiet:
            err_console.print(f"[green]Transcript saved to {transcript_output}[/green]")
    elif not is_piped:
        auto_transcript = derive_output_path(audio, "_transcript")
        write_output(t_result.text, auto_transcript, use_stdout=False)
        if not quiet:
            err_console.print(f"[green]Transcript saved to {auto_transcript}[/green]")

    # Summarize
    try:
        with _status("Summarizing...", quiet):
            s_result = summarizer.summarize(t_result.text, system_prompt=system_prompt)
    except APIError as e:
        _api_error(e)

    # Output summary
    use_stdout = output is None and is_piped
    if output is None and not use_stdout:
        output = derive_output_path(audio, "_summary")

    write_output(s_result.text, output, use_stdout)

    if not quiet and not use_stdout:
        err_console.print(f"[green]Summary saved to {output}[/green]")
