from __future__ import annotations

import importlib.resources
import os
from pathlib import Path


class PromptNotFoundError(ValueError):
    pass


def _xdg_prompts_dir() -> Path:
    xdg = os.environ.get("XDG_CONFIG_HOME")
    base = Path(xdg) if xdg else Path.home() / ".config"
    return base / "sumr" / "prompts"


def _search_dirs(extra_dirs: list[Path] | None = None) -> list[Path]:
    dirs: list[Path] = []
    if extra_dirs:
        dirs.extend(extra_dirs)
    dirs.append(_xdg_prompts_dir())
    return dirs


def _load_bundled(name: str) -> str | None:
    try:
        f = importlib.resources.files("sumr.prompts") / f"{name}.md"
        return f.read_text(encoding="utf-8")
    except FileNotFoundError, TypeError:
        return None


def load_prompt(name: str, extra_dirs: list[Path] | None = None) -> str:
    """Load a named prompt as a string (used as system message).

    Search order: extra_dirs → $XDG_CONFIG_HOME/sumr/prompts/ → bundled.
    Raises PromptNotFoundError if not found anywhere.
    """
    for search_dir in _search_dirs(extra_dirs):
        candidate = search_dir / f"{name}.md"
        if candidate.is_file():
            return candidate.read_text(encoding="utf-8")
    bundled = _load_bundled(name)
    if bundled is not None:
        return bundled
    searched = [str(d / f"{name}.md") for d in _search_dirs(extra_dirs)] + [
        f"<bundled>/{name}.md"
    ]
    raise PromptNotFoundError(
        f"Prompt '{name}' not found. Searched: {', '.join(searched)}"
    )
