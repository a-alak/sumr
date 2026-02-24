from collections.abc import Callable

from sumr.providers.base import Summarizer, Transcriber, TranscriberLimits
from sumr.providers.chunking import ChunkingTranscriber
from sumr.providers.openai import OpenAISummarizer, OpenAITranscriber
from sumr.providers.openai import get_limits as _openai_get_limits

TRANSCRIBERS: dict[str, Callable[..., Transcriber]] = {
    "openai": OpenAITranscriber,
}

SUMMARIZERS: dict[str, Callable[..., Summarizer]] = {
    "openai": OpenAISummarizer,
}

# Each entry maps a provider name to a callable that returns TranscriberLimits
# for a given model name (None = provider default).
LIMIT_GETTERS: dict[str, Callable[[str | None], TranscriberLimits]] = {
    "openai": _openai_get_limits,
}


def get_transcriber(
    provider: str, api_key: str, model: str | None = None
) -> Transcriber:
    if provider not in TRANSCRIBERS:
        raise ValueError(
            f"Unknown provider: {provider}. Available: {', '.join(TRANSCRIBERS)}"
        )
    kwargs: dict = {"api_key": api_key}
    if model is not None:
        kwargs["model"] = model
    inner = TRANSCRIBERS[provider](**kwargs)
    limits = LIMIT_GETTERS[provider](model)
    return ChunkingTranscriber(inner, limits)


def get_summarizer(provider: str, api_key: str, model: str | None = None) -> Summarizer:
    if provider not in SUMMARIZERS:
        raise ValueError(
            f"Unknown provider: {provider}. Available: {', '.join(SUMMARIZERS)}"
        )
    kwargs: dict = {"api_key": api_key}
    if model is not None:
        kwargs["model"] = model
    return SUMMARIZERS[provider](**kwargs)
