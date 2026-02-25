from collections.abc import Callable

from sumr.providers.base import Summarizer, Transcriber
from sumr.providers.chunking import ChunkingTranscriber
from sumr.providers.openai import OpenAISummarizer, OpenAITranscriber

TRANSCRIBERS: dict[str, Callable[..., Transcriber]] = {
    "openai": OpenAITranscriber,
}

SUMMARIZERS: dict[str, Callable[..., Summarizer]] = {
    "openai": OpenAISummarizer,
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
    return ChunkingTranscriber(inner)


def get_summarizer(provider: str, api_key: str, model: str | None = None) -> Summarizer:
    if provider not in SUMMARIZERS:
        raise ValueError(
            f"Unknown provider: {provider}. Available: {', '.join(SUMMARIZERS)}"
        )
    kwargs: dict = {"api_key": api_key}
    if model is not None:
        kwargs["model"] = model
    return SUMMARIZERS[provider](**kwargs)
