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

PROVIDER_MAX_UPLOAD_BYTES: dict[str, int] = {
    "openai": 25 * 1024 * 1024,
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
    max_bytes = PROVIDER_MAX_UPLOAD_BYTES.get(provider, 25 * 1024 * 1024)
    return ChunkingTranscriber(inner, max_upload_bytes=max_bytes)


def get_summarizer(provider: str, api_key: str, model: str | None = None) -> Summarizer:
    if provider not in SUMMARIZERS:
        raise ValueError(
            f"Unknown provider: {provider}. Available: {', '.join(SUMMARIZERS)}"
        )
    kwargs: dict = {"api_key": api_key}
    if model is not None:
        kwargs["model"] = model
    return SUMMARIZERS[provider](**kwargs)
