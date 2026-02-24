from pathlib import Path

from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    openai_api_key: str = ""
    default_provider: str = "openai"
    default_transcription_model: str = "gpt-4o-transcribe-diarize"
    default_summarization_model: str = "gpt-5.2"
    default_prompt_name: str = "summarize"
    sumr_prompts_dir: Path | None = None
