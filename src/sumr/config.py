from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    openai_api_key: str = ""
    default_provider: str = "openai"
    default_transcription_model: str = "gpt-4o-mini-transcribe"
    default_summarization_model: str = "gpt-4o-mini"
    default_summary_prompt: str = (
        "Summarize the following transcription concisely, "
        "capturing the key points and main ideas."
    )
