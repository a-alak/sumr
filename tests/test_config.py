from sumr.config import Settings


class TestSettings:
    def test_defaults_without_env_vars(self, monkeypatch):
        monkeypatch.delenv("OPENAI_API_KEY", raising=False)
        settings = Settings()
        assert settings.openai_api_key == ""
        assert settings.default_provider == "openai"
        assert settings.default_transcription_model == "gpt-4o-mini-transcribe"
        assert settings.default_summarization_model == "gpt-4o-mini"
        assert settings.default_prompt_name == "summarize"

    def test_env_var_overrides(self, monkeypatch):
        monkeypatch.setenv("OPENAI_API_KEY", "sk-test-key")
        monkeypatch.setenv("DEFAULT_PROVIDER", "custom")
        monkeypatch.setenv("DEFAULT_TRANSCRIPTION_MODEL", "whisper-1")
        monkeypatch.setenv("DEFAULT_SUMMARIZATION_MODEL", "gpt-4o")
        settings = Settings()
        assert settings.openai_api_key == "sk-test-key"
        assert settings.default_provider == "custom"
        assert settings.default_transcription_model == "whisper-1"
        assert settings.default_summarization_model == "gpt-4o"

    def test_missing_api_key_returns_empty_string(self, monkeypatch):
        monkeypatch.delenv("OPENAI_API_KEY", raising=False)
        settings = Settings()
        assert settings.openai_api_key == ""

    def test_sumr_prompts_dir_from_env(self, monkeypatch, tmp_path):
        monkeypatch.setenv("SUMR_PROMPTS_DIR", str(tmp_path))
        settings = Settings()
        assert settings.sumr_prompts_dir == tmp_path
