import pytest

from sumr.prompts import PromptNotFoundError, load_prompt


class TestLoadPrompt:
    def test_bundled_fallback(self):
        content = load_prompt("summarize")
        assert "You are an expert" in content

    def test_bundled_not_found_raises(self):
        with pytest.raises(PromptNotFoundError, match="nonexistent_xyz"):
            load_prompt("nonexistent_xyz")

    def test_error_message_lists_searched_paths(self):
        with pytest.raises(PromptNotFoundError) as exc_info:
            load_prompt("missing_prompt")
        msg = str(exc_info.value)
        assert "missing_prompt.md" in msg
        assert "<bundled>" in msg

    def test_filesystem_lookup(self, tmp_path):
        prompt_dir = tmp_path / "prompts"
        prompt_dir.mkdir()
        (prompt_dir / "custom.md").write_text("My custom prompt")

        content = load_prompt("custom", extra_dirs=[prompt_dir])
        assert content == "My custom prompt"

    def test_extra_dirs_take_priority_over_bundled(self, tmp_path):
        prompt_dir = tmp_path / "prompts"
        prompt_dir.mkdir()
        (prompt_dir / "summarize.md").write_text("Override summarize prompt")

        content = load_prompt("summarize", extra_dirs=[prompt_dir])
        assert content == "Override summarize prompt"

    def test_xdg_dir_takes_priority_over_bundled(self, tmp_path, monkeypatch):
        xdg_prompts = tmp_path / "sumr" / "prompts"
        xdg_prompts.mkdir(parents=True)
        (xdg_prompts / "summarize.md").write_text("XDG summarize prompt")

        monkeypatch.setenv("XDG_CONFIG_HOME", str(tmp_path))
        content = load_prompt("summarize")
        assert content == "XDG summarize prompt"

    def test_extra_dirs_take_priority_over_xdg(self, tmp_path, monkeypatch):
        xdg_prompts = tmp_path / "xdg" / "sumr" / "prompts"
        xdg_prompts.mkdir(parents=True)
        (xdg_prompts / "myp.md").write_text("XDG version")

        extra_dir = tmp_path / "extra"
        extra_dir.mkdir()
        (extra_dir / "myp.md").write_text("Extra version")

        monkeypatch.setenv("XDG_CONFIG_HOME", str(tmp_path / "xdg"))
        content = load_prompt("myp", extra_dirs=[extra_dir])
        assert content == "Extra version"

    def test_falls_through_to_bundled_when_filesystem_missing(self, tmp_path):
        empty_dir = tmp_path / "empty"
        empty_dir.mkdir()

        content = load_prompt("summarize", extra_dirs=[empty_dir])
        assert "You are an expert" in content
