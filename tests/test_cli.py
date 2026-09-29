"""CLI wiring: the LLM is opt-in, and check-note's exit code reflects failures."""

import pytest

from euro_rates_monitor import cli, note


def test_note_defaults_to_scaffold(monkeypatch):
    seen = {}
    monkeypatch.setattr(note, "write_note", lambda **kw: seen.update(kw) or "")
    cli.main(["note"])
    assert seen["use_llm"] is False


def test_llm_is_behind_an_explicit_flag(monkeypatch):
    seen = {}
    monkeypatch.setattr(note, "write_note", lambda **kw: seen.update(kw) or "")
    cli.main(["note", "--llm"])
    assert seen["use_llm"] is True


def test_check_note_exit_codes(tmp_path, monkeypatch, capsys):
    good, bad = tmp_path / "2026-09-28.md", tmp_path / "2026-10-05.md"
    good.write_text("ok")
    bad.write_text("bad")
    monkeypatch.setattr(note, "check_note", lambda p: [] if p == good else ["not in x: 90"])
    assert cli.main(["check-note", str(good)]) == 0
    with pytest.raises(SystemExit, match="1 of 2"):
        cli.main(["check-note", str(tmp_path)])
    assert "FAIL" in capsys.readouterr().out


def test_check_note_on_empty_directory_passes(tmp_path, capsys):
    assert cli.main(["check-note", str(tmp_path)]) == 0
    assert "no notes to check" in capsys.readouterr().out
