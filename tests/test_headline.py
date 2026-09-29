"""README headline: wording rules, number validation, and the marker replacement."""

import json
from pathlib import Path

import pytest

from euro_rates_monitor import headline, paths
from euro_rates_monitor.note import validate

LATEST = Path(__file__).parents[1] / "data" / "processed" / "metrics_latest.json"


@pytest.mark.parametrize("pct, phrase", [
    (100, "at the top"), (95, "near the top"), (50, "50% of the way"),
    (5, "near the bottom"), (0, "at the bottom"),
])
def test_range_phrase(pct, phrase):
    assert phrase in headline._range_phrase(pct)


@pytest.mark.skipif(not LATEST.exists(), reason="no processed metrics; run `erm build`")
def test_headline_uses_only_dictionary_numbers():
    m = json.loads(LATEST.read_text())
    validate({"headline": headline.headline_markdown(m)}, m)


@pytest.mark.skipif(not LATEST.exists(), reason="no processed metrics; run `erm build`")
def test_update_replaces_only_the_marked_block(tmp_path, monkeypatch):
    (tmp_path / "data" / "processed").mkdir(parents=True)
    (tmp_path / "data" / "processed" / "metrics_latest.json").write_text(LATEST.read_text())
    readme = tmp_path / "README.md"
    readme.write_text(f"# Title\n\n{headline.START}\nold text\n{headline.END}\n\nRest.\n")
    monkeypatch.setattr(paths, "ROOT", tmp_path)
    monkeypatch.setattr(paths, "PROCESSED", tmp_path / "data" / "processed")
    headline.update_readme()
    text = readme.read_text()
    assert "old text" not in text
    assert text.startswith("# Title") and text.endswith("Rest.\n")
    assert "## Headline (curve as of" in text


def test_missing_markers_fail_loudly(tmp_path, monkeypatch):
    (tmp_path / "data" / "processed").mkdir(parents=True)
    (tmp_path / "data" / "processed" / "metrics_latest.json").write_text(
        json.dumps({"as_of": "2026-09-28"}))
    (tmp_path / "README.md").write_text("# no markers\n")
    monkeypatch.setattr(paths, "ROOT", tmp_path)
    monkeypatch.setattr(paths, "PROCESSED", tmp_path / "data" / "processed")
    monkeypatch.setattr(headline, "headline_markdown", lambda m: "## Headline")
    with pytest.raises(RuntimeError, match="markers"):
        headline.update_readme()
