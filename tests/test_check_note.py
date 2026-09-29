"""Hand-written notes: the scaffold, `check_note`, and `write_note`'s safety rules."""

import json
from pathlib import Path

import pytest

from euro_rates_monitor import note, paths
from euro_rates_monitor.note import (
    PLACEHOLDER,
    UNREVIEWED,
    check_note,
    note_files,
    render_scaffold,
    unknown_tokens,
)

LATEST = Path(__file__).parents[1] / "data" / "processed" / "metrics_latest.json"
FIGURES = {"as_of": "2026-09-28", "aaa_10y_pct": 3.63, "aaa_10y_chg_1w_bp": 16,
           "ecb_dfr_pct": 2.5, "fwd3m_in_1y_pct": 3.42, "fwd3m_in_1y_minus_dfr_bp": 92}
FILLED = """# Euro rates weekly: 28 September 2026

## What moved this week
- The 10Y rose 16bp to 3.63%.

## What the curve prices for the ECB
The 3M forward in 1Y is 3.42%, 92bp above the 2.50% deposit rate.

![3M forward path](img/2026-09-28_forward_path.png)
"""


@pytest.fixture
def processed(tmp_path: Path) -> Path:
    d = tmp_path / "processed"
    d.mkdir()
    (d / "metrics_2026-09-28.json").write_text(json.dumps(FIGURES))
    return d


def _note(tmp_path: Path, text: str, name: str = "2026-09-28.md") -> Path:
    f = tmp_path / name
    f.write_text(text)
    return f


def test_filled_note_passes(tmp_path, processed):
    assert check_note(_note(tmp_path, FILLED), processed) == []


def test_wrong_number_fails_with_the_token(tmp_path, processed):
    problems = check_note(_note(tmp_path, FILLED.replace("92bp", "90bp")), processed)
    assert problems == ["not in metrics_2026-09-28.json: 90"]


def test_wrong_date_fails_with_the_date(tmp_path, processed):
    text = FILLED.replace("# Euro rates weekly: 28", "# Euro rates weekly: 21")
    assert check_note(_note(tmp_path, text), processed) == [
        "not in metrics_2026-09-28.json: 21 September 2026"]


def test_note_is_checked_against_the_figures_for_its_own_date(tmp_path, processed):
    # Same text, different file date: no figures file for that week.
    problems = check_note(_note(tmp_path, FILLED, "2026-10-05.md"), processed)
    assert "no figures file" in problems[0]


def test_placeholders_and_unreviewed_marker_fail(tmp_path, processed):
    text = f"{UNREVIEWED}\n{FILLED}\n- {PLACEHOLDER} something]\n"
    problems = check_note(_note(tmp_path, text), processed)
    assert any("unreviewed" in p for p in problems)
    assert any("1 unfilled" in p for p in problems)


def test_bad_file_name(tmp_path, processed):
    assert "YYYY-MM-DD" in check_note(_note(tmp_path, FILLED, "draft.md"), processed)[0]


def test_note_files_expands_directories(tmp_path):
    (tmp_path / "2026-09-28.md").write_text("x")
    (tmp_path / "README.md").write_text("x")
    assert [p.name for p in note_files([tmp_path])] == ["2026-09-28.md"]
    (tmp_path / "none").mkdir()
    assert note_files([tmp_path / "none"]) == []


@pytest.mark.skipif(not LATEST.exists(), reason="no processed metrics; run `erm build`")
def test_scaffold_has_placeholders_but_no_unverifiable_figures():
    m = json.loads(LATEST.read_text())
    text = render_scaffold(m, "img/chart.png")
    assert text.startswith(UNREVIEWED)
    # Three moves, pricing, FX, credit, hedger.
    assert text.count(PLACEHOLDER) == 7
    assert unknown_tokens(text, m) == []


@pytest.fixture
def project(tmp_path, monkeypatch):
    """A throwaway project tree with metrics and a chart, wired into `paths`."""
    for d in ("data/processed", "figures", "notes"):
        (tmp_path / d).mkdir(parents=True)
    m = json.loads(LATEST.read_text())
    (tmp_path / "data/processed/metrics_latest.json").write_text(json.dumps(m))
    (tmp_path / "figures" / note.CHART).write_bytes(b"png")
    monkeypatch.setattr(paths, "ROOT", tmp_path)
    monkeypatch.setattr(paths, "PROCESSED", tmp_path / "data/processed")
    monkeypatch.setattr(paths, "FIGURES", tmp_path / "figures")
    monkeypatch.setattr(paths, "NOTES", tmp_path / "notes")
    return tmp_path, m


@pytest.mark.skipif(not LATEST.exists(), reason="no processed metrics; run `erm build`")
def test_write_note_default_is_a_scaffold_and_never_calls_the_llm(project, monkeypatch):
    root, m = project
    monkeypatch.setattr(note, "draft_llm", lambda *_: pytest.fail("LLM must not be called"))
    note.write_note()
    out = root / "notes" / f"{m['as_of']}.md"
    assert PLACEHOLDER in out.read_text()
    assert (root / "notes" / "img" / f"{m['as_of']}_{note.CHART}").exists()


@pytest.mark.skipif(not LATEST.exists(), reason="no processed metrics; run `erm build`")
def test_write_note_never_overwrites_a_hand_written_note(project):
    root, m = project
    mine = root / "notes" / f"{m['as_of']}.md"
    mine.write_text("my note")
    with pytest.raises(FileExistsError):
        note.write_note()
    assert mine.read_text() == "my note"


@pytest.mark.skipif(not LATEST.exists(), reason="no processed metrics; run `erm build`")
def test_llm_draft_is_marked_unreviewed(project, monkeypatch):
    root, m = project
    sections = {"moved": ["a", "b", "c"], "ecb_pricing": "p", "fx_bullet": "f",
                "credit_bullet": "c", "hedging": "h"}
    monkeypatch.setattr(note, "draft_llm", lambda *_: sections)
    note.write_note(use_llm=True)
    text = (root / "notes" / f"{m['as_of']}.md").read_text()
    assert text.startswith(UNREVIEWED) and "LLM draft" in text


@pytest.mark.skipif(not LATEST.exists(), reason="no processed metrics; run `erm build`")
def test_write_note_to_another_directory_leaves_notes_untouched(project):
    root, m = project
    note.write_note(out_dir=root / "scaffold")
    assert (root / "scaffold" / f"{m['as_of']}.md").exists()
    assert list((root / "notes").iterdir()) == []
