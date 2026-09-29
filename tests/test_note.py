"""The number validator and the LLM drafting loop (with a fake client, no network)."""

import json
from types import SimpleNamespace

import pytest

from euro_rates_monitor.note import (
    NumberValidationError,
    draft_llm,
    draft_template,
    unknown_numbers,
    validate,
)

METRICS = {
    "as_of": "2026-09-28",
    "aaa_10y_pct": 3.63, "aaa_10y_chg_1w_bp": 16, "ecb_dfr_pct": 2.5, "estr_pct": 2.44,
    "hicp_month": "2026-08", "spf_survey_quarter": "2026-Q3",
    "top_moves_1w": [{"measure": "AAA 10Y yield", "chg_1w_bp": -4, "latest": 3.63,
                      "latest_unit": "%", "size_vs_typical_week": 2.3}],
}


def test_numbers_from_dictionary_pass():
    text = "The 10Y rose 16bp to 3.63%; the deposit rate is 2.50% and EUR STR 2.44%."
    assert unknown_numbers(text, METRICS) == []


def test_sign_can_become_a_word():
    assert unknown_numbers("It fell 4bp.", METRICS) == []


def test_invented_or_rerounded_numbers_fail():
    assert unknown_numbers("The 10Y is 3.6%.", METRICS) == ["3.6"]
    assert unknown_numbers("Markets price 3 hikes.", METRICS) == ["3"]
    # Arithmetic on figures is not allowed either: 3.63 - 2.5 = 1.13.
    assert unknown_numbers("A 1.13 point gap.", METRICS) == ["1.13"]


def test_labels_and_dictionary_dates_are_not_numbers():
    text = ("On 28 September 2026 the 2s10s and 10s30s moved; the 3-month rate in 1Y; "
            "HICP for August 2026; SPF Q3 2026.")
    assert unknown_numbers(text, METRICS) == []
    assert unknown_numbers("As of 2025-01-01.", METRICS) != []


def test_validate_raises_with_details():
    with pytest.raises(NumberValidationError, match="ecb_pricing: 99"):
        validate({"moved": ["ok 16bp"], "ecb_pricing": "99bp of hikes"}, METRICS)


def _fake_client(*drafts: dict) -> SimpleNamespace:
    calls = []

    def create(**kwargs):
        calls.append(kwargs)
        body = drafts[len(calls) - 1]
        return SimpleNamespace(stop_reason="end_turn", stop_details=None,
                               content=[SimpleNamespace(type="text", text=json.dumps(body))])

    return SimpleNamespace(beta=SimpleNamespace(messages=SimpleNamespace(create=create)),
                           calls=calls)


GOOD = {"moved": ["10Y rose 16bp to 3.63%.", "a", "b"], "ecb_pricing": "DFR 2.5%.",
        "fx_bullet": "x", "credit_bullet": "y", "hedging": "EUR STR 2.44%."}
BAD = {**GOOD, "ecb_pricing": "Markets price 75bp of hikes."}


def test_llm_draft_retries_once_then_succeeds():
    client = _fake_client(BAD, GOOD)
    assert draft_llm(METRICS, client=client) == GOOD
    assert len(client.calls) == 2
    assert "rejected" in client.calls[1]["messages"][0]["content"]
    assert client.calls[0]["model"]
    assert "ANTHROPIC" not in json.dumps(client.calls[0], default=str)


def test_llm_draft_fails_loudly():
    with pytest.raises(NumberValidationError, match="75"):
        draft_llm(METRICS, client=_fake_client(BAD, BAD))


def test_refusal_is_an_error():
    client = _fake_client(GOOD)
    client.beta.messages.create = lambda **_: SimpleNamespace(
        stop_reason="refusal", stop_details={"category": "x"}, content=[])
    with pytest.raises(RuntimeError, match="declined"):
        draft_llm(METRICS, client=client)


def test_template_passes_validator_on_real_shaped_metrics(tmp_path):
    from pathlib import Path

    latest = Path(__file__).parents[1] / "data" / "processed" / "metrics_latest.json"
    if not latest.exists():
        pytest.skip("no processed metrics; run `erm build`")
    m = json.loads(latest.read_text())
    validate(draft_template(m), m)
