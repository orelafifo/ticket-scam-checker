"""
Unit tests for v2.6 (run:  python -m pytest -q).   S4 evidence: code, test, debug.

Rules-only tests always run. Tests marked `needs_minilm` are skipped if the MiniLM
model can't be loaded (e.g. first run with no internet).
"""
import sys
from datetime import date
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import features as f                      # noqa: E402
import semantic_flags as sem              # noqa: E402
from extract_event import extract_event   # noqa: E402
from normalise import normalise           # noqa: E402

needs_minilm = pytest.mark.skipif(not sem.available(), reason="MiniLM not available")


# ---------------------------------------------------------------- 1. text normalisation
@pytest.mark.parametrize("raw, expected", [
    ("HMU!!!", "hit me up!"),
    ("plsease dm me", "please dm me"),
    ("pleaaase inbox me", "please inbox me"),
    ("pay via F&F", "pay via friends and family"),
    ("slide into my dms", "dm me"),
    ("2 tix 🔥🔥", "2 tickets"),
    ("pay with revolute", "pay with revolut"),
])
def test_normalise(raw, expected):
    assert normalise(raw) == expected


def test_normalise_leaves_names_alone():
    assert "olivia rodrigo" in normalise("Olivia Rodrigo tickets")


# ---------------------------------------------------------------- 2. rule phrase variants
@pytest.mark.parametrize("text", [
    "Kindly DM if interested", "kindly dm if interested", "HMU", "hmu x", "plsease dm",
    "plz dm", "inbox me", "Inbox me x", "DM for price", "price in dm", "slide into my dms",
    "hit me up if you want them", "message me", "add me on snapchat",
])
def test_off_platform_variants(text):
    assert f.text_flags(text, semantic=False)["off_platform"] == 1


@pytest.mark.parametrize("text", [
    "Revolut only", "pay on monzo", "TWINT please", "PayPal F&F", "paypal fnf",
    "paypal friends and family", "cash app me", "I'll send my sort code",
])
def test_unsafe_payment_variants(text):
    assert f.text_flags(text, semantic=False)["unsafe_payment"] == 1


@pytest.mark.parametrize("text", [
    "Pay through PayPal Goods and Services", "Transfer via the Ticketmaster app",
    "Block 112 row F", "Listed on Twickets at face value",
])
def test_ordinary_sentences_raise_no_payment_or_contact_flag(text):
    t = f.text_flags(text, semantic=False)
    assert t["unsafe_payment"] == 0 and t["off_platform"] == 0


# ---------------------------------------------------------------- 3. the AI meaning-matcher
@needs_minilm
@pytest.mark.parametrize("text, flag", [
    ("Send me a private msg babe", "off_platform"),
    ("Paypal but only the friends option", "unsafe_payment"),
    ("PayPal froze my account so need another method", "payment_switch"),
])
def test_minilm_finds_meaning(text, flag):
    assert flag in f.why_semantic(text)


@needs_minilm
@pytest.mark.parametrize("text", [
    "Tickets will be transferred through the Ticketmaster app",
    "Pay through PayPal Goods and Services please", "Doors open at 6pm",
])
def test_minilm_quiet_on_ordinary_sentences(text):
    assert f.why_semantic(text) == {}


def test_hard_stops_are_never_ai_matched():
    assert not set(sem.PROTOTYPES) & f.HARD_STOPS


# ---------------------------------------------------------------- 4. new inputs
def test_price_only_in_dms_turns_on_off_platform():
    flags = f.all_flags("Selling 2 tickets", 365, 200, 1.0, 0, price_status="dm_only", semantic=False)
    assert flags["off_platform"] == 1


def test_contact_question_flags():
    flags = f.all_flags("Selling 2 tickets", 365, 200, 1.0, 0, contact="group_chat", semantic=False)
    assert flags["stranger_group_chat"] == 1 and flags["seller_messaged_first"] == 0


def test_safety_net_whatsapp_case():
    flags = f.all_flags("Selling tickets, kindly DM", 365, 200, 1.0, 0,
                        channel={"category": "social"}, contact="group_chat", semantic=False)
    assert f.not_enough_info(flags, "not_stated", seller_known=False, contact="group_chat")


def test_safety_net_not_for_official_sites_or_priced_listings():
    flags = f.all_flags("2 tickets £50", 365, 200, 1.0, 1, channel={"category": "official"}, semantic=False)
    assert not f.not_enough_info(flags, "stated", seller_known=False, contact=None)


# ---------------------------------------------------------------- 5. event details from the text
def test_extract_event_whatsapp_listing():
    e = extract_event("Hello, I'm selling 3x Olivia Rodrigo Tickets The Unraveled Tour @ The O2 Arena, "
                      "London on Friday 9th April 2027. Kindly Dm if interested.", today=date(2026, 11, 12))
    assert e["artist"] == "Olivia Rodrigo"
    assert "O2" in e["venue"]
    assert e["date"] == date(2027, 4, 9)


def test_extract_event_football():
    e = extract_event("Arsenal v Chelsea tickets 3rd May Emirates, pm me", today=date(2026, 11, 12))
    assert e["artist"] == "Arsenal v Chelsea" and e["date"] == date(2027, 5, 3)


# ---------------------------------------------------------------- 6. event check without live data
def test_no_keys_means_not_checked(monkeypatch):
    """v2.6: with no API key the app never falls back to made-up data - the event is 'not checked'."""
    from event_check import check_event
    monkeypatch.delenv("TICKETMASTER_API_KEY", raising=False)
    monkeypatch.delenv("SKIDDLE_API_KEY", raising=False)
    e = check_event("Olivia Rodrigo", "The O2", date(2027, 4, 9))
    assert e["found"] is None and e["details_match"] is None and "not checked" in e["note"]
    flags = f.all_flags("Olivia Rodrigo tickets", 365, 200, 1.0, 1, event=e, semantic=False)
    assert flags["event_not_found"] == 0 and flags["event_mismatch"] == 0


def test_date_venue_matching_with_test_catalogue(monkeypatch):
    """The made-up test catalogue is only used here, to test the matching logic offline."""
    from event_check import check_event
    monkeypatch.delenv("TICKETMASTER_API_KEY", raising=False)
    monkeypatch.delenv("SKIDDLE_API_KEY", raising=False)
    ok = check_event("Burna Boy", "O2 Arena", date(2027, 3, 14), today=date(2026, 11, 12), use_demo=True)
    wrong = check_event("Burna Boy", "O2 Arena", date(2027, 3, 20), today=date(2026, 11, 12), use_demo=True)
    assert ok["details_match"] is True and wrong["details_match"] is False
