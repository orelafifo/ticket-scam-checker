"""
features.py - Layer 1: spot the red flags.

Turns a listing (text + seller details + event check result) into numbers the
model can use, and a human-readable list of red flags for the explanation.
Each rule is based on published scam-warning advice (Action Fraud, UK Finance,
bank guidance, Which?) - cite those sources on your slides.

Flag groups:
  A. Payment          B. Language / behaviour
  C. Seller account   D. Ticket evidence       E. Event (from event_check.py)
"""
from __future__ import annotations

import re

# ---------------------------------------------------------------------------
# Text rules: (flag name, regex pattern, plain-English description)
# ---------------------------------------------------------------------------
TEXT_RULES = [
    # --- A. Payment ---
    ("unsafe_payment",
     r"bank transfer|friends (and|&) family|\bf&f\b|\bff\b|gift ?card|crypto|bitcoin|"
     r"revolut me|monzo me|paypal family",
     "Asks for a payment method with no buyer protection"),
    ("payment_switch",
     r"paypal (is |isn'?t |is not |not )?(working|down|blocked|restricted|limited)|"
     r"can'?t (take|accept|receive) (paypal|card)|(switch|change) to (bank|revolut|monzo)|"
     r"bank (transfer )?instead|card (payment )?(isn'?t|is not|not) working",
     "Switches you away from a protected payment method"),
    ("deposit_or_hold",
     r"deposit|holding fee|hold (them|it|the tickets?) for|part[- ]?payment|half now|"
     r"pay half|reserve (them|it|the tickets?) (for|with)",
     "Asks for a deposit or 'holding' payment"),
    ("hide_reference",
     r"(don'?t|do not|dont) (put|mention|write|say|use) .{0,25}(tickets?|reference)|"
     r"leave the reference (blank|empty)|no reference|(send|mark) (it |the payment )?as (a )?gift|"
     r"reference (it )?as (gift|food|rent|dinner)",
     "Asks you to hide what the payment is for"),

    # --- B. Language / behaviour ---
    ("urgency",
     r"first to pay|loads of (people|messages)|quick(ly)?|asap|today only|gone fast|"
     r"won'?t last|serious buyers only",
     "Uses pressure or urgency language"),
    ("off_platform",
     r"whatsapp|telegram|dm me|message me on|text me on|move to",
     "Tries to move the conversation off the platform"),
    ("send_after_payment",
     r"after (payment|you pay)|once paid|send (them )?after|transfer (them )?after",
     "Will only send tickets after you pay"),
    ("sob_story",
     r"can'?t go anymore|last minute|family emergency|work came up|\bill\b|covid",
     "Gives a sympathetic reason for selling (common scam script)"),
    ("trust_claims",
     r"100% (legit|genuine|real)|not a scam|legit seller|genuine seller|trusted seller|"
     r"vouch(es|ed)?|(i'?m|im|i am) legit|real tickets",
     "Insists they are 'legit' instead of offering protection"),
    ("asks_for_codes",
     r"(verification|verify|6[- ]digit|security|whatsapp|login|log in) code|"
     r"send (me )?the code|code (i|we) (just )?sent|your (password|card details|log ?in details)",
     "Asks for a code or login details (account-takeover risk)"),

    # --- D. Ticket evidence (text part) ---
    # Only counts as a red flag when the event uses app-only tickets (see all_flags).
    ("mentions_file_ticket",
     r"\bpdfs?\b|screenshots? of the tickets?|email (you )?the tickets?|paper tickets?|"
     r"print(ed|able)? tickets?|e-?tickets? (by|via) email",
     None),  # helper, not shown to the user
]


def text_flags(text: str) -> dict:
    t = text.lower()
    return {name: int(bool(re.search(pattern, t))) for name, pattern, _ in TEXT_RULES}


# ---------------------------------------------------------------------------
# C. Seller account + D. price / seat evidence
# ---------------------------------------------------------------------------
def seller_flags(account_age_days: float, followers: float, price_ratio: float,
                 has_seat_details: int, sudden_seller: int = 0) -> dict:
    """price_ratio = asking price / face value (1.0 = face value).
    sudden_seller = an established account that has suddenly started posting
    ticket sales (typical of a hacked friend's account)."""
    return {
        "new_account": int(account_age_days < 60),
        "few_followers": int(followers < 50),
        "sudden_seller": int(bool(sudden_seller)),
        "too_cheap": int(price_ratio < 0.8),
        "no_seat_details": int(not has_seat_details),
    }


# ---------------------------------------------------------------------------
# E. Event flags - built from the dict returned by event_check.check_event()
# ---------------------------------------------------------------------------
def event_flags(event: dict | None, mentions_file_ticket: int) -> dict:
    """event keys: found (True/False/None), details_match (True/False/None),
    high_demand (bool), before_on_sale (bool), app_only (True/False/None).
    None means 'could not check' - that never raises a flag on its own."""
    e = event or {}
    return {
        "event_not_found": int(e.get("found") is False),
        "event_mismatch": int(e.get("found") is True and e.get("details_match") is False),
        "high_demand": int(bool(e.get("high_demand"))),
        "before_on_sale": int(bool(e.get("before_on_sale"))),
        "format_mismatch": int(bool(mentions_file_ticket) and e.get("app_only") is True),
    }


# ---------------------------------------------------------------------------
# Descriptions, column order, helpers
# ---------------------------------------------------------------------------
TEXT_DESCRIPTIONS = {name: desc for name, _, desc in TEXT_RULES if desc}
SELLER_DESCRIPTIONS = {
    "new_account": "Seller account is less than 2 months old",
    "few_followers": "Seller has very few followers / little history",
    "sudden_seller": "Established account suddenly selling tickets (possible hacked account)",
    "too_cheap": "Price is well below face value for an in-demand show",
    "no_seat_details": "No section, row or seat details given",
}
EVENT_DESCRIPTIONS = {
    "event_not_found": "We couldn't find this event on official listings",
    "event_mismatch": "Date or venue doesn't match the official event",
    "high_demand": "Sold-out / high-demand show (scammers target these)",
    "before_on_sale": "Tickets offered before the official general sale",
    "format_mismatch": "Offers PDF/screenshot tickets for an app-only event",
}
ALL_DESCRIPTIONS = {**TEXT_DESCRIPTIONS, **SELLER_DESCRIPTIONS, **EVENT_DESCRIPTIONS}
FLAG_COLUMNS = list(TEXT_DESCRIPTIONS) + list(SELLER_DESCRIPTIONS) + list(EVENT_DESCRIPTIONS)

# Rules that force the risk level to High, whatever the model says.
# Design choice: these are so strongly linked to fraud that we don't want the
# model to talk us out of them. The model's score is still shown honestly.
HARD_STOPS = {"asks_for_codes", "hide_reference", "event_mismatch", "format_mismatch"}


def all_flags(text, account_age_days, followers, price_ratio, has_seat_details,
              sudden_seller=0, event=None) -> dict:
    tf = text_flags(text)
    mentions_file = tf.pop("mentions_file_ticket")
    flags = {**tf,
             **seller_flags(account_age_days, followers, price_ratio, has_seat_details, sudden_seller),
             **event_flags(event, mentions_file)}
    return {k: flags[k] for k in FLAG_COLUMNS}


def triggered(flags: dict) -> list:
    """Plain-English list of the red flags that fired."""
    return [ALL_DESCRIPTIONS[k] for k, v in flags.items() if v]


def hard_stops(flags: dict) -> list:
    """Plain-English list of any hard-stop rules that fired."""
    return [ALL_DESCRIPTIONS[k] for k in HARD_STOPS if flags.get(k)]
