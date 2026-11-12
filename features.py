"""
features.py - Layer 1: spot the red flags.

Turns a listing (text + seller details + event check result) into numbers the
model can use, and a human-readable list of red flags for the explanation.
Each rule is based on published scam-warning advice (Action Fraud, UK Finance,
bank guidance, Which?) - cite those sources on your slides.

Flag groups:
  0. Where it's sold (from channels.py)
  A. Payment          B. Language / behaviour
  C. Seller account   D. Ticket evidence       E. Event (from event_check.py)
"""
from __future__ import annotations

import re

from normalise import normalise
from semantic_flags import semantic_flags

# ---------------------------------------------------------------------------
# Text rules: (flag name, regex pattern, plain-English description)
# ---------------------------------------------------------------------------
TEXT_RULES = [
    # --- A. Payment ---
    ("unsafe_payment",
     r"bank transfer|friends (and|&) family|\bf&f\b|\bff\b|gift ?card|crypto|bitcoin|\busdt\b|"
     r"revolut me|monzo me|paypal family|(amazon|itunes|apple|steam|google play) (gift )?card|"
     # v2.6: payment apps with no buyer protection for this kind of sale (gaps found in blind test #3)
     r"\brevolut\b|\bmonzo\b|\btwint\b|cash app|\bvenmo\b|\bzelle\b|\bwise\b|"
     r"sort code|account number|bank details|paypal (friends|f&f|personal)",
     "Asks for a payment method with no buyer protection"),
    ("payment_switch",
     r"paypal (is |isn'?t |is not |not )?(working|down|blocked|restricted|limited)|"
     r"can'?t (take|accept|receive) (paypal|card)|(switch|change) to (bank|revolut|monzo)|"
     r"bank (transfer )?instead|card (payment )?(isn'?t|is not|not) working",
     "Switches you away from a protected payment method"),
    ("deposit_or_hold",
     r"deposit|holding fee|hold (them|it|the tickets?) for|part[- ]?payment|half now|"
     r"pay half|reserve (them|it|the tickets?) (for|with)|up ?front|full amount|"
     r"(pay|payment) (first|before)|to (secure|reserve) (them|it|your|the)",
     "Asks for a deposit or payment up front to 'hold' the tickets"),
    ("hide_reference",
     r"(don'?t|do not|dont) (put|mention|write|say|use) .{0,25}(tickets?|reference)|"
     r"leave the reference (blank|empty)|no reference|(send|mark) (it |the payment )?as (a )?gift|"
     r"reference (it )?as (gift|food|rent|dinner)",
     "Asks you to hide what the payment is for"),

    # --- B. Language / behaviour ---
    ("urgency",
     r"first to pay|loads of (people|messages|interest)|lot of interest|quick(ly)?|asap|today only|"
     r"gone fast|won'?t last|serious buyers only|others (are )?asking|first come,? first served|"
     r"(need|want) (the )?payment today|before they'?re gone|limited (release|availability)",
     "Uses pressure or urgency language"),
    ("off_platform",
     r"whatsapp|telegram|\b(dm|pm|inbox) me\b|message me on|text me on|move to|"
     r"private message|message (me|you) privately|"
     # v2.6: wording found in real posts ("Kindly DM if interested", "hmu"); text is normalised first
     r"\b(dm|pm)\b|\binbox\b|hit me up|\bmessage me\b|text me|snapchat|add me on|"
     r"(price|details|info) (in|on|via) (dm|pm|private|message)",
     "Tries to move the conversation off the platform"),
    ("send_after_payment",
     r"after (payment|paying|you('ve)? pa(y|id)|u pay)|once paid|once (the )?payment|"
     r"(send|transfer|email)(ed)? (them |it )?(straight )?after|emailed after|"
     r"see the tickets after|as soon as (the )?payment",
     "Will only send tickets after you pay"),
    ("sob_story",
     r"can'?t go anymore|last minute|family emergency|work came up|\bill\b|covid|surgery|hospital|"
     r"\bflu\b|out of town|bailed|other plans|having a baby|unable to attend|can no longer attend",
     "Gives a sympathetic reason for selling (common scam script)"),
    ("trust_claims",
     r"100% (legit|genuine|real)|not a scam|legit seller|genuine seller|trusted seller|"
     r"vouch(es|ed)?|(i'?m|im|i am) legit|real tickets",
     "Insists they are 'legit' instead of offering protection"),
    ("asks_for_codes",
     r"(verification|verify|6[- ]digit|security|whatsapp|login|log in) code|"
     r"send (me )?the code|code (i|we) (just )?sent|your (password|card details|log ?in details)",
     "Asks for a code or login details (account-takeover risk)"),

    ("third_party_payment",
     r"(friend|mate|cousin|brother|sister|colleague|partner)'?s? (bank )?account|"
     r"account (is|was|has been) (frozen|blocked|restricted|locked)|pay (it )?into (my )?(another|a different)",
     "Asks you to pay into someone else's account"),
    ("cheap_wording",
     r"reduced price|discount(ed)?|\bcheap\b|going cheap|bargain|below face",
     "Pushes a cheap or discounted price without details"),
    ("screenshot_proof",
     r"screenshots?|pics? (attached|of the)|pictures? of the tickets|photo of the tickets",
     "Offers screenshots or photos as 'proof' (easy to fake)"),
    ("code_or_access_sale",
     r"(presale|pre-sale|access|priority|ticket) codes?|registration fee|priority access|"
     r"pre-?sale tickets",
     "Sells presale codes or 'priority access' (official codes are never sold)"),
    ("engagement_bait",
     r"comment ['\"]?(interested|me|below)|like (this|the) post|drop a comment|share (this|the) post",
     "Asks you to like or comment first (used to find targets)"),

    # --- D. Ticket evidence (text part) ---
    # Only counts as a red flag when the event uses app-only tickets (see all_flags).
    ("mentions_file_ticket",
     r"\bpdfs?\b|screenshots? of the tickets?|email (you )?the tickets?|paper tickets?|"
     r"print(ed|able)? tickets?|e-?tickets? (by|via) email",
     None),  # helper, not shown to the user
]


def text_flags(text: str, semantic: bool = True) -> dict:
    """v2.6: the text is normalised first (normalise.py: "hmu" -> "hit me up", "plz" -> "please",
    typos in key words fixed). Then each flag fires if its rule matches OR, for the soft flags,
    the AI meaning-matcher finds a sentence that means the same (semantic_flags.py)."""
    t = normalise(text)
    flags = {name: int(bool(re.search(pattern, t))) for name, pattern, _ in TEXT_RULES}
    if semantic:
        for name in semantic_flags(t):
            flags[name] = 1
    # Real scam posts are often tiny and vague: no price, no details, just "PM me".
    flags["vague_listing"] = int(len(t.split()) < 15 and not re.search(r"[£$€]\s?\d|\d+\s?(each|quid)", t))
    return flags


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
        "very_cheap": int(price_ratio < 0.5),   # v2.5: "too good to be true" price
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
# 0. Where it's being sold - built from the dict returned by channels.channel_info()
# ---------------------------------------------------------------------------
def channel_flags(channel: dict | None) -> dict:
    c = channel or {}
    cat = c.get("category")
    return {
        "social_or_messaging": int(cat == "social"),
        "secondary_market": int(cat == "secondary"),
        "unknown_website": int(cat == "unknown_site"),
        "lookalike_link": int(bool(c.get("lookalike"))),
        "new_domain": int(bool(c.get("new_domain"))),
        "insecure_or_short_link": int(bool(c.get("insecure_or_short"))),
        "official_platform": int(cat == "official"),   # protective signal (lowers risk)
    }


# ---------------------------------------------------------------------------
# v2.6: F. How the buyer came across the seller (new question in the app)
# ---------------------------------------------------------------------------
CONTACT_CHOICES = {
    "Don't know / prefer not to say": None,
    "A public post or listing (e.g. Instagram, X, Facebook, Gumtree)": "public_post",
    "A group chat with people I don't know (e.g. joined by an invite link)": "group_chat",
    "They messaged me first": "messaged_first",
    "A friend's account": "friend_account",
    "An official ticket or resale site": "official_site",
}


def contact_flags(contact: str | None) -> dict:
    return {
        "stranger_group_chat": int(contact == "group_chat"),
        "seller_messaged_first": int(contact == "messaged_first"),
        "friend_account": int(contact == "friend_account"),
    }


# ---------------------------------------------------------------------------
# Descriptions, column order, helpers
# ---------------------------------------------------------------------------
TEXT_DESCRIPTIONS = {name: desc for name, _, desc in TEXT_RULES if desc}
TEXT_DESCRIPTIONS["vague_listing"] = "Very short, vague post with no price or details"
SELLER_DESCRIPTIONS = {
    "new_account": "Seller account is less than 2 months old",
    "few_followers": "Seller has very few followers / little history",
    "sudden_seller": "Established account suddenly selling tickets (possible hacked account)",
    "too_cheap": "Price is below 80% of face value",
    "very_cheap": "Price is less than half of face value - too good to be true",
    "no_seat_details": "No section, row or seat details given",
}
EVENT_DESCRIPTIONS = {
    "event_not_found": "We couldn't find this event on official listings",
    "event_mismatch": "Date or venue doesn't match the official event",
    "high_demand": "Sold-out / high-demand show (scammers target these)",
    "before_on_sale": "Tickets offered before the official general sale",
    "format_mismatch": "Offers PDF/screenshot tickets for an app-only event",
}
CHANNEL_DESCRIPTIONS = {
    "social_or_messaging": "Sold through social media or a messaging app - no buyer protection if it goes wrong",
    "secondary_market": "Secondary resale site (e.g. viagogo, StubHub) - often above face value, "
                        "and the event can cancel resold tickets",
    "unknown_website": "Ticket website we don't recognise",
    "lookalike_link": "Link imitates a well-known ticket site (possible fake website)",
    "new_domain": "Website was registered less than 6 months ago",
    "insecure_or_short_link": "Link isn't secure (http) or hides its destination (link shortener)",
    "official_platform": "Official seller or face-value resale platform with buyer protection",
}
CONTACT_DESCRIPTIONS = {
    "stranger_group_chat": "Seller is a stranger in a group chat (e.g. joined through an open invite link)",
    "seller_messaged_first": "The seller contacted you first (cold approach)",
    "friend_account": "Sold from a friend's account - check with your friend another way, "
                      "accounts are often hacked to sell fake tickets",
}
ALL_DESCRIPTIONS = {**CHANNEL_DESCRIPTIONS, **TEXT_DESCRIPTIONS, **SELLER_DESCRIPTIONS, **EVENT_DESCRIPTIONS,
                    **CONTACT_DESCRIPTIONS}
FLAG_COLUMNS = (list(CHANNEL_DESCRIPTIONS) + list(TEXT_DESCRIPTIONS) +
                list(SELLER_DESCRIPTIONS) + list(EVENT_DESCRIPTIONS) + list(CONTACT_DESCRIPTIONS))
PROTECTIVE = {"official_platform"}  # good signs: shown with a tick, not a red flag
V25_FLAG_COLUMNS = [c for c in FLAG_COLUMNS if c not in CONTACT_DESCRIPTIONS]   # v2.5 used these 34
V24_FLAG_COLUMNS = [c for c in V25_FLAG_COLUMNS if c != "very_cheap"]           # v2.4 used these 33

# ---------------------------------------------------------------------------
# v2.5: expert evidence tiers (agreed with Orela, 9 Oct 2026). Each red flag starts
# from its tier's weight ("informative prior"); training on real examples then
# adjusts it. Sources for each tier: see TIER_SOURCES.md.
# ---------------------------------------------------------------------------
TIERS = {
    "hard_stop": ["asks_for_codes", "lookalike_link", "third_party_payment", "hide_reference",
                  "event_mismatch", "code_or_access_sale", "format_mismatch"],
    "strong": ["unsafe_payment", "payment_switch", "deposit_or_hold", "sudden_seller",
               "unknown_website", "new_domain", "very_cheap",
               "event_not_found", "before_on_sale"],
    "moderate": ["off_platform", "send_after_payment", "screenshot_proof", "vague_listing",
                 "cheap_wording", "engagement_bait", "social_or_messaging", "sob_story",
                 "insecure_or_short_link",
                 "stranger_group_chat", "seller_messaged_first"],          # v2.6 (Claude placed - confirm)
    "weak": ["new_account", "few_followers", "no_seat_details", "urgency", "trust_claims",
             "too_cheap", "high_demand", "secondary_market",
             "friend_account"],                                            # v2.6 (Claude placed - confirm)
    "good_sign": ["official_platform"],
}
TIER_PRIOR = {"hard_stop": 3.0, "strong": 2.5, "moderate": 1.5, "weak": 0.5, "good_sign": -2.0}
PRIOR_WEIGHTS = {flag: TIER_PRIOR[tier] for tier, flags in TIERS.items() for flag in flags}
assert set(PRIOR_WEIGHTS) == set(FLAG_COLUMNS), set(FLAG_COLUMNS) ^ set(PRIOR_WEIGHTS)

# Rules that force the risk level to High, whatever the model says.
# Design choice: these are so strongly linked to fraud that we don't want the
# model to talk us out of them. The model's score is still shown honestly.
HARD_STOPS = {"asks_for_codes", "hide_reference", "event_mismatch", "format_mismatch",
              "lookalike_link", "third_party_payment", "code_or_access_sale"}


def all_flags(text, account_age_days, followers, price_ratio, has_seat_details,
              sudden_seller=0, event=None, channel=None, contact=None, price_status=None,
              semantic=True, columns=None) -> dict:
    """v2.6 extra inputs:
      contact      - how the buyer came across the seller (CONTACT_CHOICES values; None = unknown)
      price_status - "stated" / "not_stated" / "dm_only" (None = unknown). "Only in DMs" means the
                     seller is moving you off the platform, so it turns on off_platform.
      semantic     - False = rules only (used to rebuild v2.5-style results for comparison)"""
    tf = text_flags(text, semantic=semantic)
    mentions_file = tf.pop("mentions_file_ticket")
    if price_status == "dm_only":
        tf["off_platform"] = 1
    flags = {**channel_flags(channel), **tf,
             **seller_flags(account_age_days, followers, price_ratio, has_seat_details, sudden_seller),
             **event_flags(event, mentions_file), **contact_flags(contact)}
    return {k: flags[k] for k in (columns or FLAG_COLUMNS)}


def why_semantic(text: str) -> dict:
    """{flag: (sentence, example it matched, similarity)} - shown in the app so the AI part is explainable."""
    return semantic_flags(normalise(text))


# ---------------------------------------------------------------------------
# v2.6: safety net - "not enough information" instead of a reassuring Low
# ---------------------------------------------------------------------------
def not_enough_info(flags: dict, price_status: str | None, seller_known: bool, contact: str | None) -> bool:
    """A listing from social media / a group chat / a cold message, with no price AND no seller
    details, is never rated Low: there is simply too little to judge (the WhatsApp case,
    Nov 2026, scored 16% Low). The app shows Medium - not enough information."""
    risky_channel = bool(flags.get("social_or_messaging")) or contact in ("group_chat", "messaged_first")
    no_price = price_status in (None, "not_stated", "dm_only")
    return risky_channel and no_price and not seller_known


def triggered(flags: dict) -> list:
    """Plain-English list of the red flags that fired."""
    return [ALL_DESCRIPTIONS[k] for k, v in flags.items() if v and k not in PROTECTIVE]


def protections(flags: dict) -> list:
    """Plain-English list of good signs (e.g. buying through official resale)."""
    return [ALL_DESCRIPTIONS[k] for k in PROTECTIVE if flags.get(k)]


def hard_stops(flags: dict) -> list:
    """Plain-English list of any hard-stop rules that fired."""
    return [ALL_DESCRIPTIONS[k] for k in HARD_STOPS if flags.get(k)]
