# Ticket scam checker reference code (code v2.3, 8 Oct 2026)

v2.3: rules widened and training data made realistic after the real-world error analysis (33 flags; new: third_party_payment, cheap_wording, screenshot_proof, code_or_access_sale, engagement_bait, vague_listing; hard stops add third_party_payment, code_or_access_sale). evaluate_real.py takes --test for an independent held-out file (template data/heldout_template.csv). build_real_test_set.py builds the 80-row development set. v2.2: platform + link checks (channels.py), multi-select official sellers.

## channels.py
```python
"""
channels.py - Layer 0: WHERE is the ticket being sold?

The same listing is far riskier on WhatsApp than on Twickets, so the checker
asks where the buyer found it (and optionally takes the link) BEFORE scoring.

Two inputs:
  1. platform - picked from a list (WhatsApp, X, viagogo, Ticketmaster resale...)
  2. link     - optional URL the buyer is about to use

What we do with a link (privacy and security by design):
  * We NEVER open or download the page. Visiting an unknown link could expose
    the user or our server to malware, and we don't need the page content.
  * We only look at the web address itself: is it a known site, a lookalike of
    a known site, a link shortener, or insecure (http)?
  * For unfamiliar websites we look up the domain's registration date using
    RDAP (the public registry service that replaced WHOIS). Brand-new domains
    are a classic fake-ticket-site sign. If the lookup fails we simply skip it.

Category meanings:
  official   - primary seller or face-value / capped resale with buyer protection
  secondary  - open resale marketplaces (viagogo, StubHub): buyer guarantees exist,
               but prices are often far above face value and promoters can cancel
               tickets resold against their terms
  social     - social media, messaging apps and classified ads: no buyer protection
  unknown_site - a website we don't recognise
"""
from __future__ import annotations

import re
from datetime import datetime, timezone
from difflib import SequenceMatcher
from urllib.parse import urlparse

# ---------------------------------------------------------------------------
# Platforms the user can pick (label -> category)
# ---------------------------------------------------------------------------
PLATFORMS = {
    "Ticketmaster / Ticketmaster Resale": "official",
    "AXS / AXS Official Resale": "official",
    "See Tickets": "official",
    "DICE": "official",
    "Twickets": "official",
    "Skiddle": "official",
    "Other official seller (Eventim, Gigantic...)": "official",
    "viagogo": "secondary",
    "StubHub": "secondary",
    "Other resale website": "secondary",
    "Instagram": "social",
    "Facebook (post, group or Marketplace)": "social",
    "X (Twitter)": "social",
    "TikTok": "social",
    "WhatsApp": "social",
    "Telegram": "social",
    "Snapchat": "social",
    "Text message / iMessage": "social",
    "Gumtree or other classified ads": "social",
    "A website I don't recognise": "unknown_site",
}
PLATFORM_CHOICES = ["Choose where you found the tickets..."] + list(PLATFORMS)

# ---------------------------------------------------------------------------
# Known web addresses (registrable domain -> (category, friendly name))
# ---------------------------------------------------------------------------
KNOWN_DOMAINS = {
    # official / face-value
    "ticketmaster.co.uk": ("official", "Ticketmaster"), "ticketmaster.com": ("official", "Ticketmaster"),
    "livenation.co.uk": ("official", "Live Nation"), "axs.com": ("official", "AXS"),
    "seetickets.com": ("official", "See Tickets"), "dice.fm": ("official", "DICE"),
    "twickets.live": ("official", "Twickets"), "twickets.co.uk": ("official", "Twickets"),
    "skiddle.com": ("official", "Skiddle"), "gigantic.com": ("official", "Gigantic"),
    "eventim.co.uk": ("official", "Eventim"), "ticketweb.uk": ("official", "TicketWeb"),
    "gigsandtours.com": ("official", "Gigs and Tours"),
    # secondary marketplaces
    "viagogo.co.uk": ("secondary", "viagogo"), "viagogo.com": ("secondary", "viagogo"),
    "stubhub.co.uk": ("secondary", "StubHub"), "stubhub.com": ("secondary", "StubHub"),
    "gigsberg.com": ("secondary", "Gigsberg"), "seatgeek.com": ("secondary", "SeatGeek"),
    "vividseats.com": ("secondary", "Vivid Seats"),
    # social media / messaging / classifieds
    "instagram.com": ("social", "Instagram"), "facebook.com": ("social", "Facebook"),
    "fb.com": ("social", "Facebook"), "fb.me": ("social", "Facebook"),
    "x.com": ("social", "X"), "twitter.com": ("social", "X"),
    "tiktok.com": ("social", "TikTok"), "reddit.com": ("social", "Reddit"),
    "discord.com": ("social", "Discord"), "discord.gg": ("social", "Discord"),
    "whatsapp.com": ("social", "WhatsApp"), "wa.me": ("social", "WhatsApp"),
    "t.me": ("social", "Telegram"), "telegram.me": ("social", "Telegram"),
    "snapchat.com": ("social", "Snapchat"), "gumtree.com": ("social", "Gumtree"),
}
SHORTENERS = {"bit.ly", "tinyurl.com", "t.co", "goo.gl", "ow.ly", "is.gd", "cutt.ly",
              "rebrand.ly", "shorturl.at", "tiny.cc", "rb.gy"}
# brand words scammers copy into fake domains
BRANDS = ["ticketmaster", "livenation", "axs", "seetickets", "dice", "twickets", "skiddle",
          "viagogo", "stubhub", "eventim", "gigantic", "theo2", "wembley"]
SECOND_LEVEL = {"co.uk", "org.uk", "ac.uk", "com.au", "co.nz", "co.za", "com.br"}


# ---------------------------------------------------------------------------
# Link helpers
# ---------------------------------------------------------------------------
def registrable_domain(url: str) -> tuple[str, str]:
    """Returns (scheme, registrable domain) e.g. ('https', 'ticketmaster.co.uk')."""
    url = url.strip()
    if not re.match(r"^[a-z]+://", url, re.I):
        url = "https://" + url  # people often paste without the https://
    parsed = urlparse(url)
    host = (parsed.hostname or "").lower().rstrip(".")
    parts = host.split(".")
    keep = 3 if len(parts) >= 3 and ".".join(parts[-2:]) in SECOND_LEVEL else 2
    scheme = parsed.scheme.lower() if "://" in url[:10] else "https"
    return scheme, ".".join(parts[-keep:])


def _label(domain: str) -> str:
    """'ticketmaster-uk.co.uk' -> 'ticketmaster-uk'"""
    return domain.split(".")[0]


def is_lookalike(domain: str) -> str | None:
    """Returns the brand being imitated, or None.
    Catches both 'ticketmaster-uk-sale.com' (brand word inside) and
    'ticketrnaster.co.uk' (spelling trick)."""
    if domain in KNOWN_DOMAINS:
        return None
    label = _label(domain).replace("-", "")
    for known, (_, name) in KNOWN_DOMAINS.items():
        known_label = _label(known)
        if len(known_label) < 5:  # short names like 'x', 'axs', 'dice' give false alarms
            continue
        if SequenceMatcher(None, label, known_label).ratio() >= 0.85:
            return name
    for brand in BRANDS:
        if len(brand) >= 5 and brand in label:
            return brand
    return None


def domain_age_days(domain: str, timeout: float = 5) -> int | None:
    """Days since the domain was registered, from RDAP. None if unknown."""
    try:
        import requests
        r = requests.get(f"https://rdap.org/domain/{domain}", timeout=timeout,
                         headers={"Accept": "application/rdap+json"})
        if r.status_code != 200:
            return None
        for event in r.json().get("events", []):
            if event.get("eventAction") == "registration":
                registered = datetime.fromisoformat(event["eventDate"].replace("Z", "+00:00"))
                return (datetime.now(timezone.utc) - registered).days
    except Exception:
        return None
    return None


def analyse_link(url: str, check_age: bool = True) -> dict:
    """Look at a web address WITHOUT visiting it."""
    scheme, domain = registrable_domain(url)
    info = dict(domain=domain, category=None, site=None, lookalike_of=None,
                insecure=(scheme == "http"), shortener=domain in SHORTENERS,
                age_days=None, notes=[])
    if not domain or "." not in domain:
        info["notes"].append("That doesn't look like a valid web address.")
        return info
    if domain in KNOWN_DOMAINS:
        info["category"], info["site"] = KNOWN_DOMAINS[domain]
        info["notes"].append(f"Recognised site: {info['site']}.")
    elif info["shortener"]:
        info["category"] = "unknown_site"
        info["notes"].append("Shortened link: it hides the real website. Ask for the full link.")
    else:
        info["category"] = "unknown_site"
        info["lookalike_of"] = is_lookalike(domain)
        if info["lookalike_of"]:
            info["notes"].append(f"'{domain}' looks like an imitation of {info['lookalike_of']}, "
                                 "but it isn't their real website.")
        else:
            info["notes"].append(f"'{domain}' isn't a ticket site we recognise.")
        if check_age:
            info["age_days"] = domain_age_days(domain)
            if info["age_days"] is not None:
                info["notes"].append(f"Website registered {info['age_days']} days ago.")
    if info["insecure"]:
        info["notes"].append("The link isn't secure (http, not https).")
    return info


# ---------------------------------------------------------------------------
# Turn platform + link into the channel dict used by features.all_flags()
# ---------------------------------------------------------------------------
def channel_info(platform: str | None, link: str = "", check_age: bool = True) -> dict:
    category = PLATFORMS.get(platform) if platform else None
    link_info = analyse_link(link, check_age) if link and link.strip() else None
    # If the link tells us more than the dropdown, trust the link
    # (e.g. user picked 'Twickets' but pasted a lookalike address).
    if link_info and link_info["category"]:
        if category is None or link_info["category"] in ("unknown_site", "social"):
            category = link_info["category"]
    li = link_info or {}
    age = li.get("age_days")
    return dict(
        category=category,                         # official / secondary / social / unknown_site / None
        lookalike=bool(li.get("lookalike_of")),
        new_domain=age is not None and age < 180,
        insecure_or_short=bool(li.get("insecure") or li.get("shortener")),
        platform=platform, link=link_info,
    )


if __name__ == "__main__":
    for p, l in [("Twickets", ""), ("WhatsApp", ""), (None, "https://www.viagogo.co.uk/Concert-Tickets/x"),
                 ("Twickets", "twickets-uk.co.uk/tickets"), (None, "http://bit.ly/abc"),
                 (None, "https://ticketrnaster.co.uk/event/123"), (None, "x.com/someone/status/1")]:
        c = channel_info(p, l, check_age=False)
        print(f"{str(p):12s} {l:45s} -> {c['category']:12s} lookalike={c['lookalike']} "
              f"short/insecure={c['insecure_or_short']}  {c['link']['notes'] if c['link'] else ''}")

```

## features.py
```python
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

# ---------------------------------------------------------------------------
# Text rules: (flag name, regex pattern, plain-English description)
# ---------------------------------------------------------------------------
TEXT_RULES = [
    # --- A. Payment ---
    ("unsafe_payment",
     r"bank transfer|friends (and|&) family|\bf&f\b|\bff\b|gift ?card|crypto|bitcoin|\busdt\b|"
     r"revolut me|monzo me|paypal family|(amazon|itunes|apple|steam|google play) (gift )?card",
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
     r"private message|message (me|you) privately",
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


def text_flags(text: str) -> dict:
    t = text.lower()
    flags = {name: int(bool(re.search(pattern, t))) for name, pattern, _ in TEXT_RULES}
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
# Descriptions, column order, helpers
# ---------------------------------------------------------------------------
TEXT_DESCRIPTIONS = {name: desc for name, _, desc in TEXT_RULES if desc}
TEXT_DESCRIPTIONS["vague_listing"] = "Very short, vague post with no price or details"
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
ALL_DESCRIPTIONS = {**CHANNEL_DESCRIPTIONS, **TEXT_DESCRIPTIONS, **SELLER_DESCRIPTIONS, **EVENT_DESCRIPTIONS}
FLAG_COLUMNS = (list(CHANNEL_DESCRIPTIONS) + list(TEXT_DESCRIPTIONS) +
                list(SELLER_DESCRIPTIONS) + list(EVENT_DESCRIPTIONS))
PROTECTIVE = {"official_platform"}  # good signs: shown with a tick, not a red flag

# Rules that force the risk level to High, whatever the model says.
# Design choice: these are so strongly linked to fraud that we don't want the
# model to talk us out of them. The model's score is still shown honestly.
HARD_STOPS = {"asks_for_codes", "hide_reference", "event_mismatch", "format_mismatch",
              "lookalike_link", "third_party_payment", "code_or_access_sale"}


def all_flags(text, account_age_days, followers, price_ratio, has_seat_details,
              sudden_seller=0, event=None, channel=None) -> dict:
    tf = text_flags(text)
    mentions_file = tf.pop("mentions_file_ticket")
    flags = {**channel_flags(channel), **tf,
             **seller_flags(account_age_days, followers, price_ratio, has_seat_details, sudden_seller),
             **event_flags(event, mentions_file)}
    return {k: flags[k] for k in FLAG_COLUMNS}


def triggered(flags: dict) -> list:
    """Plain-English list of the red flags that fired."""
    return [ALL_DESCRIPTIONS[k] for k, v in flags.items() if v and k not in PROTECTIVE]


def protections(flags: dict) -> list:
    """Plain-English list of good signs (e.g. buying through official resale)."""
    return [ALL_DESCRIPTIONS[k] for k in PROTECTIVE if flags.get(k)]


def hard_stops(flags: dict) -> list:
    """Plain-English list of any hard-stop rules that fired."""
    return [ALL_DESCRIPTIONS[k] for k in HARD_STOPS if flags.get(k)]

```

## event_check.py
```python
"""
event_check.py - Layer 1b: check the listing against the real event.

Answers four Core questions:
  1. Does this event exist?                  -> found
  2. Do the date and venue match?            -> details_match
  3. Is it a sold-out / high-demand show?    -> high_demand
  4. Is it being sold before general sale?   -> before_on_sale
It also returns app_only (for the PDF/screenshot check), the event status
(cancelled / postponed) and the official face-value range.

Sources (keys go in a .env file, see .env.example):
  * LIVE:    Ticketmaster Discovery API (free key: developer.ticketmaster.com)
             TICKETMASTER_API_KEY=...   (also covers Live Nation events)
  * LIVE:    Skiddle Events API (free key: skiddle.com/api/join.php)
             SKIDDLE_API_KEY=...        (UK gigs, clubs, festivals)
  * NOT POSSIBLE: AXS, See Tickets, DICE, Eventim, Gigantic have no public API.
             For these the user picks the official seller, and "not found" is
             treated as "couldn't check" rather than a red flag.
  * OFFLINE: data/events_demo.csv - a small DEMO catalogue so the app works on
             camera without internet. It is made-up sample data, NOT real tour
             dates. Say so in the video.

Values are True / False / None. None = "couldn't check", which never raises a
red flag by itself (we don't punish a seller because our lookup failed).

Limitation to mention: no single source covers every UK ticket agent, so
"not found" only counts as a red flag when the official seller is one we search.
"""
from __future__ import annotations

import os
import re
from datetime import date, datetime
from difflib import SequenceMatcher
from pathlib import Path

import pandas as pd

CATALOGUE = Path(__file__).parent / "data" / "events_demo.csv"
TM_URL = "https://app.ticketmaster.com/discovery/v2/events.json"
SKIDDLE_URL = "https://www.skiddle.com/api/v1/events/search/"
GENERIC_VENUE_WORDS = {"the", "arena", "stadium", "london", "manchester", "birmingham", "live", "hall"}


# ---------------------------------------------------------------------------
# Matching helpers
# ---------------------------------------------------------------------------
def _norm(s: str) -> str:
    return re.sub(r"[^a-z0-9 ]", " ", str(s).lower()).strip()


def venue_matches(a: str, b: str) -> bool:
    """'O2 Arena' should match 'The O2'. Share a meaningful word, or look similar."""
    a, b = _norm(a), _norm(b)
    if not a or not b:
        return False
    key_a = set(a.split()) - GENERIC_VENUE_WORDS
    key_b = set(b.split()) - GENERIC_VENUE_WORDS
    if key_a & key_b:
        return True
    return SequenceMatcher(None, a, b).ratio() > 0.75


def artist_matches(query: str, name: str) -> bool:
    return _norm(query) in _norm(name) or SequenceMatcher(None, _norm(query), _norm(name)).ratio() > 0.85


def _empty(source: str, note: str) -> dict:
    return dict(found=None, details_match=None, high_demand=False, before_on_sale=False,
                app_only=None, status=None, face_min=None, face_max=None,
                official_date=None, official_venue=None, source=source, note=note)


def _judge(events: list[dict], venue: str, event_date: date | None, today: date, source: str) -> dict:
    """events: list of dicts with keys date, venue, on_sale, status, high_demand, app_only, face_min, face_max."""
    if not events:
        r = _empty(source, "No official event found for this artist.")
        r["found"] = False
        return r

    def is_match(e):
        ok_venue = venue_matches(venue, e["venue"]) if venue else True
        ok_date = (e["date"] == event_date) if event_date else True
        return ok_venue and ok_date

    matches = [e for e in events if is_match(e)]
    # show the closest official event: same venue first, then nearest date
    best = matches[0] if matches else min(
        events, key=lambda e: (not (venue and venue_matches(venue, e["venue"])),
                               abs((e["date"] - (event_date or today)).days) if e["date"] else 9999))

    on_sale = best.get("on_sale")
    return dict(
        found=True,
        details_match=bool(matches),
        high_demand=bool(best.get("high_demand")),
        before_on_sale=bool(on_sale and today < on_sale),
        app_only=best.get("app_only"),
        status=best.get("status"),
        face_min=best.get("face_min"),
        face_max=best.get("face_max"),
        official_date=best["date"].isoformat() if best.get("date") else None,
        official_venue=best.get("venue"),
        source=source,
        note="Matched an official event." if matches else
             "Artist is touring, but not on this date / at this venue.",
    )


# ---------------------------------------------------------------------------
# Source 1: Ticketmaster Discovery API
# ---------------------------------------------------------------------------
def _ticketmaster(artist: str, api_key: str) -> list[dict]:
    import requests
    resp = requests.get(TM_URL, timeout=6, params={
        "apikey": api_key, "keyword": artist, "countryCode": "GB",
        "classificationName": "music", "size": 100, "sort": "date,asc"})
    resp.raise_for_status()
    events = []
    for e in resp.json().get("_embedded", {}).get("events", []):
        names = [e.get("name", "")] + [a.get("name", "") for a in e.get("_embedded", {}).get("attractions", [])]
        if not any(artist_matches(artist, n) for n in names):
            continue
        start = e.get("dates", {}).get("start", {}).get("localDate")
        status = e.get("dates", {}).get("status", {}).get("code")      # onsale / offsale / cancelled / postponed
        on_sale = e.get("sales", {}).get("public", {}).get("startDateTime")
        prices = e.get("priceRanges") or [{}]
        event_day = date.fromisoformat(start) if start else None
        events.append(dict(
            date=event_day,
            venue=(e.get("_embedded", {}).get("venues") or [{}])[0].get("name", ""),
            on_sale=datetime.fromisoformat(on_sale.replace("Z", "+00:00")).date() if on_sale else None,
            status=status,
            # 'offsale' before the show has happened usually means sold out
            high_demand=(status == "offsale" and event_day is not None and event_day >= date.today()),
            # Assumption: UK Ticketmaster arena/stadium shows use app (SafeTix) tickets. Verify per tour.
            app_only=True,
            face_min=prices[0].get("min"), face_max=prices[0].get("max"),
        ))
    return events


# ---------------------------------------------------------------------------
# Source 2: Skiddle Events API (UK gigs, clubs, festivals)
# Free key: apply at https://www.skiddle.com/api/join.php
# Terms: you must credit Skiddle (name + logo) wherever its data is shown.
# ---------------------------------------------------------------------------
def _skiddle(artist: str, api_key: str, event_date: date | None) -> list[dict]:
    import requests
    params = {"api_key": api_key, "keyword": artist, "limit": 100, "description": 1}
    if event_date:  # narrow the search to the month around the date
        params["minDate"] = date(event_date.year, event_date.month, 1).isoformat()
    resp = requests.get(SKIDDLE_URL, timeout=6, params=params)
    resp.raise_for_status()
    events = []
    for e in resp.json().get("results", []) or []:
        names = [e.get("eventname", "")] + [a.get("name", "") for a in (e.get("artists") or [])]
        if not any(artist_matches(artist, n) for n in names):
            continue
        day = e.get("date") or (e.get("startdate") or "")[:10]
        price = str(e.get("entryprice") or "").replace("£", "").split()[0] if e.get("entryprice") else None
        try:
            price = float(price) if price else None
        except ValueError:
            price = None
        events.append(dict(
            date=date.fromisoformat(day) if day else None,
            venue=(e.get("venue") or {}).get("name", ""),
            on_sale=None,                      # Skiddle doesn't give a general on-sale date
            status="cancelled" if str(e.get("cancelled", "0")) == "1" else "onsale",
            high_demand=False,                 # not available from Skiddle
            app_only=None,                     # unknown: Skiddle often uses e-tickets, so don't assume
            face_min=price, face_max=price,
        ))
    return events


# ---------------------------------------------------------------------------
# Source 3: offline demo catalogue
# ---------------------------------------------------------------------------
def _catalogue(artist: str) -> list[dict] | None:
    """Returns None if the artist isn't in the demo catalogue at all
    (a small demo list can't prove an event doesn't exist)."""
    if not CATALOGUE.exists():
        return None
    df = pd.read_csv(CATALOGUE)
    df = df[df["artist"].apply(lambda n: artist_matches(artist, n))]
    if df.empty:
        return None
    return [dict(date=date.fromisoformat(r.date), venue=r.venue,
                 on_sale=date.fromisoformat(r.on_sale_date), status=r.status,
                 high_demand=bool(r.high_demand), app_only=bool(r.app_only),
                 face_min=r.face_min, face_max=r.face_max) for r in df.itertuples()]


# ---------------------------------------------------------------------------
# Public function
# ---------------------------------------------------------------------------
# Ticket sellers we can search. AXS, See Tickets, DICE, Eventim and Gigantic
# have no public API (partner access only), so we can't confirm their events.
SEARCHABLE_SELLERS = {"Ticketmaster", "Skiddle"}
SELLER_OPTIONS = ["Ticketmaster", "AXS", "See Tickets", "DICE", "Skiddle",
                  "Eventim", "Gigantic", "Other"]   # pick all that apply; none picked = don't know


def check_event(artist: str, venue: str = "", event_date: date | None = None,
                today: date | None = None, official_sellers=None) -> dict:
    """official_sellers = list of who sells the official tickets (from the artist's
    website); empty = don't know. Gigs are often split across several sellers.
    We only say 'event not found' when EVERY official seller is one we searched;
    otherwise a missing result just means 'couldn't check' (no red flag)."""
    if isinstance(official_sellers, str):  # accept a single name too
        official_sellers = [official_sellers]
    sellers = [s for s in (official_sellers or []) if s != "Don't know"]
    today = today or date.today()
    if not artist or not artist.strip():
        return _empty("none", "No artist entered, so the event wasn't checked.")

    # 1) Ask every live source we have a key for
    sources = {"Ticketmaster": ("TICKETMASTER_API_KEY", lambda k: _ticketmaster(artist, k)),
               "Skiddle": ("SKIDDLE_API_KEY", lambda k: _skiddle(artist, k, event_date))}
    events, searched, problems, hits = [], [], [], []
    for name, (env_var, fetch) in sources.items():
        key = os.getenv(env_var)
        if not key:
            continue
        try:
            found = fetch(key)
            searched.append(name)
            events += found
            if found:
                hits.append(name)
        except Exception as e:  # network down, bad key, rate limit -> carry on, never crash
            problems.append(f"{name} lookup failed ({e.__class__.__name__})")

    if events:
        return _judge(events, venue, event_date, today, " + ".join(hits))

    if searched:
        if sellers and all(x in searched for x in sellers):
            r = _empty(" + ".join(searched),
                       f"No official event found on {' or '.join(sellers)} for this artist.")
            r["found"] = False
            return r
        unsearched = [x for x in sellers if x not in searched]
        if any(x in SEARCHABLE_SELLERS for x in unsearched):
            problems.append(f"Not found on {' or '.join(searched)}, and "
                            f"{', '.join(x for x in unsearched if x in SEARCHABLE_SELLERS)} couldn't be searched")
        else:
            who = ", ".join(unsearched) if unsearched else "the official seller"
            # don't return yet: the demo catalogue below may still know the artist
            problems.append(f"Not found on {' or '.join(searched)}. Tickets may be sold by {who}, "
                            "which we can't search, so check the artist's official website")

    # 2) Fall back to the offline demo catalogue
    reason = "; ".join(problems) or "No API keys set, used demo catalogue"
    events = _catalogue(artist)
    if events is None:
        return _empty("demo catalogue", reason + ". Event not checked.")
    result = _judge(events, venue, event_date, today, "demo catalogue")
    result["note"] = f"{result['note']} ({reason})"
    return result


if __name__ == "__main__":
    # quick manual test:  python event_check.py
    for a, v, d in [("Burna Boy", "O2 Arena", date(2027, 3, 14)),
                    ("Burna Boy", "O2 Arena", date(2027, 3, 20)),
                    ("Stray Kids", "Wembley", date(2027, 7, 3)),
                    ("Unknown Band", "", None)]:
        print(a, v, d, "->", check_event(a, v, d))

```

## step1_make_dataset.py
```python
"""
step1_make_dataset.py - build a SYNTHETIC labelled dataset of resale listings.

There is no public dataset of labelled ticket-scam listings, so we generate one
based on published scam indicators. Be upfront about this in your video: it's a
limitation, and "real data via a bank/promoter partnership" is your next phase.

Changes in v2:
  * Listings are assembled from mix-and-match phrase pools (opener + details +
    reason + payment + delivery + extras) instead of 8 fixed templates. This
    reduces "template leakage" (the model learning one template's wording).
  * New red flags appear in the text: payment switch, deposit/hold, hidden
    payment reference, "100% legit" trust claims, asking for codes, PDF tickets.
  * Each row also gets seller + event-check fields (sudden_seller, event found,
    details match, high demand, before on-sale, app-only tickets).
  * v2.3: after the real-world error analysis, added short vague scam posts,
    short genuine posts, listings with no price, and new scam tactics
    (third-party accounts, presale-code sales, like-and-comment bait,
    screenshots as proof, "up front" payments). Wording is new, not copied
    from the real-world test set.
  * v2.2: each row also records WHERE it was sold (official / secondary resale /
    social media & messaging / unknown website) plus link warning signs.

To stop an unrealistic 100%, we add overlap on purpose: messy genuine sellers
(bank transfer, urgency, WhatsApp) and sneaky scams (polished, seat details,
only one red flag), plus ~3% label noise.

Output: data/listings.csv
"""
import random
from pathlib import Path

import pandas as pd

random.seed(42)

ARTISTS = ["Burna Boy", "Tems", "SZA", "Stray Kids", "SEVENTEEN", "Beyonce",
           "Central Cee", "Rema", "Olivia Rodrigo", "BLACKPINK", "Dave", "Coldplay"]
VENUES = ["O2 Arena", "Wembley Stadium", "OVO Arena Wembley", "Co-op Live",
          "Tottenham Hotspur Stadium", "AO Arena", "Utilita Arena"]

# ---------------------------------------------------------------- phrase pools
OPENERS = [
    "Selling {n} tickets for {artist} at {venue}.",
    "{n}x {artist} {venue}.",
    "Spare tickets for {artist}, {venue}.",
    "Got {n} {artist} tickets for {venue} I can't use.",
    "{artist} @ {venue}, {n} tickets available.",
    "Anyone want {artist} tickets? {venue}.",
]
SCAM_OPENERS = OPENERS + [
    "{artist} tickets at {venue}!!",
    "Last minute {n}x {artist} at {venue}.",
    "SELLING {artist} {venue} TICKETS",
]
GENUINE_DETAILS = [
    "{section}. £{price} each, face value.",
    "{section}, paid £{price}, selling for the same.",
    "£{price} each, {section}.",
    "{section}. Selling at what I paid (£{price}).",
]
SCAM_DETAILS = [
    "Only £{price}!!",
    "£{price} each.",
    "Cheap at £{price}.",
    "{section}. £{price} each.",
]
GENUINE_REASONS = ["Can't make the date anymore.", "Can't go anymore, gutted.", "Work came up so can't make it.", "Clashes with a wedding.", "Friend dropped out.",
                   "Bought too many in the presale.", "Moving away that month.", ""]
SCAM_REASONS = ["Can't go anymore due to a family emergency.", "Work came up last minute.",
                "I'm ill and can't go.", "Can't go anymore.", ""]
PROTECTED_PAYMENT = [
    "PayPal goods and services only.",
    "Happy to use PayPal G&S so you're covered.",
    "Will list them on Twickets so you're protected.",
    "Can do it through the Ticketmaster resale so it's safe for both of us.",
    "Paying through the official exchange is fine.",
]
UNSAFE_PAYMENT = [
    "Bank transfer only.",
    "PayPal friends and family only.",
    "Revolut or bank transfer.",
    "Crypto or gift cards accepted.",
    # payment switch
    "PayPal isn't working on my account so bank transfer instead.",
    "Can't take PayPal at the moment, bank instead please.",
    # deposit / hold
    "£20 deposit to hold them for you, rest before the show.",
    "Pay half now and half when you get them.",
    # hide reference
    "Don't put tickets in the reference or PayPal holds the money.",
    "Send it as a gift so there are no fees.",
]
GENUINE_DELIVERY = [
    "Tickets transferred through the Ticketmaster app.",
    "Transfer on the official app straight away.",
    "Happy to show proof of purchase.",
    "Can transfer as soon as payment goes through the platform.",
]
SCAM_DELIVERY = [
    "Will send after payment.",
    "Tickets sent once paid.",
    "I'll email you the PDF tickets once paid.",
    "Can send screenshots of the tickets.",
    "I'll transfer them after you pay.",
]
SCAM_EXTRAS = [
    # v2.3 additions: tactics found in the real-world error analysis, worded fresh
    "Payment goes to my cousin's account as mine is restricted at the moment.",
    "Going at a discount, just want them gone to a real fan.",
    "Can send a screenshot of the booking so you know they're real.",
    "Also got a couple of presale codes going for £30 each.",
    "Like and comment below and I'll inbox you the details.",
    "Need the full amount up front to keep them for you.",
    "Few others asking so first come first served.",
    "Loads of people messaging, first to pay gets them.",
    "Serious buyers only.",
    "DM me on WhatsApp.",
    "Message me on Telegram.",
    "100% legit, can vouch.",
    "Genuine seller, not a scam.",
    "I'll send you a 6 digit code to check you're real, just send the code back.",
    "Won't last!!",
]
# Genuine-but-messy extras: realistic overlap with scam wording
MESSY_EXTRAS = [
    "Bank transfer is easiest for me but can do PayPal.",
    "Need them gone quick.",
    "Message me on WhatsApp if it's easier.",
    "Can send a screenshot of my order confirmation.",
    "Genuine seller btw, check my profile.",
    "Tickets are PDFs by email for this venue.",  # only used when event isn't app-only
]
# Sneaky scams: polished, protected-sounding, but one quiet red flag
SNEAKY_FLAGS = [
    "Happy to transfer on the app, just prefer friends and family to avoid fees.",
    "Easier to chat on WhatsApp, payment by bank transfer.",
    "Small deposit to hold them and I'll transfer on the app.",
    "PayPal is limited on my account right now so bank transfer instead.",
    "Please leave the reference blank, PayPal is funny about tickets.",
    random.choice(["Can email the PDF tickets over tonight.", "Will send the tickets as a PDF.",
                   "They're e-tickets, I'll forward you the PDF.", "PDF tickets, sent by email."]),
]


def section():
    return random.choice([f"Block {random.randint(101, 430)} Row {random.choice('ABCDEFGHJK')} "
                          f"Seats {random.randint(1, 30)}-{random.randint(31, 40)}",
                          "Standing", "Floor standing", "Lower tier"])


def pick(pool, p=1.0):
    return random.choice(pool) if random.random() < p else ""


def event_fields(is_scam: bool, mentions_pdf: bool) -> dict:
    """Simulate what event_check.py would return. found: 1 yes, 0 no, -1 couldn't check."""
    r = random.random()
    if is_scam:
        found = -1 if r < 0.10 else 0 if r < 0.25 else 1
        details_match = int(random.random() > 0.25) if found == 1 else -1
        high_demand = int(random.random() < 0.80)
        before_on_sale = int(random.random() < 0.12)
        app_only = int(random.random() < 0.85)
    else:
        found = -1 if r < 0.06 else 0 if r < 0.09 else 1       # not found = API coverage gap
        details_match = int(random.random() > 0.03) if found == 1 else -1
        high_demand = int(random.random() < 0.50)
        before_on_sale = int(random.random() < 0.01)
        app_only = 0 if mentions_pdf else int(random.random() < 0.85)
    return dict(event_found=found, event_details_match=details_match, event_high_demand=high_demand,
                event_before_on_sale=before_on_sale, event_app_only=app_only)


def channel_fields(kind: str) -> dict:
    """Simulate WHERE the listing was found (what channels.py would return).
    Shares are illustrative, guided by bank reports that most ticket scams start
    on social media; genuine fans also sell on social media, so there is overlap."""
    shares = {  # official, secondary, social, unknown_site, not given
        "genuine": [0.35, 0.15, 0.40, 0.03, 0.07],
        "messy":   [0.10, 0.10, 0.70, 0.00, 0.10],
        "scam":    [0.02, 0.06, 0.70, 0.12, 0.10],
        "sneaky":  [0.03, 0.15, 0.60, 0.15, 0.07],
        "vague":   [0.00, 0.02, 0.88, 0.02, 0.08],
        "short_genuine": [0.30, 0.05, 0.55, 0.00, 0.10],
    }[kind]
    cat = random.choices(["official", "secondary", "social", "unknown_site", "none"], weights=shares)[0]
    is_scam = kind in ("scam", "sneaky", "vague")
    site = cat == "unknown_site"
    return dict(
        ch_category=cat,
        ch_lookalike=int(site and is_scam and random.random() < 0.5),
        ch_new_domain=int(site and random.random() < (0.8 if is_scam else 0.2)),
        ch_insecure_or_short=int(site and random.random() < (0.4 if is_scam else 0.05)),
    )


# v2.3: real scam posts are often tiny and vague, and genuine posts are often short too.
VAGUE_SCAM = [
    "Tickets available, message me.",
    "{n} spare for {artist}, inbox me.",
    "Anyone need {artist} tickets? DM.",
    "Got tickets going for {artist} tonight, pm for info.",
    "Selling my {artist} tickets cheap, message me for details.",
    "Have {n} for {venue}, hit me up if interested.",
    "{artist} tickets going, DM for price.",
    "Can't make {artist} anymore, {n} tickets going at a discount, PM.",
    "Who wants {artist} tickets?? Message me x",
    "Spare {artist} tix, send me a message.",
]
SHORT_GENUINE = [
    "Spare {artist} ticket tonight, £{price} face value, DICE transfer.",
    "1 x {artist} standing, £{price}, listing it on Twickets now.",
    "{artist} at {venue}, {section}, £{price} each, Ticketmaster transfer + PayPal G&S.",
    "Selling 1 for {artist} at face (£{price}), will send via the official app.",
    "{n} tickets for {artist}, £{price}, AXS transfer, paypal goods and services.",
]


def make(kind: str) -> dict:
    """kind: genuine | messy | short_genuine | scam | sneaky | vague"""
    is_scam = kind in ("scam", "sneaky", "vague")
    face = random.choice([45, 55, 70, 85, 95, 120, 150])
    if kind == "scam":
        price = int(face * random.uniform(0.35, 0.85))
    elif kind == "sneaky":
        price = int(face * random.uniform(0.85, 1.05))
    else:
        price = int(face * random.uniform(0.9, 1.1))
    seat = section() if (kind != "scam" or random.random() < 0.3) else "Standing"

    if kind == "genuine":
        parts = [pick(OPENERS), pick(GENUINE_DETAILS), pick(GENUINE_REASONS),
                 pick(PROTECTED_PAYMENT), pick(GENUINE_DELIVERY, 0.8)]
    elif kind == "messy":
        parts = [pick(OPENERS), pick(GENUINE_DETAILS), pick(GENUINE_REASONS),
                 pick(MESSY_EXTRAS), pick(PROTECTED_PAYMENT, 0.4), pick(GENUINE_DELIVERY, 0.6)]
    elif kind == "scam":
        parts = [pick(SCAM_OPENERS), pick(SCAM_DETAILS), pick(SCAM_REASONS),
                 pick(UNSAFE_PAYMENT), pick(SCAM_DELIVERY, 0.8),
                 pick(SCAM_EXTRAS, 0.7), pick(SCAM_EXTRAS, 0.3)]
    elif kind == "vague":
        parts = [pick(VAGUE_SCAM), pick(SCAM_EXTRAS, 0.3)]
    elif kind == "short_genuine":
        parts = [pick(SHORT_GENUINE)]
    else:  # sneaky
        flag = pick(SNEAKY_FLAGS)
        if "PDF" in flag:  # vary the wording so the model can't memorise one sentence
            flag = random.choice(["Can email the PDF tickets over tonight.", "Will send the tickets as a PDF.",
                                  "They're e-tickets, I'll forward you the PDF.", "PDF tickets, sent by email."])
        parts = [pick(OPENERS), pick(GENUINE_DETAILS), pick(GENUINE_REASONS, 0.5), flag]

    text = " ".join(p for p in parts if p).format(
        n=random.choice([1, 2, 2, 3, 4]), artist=random.choice(ARTISTS),
        venue=random.choice(VENUES), section=seat, price=price)
    has_seat = int("Row" in seat and "Row" in text)
    mentions_pdf = "pdf" in text.lower()
    # v2.3: price often isn't stated in real posts -> ratio unknown (treated as 1.0)
    price_known = "£" in text and random.random() > (0.4 if is_scam else 0.15)
    ratio = round(price / face, 2) if price_known else 1.0

    # --- seller account ---
    sudden = 0
    if is_scam:
        sudden = int(random.random() < 0.25)      # hacked friend's account
        if sudden:
            age, followers = random.randint(400, 3000), random.randint(150, 1500)
        else:
            age = random.choice([random.randint(1, 45), random.randint(1, 45), random.randint(100, 900)])
            followers = random.choice([random.randint(0, 40), random.randint(0, 40), random.randint(80, 600)])
    else:
        sudden = int(random.random() < 0.03)
        age = random.choice([random.randint(200, 3000), random.randint(200, 3000), random.randint(10, 60)])
        followers = random.choice([random.randint(100, 1500), random.randint(100, 1500), random.randint(5, 40)])

    return dict(text=text, account_age_days=age, followers=followers, sudden_seller=sudden,
                price_ratio=ratio, has_seat_details=has_seat,
                **event_fields(is_scam, mentions_pdf), **channel_fields(kind), label=int(is_scam))


rows = ([make("genuine") for _ in range(240)] + [make("messy") for _ in range(70)] +
        [make("short_genuine") for _ in range(60)] +
        [make("scam") for _ in range(220)] + [make("sneaky") for _ in range(80)] +
        [make("vague") for _ in range(80)])

# ~3% label noise, because real-world labels are never perfect
df = pd.DataFrame(rows).sample(frac=1, random_state=42).reset_index(drop=True)
flip = df.sample(frac=0.03, random_state=1).index
df.loc[flip, "label"] = 1 - df.loc[flip, "label"]

Path("data").mkdir(exist_ok=True)
df.to_csv("data/listings.csv", index=False)
print(f"Saved {len(df)} listings to data/listings.csv  "
      f"(scams: {df.label.sum()}, genuine: {(1 - df.label).sum()})")

```

## step2_train_model.py
```python
"""
step2_train_model.py - Layer 2: train and test the scam classifier.

Model: TF-IDF on the listing text + all red-flag features (payment, language,
seller, ticket evidence, event checks) -> logistic regression.
Simple, fast, and explainable (you can show which words/flags push the score up).

Outputs:
  model/scam_model.joblib           - the trained model (used by the app)
  outputs/confusion_matrix.png      - put this on a slide
  outputs/top_features.png          - "what the model learned" slide
  printed precision / recall report - quote these numbers in the video
"""
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (ConfusionMatrixDisplay, classification_report,
                             precision_score, recall_score)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline

from features import FLAG_COLUMNS, HARD_STOPS, all_flags

THRESHOLD = 0.4  # below 0.5 on purpose: missing a scam costs more than a false alarm


def tri(v):
    """dataset stores 1 / 0 / -1 (couldn't check) -> True / False / None"""
    return None if v == -1 else bool(v)


def row_to_event(r) -> dict:
    return dict(found=tri(r.event_found), details_match=tri(r.event_details_match),
                high_demand=bool(r.event_high_demand), before_on_sale=bool(r.event_before_on_sale),
                app_only=bool(r.event_app_only))


def row_to_channel(r) -> dict:
    return dict(category=None if r.ch_category == "none" else r.ch_category,
                lookalike=bool(r.ch_lookalike), new_domain=bool(r.ch_new_domain),
                insecure_or_short=bool(r.ch_insecure_or_short))


df = pd.read_csv("data/listings.csv")
flags = df.apply(lambda r: all_flags(r.text, r.account_age_days, r.followers, r.price_ratio,
                                     r.has_seat_details, r.sudden_seller, row_to_event(r),
                                     row_to_channel(r)), axis=1)
# (sudden_seller is both a raw column and a flag, so drop raw copies before joining)
df = pd.concat([df.drop(columns=[c for c in FLAG_COLUMNS if c in df.columns]),
                pd.DataFrame(list(flags))], axis=1)

print("How often each flag fires (scam rate when it fires):")
summary = pd.DataFrame({"fires": df[FLAG_COLUMNS].sum(),
                        "scam_rate": [df.loc[df[c] == 1, "label"].mean() for c in FLAG_COLUMNS]})
print(summary.round(2).to_string(), "\n")

X, y = df[["text"] + FLAG_COLUMNS], df["label"]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, stratify=y, random_state=42)

model = Pipeline([
    ("features", ColumnTransformer([
        ("words", TfidfVectorizer(ngram_range=(1, 2), min_df=2), "text"),
        ("flags", "passthrough", FLAG_COLUMNS),
    ])),
    ("clf", LogisticRegression(max_iter=1000, class_weight="balanced")),
])
model.fit(X_train, y_train)

# ---- Evaluate: model alone ----
proba = model.predict_proba(X_test)[:, 1]
pred = (proba >= THRESHOLD).astype(int)
print("MODEL ONLY")
print(classification_report(y_test, pred, target_names=["genuine", "scam"]))
print(f"Scam recall (share of scams caught): {recall_score(y_test, pred):.0%}")
print(f"Scam precision (share of alerts that were real scams): {precision_score(y_test, pred):.0%}\n")

# ---- Evaluate: model + hard-stop rules ----
hard = X_test[list(HARD_STOPS)].max(axis=1).values
pred_hs = ((pred == 1) | (hard == 1)).astype(int)
print("MODEL + HARD-STOP RULES")
print(f"Scam recall: {recall_score(y_test, pred_hs):.0%}   "
      f"precision: {precision_score(y_test, pred_hs):.0%}\n")

Path("outputs").mkdir(exist_ok=True)
disp = ConfusionMatrixDisplay.from_predictions(y_test, pred, display_labels=["Genuine", "Scam"],
                                               cmap="Blues", colorbar=False)
disp.ax_.set_title("Ticket scam checker - test set results")
plt.tight_layout(); plt.savefig("outputs/confusion_matrix.png", dpi=200); plt.close()

# ---- What did the model learn? (top features pushing towards "scam") ----
names = model.named_steps["features"].get_feature_names_out()
coefs = model.named_steps["clf"].coef_[0]
top = (pd.Series(coefs, index=[n.split("__", 1)[1] for n in names])
       .sort_values(ascending=False).head(15)[::-1])
top.plot.barh(figsize=(7, 6), color="#c0392b")
plt.title("Top signals the model links to scams"); plt.xlabel("Model weight")
plt.tight_layout(); plt.savefig("outputs/top_features.png", dpi=200); plt.close()
print("Top 15 signals:", ", ".join(top.index[::-1]))

Path("model").mkdir(exist_ok=True)
joblib.dump({"model": model, "threshold": THRESHOLD}, "model/scam_model.joblib")
print("Saved model/scam_model.joblib and charts in outputs/")

```

## explainer.py
```python
"""
explainer.py - Layer 3: the LLM turns the score + red flags into plain English.

The LLM does NOT decide if it's a scam - the classifier (plus hard-stop rules)
does. The LLM only explains. If no API key is set, or the call fails, it falls
back to a template so the demo always works.

Pick ONE provider (Gemini is free via Google AI Studio):
    export GEMINI_API_KEY="your-key"        # pip install google-genai
    export ANTHROPIC_API_KEY="your-key"     # pip install anthropic
"""
import os
import time

SYSTEM_PROMPT = """You are a friendly fraud-prevention assistant for music fans buying resale tickets.
You are given a risk level from a trained model, the red flags it found, where the ticket is
being sold, any good signs, and the result of an official event check. Do NOT change the risk
level or invent new red flags. In under 90 words:
1) say the risk level in one sentence, 2) explain the red flags in plain English,
3) give one concrete safer next step that fits where they are buying (e.g. pay with PayPal Goods & Services or a credit card,
use a face-value resale site like Twickets or the official Ticketmaster exchange, never share codes).
Warm, clear, no jargon, UK English."""


# Advice that fits WHERE they're buying (category from channels.py)
CHANNEL_ADVICE = {
    "official": "You're buying through an official or face-value platform, which protects buyers, "
                "so keep the payment and ticket transfer inside that platform.",
    "secondary": "Resale sites like viagogo and StubHub have buyer guarantees, but tickets are often "
                 "far above face value and can be cancelled by the event, so compare with Twickets "
                 "or the official resale first.",
    "social": "Social media and messaging apps give you no buyer protection, so never pay by bank "
              "transfer or friends-and-family there.",
    "unknown_site": "Check the web address carefully and only buy from sites you can verify "
                    "independently, not from a link someone sent you.",
}


def fallback(level, flags, event_note="", channel_category=None):
    advice = CHANNEL_ADVICE.get(channel_category, "")
    if not flags:
        return (f"{level} risk. We didn't spot common scam signs in this listing. "
                + (advice + " " if advice else "") +
                "Pay with a protected method (PayPal Goods & Services or a credit card) "
                "and only accept tickets transferred through the official app.")
    bullets = "; ".join(f.lower() for f in flags)
    return (f"{level} risk. This listing shows signs that often appear in ticket scams: {bullets}. "
            "Safer option: don't pay by bank transfer or friends-and-family, never share a code "
            "someone sends you, and look for the same tickets on a face-value resale site like "
            "Twickets or the official exchange." + (" " + advice if advice else ""))


def _prompt(listing_text, score, level, flags, event_note, where, good_signs):
    return (f"Listing:\n{listing_text}\n\nWhere it's being sold: {where or 'not given'}\n"
            f"Model scam score: {score:.0%}\nRisk level: {level}\n"
            f"Red flags found: {flags or 'none'}\nGood signs: {good_signs or 'none'}\n"
            f"Event check: {event_note or 'not checked'}")


# Gemini model names change often (Google retires old ones), so try a few in order.
# Set GEMINI_MODEL in .env to force a specific one.
GEMINI_MODELS = [m for m in [os.getenv("GEMINI_MODEL"), "gemini-flash-latest",
                             "gemini-flash-lite-latest", "gemini-3-flash-preview",
                             "gemini-2.5-flash"] if m]
# Errors where trying again / another model can help:
#   404 NOT_FOUND = model retired, 503 UNAVAILABLE = model overloaded right now
RETRYABLE = ("404", "NOT_FOUND", "503", "UNAVAILABLE", "INCOMPLETE")


def _gemini(prompt):
    from google import genai
    client = genai.Client()
    last_error = None
    for attempt in range(2):                 # two passes through the list, short pause between
        for model in GEMINI_MODELS:
            try:
                # Newer Gemini models "think" before answering, and that thinking uses up
                # the output allowance - so give plenty of room (the prompt keeps the
                # answer itself under 90 words).
                resp = client.models.generate_content(
                    model=model, contents=prompt,
                    config=genai.types.GenerateContentConfig(system_instruction=SYSTEM_PROMPT,
                                                             max_output_tokens=2048))
                text = (resp.text or "").strip()
                finish = str(getattr((resp.candidates or [None])[0], "finish_reason", ""))
                if "MAX_TOKENS" in finish or len(text.split()) < 20:
                    raise RuntimeError("INCOMPLETE answer from Gemini")  # never show a cut-off sentence
                return text
            except Exception as e:
                last_error = e
                if any(code in str(e) for code in RETRYABLE):  # retired or busy -> try the next model
                    continue
                raise                                          # bad key, quota etc. -> stop and report
        time.sleep(2)
    raise last_error


def _short(e):
    """Readable reason for the fallback message (never includes the key)."""
    text = str(e)
    for hint, meaning in [("API_KEY_INVALID", "API key not valid - check GEMINI_API_KEY in .env"),
                          ("PERMISSION_DENIED", "key not allowed to use this model"),
                          ("RESOURCE_EXHAUSTED", "free-tier limit reached - wait a minute and retry"),
                          ("UNAVAILABLE", "Gemini is busy right now - try again in a minute"),
                          ("INCOMPLETE", "Gemini's answer was cut off, so the standard explanation is shown"),
                          ("NOT_FOUND", "model name not available - set GEMINI_MODEL in .env")]:
        if hint in text:
            return meaning
    return f"{e.__class__.__name__}: {text[:150]}"


def explain(listing_text, score, level, flags, event_note="", where="", good_signs=None,
            channel_category=None):
    prompt = _prompt(listing_text, score, level, flags, event_note, where, good_signs)
    try:
        if os.getenv("GEMINI_API_KEY"):
            return _gemini(prompt)
        if os.getenv("ANTHROPIC_API_KEY"):
            import anthropic
            msg = anthropic.Anthropic().messages.create(
                model=os.getenv("CLAUDE_MODEL", "claude-haiku-4-5"), max_tokens=300,
                system=SYSTEM_PROMPT, messages=[{"role": "user", "content": prompt}])
            return msg.content[0].text
    except Exception as e:  # never let the demo crash on camera
        return (fallback(level, flags, channel_category=channel_category)
                + f"\n\n(AI explanation unavailable: {_short(e)})")
    return fallback(level, flags, channel_category=channel_category)

```

## app.py
```python
"""
app.py - the demo app. Run with:   streamlit run app.py
"""
import os
import subprocess
import sys

import joblib
import pandas as pd
import streamlit as st

# Load API keys from a local .env file (never committed - it's in .gitignore).
# On Streamlit Cloud, keys come from the app's Secrets settings instead.
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

from channels import PLATFORM_CHOICES, PLATFORMS, channel_info
from event_check import SELLER_OPTIONS, check_event
from explainer import explain
from features import FLAG_COLUMNS, all_flags, hard_stops, protections, triggered

st.set_page_config(page_title="Ticket Scam Checker", page_icon="🎟️", layout="centered")
if not os.path.exists("model/scam_model.joblib"):  # first run online: train the model
    subprocess.run([sys.executable, "step2_train_model.py"], check=True)
bundle = joblib.load("model/scam_model.joblib")
model, threshold = bundle["model"], bundle["threshold"]

st.title("🎟️ Ticket Scam Checker")
st.caption("Paste a resale listing before you pay. Prototype - synthetic training data.")

# ---------------- 1. Where are you buying? ----------------
st.subheader("Where are you buying?")
platform = st.selectbox("Where did you find these tickets?", PLATFORM_CHOICES)
platform = None if platform == PLATFORM_CHOICES[0] else platform
link = st.text_input("Link to the listing or website (optional)",
                     placeholder="e.g. https://www.viagogo.co.uk/... or a link someone sent you")
st.caption("We never open the link. We only check the web address itself.")

# ---------------- 2. The listing ----------------
st.subheader("The listing")
text = st.text_area("Listing text (paste the post or messages)", height=140,
                    placeholder="e.g. 2x Burna Boy tickets O2 Arena, £40 each, bank transfer only...")

# ---------------- 3. The event ----------------
st.subheader("The event")
e1, e2, e3 = st.columns(3)
artist = e1.text_input("Artist", placeholder="Burna Boy")
venue = e2.text_input("Venue", placeholder="O2 Arena")
event_date = e3.date_input("Date", value=None, format="DD/MM/YYYY")
sellers = st.multiselect("Who sells the OFFICIAL tickets? Pick all that apply - check the artist's "
                         "website. Leave empty if you don't know.", SELLER_OPTIONS)
p1, p2 = st.columns(2)
price = p1.number_input("Asking price (£ per ticket)", 0.0, 5000.0, 40.0)
face = p2.number_input("Face value (£ per ticket, 0 = look it up)", 0.0, 5000.0, 0.0)
seat = st.checkbox("Listing gives section / row / seat details")

# ---------------- 4. The seller ----------------
st.subheader("The seller")
s1, s2 = st.columns(2)
age = s1.number_input("Seller account age (days)", 0, 10000, 30)
followers = s2.number_input("Seller followers", 0, 1000000, 20)
sudden = st.checkbox("It's an older account that has suddenly started selling tickets "
                     "(e.g. a friend's account posting out of character)")

clicked = st.button("Check listing", type="primary")
if clicked and not (platform or link.strip()):
    st.warning("First tell us where you're buying: pick a platform or paste the link.")
elif clicked and not text.strip():
    st.warning("Paste the listing text or the seller's messages.")
elif clicked:
    with st.spinner("Checking where it's sold and the event..."):
        channel = channel_info(platform, link)
        event = check_event(artist, venue, event_date, official_sellers=sellers)

    # Face value: user's number, else the cheapest official price, else skip the price check
    face_used = face or event.get("face_min") or 0
    price_ratio = price / face_used if face_used else 1.0

    flags = all_flags(text, age, followers, price_ratio, int(seat), int(sudden), event, channel)
    X = pd.DataFrame([{"text": text, **flags}])[["text"] + FLAG_COLUMNS]
    score = float(model.predict_proba(X)[0, 1])
    found = triggered(flags)
    good = protections(flags)
    stops = hard_stops(flags)

    level = "High" if score >= 0.7 else "Medium" if score >= threshold else "Low"
    if stops:
        level = "High"  # hard-stop rule overrides the model

    msg = f"### {level} risk - model score {score:.0%}"
    {"High": st.error, "Medium": st.warning, "Low": st.success}[level](msg)
    st.progress(score)
    if stops:
        st.caption("Set to High by a hard-stop rule: " + "; ".join(stops))

    st.subheader("Red flags found")
    if found:
        for f in found:
            st.markdown(f"- 🚩 {f}")
    else:
        st.markdown("No common red flags spotted.")
    for g in good:
        st.markdown(f"- ✅ {g}")

    st.subheader("Where it's being sold")
    where = platform or (channel["link"] or {}).get("site") or "a website"
    st.markdown(f"**{where}**" + (f" (link: `{channel['link']['domain']}`)" if channel["link"] else ""))
    if channel["link"]:
        for note in channel["link"]["notes"]:
            st.markdown(f"- {note}")
        if platform and PLATFORMS.get(platform) != channel["link"]["category"] and channel["link"]["site"]:
            st.caption(f"Note: you picked {platform}, but the link goes to {channel['link']['site']}.")

    st.subheader("Event check")
    st.markdown(f"**Source:** {event['source']}  \n{event['note']}")
    if event.get("found"):
        st.markdown(f"Official event: **{event['official_venue']}**, {event['official_date']}  \n"
                    f"Status: {event.get('status') or 'unknown'}"
                    + (f"  \nOfficial price range: £{event['face_min']:.0f}–£{event['face_max']:.0f}"
                       if event.get("face_min") else ""))
        if event.get("status") in ("cancelled", "postponed", "rescheduled"):
            st.warning(f"This event is {event['status']}. Check the official site before buying anything.")
    if "Skiddle" in event["source"]:
        st.caption("Event data from [Skiddle](https://www.skiddle.com).")  # required by Skiddle's API terms
    if not face_used:
        st.caption("No face value available, so the price check was skipped.")

    st.subheader("What this means")
    with st.spinner("Writing explanation..."):
        st.write(explain(text, score, level, found, event["note"], where=where,
                         good_signs=good, channel_category=channel["category"]))

```

## evaluate_real.py
```python
"""
evaluate_real.py - does the checker work on REAL scam examples, and does the
hybrid design beat each of its parts on their own?

Run AFTER step1 + step2 + build_real_test_set.py:
    python evaluate_real.py            # rules, text model, hybrid (development set)
    python evaluate_real.py --test data/heldout_test_set.csv   # your independent held-out set
    (with GEMINI_API_KEY in .env, the "AI only" arm runs too)

Four approaches, all judged on data/real_test_set.csv (never used in training):
  1. Rules only   - hand-written red-flag rules, no machine learning.
                    Scam if any hard-stop rule fires OR 2+ meaningful red flags.
  2. Text model   - machine learning on the listing wording only (TF-IDF +
                    logistic regression, trained on the synthetic data, no rules).
  3. AI only      - an LLM (Gemini) asked directly "is this a scam?", zero-shot.
  4. Hybrid       - the actual checker: rule flags + wording in one model,
                    plus hard-stop rules. (The LLM only explains, it never decides.)

All decision rules and thresholds were fixed BEFORE looking at the results.
The event check is switched off for every approach, because these are past or
example events that can't be looked up fairly. Website age (RDAP) is also off,
because several test links are invented examples.

Outputs (in outputs/):
  real_eval_summary.csv      - one row per approach
  real_eval_predictions.csv  - every listing, every approach, which flags fired
  real_eval_chart.png        - chart for the report
  real_eval_report.md        - tables + error analysis, ready to quote
"""
from __future__ import annotations

import json
import os
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline

from channels import channel_info
from features import FLAG_COLUMNS, HARD_STOPS, PROTECTIVE, all_flags

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

OUT = Path("outputs")
OUT.mkdir(exist_ok=True)
THRESHOLD = 0.4                                   # same threshold as the live app
WEAK_FLAGS = {"no_seat_details", "high_demand"}   # too common to count on their own (decided up front)
NEUTRAL_AGE, NEUTRAL_FOLLOWERS = 365, 200         # used when a source doesn't say


# ---------------------------------------------------------------- load data
import argparse
_args = argparse.ArgumentParser()
_args.add_argument("--test", default="data/real_test_set.csv",
                   help="test file, e.g. data/heldout_test_set.csv for the independent held-out set")
TEST_FILE = _args.parse_args().test
test = pd.read_csv(TEST_FILE)
if "source_type" not in test:
    test["source_type"] = "heldout"
if "source" not in test:
    test["source"] = ""
synthetic = pd.read_csv("data/listings.csv")


def num(v, default):
    return default if pd.isna(v) or v == "" else float(v)


def row_flags(r) -> dict:
    price, face = num(r.price, 0), num(r.face_value, 0)
    channel = channel_info(r.platform, r.link if isinstance(r.link, str) else "", check_age=False)
    return all_flags(r.text, num(r.account_age_days, NEUTRAL_AGE), num(r.followers, NEUTRAL_FOLLOWERS),
                     price / face if price and face else 1.0, int(r.has_seat_details),
                     int(r.sudden_seller), None, channel)


flags = pd.DataFrame([row_flags(r) for r in test.itertuples()])
y = test["label"].values
preds, scores = {}, {}

# ---------------------------------------------------------------- 1. rules only
meaningful = [c for c in FLAG_COLUMNS if c not in WEAK_FLAGS and c not in PROTECTIVE]
n_meaningful = flags[meaningful].sum(axis=1)
hard = flags[list(HARD_STOPS)].max(axis=1)
preds["Rules only"] = ((hard == 1) | (n_meaningful >= 2)).astype(int).values
scores["Rules only"] = np.minimum(n_meaningful / 4, 1).values

# ---------------------------------------------------------------- 2. text model only
text_model = Pipeline([("words", TfidfVectorizer(ngram_range=(1, 2), min_df=2)),
                       ("clf", LogisticRegression(max_iter=1000, class_weight="balanced"))])
text_model.fit(synthetic["text"], synthetic["label"])
scores["Text model only"] = text_model.predict_proba(test["text"])[:, 1]
preds["Text model only"] = (scores["Text model only"] >= THRESHOLD).astype(int)

# ---------------------------------------------------------------- 4. hybrid (the real checker)
bundle = joblib.load("model/scam_model.joblib")
hybrid = bundle["model"]
X = pd.concat([test[["text"]], flags], axis=1)[["text"] + FLAG_COLUMNS]
scores["Hybrid (our checker)"] = hybrid.predict_proba(X)[:, 1]
preds["Hybrid (our checker)"] = ((scores["Hybrid (our checker)"] >= THRESHOLD) | (hard == 1)).astype(int).values

# ---------------------------------------------------------------- 3. AI only (optional)
LLM_PROMPT = """You are checking a resale ticket listing for fraud.
Where it is being sold: {platform}{link}
Listing / messages:
\"\"\"{text}\"\"\"
Reply with JSON only: {{"scam_probability": <number 0 to 1>, "reason": "<max 15 words>"}}"""


def llm_only() -> np.ndarray | None:
    if not os.getenv("GEMINI_API_KEY"):
        print("AI-only arm skipped (no GEMINI_API_KEY). Add it to .env and re-run to include it.")
        return None
    from explainer import GEMINI_MODELS
    from google import genai
    client = genai.Client()
    # saved answers, so re-runs don't spend your free quota again
    cache_file = OUT / ("llm_cache.json" if TEST_FILE.endswith("real_test_set.csv")
                        else f"llm_cache_{Path(TEST_FILE).stem}.json")
    cache = json.loads(cache_file.read_text()) if cache_file.exists() else {}
    probs, failed = [], []
    for r in test.itertuples():
        if r.id not in cache:
            link = f" (link: {r.link})" if isinstance(r.link, str) and r.link else ""
            prompt = LLM_PROMPT.format(platform=r.platform, link=link, text=r.text)
            answer = None
            for model in GEMINI_MODELS:
                try:
                    resp = client.models.generate_content(
                        model=model, contents=prompt,
                        config=genai.types.GenerateContentConfig(max_output_tokens=2048,
                                                                 response_mime_type="application/json"))
                    answer = json.loads(resp.text)
                    answer["model"] = model
                    break
                except Exception as e:
                    last = e
                    continue
            if answer is None:  # not cached, so a re-run will try this listing again
                failed.append(r.id)
                print(f"  {r.id}: AI call failed ({last.__class__.__name__})")
                probs.append(np.nan)
                continue
            cache[r.id] = answer
            cache_file.write_text(json.dumps(cache, indent=1))
        probs.append(float(cache[r.id]["scam_probability"]))
    if failed:
        print(f"\nAI-only arm INCOMPLETE: {len(failed)} listings failed (often the free-tier limit). "
              "Wait a while and run evaluate_real.py again - finished answers are saved, so only "
              "the missing ones are retried. The AI-only row is left out until all succeed.")
        return None
    return np.array(probs)


ai = llm_only()
if ai is not None:
    scores["AI only (Gemini)"] = ai
    preds["AI only (Gemini)"] = (ai >= 0.5).astype(int)

ORDER = [k for k in ["Rules only", "Text model only", "AI only (Gemini)", "Hybrid (our checker)"] if k in preds]


# ---------------------------------------------------------------- metrics
def metrics(y_true, y_pred) -> dict:
    tp = int(((y_pred == 1) & (y_true == 1)).sum()); fp = int(((y_pred == 1) & (y_true == 0)).sum())
    tn = int(((y_pred == 0) & (y_true == 0)).sum()); fn = int(((y_pred == 0) & (y_true == 1)).sum())
    prec = tp / (tp + fp) if tp + fp else 0.0
    rec = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * prec * rec / (prec + rec) if prec + rec else 0.0
    return dict(scams_caught=f"{tp}/{tp + fn}", recall=rec, precision=prec, f1=f1,
                false_alarms=f"{fp}/{fp + tn}", false_alarm_rate=fp / (fp + tn), accuracy=(tp + tn) / len(y_true))


summary = pd.DataFrame({k: metrics(y, preds[k]) for k in ORDER}).T
by_type = pd.DataFrame({k: {t: preds[k][(test.source_type == t).values & (y == 1)].mean()
                            for t in ["verbatim", "reconstructed"]} for k in ORDER}).T

# synthetic-vs-real: how much does the hybrid drop outside its own data?
# (rebuilds the same features and the same train/test split as step2_train_model.py)
def _tri(v):
    return None if v == -1 else bool(v)


syn_rows = [all_flags(r.text, r.account_age_days, r.followers, r.price_ratio, r.has_seat_details,
                      r.sudden_seller,
                      dict(found=_tri(r.event_found), details_match=_tri(r.event_details_match),
                           high_demand=bool(r.event_high_demand), before_on_sale=bool(r.event_before_on_sale),
                           app_only=bool(r.event_app_only)),
                      dict(category=None if r.ch_category == "none" else r.ch_category,
                           lookalike=bool(r.ch_lookalike), new_domain=bool(r.ch_new_domain),
                           insecure_or_short=bool(r.ch_insecure_or_short)))
            for r in synthetic.itertuples()]
syn_X = pd.concat([synthetic[["text"]], pd.DataFrame(syn_rows)], axis=1)[["text"] + FLAG_COLUMNS]
_, X_syn_test, _, y_syn_test = train_test_split(syn_X, synthetic["label"], test_size=0.25,
                                                stratify=synthetic["label"], random_state=42)
syn_hard = X_syn_test[list(HARD_STOPS)].max(axis=1).values
syn_pred = ((hybrid.predict_proba(X_syn_test)[:, 1] >= THRESHOLD) | (syn_hard == 1)).astype(int)
syn = metrics(y_syn_test.values, syn_pred)

# ---------------------------------------------------------------- per-listing table
detail = test[["id", "label", "source_type", "source", "platform", "text"]].copy()
for k in ORDER:
    detail[k] = preds[k]
detail["hybrid_score"] = scores["Hybrid (our checker)"].round(2)
detail["flags_fired"] = flags.apply(lambda f: ", ".join(c for c in FLAG_COLUMNS if f[c]), axis=1)
detail.to_csv(OUT / "real_eval_predictions.csv", index=False)
summary.to_csv(OUT / "real_eval_summary.csv")

# ---------------------------------------------------------------- chart
fig, ax = plt.subplots(figsize=(8, 4.2))
colors = {"Scams caught (recall)": "#2a78d6", "Alerts correct (precision)": "#eb6834", "F1": "#1baf7a"}
vals = {"Scams caught (recall)": summary["recall"], "Alerts correct (precision)": summary["precision"],
        "F1": summary["f1"]}
w, xs = 0.26, np.arange(len(ORDER))
for i, (name, v) in enumerate(vals.items()):
    bars = ax.bar(xs + (i - 1) * w, v.astype(float), w - 0.02, color=colors[name], label=name)
    for b, val in zip(bars, v.astype(float)):
        ax.text(b.get_x() + b.get_width() / 2, val + 0.015, f"{val:.0%}", ha="center", va="bottom",
                fontsize=8, color="#333333")
ax.set_xticks(xs, ORDER)
ax.set_ylim(0, 1.12)
ax.set_yticks([0, .25, .5, .75, 1], ["0%", "25%", "50%", "75%", "100%"])
ax.spines[["top", "right"]].set_visible(False)
ax.grid(axis="y", color="#e6e6e6", linewidth=0.8)
ax.set_axisbelow(True)
ax.legend(frameon=False, ncol=3, loc="upper left", fontsize=8)
ax.set_title(f"Real-world test set ({int(y.sum())} scams, {int((1 - y).sum())} genuine)", fontsize=11, loc="left")
plt.tight_layout()
plt.savefig(OUT / "real_eval_chart.png", dpi=200)
plt.close()


# ---------------------------------------------------------------- report
def pct(v):
    return f"{float(v):.0%}"


h = "Hybrid (our checker)"
misses = detail[(detail.label == 1) & (detail[h] == 0)]
false_alarms = detail[(detail.label == 0) & (detail[h] == 1)]
lines = [
    "# Real-world evaluation",
    "",
    f"Test file: `{TEST_FILE}`",
    "",
    f"Test set: {len(test)} listings ({int(y.sum())} scams from published sources: "
    f"{int((test.source_type == 'verbatim').sum())} verbatim, {int((test.source_type == 'reconstructed').sum())} "
    f"reconstructed from documented UK cases; {int((1 - y).sum())} constructed genuine listings). "
    "None were used in training.",
    "",
    "| Approach | Scams caught | Recall | Precision | F1 | False alarms |",
    "|---|---|---|---|---|---|",
    *[f"| {k} | {r.scams_caught} | {pct(r.recall)} | {pct(r.precision)} | {pct(r.f1)} | {r.false_alarms} |"
      for k, r in summary.iterrows()],
    "",
    f"For comparison, the hybrid on its own synthetic test set: recall {pct(syn['recall'])}, "
    f"precision {pct(syn['precision'])}, F1 {pct(syn['f1'])}.",
    "",
    *(["Recall by source type:", "",
       "| Approach | Verbatim quotes | Reconstructed cases |", "|---|---|---|",
       *[f"| {k} | {pct(r.verbatim)} | {pct(r.reconstructed)} |" for k, r in by_type.iterrows()]]
      if (test.source_type == "verbatim").any() else []),
    "",
    f"## Scams the hybrid missed ({len(misses)})",
    "",
    *[f"- **{r.id}** ({r.source}; score {r.hybrid_score}): \"{r.text[:140]}\" - flags: {r.flags_fired or 'none'}"
      for r in misses.itertuples()],
    "",
    f"## Genuine listings the hybrid wrongly flagged ({len(false_alarms)})",
    "",
    *[f"- **{r.id}** ({r.platform}; score {r.hybrid_score}): \"{r.text[:140]}\" - flags: {r.flags_fired or 'none'}"
      for r in false_alarms.itertuples()],
    "",
    "## Version history on the development set",
    "- v2.2 (before fixes): Rules 30/40 (F1 79%), Text model 38/40 (F1 69%), AI only 39/40 (F1 93%), "
    "Hybrid 19/40 (F1 62%, 2 false alarms). Cause: synthetic scams always carried several obvious flags, "
    "while real ones are short and vague; several real tactics had no rule.",
    "- v2.3: widened rules (pm me, up front, third-party accounts, gift-card wording, presale-code sales, "
    "like-and-comment bait, screenshots as proof, vague posts) and made the training data more realistic "
    "(short vague scams, short genuine posts, missing prices). Results above.",
    f"- Caution: v2.3 was built after studying this set's mistakes, so results on {TEST_FILE} "
    "are optimistic unless it is a fresh held-out set. The fair test is a new set collected independently "
    "(see data/heldout_template.csv).",
    "",
    "## Method notes",
    "- Thresholds and the rules-only decision rule were fixed before running the test.",
    "- Event check and website-age lookup were switched off for all approaches (see docstring).",
    "- Unknown seller details were treated as neutral (365-day account, 200 followers).",
    "- Limitations: small test set; genuine listings are constructed, not collected; reconstructed scams were "
    "written by the developer, which risks unconscious bias towards the rules. Next step: an independent, "
    "real-world labelled set via a bank or platform partner.",
]
(OUT / "real_eval_report.md").write_text("\n".join(lines), encoding="utf-8")

print(summary[["scams_caught", "recall", "precision", "f1", "false_alarms"]].to_string())
print(f"\nHybrid on synthetic test: recall {pct(syn['recall'])}, precision {pct(syn['precision'])}")
print("\nSaved outputs/real_eval_report.md, real_eval_chart.png, real_eval_summary.csv, real_eval_predictions.csv")

```

## requirements.txt
```text
pandas
scikit-learn
matplotlib
joblib
streamlit
requests          # Ticketmaster event lookup
google-genai      # Gemini explainer (free tier)
anthropic         # optional alternative explainer
python-dotenv     # loads API keys from .env

```

